"""Nested cross-validation for cell-viability regression: one (features, model, seed, outer fold) per run.

Outer loop  : --scheme kfold   (default): 5 stratified folds of rows (similar viability distribution in each)
              --scheme holdout          : repeated random split; each seed = one stratified 80/20 train/test split
                                          (use --fold 0); results go to results_holdout/
Inner loop  : Optuna tunes on the outer-training data only, scored by 3-fold CV.
Evaluation  : the tuned model is refit on the outer-training data and scored once on the outer test fold.
Extras      : applicability domain (kNN distance) per test row; --scramble for the y-scrambling check.

    python src/cv_train.py --features C --model rf --seed 0 --fold 0 --n-trials 50
    python src/cv_train.py --scheme holdout --features C --model rf --seed 0 --fold 0 --n-trials 50
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors

from cv_models import CV_MODELS, build, params_from_best, suggest
from data import DEFAULT_DATA, FEATURE_SETS, GROUP, load_data, xy
from models import preprocessor

RESULTS = Path(__file__).resolve().parents[1] / "results_cv"
RESULTS_HOLDOUT = Path(__file__).resolve().parents[1] / "results_holdout"
N_OUTER, N_INNER = 5, 3
HOLDOUT_TEST = 0.2
optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_folds(y, n_folds, seed, n_bins=10):
    """Stratified K-fold on rows (Géron ch. 2): each fold gets a similar viability distribution (deciles)."""
    bins = pd.qcut(y, n_bins, labels=False, duplicates="drop")
    return list(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), bins))


def holdout_split(y, seed, test_size=HOLDOUT_TEST, n_bins=10):
    """One stratified random train/test split of rows (Géron ch. 2); repeated with different seeds."""
    bins = pd.qcut(y, n_bins, labels=False, duplicates="drop")
    tr, te = train_test_split(np.arange(len(y)), test_size=test_size, random_state=seed, stratify=bins)
    return [(np.sort(tr), np.sort(te))]


def outer_splits(scheme, y, seed):
    """Outer train/test splits: 5 stratified folds (kfold) or one random 80/20 split (holdout)."""
    return make_folds(y, N_OUTER, seed) if scheme == "kfold" else holdout_split(y, seed)


def results_dir(scheme):
    return RESULTS if scheme == "kfold" else RESULTS_HOLDOUT


def ccc(y, p):
    """Lin's concordance correlation coefficient (used in QSAR validation, e.g. Liu et al. 2024)."""
    my, mp = y.mean(), p.mean()
    return float(2 * np.mean((y - my) * (p - mp)) / (y.var() + p.var() + (my - mp) ** 2))


def metrics(y, p):
    return {"R2": r2_score(y, p), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAE": mean_absolute_error(y, p), "CCC": ccc(y, p), "n": int(len(y))}


def applicability_domain(Xtr, Xte, num_cols, cat_cols, k=5, pct=95):
    """Mean distance to the k nearest training rows (standardised numeric + one-hot space).

    A test row is inside the domain if its distance <= the `pct` percentile of the training rows'
    own k-NN distances (each training row left out of its own neighbour search).
    """
    prep = preprocessor(num_cols, cat_cols, scale=True).fit(Xtr)
    A, B = prep.transform(Xtr), prep.transform(Xte)
    d_test = NearestNeighbors(n_neighbors=k).fit(A).kneighbors(B)[0].mean(axis=1)
    d_ref = NearestNeighbors(n_neighbors=k + 1).fit(A).kneighbors(A)[0][:, 1:].mean(axis=1)
    thr = float(np.percentile(d_ref, pct))
    return d_test, d_test <= thr, thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="kfold", choices=["kfold", "holdout"],
                    help="kfold: 5-fold nested CV; holdout: repeated random 80/20 split (one split per seed)")
    ap.add_argument("--features", default="C", choices=list(FEATURE_SETS))
    ap.add_argument("--model", required=True, choices=CV_MODELS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, required=True, choices=range(N_OUTER))
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--scramble", action="store_true", help="y-scrambling: shuffle training targets")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = ap.parse_args()
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

    name = args.model + ("_scrambled" if args.scramble else "")
    out = results_dir(args.scheme) / f"features_{args.features}" / name / f"seed{args.seed}" / f"fold{args.fold}"
    out.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    X, y = xy(df, args.features)
    groups = df[GROUP].to_numpy()
    num_cols, cat_cols = FEATURE_SETS[args.features]

    splits = outer_splits(args.scheme, y, args.seed)
    if args.fold >= len(splits):
        raise SystemExit(f"--scheme {args.scheme} has {len(splits)} outer split(s); use --fold 0..{len(splits) - 1}")
    tr, te = splits[args.fold]
    Xtr, Xte = X.iloc[tr], X.iloc[te]
    ytr, yte = y[tr].copy(), y[te]
    print(f"fold {args.fold}/seed {args.seed}: train {len(tr)} rows, test {len(te)} rows")
    if args.scramble:
        ytr = np.random.default_rng(1000 + args.seed).permutation(ytr)

    inner = make_folds(ytr, N_INNER, args.seed + 100)

    def objective(trial):
        params = suggest(trial, args.model)
        rmse = []
        for itr, iva in inner:
            m = build(args.model, params, num_cols, cat_cols, args.seed, n_jobs).fit(Xtr.iloc[itr], ytr[itr])
            rmse.append(np.sqrt(mean_squared_error(ytr[iva], m.predict(Xtr.iloc[iva]))))
        return float(np.mean(rmse))

    study = optuna.create_study(study_name=f"{args.features}_{name}_seed{args.seed}_fold{args.fold}",
                                storage=f"sqlite:///{out / 'optuna.db'}", load_if_exists=True,
                                direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed))
    remaining = args.n_trials - len([t for t in study.trials if t.state.is_finished()])
    t0 = time.time()
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    print(f"inner tuning: {time.time() - t0:.0f}s, best inner-CV RMSE {study.best_value:.2f}")

    best = build(args.model, params_from_best(args.model, study.best_params), num_cols, cat_cols,
                 args.seed, n_jobs).fit(Xtr, ytr)
    p_tr, p_te = best.predict(Xtr), best.predict(Xte)
    d_te, in_ad, thr = applicability_domain(Xtr, Xte, num_cols, cat_cols)

    res = {"scheme": args.scheme, "features": args.features, "model": name, "seed": args.seed, "fold": args.fold,
           "n_trials": len(study.trials), "inner_cv_RMSE": study.best_value,
           **{f"train_{k}": v for k, v in metrics(ytr, p_tr).items()},
           **{f"test_{k}": v for k, v in metrics(yte, p_te).items()},
           "ad_threshold": thr, "test_frac_in_AD": float(in_ad.mean())}
    for tag, mask in [("inAD", in_ad), ("outAD", ~in_ad)]:
        if mask.sum() >= 5:
            res.update({f"test_{tag}_{k}": v for k, v in metrics(yte[mask], p_te[mask]).items()})
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}))

    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    (out / "best_params.json").write_text(json.dumps(study.best_params, indent=2))
    pd.DataFrame({"row": te, GROUP: groups[te], "y_true": yte, "y_pred": p_te,
                  "ad_distance": d_te, "in_AD": in_ad}).to_csv(out / "predictions.csv", index=False)


if __name__ == "__main__":
    main()
