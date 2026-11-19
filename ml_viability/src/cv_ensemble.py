"""Consensus and stacking ensembles for one (features, seed, outer fold), after cv_train.py has run.

The top-3 base models are chosen by their *inner-CV* RMSE for this fold (test data is never used to choose).
- consensus: plain average of their test predictions (Yan et al. 2019)
- stack: base models refit inside the outer-training data; a non-negative linear blender is trained on
  their inner out-of-fold predictions

Example:
    python src/cv_ensemble.py --features C --seed 0 --fold 0
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression

from cv_models import CV_MODELS, build, params_from_best
from cv_train import N_INNER, N_OUTER, make_folds, metrics, outer_splits, results_dir
from data import DEFAULT_DATA, FEATURE_SETS, load_data, xy

TOP_K = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="kfold", choices=["kfold", "holdout"])
    ap.add_argument("--features", default="C", choices=list(FEATURE_SETS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, required=True, choices=range(N_OUTER))
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = ap.parse_args()
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    base = results_dir(args.scheme) / f"features_{args.features}"
    run = lambda m: base / m / f"seed{args.seed}" / f"fold{args.fold}"

    done = {m: json.loads((run(m) / "metrics.json").read_text()) for m in CV_MODELS
            if (run(m) / "metrics.json").exists()}
    if len(done) < TOP_K:
        raise SystemExit(f"only {len(done)} base models finished for this fold; need {TOP_K}")
    top = sorted(done, key=lambda m: done[m]["inner_cv_RMSE"])[:TOP_K]
    print("top models by inner-CV RMSE:", {m: round(done[m]["inner_cv_RMSE"], 2) for m in top})

    df = load_data(args.data)
    X, y = xy(df, args.features)
    num_cols, cat_cols = FEATURE_SETS[args.features]
    tr, te = outer_splits(args.scheme, y, args.seed)[args.fold]
    preds = {m: pd.read_csv(run(m) / "predictions.csv") for m in top}
    ref = preds[top[0]]
    assert all((p["row"].to_numpy() == te).all() for p in preds.values()), "fold rows differ between models"

    # Consensus: average of the top-3 test predictions
    p_cons = np.mean([preds[m]["y_pred"].to_numpy() for m in top], axis=0)

    # Stacking: blender trained on inner out-of-fold predictions (same inner folds as tuning)
    inner = make_folds(y[tr], N_INNER, args.seed + 100)
    estimators = [(m, build(m, params_from_best(m, json.loads((run(m) / "best_params.json").read_text())),
                            num_cols, cat_cols, args.seed, n_jobs)) for m in top]
    stack = StackingRegressor(estimators, final_estimator=LinearRegression(positive=True), cv=inner,
                              n_jobs=1).fit(X.iloc[tr], y[tr])
    p_stack = stack.predict(X.iloc[te])
    weights = dict(zip(top, np.round(stack.final_estimator_.coef_, 3).tolist()))
    print("stack blender weights:", weights)

    in_ad = ref["in_AD"].to_numpy()
    for name, p in [("consensus", p_cons), ("stack", p_stack)]:
        out = run(name)
        out.mkdir(parents=True, exist_ok=True)
        res = {"scheme": args.scheme, "features": args.features, "model": name, "seed": args.seed, "fold": args.fold, "members": top,
               **{f"test_{k}": v for k, v in metrics(y[te], p).items()},
               "test_frac_in_AD": float(in_ad.mean())}
        if name == "stack":
            res["blender_weights"] = weights
        for tag, mask in [("inAD", in_ad), ("outAD", ~in_ad)]:
            if mask.sum() >= 5:
                res.update({f"test_{tag}_{k}": v for k, v in metrics(y[te][mask], p[mask]).items()})
        (out / "metrics.json").write_text(json.dumps(res, indent=2))
        ref.assign(y_pred=p).to_csv(out / "predictions.csv", index=False)
        print(name, json.dumps({k: round(v, 3) for k, v in res.items() if k.startswith("test_") and "_n" not in k}))


if __name__ == "__main__":
    main()
