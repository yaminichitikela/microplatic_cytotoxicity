import argparse
import json
import os
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data import DEFAULT_DATA, FEATURE_SETS, GROUP, load_data, random_split, xy
from models import MODELS, build, suggest

RESULTS = Path(__file__).resolve().parents[1] / "results"
optuna.logging.set_verbosity(optuna.logging.WARNING)


def metrics(model, X, y):
    p = model.predict(X)
    return {"R2": r2_score(y, p), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAE": mean_absolute_error(y, p)}, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--features", default="C", choices=list(FEATURE_SETS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--no-save-model", action="store_true",
                    help="don't write model.joblib (saves disk space for many repeated runs)")
    args = ap.parse_args()
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

    out = RESULTS / f"features_{args.features}" / f"{args.model}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    train, val, test = random_split(df, args.seed)
    print(f"random split: train={len(train)} val={len(val)} test={len(test)} rows")
    Xtr, ytr = xy(train, args.features)
    Xva, yva = xy(val, args.features)
    Xte, yte = xy(test, args.features)
    num_cols, cat_cols = FEATURE_SETS[args.features]
    make = lambda params: build(args.model, params, num_cols, cat_cols, args.seed, n_jobs)

    def objective(trial):
        model = make(suggest(trial, args.model)).fit(Xtr, ytr)
        return metrics(model, Xva, yva)[0]["RMSE"]

    # One SQLite file per run: array tasks never write to the same DB, and a killed job can resume
    study = optuna.create_study(
        study_name=f"{args.model}_{args.features}_seed{args.seed}",
        storage=f"sqlite:///{out / 'optuna.db'}",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    remaining = args.n_trials - len([t for t in study.trials if t.state.is_finished()])
    t0 = time.time()
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    print(f"tuning done in {time.time() - t0:.0f}s; best val RMSE = {study.best_value:.3f}")

    # Refit on the training set with the best params; val was used for selection, test is untouched until now
    best = make(study.best_params).fit(Xtr, ytr)
    res = {"model": args.model, "features": args.features, "seed": args.seed,
           "n_trials": len(study.trials), "n_train": len(train), "n_val": len(val), "n_test": len(test)}
    preds = {}
    for name, X, y in [("train", Xtr, ytr), ("val", Xva, yva), ("test", Xte, yte)]:
        m, preds[name] = metrics(best, X, y)
        res.update({f"{name}_{k}": v for k, v in m.items()})
    print(json.dumps(res, indent=2))

    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    (out / "best_params.json").write_text(json.dumps(study.best_params, indent=2))
    study.trials_dataframe().to_csv(out / "trials.csv", index=False)
    pd.concat([
        pd.DataFrame({"split": s, GROUP: d[GROUP].to_numpy(), "y_true": y, "y_pred": preds[s]})
        for s, d, y in [("train", train, ytr), ("val", val, yva), ("test", test, yte)]
    ]).to_csv(out / "predictions.csv", index=False)
    if not args.no_save_model:
        joblib.dump(best, out / "model.joblib")

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(ytr, preds["train"], s=8, color="#9ec5f4", label=f"train  R²={res['train_R2']:.2f}")
    ax.scatter(yte, preds["test"], s=14, color="#2a78d6", label=f"test   R²={res['test_R2']:.2f}")
    lim = [min(ytr.min(), yte.min()) - 5, max(ytr.max(), yte.max()) + 5]
    ax.plot(lim, lim, color="#6b7280", lw=1, ls="--")
    ax.set(xlim=lim, ylim=lim, xlabel="Observed viability (%)", ylabel="Predicted viability (%)",
           title=f"{args.model.upper()} · features {args.features} · single random split · seed {args.seed}")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "pred_vs_obs.png", dpi=150)


if __name__ == "__main__":
    main()
