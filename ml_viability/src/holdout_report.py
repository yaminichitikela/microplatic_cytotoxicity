"""Summarise the repeated random split (cv_train.py --scheme holdout): one table in the same format as the CV table.

Each repeat (seed) is one stratified random 80/20 train/test split. Models are tuned exactly as in the nested CV
(Optuna on 3-fold CV inside the training rows) and scored once on that repeat's 20% test set.
Same calculation as the nested-CV report (cv_report.py), per model:
  R2, RMSE, MAE, CCC  = computed on each repeat's held-out predictions, then mean over the repeats
  R2 95% CI           = bootstrap over papers (1,000 resamples of whole papers) on each repeat's held-out
                        predictions, averaged over the repeats -- identical to cv_report.paper_bootstrap
  R2 sd               = standard deviation over the repeats
A model is reported only when all requested repeats are finished.

Example:
    python src/holdout_report.py                 # seeds 0..4
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from cv_train import RESULTS_HOLDOUT, metrics
from data import GROUP

N_BOOT = 1000


def paper_bootstrap(p, n_boot=N_BOOT, seed=0):
    """95% CI of R2 and RMSE, resampling whole papers (same as cv_report.paper_bootstrap)."""
    rng = np.random.default_rng(seed)
    by_paper = {g: d for g, d in p.groupby(GROUP)}
    papers = np.array(list(by_paper))
    r2, rmse = [], []
    for _ in range(n_boot):
        d = pd.concat([by_paper[g] for g in rng.choice(papers, len(papers), replace=True)])
        m = metrics(d.y_true.to_numpy(), d.y_pred.to_numpy())
        r2.append(m["R2"]); rmse.append(m["RMSE"])
    return np.percentile(r2, [2.5, 97.5]), np.percentile(rmse, [2.5, 97.5])

NAMES = {"consensus": "Consensus (average of top 3)", "stack": "Stacking (top 3)", "et": "Extra-Trees",
         "xgb": "XGBoost", "rf": "Random Forest", "mlp": "MLP (neural network)", "svr": "SVR", "knn": "kNN",
         "enet": "ElasticNet", "pls": "PLS"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="C")
    ap.add_argument("--n-seeds", type=int, default=5)
    args = ap.parse_args()
    seeds = list(range(args.n_seeds))
    base = RESULTS_HOLDOUT / f"features_{args.features}"

    rows, per_run = [], []
    for model_dir in sorted(p for p in base.glob("*") if p.is_dir()):
        model = model_dir.name
        files = [model_dir / f"seed{s}" / "fold0" / "predictions.csv" for s in seeds]
        if not all(f.exists() for f in files):
            print(f"skipping {model}: {sum(f.exists() for f in files)} of {len(seeds)} repeats finished")
            continue
        m = []
        for s, f in zip(seeds, files):
            p = pd.read_csv(f)
            r = metrics(p.y_true.to_numpy(), p.y_pred.to_numpy())
            (r["R2_lo"], r["R2_hi"]), (r["RMSE_lo"], r["RMSE_hi"]) = paper_bootstrap(p)
            m.append(r)
            per_run.append({"model": model, "seed": s, **r})
        m = pd.DataFrame(m)
        rows.append({"Model": NAMES.get(model, model), "R²": m.R2.mean(), "R² 95% CI (low)": m.R2_lo.mean(),
                     "R² 95% CI (high)": m.R2_hi.mean(), "RMSE": m.RMSE.mean(), "MAE": m.MAE.mean(), "CCC": m.CCC.mean(),
                     "R² sd": m.R2.std(ddof=1), "R² min": m.R2.min(), "R² max": m.R2.max(), "Repeats": len(m),
                     "Test rows per repeat": int(m.n.iloc[0])})
    if not rows:
        raise SystemExit(f"No complete models under {base}")
    pd.DataFrame(per_run).round(4).to_csv(RESULTS_HOLDOUT / "per_repeat.csv", index=False)
    t = pd.DataFrame(rows).sort_values("R²", ascending=False)
    t.round(4).to_csv(RESULTS_HOLDOUT / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"Repeated random split ({len(seeds)} repeats, 80/20, features {args.features}): mean over repeats")
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
