"""Summarise repeated single random splits (train.py with seeds 0..N-1): mean ± sd and 95% CI over the repeats.

Each repeat is an independent stratified 80/10/10 split (Géron ch. 2): the model is tuned on the 10% validation
set and scored once on the 10% test set. Over the repeats this reports, per (feature set, model):
  - mean and standard deviation of test R2, RMSE and MAE (Géron ch. 2 / App. B: report mean and std)
  - 95% confidence interval of the mean, from the t-distribution (Géron ch. 2, scipy.stats.t.interval)
  - min / max test R2 across repeats
Only complete repeat sets are reported: a (feature set, model) is skipped until all seeds are present.

Example:
    python src/repeated_report.py                 # seeds 0..9
    python src/repeated_report.py --n-seeds 5
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path(__file__).resolve().parents[1] / "results"
NAMES = {"xgb": "XGBoost", "rf": "Random Forest", "svm": "SVM (SVR)", "knn": "kNN"}
N_FEATURES = {"C": 13, "B": 11, "E6": 6, "E5": 5, "E4": 4}


def ci95(x):
    """95% confidence interval of the mean of x (t-distribution, as in Géron ch. 2)."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan, np.nan
    return stats.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=stats.sem(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=0)
    args = ap.parse_args()
    seeds = list(range(args.first_seed, args.first_seed + args.n_seeds))

    rows = []
    for p in sorted(RESULTS.glob("features_*/*_seed*/metrics.json")):
        m = json.loads(p.read_text())
        if m["seed"] in seeds:
            rows.append(m)
    if not rows:
        print(f"No runs with seeds {seeds[0]}..{seeds[-1]} under {RESULTS}")
        return
    runs = pd.DataFrame(rows)
    runs.to_csv(RESULTS / "repeated_all_runs.csv", index=False)

    out = []
    for (feats, model), d in runs.groupby(["features", "model"]):
        if sorted(d.seed) != seeds:
            print(f"skipping {feats}/{model}: {len(d)} of {len(seeds)} repeats finished")
            continue
        r = {"Features": feats, "# features": N_FEATURES.get(feats), "Model": NAMES.get(model, model),
             "Repeats": len(d)}
        for col, name in [("test_R2", "R²"), ("test_RMSE", "RMSE"), ("test_MAE", "MAE")]:
            lo, hi = ci95(d[col])
            r.update({f"{name} mean": d[col].mean(), f"{name} sd": d[col].std(ddof=1),
                      f"{name} 95% CI low": lo, f"{name} 95% CI high": hi})
        r.update({"R² min": d.test_R2.min(), "R² max": d.test_R2.max(), "Train R² mean": d.train_R2.mean()})
        out.append(r)
    if not out:
        return
    s = pd.DataFrame(out).sort_values(["# features", "R² mean"], ascending=[False, False])
    s.round(4).to_csv(RESULTS / "repeated_summary.csv", index=False)

    show = s[["Features", "# features", "Model", "Repeats"]].copy()
    show["R² (mean ± sd)"] = s["R² mean"].map("{:.3f}".format) + " ± " + s["R² sd"].map("{:.3f}".format)
    show["R² 95% CI"] = ("[" + s["R² 95% CI low"].map("{:.3f}".format) + ", "
                         + s["R² 95% CI high"].map("{:.3f}".format) + "]")
    show["R² range"] = s["R² min"].map("{:.2f}".format) + " – " + s["R² max"].map("{:.2f}".format)
    show["RMSE"] = s["RMSE mean"].map("{:.2f}".format) + " ± " + s["RMSE sd"].map("{:.2f}".format)
    show["MAE"] = s["MAE mean"].map("{:.2f}".format) + " ± " + s["MAE sd"].map("{:.2f}".format)
    pd.set_option("display.width", 250)
    print(f"Repeated random split (seeds {seeds[0]}..{seeds[-1]}): test-set metrics, mean ± sd over repeats")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
