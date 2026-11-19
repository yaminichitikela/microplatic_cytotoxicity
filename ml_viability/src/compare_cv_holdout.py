"""Compare the methods in the two settings: nested 5-fold CV vs repeated random split (5 x 80/20), all 13 features.

Both settings use the same calculation (cv_report.py / holdout_report.py):
  R2, RMSE, MAE, CCC = computed on the held-out predictions of each repeat (seed), then mean over the repeats
  95% CI             = bootstrap over papers (1,000 resamples) per repeat, averaged over the repeats
ElasticNet and PLS are left out (not run in the repeated random split).

Outputs (best_results/8_comparison/):
  cv_vs_repeated_split.png   R2 and RMSE per method, both settings, with 95% CI error bars
  cv_vs_repeated_split.csv   the values plotted

Example:
    python src/compare_cv_holdout.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "best_results" / "8_comparison"
NAMES = {"consensus": "Consensus (average of top 3)", "stack": "Stacking (top 3)", "et": "Extra-Trees",
         "xgb": "XGBoost", "rf": "Random Forest", "mlp": "MLP (neural network)", "svr": "SVR", "knn": "kNN"}
SHORT = {"Consensus (average of top 3)": "Consensus", "Stacking (top 3)": "Stacking", "MLP (neural network)": "MLP"}


def main():
    cv = pd.read_csv(ROOT / "results_cv" / "summary.csv")
    cv = cv[(cv.features == "C") & cv.model.isin(NAMES)].assign(Model=lambda d: d.model.map(NAMES))
    cv = cv.set_index("Model")[["R2_mean", "R2_lo_mean", "R2_hi_mean", "RMSE_mean", "RMSE_lo_mean", "RMSE_hi_mean",
                                "MAE_mean", "CCC_mean"]]
    cv.columns = ["R2", "R2_lo", "R2_hi", "RMSE", "RMSE_lo", "RMSE_hi", "MAE", "CCC"]

    runs = pd.read_csv(ROOT / "results_holdout" / "per_repeat.csv")
    ho = runs[runs.model.isin(NAMES)].groupby("model")[["R2", "R2_lo", "R2_hi", "RMSE", "RMSE_lo", "RMSE_hi",
                                                        "MAE", "CCC"]].mean()
    ho.index = ho.index.map(NAMES)

    order = cv.sort_values("R2", ascending=False).index
    t = pd.concat({"CV": cv.loc[order], "Repeated split": ho.loc[order]}, axis=1)
    flat = t.copy()
    flat.columns = [f"{m} ({s})" for s, m in t.columns]
    OUT.mkdir(parents=True, exist_ok=True)
    flat.round(4).to_csv(OUT / "cv_vs_repeated_split.csv")

    x, w = np.arange(len(order)), 0.38
    style = {"CV": ("#1F4E78", "Nested 5-fold CV (3 repeats)"), "Repeated split": ("#E69F00", "Repeated random split (5 × 80/20)")}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, (m, label) in zip(axes, [("R2", "R²"), ("RMSE", "RMSE (viability %)")]):
        for i, (s, (color, name)) in enumerate(style.items()):
            d = t[s]
            err = np.vstack([d[m] - d[f"{m}_lo"], d[f"{m}_hi"] - d[m]])
            bars = ax.bar(x + (i - 0.5) * w, d[m], w, yerr=err, capsize=3, color=color, label=name,
                          error_kw={"elinewidth": 1, "ecolor": "#333333"})
            ax.bar_label(bars, labels=[f"{v:.3f}" if m == "R2" else f"{v:.2f}" for v in d[m]],
                         fontsize=7.5, label_type="center", color="white", rotation=90)
        ax.set_xticks(x, [SHORT.get(n, n) for n in order], rotation=30, ha="right")
        ax.set_ylabel(label)
        ax.set_ylim(0, 1 if m == "R2" else None)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_title(f"{label}: {'higher' if m == 'R2' else 'lower'} is better", fontsize=11)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Cell viability, all 13 features: nested CV vs repeated random split\n"
                 "(bar = mean over repeats; error bar = 95% CI from resampling papers, same calculation in both)",
                 x=0.02, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(OUT / "cv_vs_repeated_split.png", dpi=160, bbox_inches="tight")
    pd.set_option("display.width", 250)
    print(flat.round(3).to_string())


if __name__ == "__main__":
    main()
