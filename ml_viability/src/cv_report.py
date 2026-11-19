"""Summarise nested-CV results.

Per seed, the 5 outer test folds are pooled (each row predicted exactly once) and scored.
Then, per (features, model):
  - mean ± sd over seeds of R2, RMSE, MAE, CCC (+ inside/outside the applicability domain)
  - 95% confidence intervals:
      RMSE_CI_t   : Géron ch. 2 t-interval on the squared errors (assumes independent rows -> too narrow
                    when rows from the same paper are correlated)
      R2/RMSE_CI  : bootstrap over *papers* (resample whole papers with replacement); rows of one paper are
                    correlated, so this gives a more honest (wider) interval -> the interval to report
    CIs are computed per seed and averaged over seeds.
  - acceptance verdict (QSAR criteria used in Zhang et al. 2025 and Liu et al. 2024):
      "good"       R2 > 0.6 and CCC > 0.85
      "acceptable" R2 > 0.5
      "fails"      otherwise
"""
import numpy as np
import pandas as pd
from scipy import stats

from cv_train import N_OUTER, RESULTS, metrics
from data import GROUP

N_BOOT = 1000


def paper_bootstrap(p, n_boot=N_BOOT, seed=0):
    """95% CI of R2 and RMSE, resampling whole papers."""
    rng = np.random.default_rng(seed)
    by_paper = {g: d for g, d in p.groupby(GROUP)}
    papers = np.array(list(by_paper))
    r2, rmse = [], []
    for _ in range(n_boot):
        d = pd.concat([by_paper[g] for g in rng.choice(papers, len(papers), replace=True)])
        m = metrics(d.y_true.to_numpy(), d.y_pred.to_numpy())
        r2.append(m["R2"]); rmse.append(m["RMSE"])
    return np.percentile(r2, [2.5, 97.5]), np.percentile(rmse, [2.5, 97.5])


def t_interval_rmse(y, p):
    se = (y - p) ** 2
    lo, hi = stats.t.interval(0.95, len(se) - 1, loc=se.mean(), scale=stats.sem(se))
    return np.sqrt(max(lo, 0)), np.sqrt(hi)


rows = []
for seed_dir in sorted(RESULTS.glob("features_*/*/seed*")):
    folds = [seed_dir / f"fold{k}" / "predictions.csv" for k in range(N_OUTER)]
    if not all(f.exists() for f in folds):
        continue  # incomplete seed
    p = pd.concat(pd.read_csv(f) for f in folds)
    feats, model = seed_dir.parts[-3].replace("features_", ""), seed_dir.parts[-2]
    y, yp = p.y_true.to_numpy(), p.y_pred.to_numpy()
    r = {"features": feats, "model": model, "seed": seed_dir.name, **metrics(y, yp),
         "frac_in_AD": p.in_AD.mean()}
    (r["R2_lo"], r["R2_hi"]), (r["RMSE_lo"], r["RMSE_hi"]) = paper_bootstrap(p)
    r["RMSE_t_lo"], r["RMSE_t_hi"] = t_interval_rmse(y, yp)
    for tag, m in [("inAD", p.in_AD), ("outAD", ~p.in_AD)]:
        if m.sum() >= 5:
            r.update({f"{tag}_{k}": v for k, v in metrics(y[m], yp[m]).items()})
    rows.append(r)

if not rows:
    raise SystemExit(f"No complete seeds under {RESULTS}")
df = pd.DataFrame(rows)
df.to_csv(RESULTS / "pooled_per_seed.csv", index=False)

keys = ["features", "model"]
g = df.groupby(keys)
s = g.mean(numeric_only=True).add_suffix("_mean").join(g.std(numeric_only=True).add_suffix("_sd"))
s.insert(0, "n_seeds", g.size())
s["verdict"] = np.where((s.R2_mean > 0.6) & (s.CCC_mean > 0.85), "good",
                        np.where(s.R2_mean > 0.5, "acceptable", "fails"))
s = s.reset_index().sort_values(["features", "R2_mean"], ascending=[True, False])
s.round(3).to_csv(RESULTS / "summary.csv", index=False)

f2, f1 = "{:.2f}".format, "{:.1f}".format
show = s[keys + ["n_seeds"]].copy()
show["R2 (±sd)"] = s.R2_mean.map(f2) + " ± " + s.R2_sd.fillna(0).map(f2)
show["R2 95% CI"] = "[" + s.R2_lo_mean.map(f2) + ", " + s.R2_hi_mean.map(f2) + "]"
show["RMSE (±sd)"] = s.RMSE_mean.map(f1) + " ± " + s.RMSE_sd.fillna(0).map(f1)
show["RMSE 95% CI"] = "[" + s.RMSE_lo_mean.map(f1) + ", " + s.RMSE_hi_mean.map(f1) + "]"
show["MAE"] = s.MAE_mean.map(f1)
show["CCC"] = s.CCC_mean.map(f2)
show["in AD"] = s.frac_in_AD_mean.map("{:.0%}".format)
if "inAD_R2_mean" in s:
    show["R2 inAD / outAD"] = s.inAD_R2_mean.map(f2) + " / " + s.get("outAD_R2_mean", pd.Series(np.nan, s.index)).map(f2)
show["verdict"] = s.verdict
pd.set_option("display.width", 300)
print("R2/RMSE 95% CI: bootstrap over papers. Verdict: good = R2>0.6 & CCC>0.85; acceptable = R2>0.5.")
print(show.to_string(index=False))
