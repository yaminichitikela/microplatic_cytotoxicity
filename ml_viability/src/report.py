"""Summarise the original single random split (seed 42) from results/features_*/<model>_seed42/metrics.json."""
import json
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
KEYS = ["features", "model"]
METRICS = ["train_R2", "test_R2", "test_RMSE", "test_MAE"]

SEED = 42  # the original single random split; repeated splits (seeds 0-9) are summarised by repeated_report.py
rows = [m for m in (json.loads(p.read_text()) for p in sorted(RESULTS.glob("features_*/*/metrics.json")))
        if m["seed"] == SEED]
if not rows:
    raise SystemExit(f"No metrics.json found under {RESULTS}")
df = pd.DataFrame(rows)
df.to_csv(RESULTS / "all_runs.csv", index=False)

g = df.groupby(KEYS)
s = g[METRICS].agg(["mean", "std"])
s.columns = [f"{m}_{a}" for m, a in s.columns]
s.insert(0, "n_seeds", g.size())
s = s.reset_index().sort_values(["features", "test_R2_mean"], ascending=[True, False])
s.round(3).to_csv(RESULTS / "summary.csv", index=False)

show = s[KEYS + ["n_seeds"]].copy()
for m in METRICS:
    show[m.replace("test_", "")] = s[f"{m}_mean"].map("{:.2f}".format) + " ± " + s[f"{m}_std"].fillna(0).map("{:.2f}".format)
pd.set_option("display.width", 250)
print(show.to_string(index=False))
