"""Permutation feature importance on held-out data, using the tuned nested-CV models.

For every (seed, outer fold) the model is refit on the outer-training data with its saved best params and
the drop in R2 on the outer *test* fold is measured when a feature is shuffled (Géron ch. 2/7:
"analyze the best models and their errors"). Two levels:
  original : the 13 input columns (a one-hot feature is shuffled as a whole)
  encoded  : each column after log10 / one-hot encoding (e.g. "Cell line = A549")
Results are averaged over all 15 folds (3 seeds x 5 folds).

Example:
    python src/feature_importance.py --features C --model xgb
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from cv_models import CV_MODELS, build, params_from_best
from cv_train import N_OUTER, RESULTS, make_folds
from data import DEFAULT_DATA, FEATURE_SETS, load_data, xy

TOP = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="C", choices=list(FEATURE_SETS))
    ap.add_argument("--model", default="xgb", choices=CV_MODELS)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--n-repeats", type=int, default=5)
    args = ap.parse_args()
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

    df = load_data(DEFAULT_DATA)
    X, y = xy(df, args.features)
    num_cols, cat_cols = FEATURE_SETS[args.features]
    base = RESULTS / f"features_{args.features}" / args.model

    orig, enc = [], []
    for seed in range(args.n_seeds):
        folds = make_folds(y, N_OUTER, seed)
        for k, (tr, te) in enumerate(folds):
            best = json.loads((base / f"seed{seed}" / f"fold{k}" / "best_params.json").read_text())
            pipe = build(args.model, params_from_best(args.model, best), num_cols, cat_cols, seed, n_jobs)
            pipe.fit(X.iloc[tr], y[tr])
            r = permutation_importance(pipe, X.iloc[te], y[te], scoring="r2", n_repeats=args.n_repeats,
                                       random_state=seed, n_jobs=1)
            orig.append(pd.Series(r.importances_mean, index=X.columns))
            prep, est = pipe.named_steps["prep"], pipe.named_steps["model"]
            Xt = prep.transform(X.iloc[te])
            r = permutation_importance(est, Xt, y[te], scoring="r2", n_repeats=args.n_repeats,
                                       random_state=seed, n_jobs=1)
            names = [n.replace("num__", "log10 ").replace("cat__", "") for n in prep.get_feature_names_out()]
            enc.append(pd.Series(r.importances_mean, index=names))
            print(f"seed {seed} fold {k} done")

    out = RESULTS / "feature_importance"
    out.mkdir(exist_ok=True)
    tag = f"{args.features}_{args.model}"
    summ = lambda L: pd.DataFrame(L).fillna(0).agg(["mean", "std"]).T.sort_values("mean", ascending=False)
    o, e = summ(orig), summ(enc)
    o.to_csv(out / f"{tag}_original.csv")
    e.to_csv(out / f"{tag}_encoded.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), gridspec_kw={"width_ratios": [1, 1.3]})
    for ax, d, title in [(axes[0], o, "Original features"), (axes[1], e.head(TOP), f"Top {TOP} encoded features")]:
        d = d.iloc[::-1]
        ax.barh(d.index, d["mean"], xerr=d["std"], color="#2a78d6", height=0.7,
                error_kw=dict(ecolor="#9ca3af", lw=1))
        ax.axvline(0, color="#6b7280", lw=0.8)
        ax.set(title=title, xlabel="Drop in held-out R² when shuffled")
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", color="#e5e7eb")
        ax.set_axisbelow(True)
    fig.suptitle(f"Permutation importance · {args.model.upper()} · features {args.features} · random-split CV "
                 f"(mean ± sd over {len(orig)} held-out folds)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"{tag}.png", dpi=150)

    pd.set_option("display.width", 200)
    print(f"\n== {tag}: original features ==\n{o.round(4).to_string()}")
    print(f"\n== {tag}: top {TOP} encoded features ==\n{e.head(TOP).round(4).to_string()}")


if __name__ == "__main__":
    main()
