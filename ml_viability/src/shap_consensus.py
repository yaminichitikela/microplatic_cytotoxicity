"""SHAP for the consensus model (best nested-CV model: average of the top-3 models in each fold).

The consensus prediction is the plain average of its 3 member models, and SHAP values are additive, so
    consensus SHAP = average of the members' SHAP values   (exact, fold by fold)
Members are refit on the outer-training rows with their saved best params, then explained on the held-out fold:
  - XGBoost, Extra-Trees, Random Forest: exact tree SHAP (TreeExplainer)
  - MLP: model-agnostic permutation SHAP with 100 training rows as background (approximate)

Outputs (best_results/6_shap/):
  shap_bar_consensus.png, shap_beeswarm_consensus.png,
  consensus_mean_abs_shap_original.csv, consensus_mean_abs_shap_encoded.csv,
  consensus_shap_values.npz (raw SHAP values, so plots can be redrawn without recomputing)

Example:
    python src/shap_consensus.py            # all 3 seeds (bar uses all, beeswarm uses seed 0)
    python src/shap_consensus.py --seeds 0 --folds 0     # quick test on one fold
    python src/shap_consensus.py --redraw               # redraw the beeswarm from saved SHAP values (seconds)
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from cv_models import build, params_from_best
from cv_train import N_OUTER, RESULTS, make_folds, metrics
from data import DEFAULT_DATA, FEATURE_SETS, load_data, xy
from models import preprocessor
from shap_analysis import OUT, grouped, plot_bar, pretty

FEATURES = "C"
RUN = RESULTS / f"features_{FEATURES}"


def member_shap(name, pipe, Xtr, Xte, seed):
    """SHAP values (rows of Xte x encoded columns) and base value for one fitted member pipeline."""
    prep, est = pipe.named_steps["prep"], pipe.named_steps["model"]
    A, B = prep.transform(Xtr), prep.transform(Xte)
    if name == "xgb":
        ex = shap.TreeExplainer(est.model_)
        return ex.shap_values(B), float(np.ravel(ex.expected_value)[0])
    if name in ("rf", "et"):
        ex = shap.TreeExplainer(est)
        return ex.shap_values(B, check_additivity=False), float(np.ravel(ex.expected_value)[0])
    # MLP (or any other model): permutation SHAP on the encoded inputs, 100 training rows as background
    bg = A[np.random.default_rng(seed).choice(len(A), size=min(100, len(A)), replace=False)]
    ex = shap.PermutationExplainer(est.predict, shap.maskers.Independent(bg, max_samples=100), seed=seed)
    e = ex(B, silent=True)
    return e.values, float(np.mean(e.base_values))


def draw_beeswarm(S, rows, names, X, num_cols, cat_cols):
    """Beeswarm of the top 20 encoded columns; title spans the full figure width."""
    Xenc = preprocessor(num_cols, cat_cols, scale=False).fit(X).transform(X)[rows]
    plt.figure()
    shap.summary_plot(S, pd.DataFrame(Xenc, columns=[pretty(n) for n in names]), max_display=20,
                      show=False, plot_size=(9, 8))
    fig = plt.gcf()
    fig.suptitle("SHAP summary: consensus model (best), nested CV\n"
                 "(dot = one held-out experiment; red = high value / category present)",
                 x=0.02, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "shap_beeswarm_consensus.png", dpi=160)
    plt.close("all")


def redraw_from_saved():
    """Redraw the beeswarm from consensus_shap_values.npz (seed 0) without recomputing SHAP."""
    z = np.load(OUT / "consensus_shap_values.npz", allow_pickle=True)
    X, _ = xy(load_data(DEFAULT_DATA), FEATURES)
    num_cols, cat_cols = FEATURE_SETS[FEATURES]
    draw_beeswarm(z["seed0_S"], z["seed0_rows"], list(z["names"]), X, num_cols, cat_cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(N_OUTER)))
    ap.add_argument("--redraw", action="store_true", help="only redraw the beeswarm from the saved SHAP values")
    args = ap.parse_args()
    if args.redraw:
        return redraw_from_saved()
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    OUT.mkdir(parents=True, exist_ok=True)

    df = load_data(DEFAULT_DATA)
    X, y = xy(df, FEATURES)
    num_cols, cat_cols = FEATURE_SETS[FEATURES]
    names = list(preprocessor(num_cols, cat_cols, scale=False).fit(X).get_feature_names_out())
    col = {n: i for i, n in enumerate(names)}

    runs = []
    for seed in args.seeds:
        S, base, pred = np.zeros((len(y), len(names))), np.zeros(len(y)), np.zeros(len(y))
        done = np.zeros(len(y), bool)
        folds = make_folds(y, N_OUTER, seed)
        for k in args.folds:
            tr, te = folds[k]
            members = json.loads((RUN / "consensus" / f"seed{seed}" / f"fold{k}" / "metrics.json").read_text())["members"]
            parts = []
            for m in members:
                best = json.loads((RUN / m / f"seed{seed}" / f"fold{k}" / "best_params.json").read_text())
                pipe = build(m, params_from_best(m, best), num_cols, cat_cols, seed, n_jobs).fit(X.iloc[tr], y[tr])
                fnames = list(pipe.named_steps["prep"].get_feature_names_out())
                sv, b = member_shap(m, pipe, X.iloc[tr], X.iloc[te], seed)
                full = np.zeros((len(te), len(names)))
                full[:, [col[n] for n in fnames]] = sv
                parts.append((full, b, pipe.predict(X.iloc[te])))
            S[te] = np.mean([p[0] for p in parts], axis=0)           # consensus SHAP = average of members
            base[te] = np.mean([p[1] for p in parts])
            pred[te] = np.mean([p[2] for p in parts], axis=0)          # consensus prediction
            done[te] = True
            err = np.abs(base[te] + S[te].sum(1) - pred[te])
            print(f"seed {seed} fold {k}: members {members}; fold R2 {metrics(y[te], pred[te])['R2']:.3f}; "
                  f"SHAP additivity error median {np.median(err):.2f}, max {err.max():.2f} viability points")
        runs.append({"S": S[done], "pred": pred[done], "y": y[done], "rows": np.flatnonzero(done)})
        print(f"seed {seed}: consensus R2 on explained rows {metrics(y[done], pred[done])['R2']:.3f} ({done.sum()} rows)")

    np.savez_compressed(OUT / "consensus_shap_values.npz", names=np.array(names),
                        **{f"seed{s}_{k}": r[k] for s, r in zip(args.seeds, runs) for k in ("S", "pred", "y", "rows")})
    imp_all = pd.concat([grouped(r["S"], names, cat_cols).abs().mean() for r in runs], axis=1)
    imp, sd = imp_all.mean(axis=1).sort_values(ascending=False), (imp_all.std(axis=1) if len(runs) > 1 else None)
    imp.to_frame("mean_abs_SHAP").join(imp_all.std(axis=1).rename("sd_over_seeds")).to_csv(
        OUT / "consensus_mean_abs_shap_original.csv")
    enc = pd.concat([pd.Series(np.abs(r["S"]).mean(0), index=[pretty(n) for n in names]) for r in runs], axis=1)
    enc.mean(axis=1).sort_values(ascending=False).to_csv(OUT / "consensus_mean_abs_shap_encoded.csv",
                                                       header=["mean_abs_SHAP"])

    n_rows = len(runs[0]["rows"])
    plot_bar(imp, sd, "SHAP importance: consensus model (best, R² 0.795)\n"
                      f"nested CV, all 13 features; mean ± sd over {len(runs)} CV repeats", OUT / "shap_bar_consensus.png")

    draw_beeswarm(runs[0]["S"], runs[0]["rows"], names, X, num_cols, cat_cols)
    print("\nmean |SHAP| consensus:\n", imp.round(2).to_string())


if __name__ == "__main__":
    main()
