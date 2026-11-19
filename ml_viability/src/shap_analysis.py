"""SHAP interpretation of the viability model (method as in Zhang et al. 2025 Fig. 4a/c and Liu et al. 2024 Fig. 7c/d).

Outputs (best_results/6_shap/):
  shap_bar.png       global importance: mean |SHAP| per feature (one-hot columns summed per original feature)
  shap_bar_single_split.png  the same for the single random-split model (train.py, seed 42)
  shap_beeswarm.png  direction of each feature's effect: one dot per experiment, red = high value / category present
  + CSV tables: importance per feature, per one-hot column, the single-split model, and a consensus of ranking methods

Main model: XGBoost, feature set C, nested cross-validation with random stratified folds (results_cv/).
SHAP values are computed on *held-out* rows (each row explained by the fold model that never saw it), 3 seeds.

Example:
    python src/shap_analysis.py
"""
import json
import os
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from cv_models import build, params_from_best
from cv_train import N_OUTER, RESULTS, make_folds, metrics
from data import DEFAULT_DATA, FEATURE_SETS, load_data, random_split, xy
from models import preprocessor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "best_results" / "6_shap"
SINGLE_SPLIT_MODEL = ROOT / "results" / "features_C" / "xgb_seed42" / "model.joblib"   # from train.py
FEATURES, MODEL, SEEDS = "C", "xgb", (0, 1, 2)
BLUE, MUTED, GRID = "#2a78d6", "#6b7280", "#e5e7eb"
SHORT = {"Concentration(μg/mL)": "Concentration", "Diameter(nm)": "Diameter", "Exposure time": "Exposure time"}

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRID,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.dpi": 110})


def original_of(name, cat_cols):
    """Encoded column name -> original feature ('num__Diameter(nm)' -> 'Diameter', 'cat__Cell line_A549' -> 'Cell line')."""
    body = name.split("__", 1)[1]
    if name.startswith("num__"):
        return SHORT.get(body, body)
    return max((c for c in cat_cols if body.startswith(c + "_")), key=len)


def pretty(name):
    body = name.split("__", 1)[1]
    if name.startswith("num__"):
        return f"log10 {SHORT.get(body, body)}"
    return body.replace("_", " = ", 1)


def best_params(seed, k):
    return json.loads((RESULTS / f"features_{FEATURES}" / MODEL / f"seed{seed}" / f"fold{k}" / "best_params.json").read_text())


def grouped(S, names, cat_cols):
    orig = [original_of(n, cat_cols) for n in names]
    return pd.DataFrame(S, columns=names).T.groupby(orig).sum().T  # per-row SHAP summed per original feature


def held_out_shap(X, y, num_cols, cat_cols, n_jobs):
    """SHAP values of every row from the CV fold model that did not train on it, for each seed (common column space)."""
    names = list(preprocessor(num_cols, cat_cols, scale=False).fit(X).get_feature_names_out())
    col = {n: i for i, n in enumerate(names)}
    runs = []
    for seed in SEEDS:
        S = np.zeros((len(y), len(names)))
        base, pred, gain = np.zeros(len(y)), np.zeros(len(y)), []
        for k, (tr, te) in enumerate(make_folds(y, N_OUTER, seed)):
            pipe = build(MODEL, params_from_best(MODEL, best_params(seed, k)), num_cols, cat_cols, seed,
                         n_jobs).fit(X.iloc[tr], y[tr])
            prep, booster = pipe.named_steps["prep"], pipe.named_steps["model"].model_
            fnames = list(prep.get_feature_names_out())
            ex = shap.TreeExplainer(booster)
            S[np.ix_(te, [col[n] for n in fnames])] = ex.shap_values(prep.transform(X.iloc[te]))
            base[te] = float(np.ravel(ex.expected_value)[0])
            pred[te] = pipe.predict(X.iloc[te])
            g = booster.get_booster().get_score(importance_type="gain")
            gain.append(pd.Series({fnames[int(f[1:])]: v for f, v in g.items()}))
        err = np.abs(base + S.sum(1) - pred).max()  # SHAP additivity: base + sum(SHAP) = prediction
        runs.append({"S": S, "gain": pd.DataFrame(gain).fillna(0).mean()})
        print(f"CV seed {seed}: held-out R2 {metrics(y, pred)['R2']:.3f}; SHAP additivity max error {err:.2e}")
    enc = pd.DataFrame(preprocessor(num_cols, cat_cols, scale=False).fit(X).transform(X), columns=names)
    return names, enc, runs


def single_split_shap(df, cat_cols):
    """Mean |SHAP| of the single random-split XGBoost model (train.py, seed 42) on its own 10% test set."""
    pipe = joblib.load(SINGLE_SPLIT_MODEL)
    _, _, test = random_split(df, 42)
    Xte, _ = xy(test, FEATURES)
    prep, booster = pipe.named_steps["prep"], pipe.named_steps["model"]
    S = shap.TreeExplainer(booster).shap_values(prep.transform(Xte))
    return grouped(S, list(prep.get_feature_names_out()), cat_cols).abs().mean()


def plot_bar(imp, sd, title, path):
    """Horizontal bar chart of mean |SHAP| per original feature (sd shown as error bars when given)."""
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    s = imp.iloc[::-1]
    ax.barh(s.index, s.values, xerr=None if sd is None else sd.reindex(s.index).values, color=BLUE, height=0.7,
            error_kw=dict(ecolor="#9ca3af", lw=1))
    for i, v in enumerate(s.values):
        ax.text(v, i, f"  {v:.1f}", va="center", fontsize=7, color=MUTED)
    ax.margins(x=0.15)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.set(xlabel="Mean |SHAP| (viability percentage points)", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_single_split(df, cat_cols, order=None):
    """Bar chart for the single random-split model (train.py, seed 42), on its 191 test rows."""
    single = single_split_shap(df, cat_cols)
    single = single.reindex(order) if order is not None else single.sort_values(ascending=False)
    single.to_frame("mean_abs_SHAP").to_csv(OUT / "single_split_mean_abs_shap_original.csv")
    plot_bar(single.sort_values(ascending=False), None,
             "SHAP feature importance: XGBoost, all 13 features, single random split\n"
             "(seed 42; 191 held-out test rows; R² 0.86)", OUT / "shap_bar_single_split.png")
    return single


def main():
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_data(DEFAULT_DATA)
    X, y = xy(df, FEATURES)
    num_cols, cat_cols = FEATURE_SETS[FEATURES]

    names, enc, runs = held_out_shap(X, y, num_cols, cat_cols, n_jobs)
    imp_all = pd.concat([grouped(r["S"], names, cat_cols).abs().mean() for r in runs], axis=1)
    imp, imp_sd = imp_all.mean(axis=1).sort_values(ascending=False), imp_all.std(axis=1)
    imp.to_frame("mean_abs_SHAP").join(imp_sd.rename("sd_over_seeds")).to_csv(OUT / "mean_abs_shap_original.csv")
    enc_imp = pd.concat([pd.Series(np.abs(r["S"]).mean(0), index=names) for r in runs], axis=1).mean(axis=1)
    enc_imp.index = [pretty(n) for n in enc_imp.index]
    enc_imp.sort_values(ascending=False).to_csv(OUT / "mean_abs_shap_encoded.csv", header=["mean_abs_SHAP"])

    # Bar: global importance per original feature (nested CV)
    plot_bar(imp, imp_sd, "SHAP feature importance: XGBoost, all 13 features, nested CV\n"
                          "(all 1,901 rows held out; mean ± sd over 3 CV repeats; R² 0.77)", OUT / "shap_bar.png")

    # Beeswarm: direction of each effect, top 20 one-hot / numeric columns (seed 0)
    plt.figure()
    shap.summary_plot(runs[0]["S"], pd.DataFrame(enc.values, columns=[pretty(n) for n in names]), max_display=20,
                      show=False, plot_size=(9, 8))
    plt.title("SHAP summary: nested CV, XGBoost\n(dot = one held-out experiment; red = high value / category present)",
              fontsize=9, loc="left")
    plt.tight_layout()
    plt.savefig(OUT / "shap_beeswarm.png", dpi=160)
    plt.close("all")

    # Tables: single-split model, and consensus of ranking methods (Liu 2024 used 3 methods)
    gain = pd.concat([r["gain"].groupby(lambda n: original_of(n, cat_cols)).sum() for r in runs], axis=1)
    gain = (gain / gain.sum()).mean(axis=1)
    perm = pd.read_csv(RESULTS / "feature_importance" / f"{FEATURES}_{MODEL}_original.csv", index_col=0)["mean"]
    perm.index = [SHORT.get(i, i) for i in perm.index]
    cons = {"SHAP, nested CV": imp.rank(ascending=False), "XGBoost gain (built-in)": gain.rank(ascending=False),
            "Permutation, nested CV": perm.rank(ascending=False)}
    if SINGLE_SPLIT_MODEL.exists():
        single = plot_single_split(df, cat_cols, order=imp.index)
        cons["SHAP, single split"] = single.rank(ascending=False)
    cons = pd.DataFrame(cons).astype(int).sort_values("SHAP, nested CV")
    cons.to_csv(OUT / "consensus_ranks.csv")
    print("\nmean |SHAP| nested CV:\n", imp.round(2).to_string())
    print("\nImportance ranks (1 = most important):\n", cons.to_string())


if __name__ == "__main__":
    main()
