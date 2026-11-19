"""Save the best repeated-random-split model (train.py runs) for later figures, e.g. predicted-vs-observed or SHAP.

Selection (no test-set snooping, Géron ch. 2):
  1. best model type  = highest mean test R2 over the repeats (seeds) for the chosen feature set
  2. saved repeat     = the seed of that model with the best *validation* RMSE (the test set is not used to choose)
The model is then refit from its saved best_params on that seed's training rows. Refitting (instead of copying a
pickle made elsewhere) keeps the saved model compatible with the Python / library versions of this machine.

Outputs (best_results/7_best_models/<features>_<model>_seed<seed>/):
  model.joblib       fitted pipeline (preprocessing + model); load with joblib.load
  info.json          features, seed, hyperparameters, selection rule, train/val/test metrics, library versions
  split_rows.json    row numbers of the train / validation / test sets (rows of load_data())
  predictions.csv    true vs predicted viability for every row, with its split

Examples:
    python src/save_best_model.py                                   # features C, seeds 0..9, best model type
    python src/save_best_model.py --features E6
    python src/save_best_model.py --model xgb                       # force a model type
    python src/save_best_model.py --first-seed 42 --n-seeds 1       # the original single split
"""
import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data import FEATURE_SETS, GROUP, TARGET, load_data, random_split, xy
from models import MODELS, build

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "best_results" / "7_best_models"


def scores(y, p):
    return {"R2": float(r2_score(y, p)), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAE": float(mean_absolute_error(y, p))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="C", choices=list(FEATURE_SETS))
    ap.add_argument("--model", choices=MODELS, help="model type to save (default: best mean test R2)")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=0)
    args = ap.parse_args()
    seeds = list(range(args.first_seed, args.first_seed + args.n_seeds))

    rows = []
    for m in MODELS:
        for s in seeds:
            p = RESULTS / f"features_{args.features}" / f"{m}_seed{s}" / "metrics.json"
            if p.exists():
                rows.append(json.loads(p.read_text()))
    runs = pd.DataFrame(rows)
    if runs.empty:
        raise SystemExit(f"No runs for features {args.features}, seeds {seeds[0]}..{seeds[-1]} under {RESULTS}")
    complete = runs.groupby("model").seed.nunique()
    complete = complete[complete == len(seeds)].index
    runs = runs[runs.model.isin(complete)]
    if runs.empty:
        raise SystemExit("No model type has all repeats finished yet")

    # 1. model type: best mean test R2 over the repeats
    mean_r2 = runs.groupby("model").test_R2.mean().sort_values(ascending=False)
    model = args.model or mean_r2.index[0]
    print("Mean test R² over repeats:", mean_r2.round(3).to_dict(), "->", model)
    # 2. repeat: best validation RMSE of that model (test set not used for this choice)
    cand = runs[runs.model == model].sort_values("val_RMSE")
    seed = int(cand.iloc[0].seed)
    params = json.loads((RESULTS / f"features_{args.features}" / f"{model}_seed{seed}" / "best_params.json").read_text())
    print(f"Saved repeat: seed {seed} (lowest validation RMSE {cand.iloc[0].val_RMSE:.2f}; "
          f"its test R² {cand.iloc[0].test_R2:.3f})")

    # 3. refit on that seed's training rows with the saved best hyperparameters
    df = load_data()
    train, val, test = random_split(df, seed)
    num_cols, cat_cols = FEATURE_SETS[args.features]
    Xtr, ytr = xy(train, args.features)
    pipe = build(model, params, num_cols, cat_cols, seed, n_jobs=-1).fit(Xtr, ytr)
    metrics, preds = {}, []
    for name, part in [("train", train), ("val", val), ("test", test)]:
        X, y = xy(part, args.features)
        p = pipe.predict(X)
        metrics[name] = scores(y, p)
        preds.append(pd.DataFrame({"row": part.index, "split": name, GROUP: part[GROUP].to_numpy(),
                                   "y_true": y, "y_pred": p}))
    orig = cand.iloc[0]
    print(f"Refit check: test R² {metrics['test']['R2']:.3f} (original run {orig.test_R2:.3f})")

    # 4. save
    out = OUT / f"{args.features}_{model}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / "model.joblib")
    pd.concat(preds).to_csv(out / "predictions.csv", index=False)
    (out / "split_rows.json").write_text(json.dumps(
        {"train": train.index.tolist(), "val": val.index.tolist(), "test": test.index.tolist()}))
    (out / "info.json").write_text(json.dumps({
        "target": TARGET, "feature_set": args.features, "numeric_features": num_cols, "categorical_features": cat_cols,
        "model": model, "seed": seed, "hyperparameters": params,
        "selection": {"model_type": "highest mean test R2 over repeats", "repeat": "lowest validation RMSE",
                      "seeds": seeds, "mean_test_R2_by_model": mean_r2.round(4).to_dict()},
        "metrics_refit": metrics,
        "metrics_original_run": {"test_R2": orig.test_R2, "test_RMSE": orig.test_RMSE, "test_MAE": orig.test_MAE},
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "xgboost": xgboost.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "load_with": "import joblib; model = joblib.load('model.joblib'); model.predict(X)  "
                     "# X = DataFrame with the feature columns above; use the same library versions",
    }, indent=2))
    print(f"Saved to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
