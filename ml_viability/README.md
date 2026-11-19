# Cell viability prediction for micro(nano)plastics

**Endpoint:** cell viability (%), predicted by regression from experimental features.
**Data:** `data/cleaned/model_data_with_ref.csv`, the same 1,902 rows as `model_data (2).xlsx` plus each row's source paper (`ref`), recovered from the raw database in `Data_analysis/data_cleaning.ipynb`. Loading cleans label whitespace/case and drops 1 exact duplicate, leaving 1,901 rows from 71 papers.

**Best results so far:** see [`best_results/README.md`](best_results/README.md).

## Project structure

```
microplastic/
├── data/
│   ├── model_data (2).xlsx, raw-micro(nano)plastics_database.xlsx      original files
│   └── cleaned/model_data_with_ref.csv                                 model input
├── Data_analysis/data_cleaning.ipynb       EDA, duplicates, raw <-> model_data merge
├── papers/, book/                          references
└── ml_viability/
    ├── src/
    │   ├── data.py                load + clean, feature sets A-D, single random split
    │   ├── models.py              shared preprocessing; RF/SVR/XGB/kNN for train.py
    │   ├── cv_models.py           8 models for nested CV (ElasticNet, PLSR, kNN, SVR, RF, Extra-Trees, XGBoost, MLP)
    │   ├── cv_train.py            MAIN: nested CV, one (split, features, model, seed, outer fold) per run
    │   ├── cv_ensemble.py         consensus + stacking of the top-3 models per fold
    │   ├── cv_report.py           pooled metrics, 95% CIs, verdict  -> results_cv/summary.csv
    │   ├── feature_importance.py  permutation importance on held-out folds
    │   ├── shap_analysis.py       SHAP interpretation (Yan group protocol)
    │   └── train.py, report.py    single random 80/10/10 split -> results/
    ├── slurm/                     Wahab job scripts (cv_task, cv_report; tune_array, report for train.py)
    ├── submit_cv.sh               submit the nested-CV pipeline on Wahab
    ├── submit.sh                  submit single-split runs on Wahab
    ├── results_cv/                nested-CV outputs (from Wahab)
    ├── results/                   single random-split outputs (train.py)
    ├── logs_wahab/                Wahab job logs
    ├── best_results/              curated summary of the best results
    └── archive/                   snapshot of the code before the group split was removed (2026-09-24)
```

## Method (nested cross-validation, `cv_train.py`)

- **Outer loop:** 5 stratified folds of rows (similar viability distribution in each), repeated with 3 seeds; the same kind of random split as Yan et al. 2025.
- **Inner loop:** Optuna (TPE) tunes on 3-fold CV of the outer-training data only (objective: RMSE).
- **Evaluation:** tuned model refit on outer-train, scored once on the outer test fold. R², RMSE, MAE, CCC; 95% CI by bootstrapping papers; verdict with QSAR criteria (good: R² > 0.6 and CCC > 0.85; acceptable: R² > 0.5).
- **Extras:** applicability domain (kNN distance, each test row flagged in/out); `--scramble` for the y-scrambling check; consensus/stacking ensembles (`cv_ensemble.py`).
- **Preprocessing** (fitted inside each fold): log10 of concentration, diameter, exposure time; one-hot categoricals; standard scaling for linear/kernel/distance/neural models.

## Feature sets (`data.py`)

All sets use the 3 numeric features (log10): concentration, diameter, exposure time.

| Set | # features | Categorical features | Used in |
|---|---|---|---|
| **C** | 13 | Core, Shape, Surface modification, Plastic source, Cell source species, Cell anatomical type, Cell-tissue-organ-origin, Cell origin, Cell line, Assay type (all 13 original features, as in Yan et al. 2025) | nested CV, single split, SHAP |
| **B** | 11 | C without Cell line and Assay type | nested CV |
| E4 | 4 | Surface modification | single split |
| E5 | 5 | Surface modification, Core | single split |
| E6 | 6 | Surface modification, Core, Cell line | single split |

E4–E6 are small sets picked from the SHAP ranking. Nested CV runs B and C by default.

## Setup on Wahab (one time)

Wahab's login shell is tcsh, so run `bash` first.

```bash
bash
cd ~/microplastic/ml_viability
~/miniconda3/bin/python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy code from the Mac, without results (Mac terminal):

```bash
rsync -av --exclude 'results*/' --exclude 'logs*/' --exclude '.venv/' --exclude 'tasks/' \
  ~/Desktop/projects/microplastic/ml_viability/ ychit001@wahab.hpc.odu.edu:~/microplastic/ml_viability/
```

## Run

```bash
# quick test
python src/cv_train.py --features C --model xgb --seed 0 --fold 0 --n-trials 2
rm -rf results_cv

# full nested CV: 250 model tasks -> 30 ensemble tasks -> report
./submit_cv.sh
N_SEEDS=5 N_TRIALS=100 ./submit_cv.sh     # bigger run

# after it finishes
cat logs/viab_cv_report_<jobid>.out
python src/feature_importance.py --features C --model xgb
python src/shap_analysis.py
```

Pull results to the Mac (Mac terminal):

```bash
rsync -av ychit001@wahab.hpc.odu.edu:~/microplastic/ml_viability/results_cv/ \
  ~/Desktop/projects/microplastic/ml_viability/results_cv/
rsync -av ychit001@wahab.hpc.odu.edu:~/microplastic/ml_viability/logs/ \
  ~/Desktop/projects/microplastic/ml_viability/logs_wahab/
```

## Outputs: `results_cv/features_<B|C>/<model>/seed<s>/fold<k>/`

| File | Content |
|---|---|
| `metrics.json` | train/test R², RMSE, MAE, CCC; inside/outside applicability domain; papers per part |
| `best_params.json` | tuned hyperparameters |
| `predictions.csv` | row, ref, y_true, y_pred, AD distance, in_AD for every test row |
| `optuna.db` | Optuna study; rerunning the same task resumes it |

`results_cv/summary.csv` (mean ± sd over seeds, CIs, verdict) and `pooled_per_seed.csv` are written by `cv_report.py`.

## Reading the results

- R² = 0 means no better than predicting the mean viability; `--scramble` (y-scrambling) shows this chance level.
- Rows are split at random, so rows from the same study can be in both training and test. Scores describe prediction within studies represented in the data, as in the published papers.
- Report the nested-CV numbers. A single split (`train.py`) is noisy: it varies ±0.1 in R² between seeds.
