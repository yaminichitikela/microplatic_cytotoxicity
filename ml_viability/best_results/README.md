# Best results: cell viability of micro(nano)plastics

- **Endpoint:** cell viability (%), regression.
- **Data:** `model_data (2).xlsx` as-is: 1,902 rows, 1,901 after dropping 1 exact duplicate; label whitespace/case cleaned; 13 experimental features (set C, the same as Yan et al. 2025).
- **Evaluation:** random splits of rows, the same kind of split as Yan et al. 2025.

Last updated: 2026-09-25.

## 1. Main result: nested cross-validation (`1_nested_cv/`)

- **Outer loop:** 5 stratified folds (similar viability distribution in each) × 3 repeats (seeds 0–2). Every row is predicted once per repeat by a model that never saw it.
- **Inner loop:** Optuna (50 trials) on 3-fold CV of the outer-training data only.
- **95% CI:** bootstrap over papers. **Verdict:** good = R² > 0.6 and CCC > 0.85; acceptable = R² > 0.5 (QSAR criteria, Zhang 2025 / Liu 2024).

| Model (features C) | R² [95% CI] | RMSE | MAE | CCC | Verdict |
|---|---|---|---|---|---|
| **Consensus** (mean of top 3) | **0.80** [0.72, 0.84] | 10.8 | 6.8 | 0.88 | good |
| Stacking | 0.79 [0.72, 0.84] | 10.9 | 6.9 | 0.88 | good |
| Extra-Trees | 0.78 [0.71, 0.83] | 11.0 | 6.9 | 0.88 | good |
| XGBoost | 0.77 [0.68, 0.82] | 11.5 | 7.5 | 0.87 | good |
| Random forest | 0.76 [0.66, 0.81] | 11.7 | 7.4 | 0.86 | good |
| MLP | 0.72 | 12.5 | 7.9 | 0.85 | acceptable |
| SVR | 0.70 | 13.0 | 8.3 | 0.83 | acceptable |
| kNN | 0.69 | 13.3 | 8.2 | 0.82 | acceptable |
| ElasticNet / PLS | 0.33 | 19.4 | 14.0 | 0.51–0.53 | fails |

With features B (11 features, no cell line or assay type), the best model is consensus at R² 0.72. Full table: `summary.csv`; each seed: `pooled_per_seed.csv`; printout: `report.txt`.

**Comparison with Yan et al. 2025 (SI Table S4, single 80/20 split):** XGBoost R² 0.81, RMSE 10.67, MAE 6.79; RF 0.78; kNN 0.61; ANN 0.40. Our nested CV reproduces this level (consensus 0.80, RMSE 10.8, MAE 6.8).

**Single random split (`../results/`):** `train.py`, 80/10/10 split, seed 42, 100 Optuna trials, features C. Test R² on 191 rows: XGBoost 0.86 (RMSE 8.5, MAE 6.2), RF 0.81, kNN 0.66, SVM 0.65.
One split is noisy: over seeds 0–4, XGBoost gave 0.73 ± 0.11. **Report the nested-CV numbers.**

**Leakage:** no classical leakage. Preprocessing and tuning use training data only, and each test fold is scored once. Because rows are split at random, rows of the same study can appear in both training and test. The scores therefore describe prediction within studies represented in the data, as in Yan et al. 2025.

## 2. Permutation feature importance (`2_feature_importance/`)

XGBoost, features C, held-out folds: the drop in R² when a feature is shuffled. The top 3 are **concentration (0.82), surface modification (0.58) and diameter (0.51)**, then exposure time, cell line and assay type.

## 3. Out-of-fold predictions (`3_predictions/`)

`C_consensus_oof.csv`, `C_xgb_oof.csv`: every row predicted once per seed by a model that never saw it.
Columns: row, ref, y_true, y_pred, ad_distance, in_AD, seed, fold.

## 4. Tuned hyperparameters (`4_best_params/`)

Best Optuna parameters for every features / seed / fold, for XGBoost, SVR, RF and Extra-Trees.

## 6. SHAP interpretation (`6_shap/`)

Plots as in Zhang et al. 2025 (Fig. 4a, 4c) and Liu et al. 2024 (Fig. 7c, 7d):

| File | Model | Shows |
|---|---|---|
| **`shap_bar_consensus.png`** | **Consensus (best model, R² 0.795)** | Which features matter most (mean \|SHAP\|, one-hot columns summed per original feature) |
| **`shap_beeswarm_consensus.png`** | **Consensus** | Direction: one dot per experiment; red = high value / category present; right = raises viability, left = lowers it. Top 20 of ~159 encoded columns |
| `shap_bar.png`, `shap_beeswarm.png` | XGBoost (best single model, R² 0.766) | The same, for comparison |
| `shap_bar_single_split.png` | XGBoost, single random split (seed 42) | The same bar chart, single split |

**How the consensus SHAP is computed (`src/shap_consensus.py`):** the consensus prediction is the average of the top-3 models in each fold, and SHAP values add up, so the consensus SHAP is the average of the 3 members' SHAP values. That's exact. Tree members (XGBoost, Extra-Trees, RF) use exact tree SHAP; the MLP uses permutation SHAP with 100 background rows (approximate). The SHAP values add up to the consensus predictions (error < 0.01 points), and the refitted members reproduce the saved consensus R² (0.792 / 0.797 / 0.796 per repeat). The raw SHAP values are saved in `consensus_shap_values.npz`.

**Bar vs beeswarm:** the bar has the **13 original features** (all dummy columns of a feature summed). The beeswarm has **individual categories** (e.g. "Cell source species = Protozoa") and shows only the **top 20 of ~159**. So the last beeswarm row is the 20th most important category, not the least important feature.

| Rank | Feature | Mean \|SHAP\| consensus (XGBoost CV / single split) | Direction |
|---|---|---|---|
| 1 | Concentration | **7.8** (9.2 / 7.0) | Above ~20–30 μg/mL lowers viability, by up to ~35 points at ≥ 1,000 μg/mL |
| 2 | Surface modification | **5.2** (5.3 / 5.2) | Amino-modified lowers viability by 10–38 points; carboxyl/unmodified slightly raise it |
| 3 | Diameter | **3.5** (4.2 / 4.4) | Smallest particles (~20–30 nm) lower viability, by up to ~50 points; > 100 nm roughly neutral |
| 4 | Cell line | **3.0** (3.0 / 2.5) | A549, RAW 264.7 lower |
| 5 | Exposure time | **2.8** (2.7 / 2.4) | Short (≤ 8 h) raises; long (≥ 48–72 h) lowers, by up to ~20 points |
| 6 | Assay type | **2.2** (1.9 / 1.9) | CCK-8, MTT, CellTiter-Glo lower; XTT higher |
| 7–13 | Species, tissue, core, anatomical type, origin, source, shape | 0.5–1.6 | Small; PET core raises viability (+5 to +15) |

- **Consensus vs XGBoost:** the same top 6 in the same order. The consensus spreads importance a little more evenly (concentration 7.8 vs 9.2), as expected for an average of different models.
- **CV vs single split:** the same top 7 in the same order. The single split gives concentration less weight (7.0 vs 9.2).
- **Consensus:** SHAP and permutation importance agree on the top 5. XGBoost's built-in gain over-ranks the categorical features, a known bias.
- **Agreement with Yan 2025 (Fig. 5c):** concentration first, and size, surface modification and cell type among the top features.
- Note: the regression isn't bounded, so it can predict below 0% for extreme cases (lowest prediction −25% for a row observed at 0%).

## Reproduce

```
./submit_cv.sh                                   # Wahab: nested CV (250 + 30 tasks + report)
python src/cv_report.py                          # summary tables
python src/feature_importance.py --features C --model xgb
python src/train.py --model xgb --features C --seed 42    # single random split
python src/shap_analysis.py                      # SHAP (needs results_cv/ and the single-split model)
python src/repeated_report.py                    # repeated random splits: mean ± sd, 95% CI
python src/save_best_model.py --features C       # save the best model for figures -> 7_best_models/
```
