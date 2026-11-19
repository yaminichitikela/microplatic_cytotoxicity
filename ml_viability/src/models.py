"""Shared preprocessing + pipelines and Optuna search spaces for RF, SVR, XGBoost and kNN (single-split train.py)."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

MODELS = ["rf", "svm", "xgb", "knn"]


def preprocessor(num_cols, cat_cols, scale):
    """log10 on numeric columns (+ scaling for distance/kernel models); one-hot on categoricals."""
    num_steps = [("log10", FunctionTransformer(np.log10, feature_names_out="one-to-one"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])


def suggest(trial, name):
    """Hyperparameter search space for each model."""
    if name == "rf":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 1000, step=50),
            max_depth=trial.suggest_categorical("max_depth", [None, 5, 10, 15, 20, 30]),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0]),
            bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
        )
    if name == "svm":
        return dict(
            C=trial.suggest_float("C", 1e-1, 1e3, log=True),
            epsilon=trial.suggest_float("epsilon", 1e-2, 10, log=True),
            gamma=trial.suggest_float("gamma", 1e-4, 1, log=True),
        )
    if name == "xgb":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 2000, step=50),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 12),
            min_child_weight=trial.suggest_float("min_child_weight", 1, 10),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            gamma=trial.suggest_float("gamma", 0, 5),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        )
    if name == "knn":
        return dict(
            n_neighbors=trial.suggest_int("n_neighbors", 1, 50),
            weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
            p=trial.suggest_categorical("p", [1, 2]),
        )
    raise ValueError(name)


def build(name, params, num_cols, cat_cols, seed, n_jobs):
    if name == "rf":
        est = RandomForestRegressor(**params, random_state=seed, n_jobs=n_jobs)
    elif name == "svm":
        est = SVR(kernel="rbf", **params)
    elif name == "xgb":
        est = XGBRegressor(**params, random_state=seed, n_jobs=n_jobs, tree_method="hist")
    elif name == "knn":
        est = KNeighborsRegressor(**params, n_jobs=n_jobs)
    else:
        raise ValueError(name)
    return Pipeline([("prep", preprocessor(num_cols, cat_cols, name in ("svm", "knn"))), ("model", est)])
