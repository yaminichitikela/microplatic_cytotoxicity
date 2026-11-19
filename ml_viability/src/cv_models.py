"""Regression models + Optuna search spaces for the nested cross-validation pipeline (cv_train.py).

Sources: RF/SVR/kNN/XGB (original pipeline); ElasticNet, Extra-Trees, MLP (Géron, Hands-On ML ch. 4, 7, 10);
PLSR (Yang et al. 2026; Wang et al. 2023).
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from models import preprocessor

CV_MODELS = ["enet", "pls", "knn", "svr", "rf", "et", "xgb", "mlp"]
SCALED = {"enet", "pls", "knn", "svr", "mlp"}  # models that need standardised numeric inputs


class MonotoneXGB(RegressorMixin, BaseEstimator):
    """XGBoost with viability constrained to be non-increasing in concentration (first input column)."""

    def __init__(self, n_jobs=1, random_state=0, **params):
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.params = params

    def get_params(self, deep=True):
        return {"n_jobs": self.n_jobs, "random_state": self.random_state, **self.params}

    def set_params(self, **p):
        for k in ("n_jobs", "random_state"):
            if k in p:
                setattr(self, k, p.pop(k))
        self.params.update(p)
        return self

    def fit(self, X, y):
        constraints = (-1,) + (0,) * (X.shape[1] - 1)  # column 0 = log10 concentration
        self.model_ = XGBRegressor(**self.params, monotone_constraints=constraints, tree_method="hist",
                                   n_jobs=self.n_jobs, random_state=self.random_state)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class PLS(RegressorMixin, BaseEstimator):
    """PLSRegression that returns 1-D predictions and caps n_components at the number of inputs."""

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y):
        self.model_ = PLSRegression(n_components=min(self.n_components, X.shape[1]), scale=False).fit(X, y)
        return self

    def predict(self, X):
        return np.ravel(self.model_.predict(X))


def suggest(trial, name):
    if name == "enet":
        return dict(alpha=trial.suggest_float("alpha", 1e-4, 10, log=True),
                    l1_ratio=trial.suggest_float("l1_ratio", 0.0, 1.0))
    if name == "pls":
        return dict(n_components=trial.suggest_int("n_components", 1, 25))
    if name == "knn":
        return dict(n_neighbors=trial.suggest_int("n_neighbors", 1, 50),
                    weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
                    p=trial.suggest_categorical("p", [1, 2]))
    if name == "svr":
        return dict(C=trial.suggest_float("C", 1e-1, 1e3, log=True),
                    gamma=trial.suggest_float("gamma", 1e-4, 1, log=True),
                    epsilon=trial.suggest_float("epsilon", 1e-2, 10, log=True))
    if name in ("rf", "et"):
        return dict(n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
                    max_depth=trial.suggest_categorical("max_depth", [None, 5, 10, 15, 20, 30]),
                    min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                    max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0]),
                    bootstrap=trial.suggest_categorical("bootstrap", [True, False]))
    if name == "xgb":
        return dict(n_estimators=trial.suggest_int("n_estimators", 100, 1500, step=50),
                    learning_rate=trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
                    max_depth=trial.suggest_int("max_depth", 2, 10),
                    min_child_weight=trial.suggest_float("min_child_weight", 1, 10),
                    subsample=trial.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
                    gamma=trial.suggest_float("gamma", 0, 5),
                    reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10, log=True))
    if name == "mlp":
        n_layers = trial.suggest_int("n_layers", 1, 3)
        width = trial.suggest_categorical("width", [16, 32, 64, 128, 256])
        return dict(hidden_layer_sizes=(width,) * n_layers,
                    alpha=trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
                    learning_rate_init=trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
                    batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]))
    raise ValueError(name)


def params_from_best(name, best):
    """Turn Optuna's stored best_params back into estimator kwargs (MLP stores layers/width separately)."""
    if name == "mlp":
        best = dict(best)
        best["hidden_layer_sizes"] = (best.pop("width"),) * best.pop("n_layers")
    return best


def build(name, params, num_cols, cat_cols, seed, n_jobs):
    if name == "enet":
        est = ElasticNet(**params, max_iter=20000)
    elif name == "pls":
        est = PLS(**params)
    elif name == "knn":
        est = KNeighborsRegressor(**params, n_jobs=n_jobs)
    elif name == "svr":
        est = SVR(kernel="rbf", **params)
    elif name == "rf":
        est = RandomForestRegressor(**params, random_state=seed, n_jobs=n_jobs)
    elif name == "et":
        est = ExtraTreesRegressor(**params, random_state=seed, n_jobs=n_jobs)
    elif name == "xgb":
        est = MonotoneXGB(n_jobs=n_jobs, random_state=seed, **params)
    elif name == "mlp":
        # Early stopping (Géron ch. 4/11) on a 10% internal validation split; target standardised
        mlp = MLPRegressor(**params, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
                           max_iter=2000, random_state=seed)
        est = TransformedTargetRegressor(regressor=mlp, transformer=StandardScaler())
    else:
        raise ValueError(name)
    return Pipeline([("prep", preprocessor(num_cols, cat_cols, scale=name in SCALED)), ("model", est)])
