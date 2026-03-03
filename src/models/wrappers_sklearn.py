"""Sklearn and baseline wrappers for tabular experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.models.wrappers_optional import fit_predict_proba_optional


def build_preprocessor(x: pd.DataFrame, cat_cols: Optional[list] = None) -> ColumnTransformer:
    cat_cols = cat_cols or []
    num_cols = [col for col in x.columns if col not in cat_cols]

    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
        sparse_threshold=0.3,
    )


def make_histgb_classifier(params: Dict[str, Any]):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(**params)


def make_catboost_classifier(params: Dict[str, Any]):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(**params)


@dataclass
class SklearnModelBundle:
    name: str
    pipeline: Any
    cat_cols: list

    def fit(self, x: pd.DataFrame, y: np.ndarray):
        self.pipeline.fit(x, y)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(x)[:, 1]


def build_model(model_name: str, overrides: dict[str, Any] | None = None):
    params = dict(overrides or {})

    if model_name == "histgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        defaults = {"max_depth": None, "max_iter": 400, "learning_rate": 0.05, "random_state": 42}
        defaults.update(params)
        return HistGradientBoostingClassifier(**defaults)

    if model_name == "realmlp":
        defaults = {
            "hidden_layer_sizes": (256, 128, 64),
            "activation": "relu",
            "solver": "adam",
            "alpha": 1e-4,
            "learning_rate_init": 1e-3,
            "max_iter": 300,
            "early_stopping": True,
            "random_state": 42,
        }
        defaults.update(params)
        return MLPClassifier(**defaults)

    if model_name == "catboost":
        from catboost import CatBoostClassifier

        defaults = {"depth": 6, "learning_rate": 0.05, "iterations": 600, "loss_function": "Logloss", "verbose": False}
        defaults.update(params)
        return CatBoostClassifier(**defaults)

    if model_name == "tabpfn":
        from tabpfn import TabPFNClassifier

        defaults = {"device": "cpu", "N_ensemble_configurations": 8}
        defaults.update(params)
        return TabPFNClassifier(**defaults)

    raise ValueError(f"No builder implemented for model: {model_name}")


def fit_predict_proba(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    params: dict[str, Any] | None = None,
) -> tuple[Any, np.ndarray]:
    if model_name in {"tabnet", "saint", "ft_transformer", "tabtransformer", "dcnv2", "node", "carte"}:
        return fit_predict_proba_optional(
            model_name=model_name,
            x_train=x_train,
            y_train=y_train,
            x_eval=x_eval,
            params=params,
        )

    model = build_model(model_name=model_name, overrides=params)
    model.fit(x_train, y_train)
    y_prob = predict_proba(model, x_eval)
    return model, y_prob


def predict_proba(model: Any, x_eval: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        pred = model.predict_proba(x_eval)
        if pred.ndim == 2:
            return pred[:, 1]
        return pred

    if hasattr(model, "decision_function"):
        logits = model.decision_function(x_eval)
        return 1.0 / (1.0 + np.exp(-logits))

    raise ValueError("Model does not implement predict_proba or decision_function")
