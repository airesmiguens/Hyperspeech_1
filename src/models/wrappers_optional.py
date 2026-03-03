"""Optional baseline adapters for heavier tabular deep learning models."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd


def _import_attr(module_path: str, attr_name: str):
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def _first_available_attr(candidates: list[tuple[str, str]]):
    last_exc: Exception | None = None
    for module_path, attr_name in candidates:
        try:
            return _import_attr(module_path, attr_name)
        except Exception as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise ImportError("No import candidates provided.")


class OptionalModelUnavailableError(ImportError):
    pass


class PytorchTabularBinaryAdapter:
    def __init__(self, model_name: str, params: dict[str, Any] | None = None):
        self.model_name = model_name
        self.params = dict(params or {})
        self._feature_cols: list[str] | None = None
        self._target_col = "target"
        self._model = None

    def _resolve_model_config_class(self):
        mapping = {
            "saint": [("pytorch_tabular.models.saint.config", "SAINTConfig")],
            "ft_transformer": [("pytorch_tabular.models.ft_transformer.config", "FTTransformerConfig")],
            "tabtransformer": [("pytorch_tabular.models.tab_transformer.config", "TabTransformerConfig")],
            "dcnv2": [("pytorch_tabular.models.dcn.config", "DeepCrossConfig")],
            "node": [("pytorch_tabular.models.node.config", "NODEConfig")],
        }
        if self.model_name not in mapping:
            raise OptionalModelUnavailableError(f"Unsupported pytorch-tabular model: {self.model_name}")
        try:
            return _first_available_attr(mapping[self.model_name])
        except Exception as exc:
            raise OptionalModelUnavailableError(
                f"Could not import config for {self.model_name} from pytorch-tabular. "
                "Install/upgrade with `pip install -U pytorch-tabular`.") from exc

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        try:
            from sklearn.model_selection import train_test_split
            from pytorch_tabular import TabularModel
            from pytorch_tabular.config import DataConfig
            from pytorch_tabular.config import OptimizerConfig
            from pytorch_tabular.config import TrainerConfig
        except Exception as exc:
            raise OptionalModelUnavailableError(
                "pytorch-tabular is required for SAINT/FT-Transformer/TabTransformer/DCNv2/NODE. "
                "Install with `pip install pytorch-tabular`.") from exc

        self._feature_cols = [f"f_{idx}" for idx in range(x_train.shape[1])]
        train_df = pd.DataFrame(x_train, columns=self._feature_cols)
        train_df[self._target_col] = y_train.astype(int)

        stratify = y_train if len(np.unique(y_train)) > 1 else None
        train_part, val_part = train_test_split(train_df, test_size=0.15, random_state=42, stratify=stratify)

        model_config_cls = self._resolve_model_config_class()
        model_config = model_config_cls(task="classification", **self.params)
        data_config = DataConfig(target=[self._target_col], continuous_cols=self._feature_cols)
        trainer_config = TrainerConfig(max_epochs=100, batch_size=256, accelerator="cpu")
        optimizer_config = OptimizerConfig()

        self._model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            trainer_config=trainer_config,
            optimizer_config=optimizer_config,
        )
        self._model.fit(train=train_part, validation=val_part)
        return self

    def predict_proba(self, x_eval: np.ndarray) -> np.ndarray:
        if self._model is None or self._feature_cols is None:
            raise RuntimeError("Model has not been fitted yet.")

        eval_df = pd.DataFrame(x_eval, columns=self._feature_cols)
        preds = self._model.predict(eval_df)

        prob_cols = [col for col in preds.columns if "probability" in col.lower()]
        if prob_cols:
            if len(prob_cols) == 1:
                return preds[prob_cols[0]].to_numpy()

            preferred = [col for col in prob_cols if col.lower().endswith("_1")]
            if preferred:
                return preds[preferred[0]].to_numpy()
            return preds[sorted(prob_cols)[-1]].to_numpy()

        if "prediction" in preds.columns:
            return preds["prediction"].to_numpy(dtype=float)
        raise RuntimeError("Could not locate probability column in pytorch-tabular prediction output.")


class TabNetAdapter:
    def __init__(self, params: dict[str, Any] | None = None):
        params = dict(params or {})
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
        except ImportError as exc:
            raise OptionalModelUnavailableError("tabnet requires pytorch-tabnet. Install with `pip install pytorch-tabnet`.") from exc
        self.model = TabNetClassifier(**params)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(x_train, y_train.reshape(-1, 1), max_epochs=100, patience=10, batch_size=1024)
        return self

    def predict_proba(self, x_eval: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x_eval)[:, 1]


class CARTEAdapter:
    def __init__(self, params: dict[str, Any] | None = None):
        params = dict(params or {})
        candidates = [
            ("carte", "CARTEClassifier"),
            ("carte_ai", "CARTEClassifier"),
            ("carte.model", "CARTEClassifier"),
        ]
        try:
            cls = _first_available_attr(candidates)
        except Exception as exc:
            raise OptionalModelUnavailableError(
                "CARTE adapter requires an install exposing `CARTEClassifier` (e.g., your chosen CARTE package)."
            ) from exc
        self.model = cls(**params)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(x_train, y_train)
        return self

    def predict_proba(self, x_eval: np.ndarray) -> np.ndarray:
        probs = self.model.predict_proba(x_eval)
        if probs.ndim == 2:
            return probs[:, 1]
        return probs


def build_optional_model(model_name: str, params: dict[str, Any] | None = None):
    params = params or {}

    if model_name == "tabnet":
        return TabNetAdapter(params)

    if model_name in {"saint", "ft_transformer", "tabtransformer", "dcnv2", "node"}:
        return PytorchTabularBinaryAdapter(model_name=model_name, params=params)

    if model_name == "carte":
        return CARTEAdapter(params)

    raise ValueError(f"Unknown optional model: {model_name}")


def fit_predict_proba_optional(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    params: dict[str, Any] | None = None,
):
    model = build_optional_model(model_name=model_name, params=params)
    model.fit(x_train, y_train)
    y_prob = model.predict_proba(x_eval)
    return model, np.asarray(y_prob, dtype=float)
