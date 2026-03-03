"""Sklearn and optional external baseline wrappers for HyperSpeech."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from src.models.wrappers_optional import fit_predict_proba_optional


@dataclass
class ModelSpec:
	name: str
	backend: str
	params: dict[str, Any]


def supported_baseline_specs() -> dict[str, ModelSpec]:
	return {
		"histgb": ModelSpec(
			name="histgb",
			backend="sklearn",
			params={"max_depth": None, "max_iter": 400, "learning_rate": 0.05, "random_state": 42},
		),
		"realmlp": ModelSpec(
			name="realmlp",
			backend="sklearn",
			params={
				"hidden_layer_sizes": (256, 128, 64),
				"activation": "relu",
				"solver": "adam",
				"alpha": 1e-4,
				"learning_rate_init": 1e-3,
				"max_iter": 300,
				"early_stopping": True,
				"random_state": 42,
			},
		),
		"catboost": ModelSpec(
			name="catboost",
			backend="catboost",
			params={"depth": 6, "learning_rate": 0.05, "iterations": 600, "loss_function": "Logloss", "verbose": False},
		),
		"tabpfn": ModelSpec(
			name="tabpfn",
			backend="tabpfn",
			params={"device": "cpu", "N_ensemble_configurations": 8},
		),
		"tabnet": ModelSpec(name="tabnet", backend="optional", params={}),
		"saint": ModelSpec(name="saint", backend="optional", params={}),
		"ft_transformer": ModelSpec(name="ft_transformer", backend="optional", params={}),
		"tabtransformer": ModelSpec(name="tabtransformer", backend="optional", params={}),
		"dcnv2": ModelSpec(name="dcnv2", backend="optional", params={}),
		"node": ModelSpec(name="node", backend="optional", params={}),
		"carte": ModelSpec(name="carte", backend="optional", params={}),
	}


def build_model(model_name: str, overrides: dict[str, Any] | None = None):
	specs = supported_baseline_specs()
	if model_name not in specs:
		raise ValueError(f"Unknown model: {model_name}. Available: {list(specs.keys())}")

	spec = specs[model_name]
	params = dict(spec.params)
	params.update(overrides or {})

	if model_name == "histgb":
		return HistGradientBoostingClassifier(**params)

	if model_name == "realmlp":
		return MLPClassifier(**params)

	if model_name == "catboost":
		try:
			from catboost import CatBoostClassifier
		except ImportError as exc:
			raise ImportError("catboost is not installed. Install with `pip install catboost`.") from exc
		return CatBoostClassifier(**params)

	if model_name == "tabpfn":
		try:
			from tabpfn import TabPFNClassifier
		except ImportError as exc:
			raise ImportError("tabpfn is not installed. Install with `pip install tabpfn`.") from exc
		return TabPFNClassifier(**params)

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
