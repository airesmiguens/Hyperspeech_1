"""Calibration utilities for post-hoc probability calibration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


CalibMethod = Literal["platt", "isotonic"]


def fit_calibrator(y_true: np.ndarray, y_prob: np.ndarray, method: CalibMethod = "platt"):
    x = np.asarray(y_prob).reshape(-1, 1)
    y = np.asarray(y_true).astype(int)

    if method == "platt":
        model = LogisticRegression(max_iter=2000)
        model.fit(x, y)
        return model

    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(np.asarray(y_prob), y)
        return model

    raise ValueError(f"Unknown calibration method: {method}")


def apply_calibrator(model, y_prob: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(np.asarray(y_prob).reshape(-1, 1))[:, 1]
    return model.predict(np.asarray(y_prob))


def save_calibrator(model, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)


def load_calibrator(path: str | Path):
    return joblib.load(path)
