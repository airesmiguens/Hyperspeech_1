"""Calibration utilities for post-hoc probability calibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from typing import Optional

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


CalibMethod = Literal["platt", "isotonic"]


@dataclass
class Calibrator:
    method: CalibMethod = "platt"
    model: Optional[object] = None

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> "Calibrator":
        x = np.asarray(y_prob).reshape(-1, 1)
        y = np.asarray(y_true).astype(int)

        if self.method == "platt":
            self.model = LogisticRegression(solver="lbfgs", max_iter=1000)
            self.model.fit(x, y)
            return self

        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip")
            self.model.fit(np.asarray(y_prob), y)
            return self

        raise ValueError(f"Unknown calibration method: {self.method}")

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Calibrator has not been fit.")

        if self.method == "platt":
            return self.model.predict_proba(np.asarray(y_prob).reshape(-1, 1))[:, 1]

        if self.method == "isotonic":
            return self.model.predict(np.asarray(y_prob))

        raise ValueError(f"Unknown calibration method: {self.method}")


def fit_calibrator(y_true: np.ndarray, y_prob: np.ndarray, method: CalibMethod = "platt"):
    cal = Calibrator(method=method).fit(y_true, y_prob)
    return cal.model


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
