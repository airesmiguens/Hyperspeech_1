"""Binary classification metric utilities for HyperSpeech experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class MetricResult:
    roc_auc: float
    pr_auc: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    specificity: float
    fpr: float
    brier: Optional[float] = None


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    try:
        roc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc = float("nan")
    try:
        pr = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr = float("nan")
    return roc, pr


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> MetricResult:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= float(threshold)).astype(int)

    roc, pr = _safe_auc(y_true, y_prob)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))

    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan")

    try:
        brier = float(brier_score_loss(y_true, y_prob))
    except Exception:
        brier = None

    return MetricResult(
        roc_auc=roc,
        pr_auc=pr,
        f1=f1,
        precision=prec,
        recall=rec,
        accuracy=acc,
        specificity=spec,
        fpr=fpr,
        brier=brier,
    )


def to_dict(metric: MetricResult) -> dict[str, float]:
    out = {
        "roc_auc": metric.roc_auc,
        "pr_auc": metric.pr_auc,
        "f1": metric.f1,
        "precision": metric.precision,
        "recall": metric.recall,
        "accuracy": metric.accuracy,
        "specificity": metric.specificity,
        "fpr": metric.fpr,
    }
    if metric.brier is not None:
        out["brier"] = metric.brier
    return out


def evaluate_binary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    return to_dict(compute_binary_metrics(y_true, y_prob, threshold))
