"""Threshold selection helpers for model operating points."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve


def threshold_for_best_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if thresholds.size == 0:
        return 0.5

    f1 = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx])


def threshold_for_target_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 0.95,
) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    p_sorted = y_prob[order]

    positives = max(1, int(y_sorted.sum()))
    tp_running = np.cumsum(y_sorted)
    recall_running = tp_running / positives

    eligible = np.where(recall_running >= target_recall)[0]
    if eligible.size == 0:
        return float(p_sorted[-1])

    idx = int(eligible[0])
    return float(p_sorted[idx])
