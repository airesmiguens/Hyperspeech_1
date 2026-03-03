"""Threshold selection helpers for model operating points."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score


@dataclass(frozen=True)
class ThresholdResult:
    threshold_f1: float
    threshold_screening: float
    achieved_recall_at_screening: float


def _grid() -> np.ndarray:
    return np.unique(np.clip(np.linspace(0.0, 1.0, 1001), 0.0, 1.0))


def pick_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    best_t = 0.5
    best = -1.0
    for threshold in _grid():
        f1 = f1_score(y_true, (y_prob >= threshold).astype(int), zero_division=0)
        if f1 > best:
            best = f1
            best_t = float(threshold)
    return best_t


def pick_recall_constrained_threshold(y_true: np.ndarray, y_prob: np.ndarray, min_recall: float) -> tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    candidates: list[tuple[float, float]] = []
    for threshold in _grid():
        recall = recall_score(y_true, (y_prob >= threshold).astype(int), zero_division=0)
        if recall >= min_recall:
            candidates.append((float(threshold), float(recall)))

    if candidates:
        threshold_best, recall_best = sorted(candidates, key=lambda pair: pair[0], reverse=True)[0]
        return threshold_best, recall_best

    best_t = 0.0
    best_recall = -1.0
    for threshold in _grid():
        recall = recall_score(y_true, (y_prob >= threshold).astype(int), zero_division=0)
        if (recall > best_recall) or (recall == best_recall and threshold > best_t):
            best_recall = float(recall)
            best_t = float(threshold)
    return best_t, best_recall


def choose_thresholds(y_true: np.ndarray, y_prob: np.ndarray, min_recall: float) -> ThresholdResult:
    t_f1 = pick_f1_threshold(y_true, y_prob)
    t_screen, r_screen = pick_recall_constrained_threshold(y_true, y_prob, min_recall=min_recall)
    return ThresholdResult(threshold_f1=t_f1, threshold_screening=t_screen, achieved_recall_at_screening=r_screen)


def threshold_for_best_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return pick_f1_threshold(y_true, y_prob)


def threshold_for_target_recall(y_true: np.ndarray, y_prob: np.ndarray, target_recall: float = 0.95) -> float:
    t, _ = pick_recall_constrained_threshold(y_true, y_prob, min_recall=target_recall)
    return t
