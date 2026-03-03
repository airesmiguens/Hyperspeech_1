"""Cross-validation split helpers for leakage-safe subject-aware evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def make_outer_splits(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> list[dict[str, list[int]]]:
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    x_stub = np.zeros((len(y), 1), dtype=float)

    folds: list[dict[str, list[int]]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(x_stub, y, groups=groups)):
        folds.append(
            {
                "fold": fold_idx,
                "train_idx": train_idx.tolist(),
                "test_idx": test_idx.tolist(),
            }
        )
    return folds


def save_splits(splits: list[dict[str, list[int]]], out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)


def load_splits(path: str | Path) -> list[dict[str, list[int]]]:
    with Path(path).open("r", encoding="utf-8") as f:
        splits = json.load(f)
    return splits
