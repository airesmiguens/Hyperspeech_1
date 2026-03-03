"""Cross-validation split helpers for leakage-safe subject-aware evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedGroupKFold


@dataclass(frozen=True)
class SplitSpec:
    outer_n_splits: int = 5
    outer_shuffle: bool = True
    outer_random_state: int = 42
    inner_n_splits: int = 3


def make_outer_splits(
    y,
    groups,
    spec: SplitSpec | None = None,
    n_splits: int | None = None,
    seed: int | None = None,
) -> list[dict[str, list[int]]]:
    y_np = np.asarray(y)
    groups_np = np.asarray(groups)

    if spec is None:
        spec = SplitSpec(
            outer_n_splits=n_splits or 5,
            outer_shuffle=True,
            outer_random_state=42 if seed is None else seed,
        )

    cv = StratifiedGroupKFold(
        n_splits=spec.outer_n_splits,
        shuffle=spec.outer_shuffle,
        random_state=spec.outer_random_state,
    )
    x_stub = np.zeros((len(y_np), 1), dtype=float)

    folds: list[dict[str, list[int]]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(x_stub, y_np, groups_np)):
        folds.append({"fold": fold_idx, "train_idx": train_idx.tolist(), "test_idx": test_idx.tolist()})
    return folds


def make_inner_splits(y_train: pd.Series, groups_train: pd.Series, spec: SplitSpec):
    gkf = GroupKFold(n_splits=spec.inner_n_splits)
    return list(gkf.split(np.zeros(len(y_train)), y_train, groups_train))


def save_splits(splits: list[dict[str, list[int]]], out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2), encoding="utf-8")


def load_splits(path: str | Path) -> list[dict[str, list[int]]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
