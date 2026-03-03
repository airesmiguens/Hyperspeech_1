"""Data loading and feature grouping utilities for HyperSpeech."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass
class DatasetBundle:
    df: pd.DataFrame
    feature_cols: list[str]
    target_col: str
    subject_col: str


def load_dataset(
    csv_path: str | Path,
    target_col: str,
    subject_col: str,
    drop_cols: Iterable[str] | None = None,
) -> DatasetBundle:
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")
    if subject_col not in df.columns:
        raise ValueError(f"Missing subject column: {subject_col}")

    excluded = set(drop_cols or []) | {target_col, subject_col}
    feature_cols = [col for col in df.columns if col not in excluded]
    if not feature_cols:
        raise ValueError("No feature columns remaining after exclusions.")

    return DatasetBundle(df=df, feature_cols=feature_cols, target_col=target_col, subject_col=subject_col)


def infer_feature_groups(feature_cols: list[str]) -> dict[str, list[str]]:
    groups = {
        "prosody": ["pitch", "f0", "prosody", "inton"],
        "jitter_shimmer": ["jitter", "shimmer"],
        "formants": ["formant", "f1", "f2", "f3", "f4"],
        "spectral": ["spectral", "mfcc", "centroid", "bandwidth", "rolloff"],
        "rate_pause": ["rate", "pause", "duration", "speechrate"],
        "demographics": ["age", "sex", "gender"],
    }
    out: dict[str, list[str]] = {name: [] for name in groups}
    out["other"] = []

    for col in feature_cols:
        col_l = col.lower()
        placed = False
        for group_name, keys in groups.items():
            if any(key in col_l for key in keys):
                out[group_name].append(col)
                placed = True
                break
        if not placed:
            out["other"].append(col)
    return out


def groups_to_feature_index(groups: dict[str, list[str]], feature_cols: list[str]) -> list[int]:
    index_by_feature = {name: idx for idx, name in enumerate(feature_cols)}
    group_names = list(groups.keys())
    out = [0] * len(feature_cols)
    for group_idx, group_name in enumerate(group_names):
        for col in groups[group_name]:
            if col in index_by_feature:
                out[index_by_feature[col]] = group_idx
    return out
