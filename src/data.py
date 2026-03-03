"""Data loading and feature-set utilities for HyperSpeech."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd


@dataclass(frozen=True)
class DataSpec:
    csv_path: str
    id_col: str
    target_col: str
    sbp_2classes_col: str = "SBP-2CLASSES"
    positive_class_value: int = 2
    sbp_col: str = "SBP"
    sbp_threshold: float = 130.0
    leakage_cols: Tuple[str, ...] = (
        "PAT_ID",
        "sbp_binary",
        "SBP",
        "DBP",
        "BPM",
        "SBP-2CLASSES",
        "DBP-2CLASSES",
        "BP-2CLASSES",
    )
    demographic_cols: Tuple[str, ...] = ("AGE", "GENDER", "HEIGHT", "WEIGHT")
    categorical_cols: Tuple[str, ...] = ("GENDER",)


@dataclass
class DatasetBundle:
    df: pd.DataFrame
    feature_cols: list[str]
    target_col: str
    subject_col: str


def _infer_target(df: pd.DataFrame, spec: DataSpec) -> pd.Series:
    if spec.target_col in df.columns:
        y = df[spec.target_col]
        if y.dtype.kind not in ("i", "b"):
            y = (y.astype(float) > 0.5).astype(int)
        return y.astype(int)

    if spec.sbp_2classes_col in df.columns:
        return (df[spec.sbp_2classes_col] == spec.positive_class_value).astype(int)

    if spec.sbp_col in df.columns:
        return (df[spec.sbp_col].astype(float) >= float(spec.sbp_threshold)).astype(int)

    raise ValueError(
        f"Target '{spec.target_col}' not found and could not be derived. "
        f"Expected either '{spec.sbp_2classes_col}' or '{spec.sbp_col}'."
    )


def load_dataframe(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at {path.resolve()}")
    return pd.read_csv(path)


def build_feature_sets(df: pd.DataFrame, spec: DataSpec, include_demographics: bool):
    if spec.id_col not in df.columns:
        raise ValueError(f"id_col '{spec.id_col}' not found in dataframe columns.")

    y = _infer_target(df, spec)
    groups = df[spec.id_col].astype(str)

    leakage = set([col for col in spec.leakage_cols if col in df.columns])
    dem_cols = [col for col in spec.demographic_cols if col in df.columns]
    cat_cols = [col for col in spec.categorical_cols if col in df.columns]

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in leakage and col != spec.id_col]
    acoustic_cols = [col for col in numeric_cols if col not in dem_cols]

    use_cols = acoustic_cols + (dem_cols if include_demographics else [])
    if include_demographics:
        for col in cat_cols:
            if col not in use_cols:
                use_cols.append(col)

    x = df[use_cols].copy()
    feature_info = {
        "acoustic": acoustic_cols,
        "demographics": dem_cols,
        "categorical": cat_cols,
        "all": use_cols,
    }
    return x, y, groups, feature_info, cat_cols


def load_dataset(
    csv_path: str | Path,
    target_col: str,
    subject_col: str,
    drop_cols: tuple[str, ...] | None = None,
) -> DatasetBundle:
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")
    if subject_col not in df.columns:
        raise ValueError(f"Missing subject column: {subject_col}")

    excluded = set(drop_cols or ()) | {target_col, subject_col}
    feature_cols = [col for col in df.columns if col not in excluded]
    return DatasetBundle(df=df, feature_cols=feature_cols, target_col=target_col, subject_col=subject_col)
