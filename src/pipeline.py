"""Pipeline orchestration helpers for train-cache-compare workflow."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.artifacts_io import save_json
from src.artifacts_io import save_predictions
from src.calibration import apply_calibrator
from src.calibration import fit_calibrator
from src.calibration import save_calibrator
from src.metrics import evaluate_binary
from src.models.hyperspeech_tokenmixer import HyperSpeechTokenMixer
from src.models.wrappers_sklearn import fit_predict_proba
from src.models.wrappers_sklearn import predict_proba
from src.models.wrappers_torch import TorchTrainConfig
from src.models.wrappers_torch import predict_proba_binary
from src.models.wrappers_torch import train_binary_tabular_model
from src.thresholding import threshold_for_best_f1
from src.thresholding import threshold_for_target_recall


def run_baseline_fold(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    subject_col: str,
    train_idx: list[int],
    test_idx: list[int],
    model_name: str,
    fold: int,
    out_dir: str | Path,
) -> dict:
    x_train = df.iloc[train_idx][feature_cols].to_numpy(dtype=float)
    y_train = df.iloc[train_idx][target_col].to_numpy(dtype=int)
    x_test = df.iloc[test_idx][feature_cols].to_numpy(dtype=float)
    y_test = df.iloc[test_idx][target_col].to_numpy(dtype=int)

    model, prob_test = fit_predict_proba(model_name, x_train, y_train, x_test)
    prob_train = predict_proba(model, x_train)

    thr_f1 = threshold_for_best_f1(y_train, prob_train)
    thr_screen = threshold_for_target_recall(y_train, prob_train, target_recall=0.95)

    calib = fit_calibrator(y_train, prob_train, method="platt")
    prob_test_cal = apply_calibrator(calib, prob_test)

    metrics_f1 = evaluate_binary(y_test, prob_test_cal, thr_f1)
    metrics_screen = evaluate_binary(y_test, prob_test_cal, thr_screen)

    fold_dir = Path(out_dir) / model_name / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    model_path = fold_dir / "model.joblib"
    calib_path = fold_dir / "calibrator.joblib"
    preds_path = fold_dir / "preds_subject.csv"
    metrics_path = fold_dir / "metrics.json"

    joblib.dump(model, model_path)
    save_calibrator(calib, calib_path)

    save_predictions(
        pd.DataFrame(
            {
                "row_idx": test_idx,
                subject_col: df.iloc[test_idx][subject_col].to_numpy(),
                "y_true": y_test,
                "y_prob": prob_test,
                "y_prob_cal": prob_test_cal,
            }
        ),
        preds_path,
    )
    payload = {
        "model": model_name,
        "fold": fold,
        "threshold_f1": thr_f1,
        "threshold_screen": thr_screen,
        "f1_mode": metrics_f1,
        "screen_mode": metrics_screen,
    }
    save_json(payload, metrics_path)
    return payload


def run_hyperspeech_tokenmixer_fold(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    subject_col: str,
    train_idx: list[int],
    test_idx: list[int],
    fold: int,
    out_dir: str | Path,
    device: str = "cpu",
) -> dict:
    x_train = df.iloc[train_idx][feature_cols].to_numpy(dtype=float)
    y_train = df.iloc[train_idx][target_col].to_numpy(dtype=int)
    x_test = df.iloc[test_idx][feature_cols].to_numpy(dtype=float)
    y_test = df.iloc[test_idx][target_col].to_numpy(dtype=int)

    model = HyperSpeechTokenMixer(n_features=x_train.shape[1], d_token=64, n_blocks=3, dropout=0.15)
    train_cfg = TorchTrainConfig(epochs=120, batch_size=128, lr=1e-3, weight_decay=1e-4, device=device)
    model = train_binary_tabular_model(model, x_train, y_train, train_cfg)

    prob_train = predict_proba_binary(model, x_train, device=device)
    prob_test = predict_proba_binary(model, x_test, device=device)

    thr_f1 = threshold_for_best_f1(y_train, prob_train)
    thr_screen = threshold_for_target_recall(y_train, prob_train, target_recall=0.95)

    calib = fit_calibrator(y_train, prob_train, method="platt")
    prob_test_cal = apply_calibrator(calib, prob_test)

    metrics_f1 = evaluate_binary(y_test, prob_test_cal, thr_f1)
    metrics_screen = evaluate_binary(y_test, prob_test_cal, thr_screen)

    fold_dir = Path(out_dir) / "hyperspeech_tokenmixer" / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), fold_dir / "model.pt")
    save_calibrator(calib, fold_dir / "calibrator.joblib")
    save_predictions(
        pd.DataFrame(
            {
                "row_idx": test_idx,
                subject_col: df.iloc[test_idx][subject_col].to_numpy(),
                "y_true": y_test,
                "y_prob": prob_test,
                "y_prob_cal": prob_test_cal,
            }
        ),
        fold_dir / "preds_subject.csv",
    )

    payload = {
        "model": "hyperspeech_tokenmixer",
        "fold": fold,
        "threshold_f1": thr_f1,
        "threshold_screen": thr_screen,
        "f1_mode": metrics_f1,
        "screen_mode": metrics_screen,
    }
    save_json(payload, fold_dir / "metrics.json")
    return payload
