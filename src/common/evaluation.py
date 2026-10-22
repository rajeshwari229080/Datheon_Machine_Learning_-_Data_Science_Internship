"""Comprehensive model evaluation module for classification and regression.

Provides metrics computation, confusion matrix extraction, residual diagnostics,
and structured JSON serialization for reproducible reporting.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def _convert_numpy_types(obj: Any) -> Any:
    """Recursively convert NumPy data types to standard Python types for JSON compatibility."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_numpy_types(x) for x in obj]
    return obj


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    labels: Optional[List[Any]] = None,
    target_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute exhaustive classification evaluation metrics.

    Calculates accuracy, macro/weighted precision/recall/F1, per-class metrics,
    confusion matrix, ROC-AUC (when probabilities provided), and full report.
    """
    acc = float(accuracy_score(y_true, y_pred))
    macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    weighted_p = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_r = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    report_dict = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, zero_division=0
    )

    metrics: Dict[str, Any] = {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
        "classification_report": report_dict,
        "report_text": report_text,
    }

    # ROC-AUC computation
    if y_prob is not None:
        try:
            if y_prob.ndim == 1 or y_prob.shape[1] == 2:
                # Binary classification
                probs = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
            else:
                # Multi-class
                metrics["roc_auc_ovr"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                )
        except Exception as e:
            logger.debug(f"ROC-AUC could not be computed: {e}")
            metrics["roc_auc"] = None

    return metrics


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute regression metrics and residual diagnostics."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))

    residuals = y_true - y_pred
    res_mean = float(np.mean(residuals))
    res_std = float(np.std(residuals))
    max_error = float(np.max(np.abs(residuals)))

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mse": mse,
        "medae": medae,
        "residual_mean": res_mean,
        "residual_std": res_std,
        "max_error": max_error,
    }


def save_metrics_json(metrics: Dict[str, Any], filepath: str) -> None:
    """Save metrics dictionary to formatted JSON file with numpy conversion."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    clean_metrics = _convert_numpy_types(metrics)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(clean_metrics, f, indent=2)
    logger.info(f"Saved evaluation metrics to: {filepath}")
