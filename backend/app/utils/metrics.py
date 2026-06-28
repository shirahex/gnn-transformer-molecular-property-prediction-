"""Evaluation metrics for molecular property prediction."""

import logging
from typing import Optional

import numpy as np
from sklearn.metrics import (
    auc,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str = "regression",
) -> dict[str, float]:
    """Compute task-appropriate metrics.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        task_type: 'regression' or 'classification'.

    Returns:
        Dictionary of metric name -> value.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) == 0:
        return {"error": float("nan")}

    metrics: dict[str, float] = {}

    if task_type == "regression":
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        # R^2
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics["r2"] = float(1 - ss_res / (ss_tot + 1e-8))

    elif task_type == "classification":
        # Binary classification metrics
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred))
        except ValueError:
            metrics["roc_auc"] = 0.5

        # Precision-Recall AUC
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            metrics["pr_auc"] = float(auc(recall, precision))
        except ValueError:
            metrics["pr_auc"] = 0.0

        # Accuracy with 0.5 threshold
        y_binary = (y_pred > 0.5).astype(int)
        metrics["accuracy"] = float(np.mean(y_binary == y_true))

    return metrics


def format_metrics(metrics: dict[str, float]) -> str:
    """Format metrics dict as human-readable string."""
    parts = []
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    return " | ".join(parts)
