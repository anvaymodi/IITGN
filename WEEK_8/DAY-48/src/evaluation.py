"""
Evaluation utilities.

- Regression metrics for stock forecasting (Sub-step 3 & 6).
- Classification metrics for churn with class imbalance (Sub-step 4).
- Cost-model threshold selection for outreach policy (Sub-step 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# Default cost parameters for Sub-step 5 — documented and overridable
DEFAULT_COST_FALSE_POSITIVE = 5.0      # cost of contacting a non-churner
DEFAULT_COST_FALSE_NEGATIVE = 80.0     # cost of missing a churner (lost LTV)
DEFAULT_BENEFIT_TRUE_POSITIVE = 0.0    # already covered by avoiding FN


@dataclass
class RegressionReport:
    rmse: float
    mae: float
    directional_accuracy: float  # fraction of days we got up/down right


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    previous_close: np.ndarray | None = None,
) -> RegressionReport:
    """RMSE + MAE in original units, plus directional accuracy if prev_close supplied."""
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    if previous_close is not None and len(previous_close) == len(y_true):
        true_dir = np.sign(y_true - previous_close)
        pred_dir = np.sign(y_pred - previous_close)
        nonflat = true_dir != 0
        da = float((true_dir[nonflat] == pred_dir[nonflat]).mean()) if nonflat.any() else float("nan")
    else:
        da = float("nan")
    return RegressionReport(rmse=rmse, mae=mae, directional_accuracy=da)


def classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Return precision, recall, F1, PR-AUC, ROC-AUC — pure numpy."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": _roc_auc(y_true, y_score),
        "pr_auc": _pr_auc(y_true, y_score),
    }


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mann-Whitney U based ROC-AUC (no sklearn)."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Rank all scores with average-rank for ties
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    unique, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    sum_ranks = np.zeros_like(unique, dtype=float)
    np.add.at(sum_ranks, inv, ranks)
    avg_ranks = sum_ranks / counts
    final_ranks = avg_ranks[inv]
    rank_sum_pos = final_ranks[pos_mask].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision (PR-AUC) via trapezoidal rule."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    total_pos = int(y_sorted.sum())
    if total_pos == 0:
        return float("nan")
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / total_pos
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    # np.trapz was removed in NumPy 2.0; use trapezoid with fallback for older versions
    trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trap(precision, recall))


def optimal_threshold_by_cost(
    y_true: np.ndarray,
    y_score: np.ndarray,
    cost_fp: float = DEFAULT_COST_FALSE_POSITIVE,
    cost_fn: float = DEFAULT_COST_FALSE_NEGATIVE,
) -> Tuple[float, dict]:
    """Sweep thresholds 0.01 -> 0.99, pick the one with lowest total cost."""
    best_thr, best_cost, best_metrics = 0.5, float("inf"), None
    for thr in np.arange(0.01, 1.0, 0.01):
        m = classification_metrics(y_true, y_score, threshold=float(thr))
        cost = cost_fp * m["fp"] + cost_fn * m["fn"]
        if cost < best_cost:
            best_cost = cost
            best_thr = float(thr)
            best_metrics = m
    best_metrics["total_cost"] = best_cost
    best_metrics["cost_fp"] = cost_fp
    best_metrics["cost_fn"] = cost_fn
    return best_thr, best_metrics
