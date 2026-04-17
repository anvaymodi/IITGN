"""
Sequence construction for time-series stock price prediction.

Core rule we defend in Sub-step 1: the train/test split is CHRONOLOGICAL,
not random. A random split leaks future information, which inflates
reported performance and is useless in production.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


DEFAULT_WINDOW_SIZE = 30
DEFAULT_TEST_FRACTION = 0.15
DEFAULT_VAL_FRACTION = 0.15


def build_sequences(
    prices: np.ndarray,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Turn a 1D price series into (X, y) for next-day prediction."""
    if len(prices) <= window_size:
        raise ValueError(f"Need more than {window_size} observations, got {len(prices)}")
    xs, ys = [], []
    for i in range(len(prices) - window_size):
        xs.append(prices[i : i + window_size])
        ys.append(prices[i + window_size])
    X = np.array(xs).reshape(-1, window_size, 1)
    y = np.array(ys)
    return X, y


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chronological split preserving temporal order."""
    n = len(X)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    n_train = n - n_test - n_val
    if n_train <= 0:
        raise ValueError("Split fractions leave no training data.")
    return (
        X[:n_train], y[:n_train],
        X[n_train : n_train + n_val], y[n_train : n_train + n_val],
        X[n_train + n_val:], y[n_train + n_val:],
    )


def normalize_train_first(
    X_train, X_val, X_test, y_train, y_val, y_test,
):
    """Fit scaling stats on TRAIN ONLY."""
    mean = float(X_train.mean())
    std = float(X_train.std() + 1e-8)
    stats = {"mean": mean, "std": std}

    def scale(arr):
        return (arr - mean) / std

    return (
        scale(X_train), scale(X_val), scale(X_test),
        scale(y_train), scale(y_val), scale(y_test),
        stats,
    )


def inverse_scale(values: np.ndarray, stats: dict) -> np.ndarray:
    """Undo normalize_train_first scaling for human-readable metrics."""
    return values * stats["std"] + stats["mean"]
