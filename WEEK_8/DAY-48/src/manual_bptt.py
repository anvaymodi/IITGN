"""
Manual backpropagation through time (BPTT) for a single-layer RNN.

Sub-step 7:
  1. Compute gradients for W_xh, W_hh by hand (no autograd).
  2. Verify they match PyTorch autograd.
  3. Empirically demonstrate vanishing gradient as T goes 5 -> 50.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


def rnn_forward_manual(
    xs: np.ndarray,
    W_xh: np.ndarray,
    W_hh: np.ndarray,
    b_h: np.ndarray,
    h0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vanilla tanh RNN forward pass. Returns (hs, pre_acts) with shape (T, hidden_dim)."""
    T = xs.shape[0]
    hidden_dim = W_hh.shape[0]
    hs = np.zeros((T, hidden_dim))
    pre_acts = np.zeros((T, hidden_dim))
    h_prev = h0
    for t in range(T):
        z = W_xh @ xs[t] + W_hh @ h_prev + b_h
        h_t = np.tanh(z)
        pre_acts[t] = z
        hs[t] = h_t
        h_prev = h_t
    return hs, pre_acts


def bptt_manual(
    xs: np.ndarray,
    target: float,
    W_xh: np.ndarray,
    W_hh: np.ndarray,
    b_h: np.ndarray,
    W_hy: np.ndarray,
    h0: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Full BPTT for loss = 0.5 * (y_hat - target)^2 where y_hat = W_hy @ h_T.

    Returns dict with gradients for every parameter plus per-timestep
    ||dL/dh_t|| for the vanishing-gradient plot.
    """
    T = xs.shape[0]
    hs, _pre_acts = rnn_forward_manual(xs, W_xh, W_hh, b_h, h0)
    h_final = hs[-1]

    y_hat = (W_hy @ h_final).item()
    loss = 0.5 * (y_hat - target) ** 2

    dL_dyhat = y_hat - target
    dL_dWhy = dL_dyhat * h_final[np.newaxis, :]
    dL_dh = (W_hy.T * dL_dyhat).flatten()

    dW_xh = np.zeros_like(W_xh)
    dW_hh = np.zeros_like(W_hh)
    db_h = np.zeros_like(b_h)
    grad_norms_per_t = np.zeros(T)

    for t in reversed(range(T)):
        grad_norms_per_t[t] = float(np.linalg.norm(dL_dh))
        # Through tanh: dh/dz = 1 - h^2
        dz = dL_dh * (1 - hs[t] ** 2)
        dW_xh += np.outer(dz, xs[t])
        h_prev = hs[t - 1] if t > 0 else h0
        dW_hh += np.outer(dz, h_prev)
        db_h += dz
        dL_dh = W_hh.T @ dz

    return {
        "loss": float(loss),
        "dW_xh": dW_xh,
        "dW_hh": dW_hh,
        "db_h": db_h,
        "dW_hy": dL_dWhy,
        "grad_norm_per_timestep": grad_norms_per_t,
    }


def bptt_autograd_reference(
    xs: np.ndarray,
    target: float,
    W_xh: np.ndarray,
    W_hh: np.ndarray,
    b_h: np.ndarray,
    W_hy: np.ndarray,
    h0: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Equivalent forward/backward using PyTorch autograd as ground truth."""
    W_xh_t = torch.tensor(W_xh, requires_grad=True, dtype=torch.float64)
    W_hh_t = torch.tensor(W_hh, requires_grad=True, dtype=torch.float64)
    b_h_t = torch.tensor(b_h, requires_grad=True, dtype=torch.float64)
    W_hy_t = torch.tensor(W_hy, requires_grad=True, dtype=torch.float64)
    xs_t = torch.tensor(xs, dtype=torch.float64)
    h = torch.tensor(h0, dtype=torch.float64)
    for t in range(xs.shape[0]):
        h = torch.tanh(W_xh_t @ xs_t[t] + W_hh_t @ h + b_h_t)
    y_hat = (W_hy_t @ h).squeeze()
    loss = 0.5 * (y_hat - target) ** 2
    loss.backward()
    return {
        "loss": float(loss.item()),
        "dW_xh": W_xh_t.grad.numpy(),
        "dW_hh": W_hh_t.grad.numpy(),
        "db_h": b_h_t.grad.numpy(),
        "dW_hy": W_hy_t.grad.numpy(),
    }


def gradient_norms_vs_sequence_length(
    lengths: list,
    hidden_dim: int = 8,
    input_dim: int = 2,
    spectral_scale: float = 1.0,
    seed: int = 0,
) -> Dict[int, float]:
    """
    For each T in `lengths`, record ||dL/dh_1|| — gradient that has
    propagated all the way back to timestep 1.
    """
    rng = np.random.default_rng(seed)
    W_xh = rng.normal(0, 0.3, (hidden_dim, input_dim))
    W_hh = rng.normal(0, spectral_scale / np.sqrt(hidden_dim), (hidden_dim, hidden_dim))
    b_h = np.zeros(hidden_dim)
    W_hy = rng.normal(0, 0.3, (1, hidden_dim))
    h0 = np.zeros(hidden_dim)

    results: Dict[int, float] = {}
    for T in lengths:
        xs = rng.normal(0, 1.0, (T, input_dim))
        target = float(rng.normal())
        out = bptt_manual(xs, target, W_xh, W_hh, b_h, W_hy, h0)
        results[T] = float(out["grad_norm_per_timestep"][0])
    return results
