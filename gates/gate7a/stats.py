"""Paired bootstrap over the benchmark's own resampling unit, vectorised.

Gate 6 resampled by looping 10,000 times over the drawn units. Gate 7A needs the same
intervals for 2 conditions x 2 constructions x 6 radii x 2 metrics, so the resample is
expressed once as a **multiplicity matrix** and applied to every count block with a single
matrix multiply. The draws are identical to Gate 6's: ``np.random.default_rng(seed)`` then
``integers(0, U, size=(n_boot, U))``.

The unit on the two single-sequence benchmarks is a contiguous block of clips from **one
drive**. Blocks from one drive are not independent scenes and are never described as such.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


def multiplicities(n_units: int, n_boot: int, seed: int) -> np.ndarray:
    """``[n_boot, n_units]`` how many times each unit appears in each resample."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_units, size=(n_boot, n_units))
    out = np.zeros((n_boot, n_units), np.float64)
    for b in range(n_boot):
        out[b] = np.bincount(draws[b], minlength=n_units)
    return out


def resample(block: np.ndarray, mult: np.ndarray) -> np.ndarray:
    """Sum a per-unit count block over every resample. ``block`` is ``[U, ...]``."""
    U = block.shape[0]
    flat = np.asarray(block, np.float64).reshape(U, -1)
    return (mult @ flat).reshape((mult.shape[0],) + block.shape[1:])


def binary_iou(b: np.ndarray) -> np.ndarray:
    """``[..., 3]`` of (tp, fp, fn) -> IoU, with an empty union scoring 0."""
    tp, fp, fn = b[..., 0], b[..., 1], b[..., 2]
    den = tp + fp + fn
    return np.where(den > 0, tp / np.maximum(den, 1e-12), 0.0)


def miou(pc: np.ndarray) -> np.ndarray:
    """``[..., C, 3]`` of (tp, fp, fn) -> mean IoU over the classes with a non-zero union.

    Exactly ``gate6.metrics.summarize``: a class with ``tp+fp+fn == 0`` is not scored, and
    the mean is over the scored classes only.
    """
    tp, fp, fn = pc[..., 0], pc[..., 1], pc[..., 2]
    den = tp + fp + fn
    iou = np.where(den > 0, tp / np.maximum(den, 1e-12), 0.0)
    n = (den > 0).sum(axis=-1)
    return np.where(n > 0, iou.sum(axis=-1) / np.maximum(n, 1), 0.0)


def interval(samples: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return float(np.percentile(samples, lo)), float(np.percentile(samples, hi))


def paired(point_a: float, point_b: float, samples_a: np.ndarray,
           samples_b: np.ndarray, alpha: float = 0.05) -> Dict[str, object]:
    """The paired difference ``b - a`` with a percentile interval."""
    d = samples_b - samples_a
    lo, hi = interval(d, alpha)
    return {"a": float(point_a), "b": float(point_b),
            "difference": float(point_b - point_a), "ci": [lo, hi],
            "excludes_zero": bool(lo > 0 or hi < 0)}


__all__ = ["multiplicities", "resample", "binary_iou", "miou", "interval", "paired"]
