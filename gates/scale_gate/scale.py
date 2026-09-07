"""Robust estimation of the single metric scalar ``s`` for one short clip.

LingBot-Map emits depth and camera translation in one arbitrary canonical scale.
Recovering metric geometry needs exactly one positive scalar per clip, applied to
**both** depth and translation and to **neither** rotation nor intrinsics:

    D_metric[t] = s * D_tilde[t]
    t_metric[t] = s * t_tilde[t]
    R_metric[t] = R[t]

Everything here works in log space, where a scale is a shift and robust statistics are
symmetric in ``s`` and ``1/s``.

Three independent estimators are provided, and they are kept separate on purpose: their
*disagreement* is the diagnostic that tells you whether one scalar is physically
coherent at all (see :func:`agreement_error`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Weighted robust statistics
# --------------------------------------------------------------------------- #
def weighted_median(x: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    """Weighted median of ``x``. Falls back to the plain median when ``w`` is None."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return float("nan")
    if w is None:
        return float(np.median(x))
    w = np.asarray(w, dtype=np.float64).ravel()
    if w.size != x.size:
        raise ValueError(f"weight/value length mismatch: {w.size} vs {x.size}")
    if not np.all(w >= 0):
        raise ValueError("weights must be non-negative")
    tot = w.sum()
    if tot <= 0:
        return float(np.median(x))
    order = np.argsort(x, kind="mergesort")
    xs, ws = x[order], w[order]
    c = np.cumsum(ws) / tot
    # Lower weighted median: first point where the cumulative weight reaches 0.5.
    return float(xs[int(np.searchsorted(c, 0.5, side="left"))])


def trim_quantiles(x: np.ndarray, lo: float, hi: float,
                   w: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
    """Drop values outside the ``[lo, hi]`` quantile range. Returns ``(x, w, n_dropped)``."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0 or (lo <= 0.0 and hi >= 1.0):
        return x, w, 0
    a, b = np.quantile(x, [lo, hi])
    keep = (x >= a) & (x <= b)
    return x[keep], (None if w is None else np.asarray(w).ravel()[keep]), int((~keep).sum())


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class ScaleEstimate:
    """One scale estimate plus everything needed to judge whether to trust it."""

    log_s: float
    n: int                       # samples used after trimming
    n_raw: int                   # samples before trimming
    mad_log: float               # median absolute deviation of the log residual
    source: str                  # "depth" | "pose" | "joint"
    valid: bool = True
    reason: str = ""
    per_frame_log_s: Optional[np.ndarray] = None

    @property
    def s(self) -> float:
        return float(np.exp(self.log_s))

    def to_dict(self) -> Dict[str, object]:
        d = {"log_s": self.log_s, "s": self.s, "n": self.n, "n_raw": self.n_raw,
             "mad_log": self.mad_log, "source": self.source,
             "valid": bool(self.valid), "reason": self.reason}
        if self.per_frame_log_s is not None and len(self.per_frame_log_s):
            v = np.asarray(self.per_frame_log_s, dtype=np.float64)
            v = v[np.isfinite(v)]
            if v.size:
                d["per_frame_log_s_std"] = float(v.std())
                d["per_frame_s_min"] = float(np.exp(v.min()))
                d["per_frame_s_max"] = float(np.exp(v.max()))
        return d


INVALID = lambda src, why: ScaleEstimate(  # noqa: E731
    log_s=float("nan"), n=0, n_raw=0, mad_log=float("nan"),
    source=src, valid=False, reason=why)


# --------------------------------------------------------------------------- #
# Depth-derived scale
# --------------------------------------------------------------------------- #
def depth_scale(
    d_pred: np.ndarray, d_gt: np.ndarray, weights: Optional[np.ndarray] = None,
    frame_index: Optional[np.ndarray] = None,
    trim: Tuple[float, float] = (0.01, 0.99), min_pixels: int = 200,
) -> ScaleEstimate:
    """Scale aligning predicted depth to metric depth at projected LiDAR pixels.

    ``log s = weighted_median( log d_gt - log d_pred )`` -- the scale-only alignment
    that minimises the absolute log-depth residual.

    Args:
        d_pred, d_gt: matched positive depths, same length and same convention.
        weights: optional per-sample weights (e.g. LingBot depth confidence).
        frame_index: optional per-sample frame id, enabling per-frame dispersion.
        trim: quantile range retained before the median.
        min_pixels: below this many valid samples the estimate is marked invalid.
    """
    d_pred = np.asarray(d_pred, dtype=np.float64).ravel()
    d_gt = np.asarray(d_gt, dtype=np.float64).ravel()
    if d_pred.shape != d_gt.shape:
        raise ValueError(f"shape mismatch: {d_pred.shape} vs {d_gt.shape}")
    ok = np.isfinite(d_pred) & np.isfinite(d_gt) & (d_pred > 0) & (d_gt > 0)
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64).ravel()
        ok &= np.isfinite(weights) & (weights >= 0)
    n_raw = int(ok.sum())
    if n_raw < min_pixels:
        e = INVALID("depth", f"only {n_raw} valid pixels (< {min_pixels})")
        e.n_raw = n_raw
        return e

    r = np.log(d_gt[ok]) - np.log(d_pred[ok])
    w = None if weights is None else weights[ok]
    fi = None if frame_index is None else np.asarray(frame_index).ravel()[ok]

    rt, wt, _ = trim_quantiles(r, trim[0], trim[1], w)
    log_s = weighted_median(rt, wt)
    mad = float(np.median(np.abs(rt - log_s))) if rt.size else float("nan")

    per_frame = None
    if fi is not None and fi.size:
        per_frame = np.array([weighted_median(r[fi == f], None if w is None else w[fi == f])
                              for f in np.unique(fi)], dtype=np.float64)
    return ScaleEstimate(log_s=float(log_s), n=int(rt.size), n_raw=n_raw,
                         mad_log=mad, source="depth", per_frame_log_s=per_frame)


# --------------------------------------------------------------------------- #
# Pose-derived scale
# --------------------------------------------------------------------------- #
def pose_scale(
    t_pred: np.ndarray, t_gt: np.ndarray, min_translation_m: float = 0.5,
    trim: Tuple[float, float] = (0.0, 1.0), min_pairs: int = 1,
) -> ScaleEstimate:
    """Scale from the ratio of metric to predicted relative translations.

    ``s_pair = ||dt_gt|| / ||dt_pred||`` over frame pairs, aggregated as a median in
    log space. Pairs whose *metric* motion is below ``min_translation_m`` are dropped:
    the ratio of two near-zero translations is dominated by noise.

    **This is deliberately not a Sim(3) fit.** A Sim(3) alignment estimates and then
    absorbs the scale, so its residual cannot be used as a scale target; the previous
    study's oracle used exactly that and is re-examined against this estimator.

    Args:
        t_pred, t_gt: ``[N, 3]`` relative translations for the *same* frame pairs, in
            the same coordinate convention and origin.
    """
    t_pred = np.asarray(t_pred, dtype=np.float64).reshape(-1, 3)
    t_gt = np.asarray(t_gt, dtype=np.float64).reshape(-1, 3)
    if t_pred.shape != t_gt.shape:
        raise ValueError(f"shape mismatch: {t_pred.shape} vs {t_gt.shape}")
    n_pred = np.linalg.norm(t_pred, axis=1)
    n_gt = np.linalg.norm(t_gt, axis=1)
    ok = np.isfinite(n_pred) & np.isfinite(n_gt) & (n_pred > 1e-9) & (n_gt >= min_translation_m)
    n_raw = int(ok.sum())
    if n_raw < min_pairs:
        e = INVALID("pose", f"only {n_raw} pairs above {min_translation_m} m of motion")
        e.n_raw = n_raw
        return e
    r = np.log(n_gt[ok]) - np.log(n_pred[ok])
    rt, _, _ = trim_quantiles(r, trim[0], trim[1])
    log_s = weighted_median(rt)
    mad = float(np.median(np.abs(rt - log_s))) if rt.size else float("nan")
    return ScaleEstimate(log_s=float(log_s), n=int(rt.size), n_raw=n_raw,
                         mad_log=mad, source="pose", per_frame_log_s=r)


# --------------------------------------------------------------------------- #
# Agreement and joint solution
# --------------------------------------------------------------------------- #
def agreement_error(a: ScaleEstimate, b: ScaleEstimate) -> float:
    """``|log(s_a / s_b)|`` -- 0 means the two estimators agree exactly."""
    if not (a.valid and b.valid):
        return float("nan")
    return float(abs(a.log_s - b.log_s))


def joint_scale(
    depth: ScaleEstimate, pose: ScaleEstimate, w_depth: float = 0.5,
) -> ScaleEstimate:
    """Combine the depth and pose solutions in log space.

    Both terms are reduced to a single robust log-scale *before* combining, so the
    hundreds of thousands of LiDAR pixels cannot drown out the handful of pose pairs.
    ``w_depth`` is the weight on the depth term; the pose term gets ``1 - w_depth``.
    """
    if not 0.0 <= w_depth <= 1.0:
        raise ValueError("w_depth must lie in [0, 1]")
    if depth.valid and pose.valid:
        log_s = w_depth * depth.log_s + (1.0 - w_depth) * pose.log_s
        return ScaleEstimate(log_s=float(log_s), n=depth.n + pose.n,
                             n_raw=depth.n_raw + pose.n_raw,
                             mad_log=float(np.nanmean([depth.mad_log, pose.mad_log])),
                             source="joint")
    if depth.valid:
        return ScaleEstimate(depth.log_s, depth.n, depth.n_raw, depth.mad_log,
                             "joint", reason="pose term unavailable")
    if pose.valid:
        return ScaleEstimate(pose.log_s, pose.n, pose.n_raw, pose.mad_log,
                             "joint", reason="depth term unavailable")
    return INVALID("joint", "neither depth nor pose term is valid")


# --------------------------------------------------------------------------- #
# Applying a scale
# --------------------------------------------------------------------------- #
def apply_scale_to_poses(pose_c2w: np.ndarray, s: float, anchor: int = 0) -> np.ndarray:
    """Scale camera translations about ``anchor``, leaving rotations untouched.

    Args:
        pose_c2w: ``[T, 3, 4]`` or ``[T, 4, 4]`` camera-to-world transforms.
        s: positive scalar.
        anchor: index whose camera centre is held fixed.

    Returns:
        The same shape, with ``t_i -> t_anchor + s (t_i - t_anchor)`` and ``R`` intact.
    """
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"scale must be finite and positive, got {s}")
    out = np.array(pose_c2w, dtype=np.float64, copy=True)
    if out.ndim != 3 or out.shape[-2] not in (3, 4) or out.shape[-1] != 4:
        raise ValueError(f"expected [T, 3|4, 4], got {out.shape}")
    t0 = out[anchor, :3, 3].copy()
    out[:, :3, 3] = t0 + s * (out[:, :3, 3] - t0)
    return out


def apply_scale_to_depth(depth: np.ndarray, s: float) -> np.ndarray:
    """Multiply depth by ``s``; intrinsics are unchanged by a pure scale."""
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"scale must be finite and positive, got {s}")
    return np.asarray(depth, dtype=np.float64) * float(s)


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def bootstrap_ci(values: Sequence[float], n_boot: int = 10000, seed: int = 0,
                 alpha: float = 0.05) -> Dict[str, float]:
    """Clip-level bootstrap CI for the mean of ``values`` (e.g. per-clip IoU deltas)."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    boot = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi),
            "n": int(v.size), "excludes_zero": bool(lo > 0 or hi < 0)}
