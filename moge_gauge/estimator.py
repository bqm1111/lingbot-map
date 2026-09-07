"""The fixed MoGe metric-gauge scale estimator. One scalar per clip, declared in advance.

    r_i(p)     = log(D_moge_i(p)) - log(D_lingbot_i(p))        valid pixels only
    log_s_moge = weighted_median over ALL valid pixels of ALL five frames
    s_moge     = exp(log_s_moge)

The weighted median is ``scale_gate.scale.weighted_median`` -- the identical routine the
Gate-4/4.1 LiDAR oracle-scale diagnostic uses -- weighted by LingBot confidence.

Validity is the intersection of:
  * MoGe's own predicted mask,
  * finite positive MoGe z-depth,
  * finite positive unscaled LingBot depth,
  * the frozen LingBot confidence criterion (conf >= 1.5),
  * the frozen metric depth range applied to the *MoGe* depth, which is already metric
    (1 m < D_moge < 60 m). It is deliberately NOT applied to LingBot depth, because that
    would require the scale this estimator is trying to find.

No semantic, ground, dynamic-object or target-derived mask is used.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from gates.scale_gate.scale import weighted_median                                 # noqa: E402


def frame_valid(moge_depth: np.ndarray, moge_mask: np.ndarray,
                lingbot_depth: np.ndarray, lingbot_conf: np.ndarray,
                conf_threshold: float, min_depth_m: float,
                max_depth_m: float) -> np.ndarray:
    return (moge_mask
            & np.isfinite(moge_depth) & (moge_depth > 0)
            & (moge_depth > min_depth_m) & (moge_depth < max_depth_m)
            & np.isfinite(lingbot_depth) & (lingbot_depth > 0)
            & (lingbot_conf >= conf_threshold))


def estimate_clip_scale(moge_depth: np.ndarray, moge_mask: np.ndarray,
                        lingbot_depth: np.ndarray, lingbot_conf: np.ndarray,
                        conf_threshold: float, min_depth_m: float, max_depth_m: float,
                        min_valid_pixels: int) -> Dict[str, object]:
    """All arrays ``(T, H, W)``. Returns one clip scalar plus per-frame diagnostics."""
    T = moge_depth.shape[0]
    ratios, weights, per_frame = [], [], []
    for t in range(T):
        v = frame_valid(moge_depth[t], moge_mask[t], lingbot_depth[t],
                        lingbot_conf[t], conf_threshold, min_depth_m, max_depth_m)
        n = int(v.sum())
        if n:
            r = np.log(moge_depth[t][v].astype(np.float64)) - \
                np.log(lingbot_depth[t][v].astype(np.float64))
            w = lingbot_conf[t][v].astype(np.float64)
            ratios.append(r); weights.append(w)
            s_f = float(np.exp(weighted_median(r, w)))
            mad = float(np.median(np.abs(r - np.median(r))))
        else:
            s_f, mad = float("nan"), float("nan")
        per_frame.append({"frame": t, "n_valid": n, "s_frame": s_f, "mad_log": mad,
                          "moge_mask_fraction": float(moge_mask[t].mean()),
                          "lingbot_conf_fraction": float(
                              (lingbot_conf[t] >= conf_threshold).mean())})
    n_total = int(sum(p["n_valid"] for p in per_frame))
    ok = n_total >= int(min_valid_pixels)
    if ok:
        R = np.concatenate(ratios); W = np.concatenate(weights)
        log_s = float(weighted_median(R, W))
        s = float(np.exp(log_s))
        mad_clip = float(np.median(np.abs(R - np.median(R))))
    else:
        log_s, s, mad_clip = float("nan"), float("nan"), float("nan")
    sf = np.array([p["s_frame"] for p in per_frame], dtype=np.float64)
    fin = np.isfinite(sf)
    disp = float(np.std(np.log(sf[fin]))) if fin.sum() > 1 else float("nan")
    rng = (float(np.max(sf[fin]) / np.min(sf[fin])) if fin.sum() > 1 else float("nan"))
    return {"s_moge": s, "log_s_moge": log_s, "ok": bool(ok),
            "n_valid_total": n_total, "mad_log_clip": mad_clip,
            "per_frame_log_scale_std": disp, "per_frame_scale_ratio_max_min": rng,
            "per_frame": per_frame}
