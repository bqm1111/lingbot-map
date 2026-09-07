"""Frozen C3 geometry: the input this gate is allowed to correct.

Gate 2 concluded ``SCALE_CORRECTION_ONLY`` -- the ``depth_cnn`` residual head is a
clip-level metric-scale estimator, and its median-zero spatial part ``r_shape`` buys
nothing.  C3 is therefore the deployable geometry that keeps only the scalar::

    r          = depth_cnn(s0 * D_lingbot, conf, valid, x, y)      (frozen head)
    a_clip     = median of r over the unchanged C0 five-frame fusion support
    s_learned  = s0 * exp(a_clip)
    depth      = s_learned * D_lingbot
    pose       = LingBot pose, relative translations scaled by s_learned

``r_shape`` is never formed here.  Only the scalar leaves this module.

Everything is imported from the validated Gate-2 implementation
(``tools/depth_gate/decompose_residual.py``) rather than reimplemented, so the C3
reproduced by this gate is the same code path that produced IoU 0.0778.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # gates/<pkg>/ -> repo root
for _p in (_ROOT, os.path.join(_ROOT, "tools", "depth_gate"),
           os.path.join(_ROOT, "tools", "geometry_gate")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from decompose_residual import (                                            # noqa: E402
    as4x4, check_pose_scaling, fuse, scaled_relative_pose,
)
from gates.depth_gate.models import build_inputs, build_model                     # noqa: E402


@dataclass
class C3Clip:
    """Frozen C3 reconstruction of one five-frame clip, in the anchor camera frame."""

    clip_id: str
    sequence: str
    anchor_frame: int
    a_clip: float
    s_learned: float
    points: np.ndarray            # (N, 3) anchor-camera metric points
    point_frame: np.ndarray       # (N,) index of the contributing frame
    point_conf: np.ndarray        # (N,) LingBot confidence at the source pixel
    point_depth: np.ndarray       # (N,) metric depth at the source pixel
    n_support_px: int


def load_head(run_dir: str, device: torch.device):
    """Load the frozen Gate-2 refinement head. It is never updated in this gate."""
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu", weights_only=False)
    model = build_model(ck["arch"], ck["in_ch"], ck["base_channels"], ck["max_log_residual"])
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ck


def clip_scale(model, ck, dep: np.ndarray, conf: np.ndarray, s0: float,
               conf_thr: float, dmin: float, dmax: float, rgb: Optional[np.ndarray],
               device: torch.device):
    """``(a_clip, s_learned, support)`` -- the only thing the frozen head contributes.

    ``support`` is the unchanged C0 fusion support: LingBot confidence and metric depth
    range on ``s0 * D_lingbot``. No LiDAR, no ground-truth pose, no oracle scale.
    """
    depf = dep.astype(np.float64)
    base0 = depf * s0
    batch = {
        "base_depth": torch.from_numpy(base0.astype(np.float32)).unsqueeze(1).to(device),
        "lingbot_confidence": torch.from_numpy(conf).unsqueeze(1).to(device),
        "valid_lingbot_mask": torch.from_numpy(
            np.isfinite(base0) & (base0 > 0)).unsqueeze(1).to(device),
    }
    if ck["use_rgb"]:
        batch["rgb"] = torch.from_numpy(rgb).to(device)
    with torch.no_grad():
        r = model(build_inputs(batch, ck["stats"], ck["use_rgb"]))[:, 0].double().cpu().numpy()
    support = (conf >= conf_thr) & np.isfinite(base0) & (base0 > dmin) & (base0 < dmax)
    a_clip = float(np.median(r[support]))
    return a_clip, s0 * float(np.exp(a_clip)), support


def c3_points(dep: np.ndarray, conf: np.ndarray, K: np.ndarray, pose_pred: np.ndarray,
              s_learned: float, conf_thr: float, dmin: float, dmax: float):
    """Fuse the C3 clip into the anchor camera, carrying per-point provenance.

    Depth is scaled by ``s_learned`` and the pose *translations* are scaled by the same
    ``s_learned`` (:func:`scaled_relative_pose`), so the fused cloud is a pure similarity
    of the canonical reconstruction -- the coupling Gate 2 showed matters.
    """
    depth_m = s_learned * dep.astype(np.float64)
    mask = (conf >= conf_thr) & np.isfinite(depth_m) & (depth_m > dmin) & (depth_m < dmax)
    T, H, W = depth_m.shape
    anchor = T - 1
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    pts, fr, cf, dp = [], [], [], []
    for f in range(T):
        m = mask[f]
        if not m.any():
            continue
        d = depth_m[f][m]
        p = np.stack([(u[m] - K[f][0, 2]) * d / K[f][0, 0],
                      (v[m] - K[f][1, 2]) * d / K[f][1, 1], d], axis=-1)
        if f != anchor:
            Trel = scaled_relative_pose(pose_pred, f, anchor, s_learned)
            p = p @ Trel[:3, :3].T + Trel[:3, 3]
        pts.append(p)
        fr.append(np.full(len(p), f, np.int8))
        cf.append(conf[f][m].astype(np.float32))
        dp.append(d.astype(np.float32))
    if not pts:
        z = np.zeros((0,), np.float32)
        return np.zeros((0, 3)), np.zeros((0,), np.int8), z, z, mask
    return (np.concatenate(pts, 0), np.concatenate(fr), np.concatenate(cf),
            np.concatenate(dp), mask)


def visible_ceiling_points(D, pose_gt: np.ndarray):
    """Gate-1 configuration A: metric LiDAR depth + GT poses, on the LiDAR support.

    This defines the **five-frame visible-reconstruction ceiling** (IoU 0.1255). It is
    used only to build training targets and diagnostics -- never as a model input and
    never as an inference-time gate.
    """
    dep = D["depth"].astype(np.float64)
    mask = D["valid"].astype(bool) & (dep > 0)
    K = np.tile(D["K_processed"].astype(np.float64), (len(dep), 1, 1))
    return fuse(dep, mask, K, as4x4(pose_gt), 1.0)


__all__ = ["C3Clip", "load_head", "clip_scale", "c3_points", "visible_ceiling_points",
           "scaled_relative_pose", "check_pose_scaling", "as4x4", "fuse"]
