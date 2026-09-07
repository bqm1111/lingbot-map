"""Gate 4.1 — diagnostic 2x2 factorization of the Occ3D-nuScenes geometry error.

Every model, threshold, radius, normalisation statistic and control is the frozen Gate-4
object; this module only changes *which geometry* is presented to them:

    depth scale  in {frozen constant s0, per-clip LiDAR-measured oracle}
    poses        in {LingBot predicted, nuScenes ground truth}

**Scaling discipline**, asserted in tests:

* LingBot translations are canonical and are multiplied by the selected scale **exactly
  once**, inside :func:`scaled_relative_pose` (the frozen Gate-2 implementation).
* nuScenes ground-truth translations are already metric and are **never** multiplied by
  ``s0``, the C3 scale or the oracle scale.
* Rotations are never scaled, in either branch.
* Predicted depth is always multiplied by the selected scale, in both pose branches.

Branches using LiDAR-measured scale or ground-truth poses are **non-deployable diagnostic
oracles** and are labelled as such everywhere they appear.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_ROOT, os.path.join(_ROOT, "tools", "depth_gate")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from decompose_residual import scaled_relative_pose                          # noqa: E402

ORACLE_CELLS = ("G01", "G10", "G11")          # every cell that consumes privileged data
DEPLOYABLE_CELLS = ("G00",)


# --------------------------------------------------------------------------- #
# Relative transforms into the anchor camera
# --------------------------------------------------------------------------- #
def predicted_relative(pose_pred: np.ndarray, f: int, anchor: int,
                       scale: float) -> np.ndarray:
    """LingBot frame ``f`` -> anchor camera, translation scaled **once** by ``scale``."""
    return scaled_relative_pose(pose_pred, f, anchor, float(scale))


def gt_camera_to_world(frames) -> np.ndarray:
    """``T_camera_to_world[t]`` from nuScenes metadata: ego pose at the camera timestamp
    composed with the static camera extrinsic. Metric, never scaled."""
    return np.stack([f.T_ego_cam_to_world @ f.T_camera_to_ego_cam for f in frames])


def gt_relative(cam2world: np.ndarray, f: int, anchor: int) -> np.ndarray:
    """Ground-truth frame ``f`` -> anchor camera. **Metric already; never scaled.**"""
    return np.linalg.inv(cam2world[anchor]) @ cam2world[f]


def relative_transforms(pose_pred: np.ndarray, cam2world_gt: np.ndarray, anchor: int,
                        pose_mode: str, scale: float) -> List[np.ndarray]:
    """One 4x4 per frame taking its camera points into the anchor camera frame."""
    T = pose_pred.shape[0]
    if pose_mode == "pred":
        return [np.eye(4) if f == anchor else predicted_relative(pose_pred, f, anchor, scale)
                for f in range(T)]
    if pose_mode == "gt":
        # scale must NOT reach this branch: GT translations are metric
        return [np.eye(4) if f == anchor else gt_relative(cam2world_gt, f, anchor)
                for f in range(T)]
    raise ValueError(f"unknown pose_mode {pose_mode!r}")


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #
def fuse(dep: np.ndarray, conf: np.ndarray, K: np.ndarray,
         rel: Sequence[np.ndarray], T_anchor_camera_to_ego: np.ndarray,
         depth_scale: float, conf_thr: float, dmin: float, dmax: float):
    """Frozen fusion with externally supplied relative transforms.

    Identical in every respect to ``occ3d_zeroshot.pipeline.fuse_to_ego`` except that the
    relative transforms are passed in, so the predicted- and GT-pose branches share one
    code path and cannot diverge in unroll order, masking or intrinsics.
    """
    depth_m = float(depth_scale) * dep.astype(np.float64)
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
            Tr = rel[f]
            p = p @ Tr[:3, :3].T + Tr[:3, 3]
        pts.append(p)
        fr.append(np.full(len(p), f, np.int8))
        cf.append(conf[f][m].astype(np.float32))
        dp.append(d.astype(np.float32))
    if not pts:
        z = np.zeros((0,), np.float32)
        return np.zeros((0, 3)), np.zeros((0,), np.int8), z, z
    P = np.concatenate(pts, 0)
    R, t = T_anchor_camera_to_ego[:3, :3], T_anchor_camera_to_ego[:3, 3]
    return (P @ R.T + t, np.concatenate(fr), np.concatenate(cf), np.concatenate(dp))


# --------------------------------------------------------------------------- #
# Pose error diagnostics
# --------------------------------------------------------------------------- #
def pose_errors(pose_pred: np.ndarray, cam2world_gt: np.ndarray, anchor: int,
                scale: float) -> List[Dict[str, float]]:
    """Scaled LingBot relative pose vs the nuScenes ground-truth relative pose."""
    out = []
    for f in range(pose_pred.shape[0]):
        if f == anchor:
            continue
        Tp = predicted_relative(pose_pred, f, anchor, scale)
        Tg = gt_relative(cam2world_gt, f, anchor)
        dR = Tg[:3, :3].T @ Tp[:3, :3]
        rot = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1.0, 1.0))))
        tp, tg = Tp[:3, 3], Tg[:3, 3]
        np_, ng = float(np.linalg.norm(tp)), float(np.linalg.norm(tg))
        if np_ > 1e-9 and ng > 1e-9:
            cos = float(np.clip(np.dot(tp, tg) / (np_ * ng), -1.0, 1.0))
            direction = float(np.degrees(np.arccos(cos)))
        else:
            direction = float("nan")
        out.append({"frame": f, "offset": f - anchor,
                    "rot_err_deg": rot,
                    "trans_err_m": float(np.linalg.norm(tp - tg)),
                    "trans_mag_pred_m": np_, "trans_mag_gt_m": ng,
                    "trans_mag_ratio": (np_ / ng) if ng > 1e-9 else float("nan"),
                    "trans_dir_err_deg": direction})
    return out


# --------------------------------------------------------------------------- #
# Depth-shape diagnostics (LiDAR is ground truth here, never a control input)
# --------------------------------------------------------------------------- #
def depth_metrics(pred_canonical: np.ndarray, gt_m: np.ndarray, valid: np.ndarray,
                  scale: float) -> Dict[str, float]:
    """Metrics of ``scale * pred`` against projected metric LiDAR at valid pixels."""
    p = float(scale) * pred_canonical.astype(np.float64)
    g = gt_m.astype(np.float64)
    m = valid & np.isfinite(p) & (p > 0) & (g > 0)
    if m.sum() < 1:
        return {k: float("nan") for k in
                ("abs_rel", "median_abs_rel", "median_abs_log", "rmse_m", "delta1")} | \
               {"n_valid": 0}
    p, g = p[m], g[m]
    rel = np.abs(p - g) / g
    ratio = np.maximum(p / g, g / p)
    return {"abs_rel": float(rel.mean()),
            "median_abs_rel": float(np.median(rel)),
            "median_abs_log": float(np.median(np.abs(np.log(p) - np.log(g)))),
            "rmse_m": float(np.sqrt(((p - g) ** 2).mean())),
            "delta1": float((ratio < 1.25).mean()),
            "n_valid": int(m.sum())}


def ground_height(points_ego: np.ndarray, x_range: Tuple[float, float],
                  abs_y_max: float, percentile: float) -> float:
    """Low percentile of ego z over a front-of-vehicle band. Ground should sit near 0."""
    if len(points_ego) == 0:
        return float("nan")
    m = ((points_ego[:, 0] > x_range[0]) & (points_ego[:, 0] < x_range[1])
         & (np.abs(points_ego[:, 1]) < abs_y_max))
    if m.sum() < 50:
        return float("nan")
    return float(np.percentile(points_ego[m, 2], percentile))

