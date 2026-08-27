"""Evaluation metrics.

The primary metrics are deliberately computed **without any scale alignment**,
because scale is exactly the quantity under test.  Sim(3)-aligned numbers are
reported alongside, clearly labelled as diagnostics.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .conventions import (
    Sim3,
    camera_to_world,
    depth_to_camera_points,
    rotation_geodesic_deg,
    umeyama_sim3,
)

POINT_THRESHOLDS_M = (0.05, 0.10, 0.25, 0.50)


# --------------------------------------------------------------------------- #
# Depth
# --------------------------------------------------------------------------- #
def depth_metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray,
                  max_depth: float = np.inf) -> Dict[str, float]:
    """Per-pixel metric depth error with NO scale alignment.

    Args:
        pred: ``(..., H, W)`` corrected metric depth.
        gt: ``(..., H, W)`` ground-truth metric depth.
        valid: ``(..., H, W)`` bool.
    """
    m = valid & np.isfinite(pred) & np.isfinite(gt) & (gt > 1e-3) & (pred > 1e-6) & (gt < max_depth)
    if not m.any():
        return {k: float("nan") for k in
                ("abs_rel", "rmse", "rmse_log", "sq_rel", "mae", "delta_1_25", "log_scale_bias", "n")}
    p, g = pred[m].astype(np.float64), gt[m].astype(np.float64)
    diff = p - g
    ratio = np.maximum(p / g, g / p)
    return {
        "abs_rel": float(np.mean(np.abs(diff) / g)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "rmse_log": float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))),
        "sq_rel": float(np.mean(diff ** 2 / g)),
        "mae": float(np.mean(np.abs(diff))),
        "delta_1_25": float(np.mean(ratio < 1.25)),
        "log_scale_bias": float(np.median(np.log(g) - np.log(p))),
        "n": int(m.sum()),
    }


def per_frame_depth_scale_error(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """``(S,)`` per-frame ``log(gt / pred)`` median -- the residual scale error."""
    out = np.full(pred.shape[0], np.nan)
    for i in range(pred.shape[0]):
        m = valid[i] & (gt[i] > 1e-3) & (pred[i] > 1e-6)
        if m.sum() >= 16:
            out[i] = float(np.median(np.log(gt[i][m].astype(np.float64))
                                     - np.log(pred[i][m].astype(np.float64))))
    return out


# --------------------------------------------------------------------------- #
# Trajectory
# --------------------------------------------------------------------------- #
def ate(pred_centres: np.ndarray, gt_centres: np.ndarray) -> Dict[str, float]:
    """Absolute trajectory error with NO alignment of any kind."""
    d = np.linalg.norm(pred_centres - gt_centres, axis=1)
    return {"ate_rmse_m": float(np.sqrt(np.mean(d ** 2))),
            "ate_mean_m": float(np.mean(d)),
            "ate_median_m": float(np.median(d)),
            "ate_max_m": float(np.max(d))}


def ate_aligned(pred_centres: np.ndarray, gt_centres: np.ndarray, with_scale: bool) -> Dict[str, float]:
    """ATE after a global SE(3) (``with_scale=False``) or Sim(3) alignment."""
    S = umeyama_sim3(pred_centres, gt_centres, with_scale=with_scale)
    d = np.linalg.norm(S.apply_points(pred_centres) - gt_centres, axis=1)
    tag = "sim3" if with_scale else "se3"
    return {f"ate_{tag}_rmse_m": float(np.sqrt(np.mean(d ** 2))),
            f"ate_{tag}_median_m": float(np.median(d)),
            f"alignment_scale_{tag}": float(S.s)}


def relative_pose_error(pred_poses: np.ndarray, gt_poses: np.ndarray, delta: int = 1) -> Dict[str, float]:
    """RPE over a fixed frame gap, for camera-to-world poses.

    Translation RPE is reported both in metres and as a percentage of the
    ground-truth motion over the same gap, so it is comparable across datasets.
    """
    n = len(pred_poses)
    if n <= delta:
        return {"rpe_trans_m": float("nan"), "rpe_trans_pct": float("nan"),
                "rpe_rot_deg": float("nan")}
    tr, rr, gt_step = [], [], []
    for i in range(n - delta):
        j = i + delta
        Rp_i, tp_i = pred_poses[i][:3, :3], pred_poses[i][:3, 3]
        Rp_j, tp_j = pred_poses[j][:3, :3], pred_poses[j][:3, 3]
        Rg_i, tg_i = gt_poses[i][:3, :3], gt_poses[i][:3, 3]
        Rg_j, tg_j = gt_poses[j][:3, :3], gt_poses[j][:3, 3]
        dp_t = Rp_i.T @ (tp_j - tp_i)
        dg_t = Rg_i.T @ (tg_j - tg_i)
        tr.append(np.linalg.norm(dp_t - dg_t))
        rr.append(rotation_geodesic_deg(Rp_i.T @ Rp_j, Rg_i.T @ Rg_j))
        gt_step.append(np.linalg.norm(dg_t))
    tr = np.asarray(tr)
    gt_step = np.asarray(gt_step)
    total = float(gt_step.sum())
    return {"rpe_trans_m": float(np.sqrt(np.mean(tr ** 2))),
            "rpe_trans_pct": float(100.0 * tr.sum() / total) if total > 1e-9 else float("nan"),
            "rpe_rot_deg": float(np.sqrt(np.mean(np.asarray(rr) ** 2)))}


def rotation_error_deg(pred_poses: np.ndarray, gt_poses: np.ndarray) -> Dict[str, float]:
    ang = rotation_geodesic_deg(pred_poses[:, :3, :3], gt_poses[:, :3, :3])
    return {"abs_rot_rmse_deg": float(np.sqrt(np.mean(ang ** 2))),
            "abs_rot_median_deg": float(np.median(ang))}


def travelled_distance(gt_centres: np.ndarray) -> np.ndarray:
    """``(S,)`` cumulative ground-truth path length at each frame."""
    step = np.linalg.norm(np.diff(gt_centres, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(step)])


def error_vs_distance(pred_centres: np.ndarray, gt_centres: np.ndarray,
                      bins_m: Sequence[float] = (10, 25, 50, 100, 200, 400, 800)) -> Dict[str, float]:
    """Position error binned by how far the camera has travelled."""
    dist = travelled_distance(gt_centres)
    err = np.linalg.norm(pred_centres - gt_centres, axis=1)
    out = {}
    lo = 0.0
    for hi in bins_m:
        m = (dist >= lo) & (dist < hi)
        out[f"ate_m_at_{int(lo)}_{int(hi)}m"] = float(np.sqrt(np.mean(err[m] ** 2))) if m.any() else float("nan")
        lo = hi
    m = dist >= lo
    out[f"ate_m_at_{int(lo)}m_plus"] = float(np.sqrt(np.mean(err[m] ** 2))) if m.any() else float("nan")
    return out


# --------------------------------------------------------------------------- #
# Frames since the last prompt
# --------------------------------------------------------------------------- #
def frames_since_prompt(prompted: np.ndarray) -> np.ndarray:
    """``(S,)`` number of frames since the most recent prompt (``-1`` before any)."""
    out = np.full(len(prompted), -1, dtype=int)
    last = None
    for i, p in enumerate(prompted):
        if p:
            last = i
        out[i] = -1 if last is None else i - last
    return out


def error_vs_frames_since_prompt(
    err: np.ndarray, since: np.ndarray,
    bins: Sequence[Tuple[int, int]] = ((0, 0), (1, 4), (5, 9), (10, 29), (30, 99), (100, 10 ** 9)),
) -> Dict[str, float]:
    out = {}
    for lo, hi in bins:
        m = (since >= lo) & (since <= hi) & np.isfinite(err)
        label = f"{lo}" if lo == hi else (f"{lo}plus" if hi > 10 ** 8 else f"{lo}_{hi}")
        out[f"gap_{label}"] = float(np.sqrt(np.mean(err[m] ** 2))) if m.any() else float("nan")
        out[f"gap_{label}_n"] = int(m.sum())
    return out


# --------------------------------------------------------------------------- #
# Point cloud
# --------------------------------------------------------------------------- #
def build_point_cloud(depth: np.ndarray, valid: np.ndarray, K: np.ndarray,
                      poses_c2w: np.ndarray, frame_stride: int = 20,
                      points_per_frame: int = 4000, seed: int = 0,
                      max_depth: float = np.inf) -> np.ndarray:
    """Sub-sampled world-frame point cloud from a depth sequence."""
    rng = np.random.default_rng(seed)
    chunks = []
    for i in range(0, len(depth), frame_stride):
        d = depth[i].astype(np.float64)
        m = valid[i] & np.isfinite(d) & (d > 1e-3) & (d < max_depth)
        if not m.any():
            continue
        cam = depth_to_camera_points(d, K)
        pts = cam[m]
        if pts.shape[0] > points_per_frame:
            pts = pts[rng.choice(pts.shape[0], points_per_frame, replace=False)]
        chunks.append(camera_to_world(pts, poses_c2w[i]))
    return np.concatenate(chunks, 0) if chunks else np.zeros((0, 3))


def point_cloud_metrics(pred_pts: np.ndarray, gt_pts: np.ndarray,
                        thresholds: Sequence[float] = POINT_THRESHOLDS_M,
                        query_workers: int = 2) -> Dict[str, float]:
    """Accuracy / completeness / F1 with NO scale alignment.

    ``query_workers`` is kept small on purpose: this runs inside a process pool,
    so letting SciPy grab every core would oversubscribe the machine.
    """
    from scipy.spatial import cKDTree

    if pred_pts.shape[0] < 10 or gt_pts.shape[0] < 10:
        return {"acc_mean_m": float("nan"), "comp_mean_m": float("nan"),
                **{f"f1_at_{t}": float("nan") for t in thresholds}}
    d_pred = cKDTree(gt_pts).query(pred_pts, k=1, workers=query_workers)[0]   # accuracy
    d_gt = cKDTree(pred_pts).query(gt_pts, k=1, workers=query_workers)[0]     # completeness
    out = {"acc_mean_m": float(np.mean(d_pred)), "acc_median_m": float(np.median(d_pred)),
           "comp_mean_m": float(np.mean(d_gt)), "comp_median_m": float(np.median(d_gt)),
           "chamfer_m": float(0.5 * (np.mean(d_pred) + np.mean(d_gt)))}
    for t in thresholds:
        prec = float(np.mean(d_pred < t))
        rec = float(np.mean(d_gt < t))
        out[f"precision_at_{t}"] = prec
        out[f"recall_at_{t}"] = rec
        out[f"f1_at_{t}"] = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return out


# --------------------------------------------------------------------------- #
def aggregate(rows: List[Dict[str, float]]) -> Dict[str, float]:
    """Mean over sequences of every finite numeric field."""
    if not rows:
        return {}
    keys = set().union(*(r.keys() for r in rows))
    out = {}
    for k in sorted(keys):
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float, np.floating))
                and np.isfinite(r[k])]
        if vals:
            out[k] = float(np.mean(vals))
            out[f"{k}__sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            out[f"{k}__n"] = len(vals)
    return out
