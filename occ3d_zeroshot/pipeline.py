"""The frozen SemanticKITTI stack, applied unchanged to Occ3D-nuScenes.

Nothing here is fitted, tuned or selected on nuScenes. Every scalar comes from Gate 0-3.1:

    s0    = 27.3665   constant scale fitted on KITTI sequences 00-07
    tau   = 0.45      threshold selected on KITTI sequences 09-10
    r     = 3 voxels at 0.2 m = 0.6 m physical correction radius
    conf >= 1.5, 1 m < depth < 60 m

**Coordinate chain** (each transform named, none ambiguous):

    p_camera[t]              unproject(depth[t] * s, K_pred[t])
    T_camera_to_lingbot_world[t]
                             LingBot pred_pose_c2w[t], canonical translation units
    T_frame_to_anchor[t]     inv(P[anchor]) @ P[t] with translation scaled by s
                             (rotations untouched; anchor is a fixed point)
    T_anchor_camera_to_ego   inv(T_ego_kf_to_world) @ T_ego_cam_to_world
                             @ T_camera_to_ego_cam, read at the ANCHOR frame only
    T_ego_to_occ_grid        floor((p_ego - origin) / voxel_size), origin (-40,-40,-1)

LingBot supplies every relative pose between the five frames. The only ground-truth
calibration used is the anchor frame's static camera extrinsic plus its single-timestamp
ego correction, which places the finished reconstruction in the official evaluation frame.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_ROOT, os.path.join(_ROOT, "tools", "depth_gate")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from decompose_residual import scaled_relative_pose                          # noqa: E402
from gates.depth_gate.models import build_inputs, build_model                      # noqa: E402
from gates.voxel_gate.voxels import build_input, dense_from_sparse, dilate, sparse_voxel_features  # noqa: E402
from gates.voxel_gate.models import VoxelCorrector3D, apply_region                 # noqa: E402

from .grid import CANONICAL, occupancy_canonical                            # noqa: E402


# --------------------------------------------------------------------------- #
# Frozen clip scale from the Gate-2 depth head (scalar only; r_shape discarded)
# --------------------------------------------------------------------------- #
def load_depth_head(path: str, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = build_model(ck["arch"], ck["in_ch"], ck["base_channels"], ck["max_log_residual"])
    m.load_state_dict(ck["state_dict"])
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, ck


def clip_scale(model, ck, dep: np.ndarray, conf: np.ndarray, s0: float, conf_thr: float,
               dmin: float, dmax: float, device) -> Tuple[float, float, np.ndarray, float]:
    """``(a_clip, s_learned, support, saturation)``. Identical to Gate 3.1's extraction."""
    depf = dep.astype(np.float64)
    base0 = depf * s0
    batch = {
        "base_depth": torch.from_numpy(base0.astype(np.float32)).unsqueeze(1).to(device),
        "lingbot_confidence": torch.from_numpy(conf).unsqueeze(1).to(device),
        "valid_lingbot_mask": torch.from_numpy(
            np.isfinite(base0) & (base0 > 0)).unsqueeze(1).to(device)}
    if ck["use_rgb"]:
        raise RuntimeError("frozen depth head must be the geometry-only depth_cnn")
    with torch.no_grad():
        r = model(build_inputs(batch, ck["stats"], ck["use_rgb"]))[:, 0].double().cpu().numpy()
    support = (conf >= conf_thr) & np.isfinite(base0) & (base0 > dmin) & (base0 < dmax)
    if not support.any():
        return float("nan"), float("nan"), support, 0.0
    a_clip = float(np.median(r[support]))
    # the head's residual is max_log_residual * tanh(.), so |r| near the bound is saturation
    sat = float(np.mean(np.abs(r[support]) > 0.95 * float(ck["max_log_residual"])))
    return a_clip, s0 * float(np.exp(a_clip)), support, sat


# --------------------------------------------------------------------------- #
# Fusion into the anchor camera, then into the official ego grid
# --------------------------------------------------------------------------- #
def fuse_to_ego(dep: np.ndarray, conf: np.ndarray, K: np.ndarray, pose_pred: np.ndarray,
                T_anchor_camera_to_ego: np.ndarray, scale: float, conf_thr: float,
                dmin: float, dmax: float):
    """Frozen C0/C3 fusion. Returns ego-frame points and per-point provenance."""
    depth_m = scale * dep.astype(np.float64)
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
            Trel = scaled_relative_pose(pose_pred, f, anchor, scale)
            p = p @ Trel[:3, :3].T + Trel[:3, 3]
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


def canonical_features(points_ego: np.ndarray, frame: np.ndarray, conf: np.ndarray,
                       depth: np.ndarray, device):
    """Frozen Gate-3.1 evidence channels, built on the canonical 0.2 m grid."""
    from prompted_lingbot.occupancy import VoxelGrid                          # local import
    import gates.voxel_gate.voxels as V
    saved = V.G
    V.G = CANONICAL                                    # same code, canonical dimensions
    try:
        sp = sparse_voxel_features(points_ego, frame, conf, depth)
        feat5 = dense_from_sparse(sp, device)
    finally:
        V.G = saved
    return feat5, sp


def run_corrector(model, feat5: torch.Tensor, region: torch.Tensor, norm: Dict[str, float],
                  tau: float, occ_only: bool):
    x = build_input(feat5, region, norm)
    if occ_only:
        x = x.clone()
        x[1:5] = 0.0
    with torch.no_grad():
        p = torch.sigmoid(model(x[None])[0, 0])
    return apply_region(p >= tau, feat5[0] > 0, region)


def load_corrector(path: str, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = VoxelCorrector3D(6, ck["channels"], ck["n_blocks"], ck["kernel"])
    m.load_state_dict(ck["state_dict"])
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, ck


def region_from(occupied: torch.Tensor, radius_voxels: int) -> torch.Tensor:
    """``R_infer`` — a pure dilation of the frozen geometry. No target array involved."""
    return dilate(occupied, radius_voxels)
