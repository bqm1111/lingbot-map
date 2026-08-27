"""Causal occupancy evaluation over cached LingbotMap predictions.

Protocol, stated precisely because the wording matters
-----------------------------------------------------
LingbotMap is run **once per sequence in streaming mode**, which is causal: the
prediction for frame ``f`` uses only frames ``<= f``.  ``history`` here is the
**aggregation window**: how many causal frames' depth maps are fused into the
volume anchored at frame ``t``, i.e. frames ``[t-history+1 .. t]``.  It is not a
re-run of the backbone with a truncated context.  Both are causal; they answer
different questions, and this one -- "does fusing more causal frames improve
occupancy?" -- is the one that governs recall.  ``--clip-inference`` in the
driver re-runs the backbone per anchor for a spot-check.

A structural fact that shapes the whole experiment
--------------------------------------------------
The volume is built in the **anchor frame's own camera/LiDAR coordinates**.
Under that construction a *global* Sim(3) correction contributes **only its
scale**: its rotation and translation cancel between the points and the pose they
are expressed relative to.  ``test_occupancy.py::
test_a_global_sim3_cancels_in_the_anchor_frame_except_for_scale`` proves it.
So pose prompts can only influence occupancy through the scale they imply --
which is a real and reportable finding, not a defect.

Prompt-point leakage
--------------------
Prompts never contribute points.  Anchors consume ``Prompt`` objects and emit a
``Sim3``; the point cloud is built exclusively from predicted depth.  This is
enforced by construction and asserted by
``test_occ_eval.py::test_removing_prompt_pixels_does_not_change_occupancy``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .anchors import MetricAnchor
from .conventions import Sim3, rotation_geodesic_deg
from .occ_datasets import apply_transform, camera_points_from_depth, relative_c2w
from .occupancy import (
    VoxelGrid, accumulate_scores, binary_occupancy_scores, occupancy_from_points,
)
from .runner import CachedSequence


@dataclass
class OccPointConfig:
    """How predicted depth becomes candidate occupied points."""

    conf_threshold: float = 1.5
    min_depth: float = 0.5          # metres, after scaling
    max_depth: float = 60.0         # metres, after scaling
    pixel_stride: int = 1
    min_points_per_voxel: int = 1
    intrinsics: str = "predicted"   # "predicted" | "dataset" | "focal_corrected"


def _intrinsics(seq: CachedSequence, frame: int, mode: str) -> np.ndarray:
    K_pred = seq.pred_K[frame].copy()
    if mode == "predicted":
        return K_pred
    K_gt = seq.gt_K.copy()
    if mode == "dataset":
        return K_gt
    if mode == "focal_corrected":
        r = float(K_pred[0, 0] / K_gt[0, 0])
        K = K_pred.copy()
        K[0, 0] /= r
        K[1, 1] /= r
        return K
    raise ValueError(f"unknown intrinsics mode {mode!r}")


def points_in_anchor_camera(
    seq: CachedSequence, anchor: int, history: int, scale: float,
    cfg: OccPointConfig, frame_lo: int = 0,
) -> np.ndarray:
    """Fuse frames ``[anchor-history+1 .. anchor]`` into the anchor camera frame.

    Everything is causal: no frame after ``anchor`` is touched.  Depth is scaled
    by ``scale`` (the anchor's estimate) *after* the rigid inter-frame transform,
    which correctly scales both the depth and the inter-frame translation.
    """
    if history < 1:
        raise ValueError("history must be >= 1")
    start = max(frame_lo, anchor - history + 1)
    pose_t = np.eye(4)
    pose_t[:3, :4] = seq.pred_pose_c2w[anchor]

    chunks = []
    for f in range(start, anchor + 1):
        depth = seq.pred_depth[f].astype(np.float32)
        conf = seq.pred_depth_conf[f].astype(np.float32)
        if cfg.pixel_stride > 1:
            depth = depth[::cfg.pixel_stride, ::cfg.pixel_stride]
            conf = conf[::cfg.pixel_stride, ::cfg.pixel_stride]
        K = _intrinsics(seq, f, cfg.intrinsics).copy()
        if cfg.pixel_stride > 1:
            st = cfg.pixel_stride
            K[0, 0] /= st; K[1, 1] /= st
            K[0, 2] = (K[0, 2] - 0.5 * (st - 1)) / st
            K[1, 2] = (K[1, 2] - 0.5 * (st - 1)) / st
        # Depth thresholds are metric, so convert them into model units.
        p_cam_f = camera_points_from_depth(
            depth, K, conf, cfg.conf_threshold,
            cfg.min_depth / max(scale, 1e-9), cfg.max_depth / max(scale, 1e-9))
        if p_cam_f.shape[0] == 0:
            continue
        if f != anchor:
            pose_f = np.eye(4)
            pose_f[:3, :4] = seq.pred_pose_c2w[f]
            p_cam_f = apply_transform(relative_c2w(pose_f, pose_t), p_cam_f)
        chunks.append(p_cam_f)

    if not chunks:
        return np.zeros((0, 3))
    return np.concatenate(chunks, 0) * float(scale)


@dataclass
class AnchorFrameResult:
    frame: int
    history: int
    scores: Dict[str, float]
    n_points: int


def evaluate_anchor_frame(
    seq: CachedSequence, anchor: int, history: int, correction: Sim3,
    grid: VoxelGrid, cam_to_grid: np.ndarray, target: np.ndarray,
    valid: Optional[np.ndarray], cfg: OccPointConfig, frame_lo: int = 0,
) -> AnchorFrameResult:
    """One (anchor frame, history) evaluation."""
    pts_cam = points_in_anchor_camera(seq, anchor, history, correction.s, cfg,
                                      frame_lo=frame_lo)
    pts_grid = apply_transform(cam_to_grid, pts_cam) if pts_cam.shape[0] else pts_cam
    vol = occupancy_from_points(pts_grid, grid,
                                min_points_per_voxel=cfg.min_points_per_voxel)
    scores = binary_occupancy_scores(vol, target, grid, valid=valid)
    return AnchorFrameResult(frame=anchor, history=history, scores=scores,
                             n_points=int(pts_cam.shape[0]))


def pose_errors(seq: CachedSequence, corrections: Sequence[Sim3]) -> Dict[str, float]:
    """Position ATE and absolute orientation RMSE, reported together.

    An anchor is never selected on ATE alone; the orientation term is what caught
    the sliding-window degeneracy.
    """
    C = np.stack([c.apply_centres(seq.pred_pose_c2w[t, :3, 3])
                  for t, c in enumerate(corrections)])
    R = np.stack([c.apply_rotations(seq.pred_pose_c2w[t, :3, :3])
                  for t, c in enumerate(corrections)])
    ate = float(np.sqrt((np.linalg.norm(C - seq.gt_pose_c2w[:, :3, 3], axis=1) ** 2).mean()))
    rot = rotation_geodesic_deg(R, seq.gt_pose_c2w[:, :3, :3])
    return {"ate_rmse_m": ate,
            "abs_rot_rmse_deg": float(np.sqrt((rot ** 2).mean())),
            "abs_rot_median_deg": float(np.median(rot)),
            "final_scale": float(corrections[-1].s)}


# --------------------------------------------------------------------------- #
# Frozen external depth + LingbotMap poses
# --------------------------------------------------------------------------- #
def external_points_in_anchor_camera(
    seq: CachedSequence, external_depth: np.ndarray, anchor: int, history: int,
    translation_scale: float, cfg: OccPointConfig, frame_lo: int = 0,
    poses_c2w: Optional[np.ndarray] = None, K_override: Optional[np.ndarray] = None,
    gate_on_lingbot_conf: bool = False,
) -> np.ndarray:
    """Fuse **external metric depth** into the anchor camera frame.

    The split that defines this experiment:

    * points come from ``external_depth``, which is **already metric** and is never
      rescaled;
    * the inter-frame motion comes from ``poses_c2w`` (LingbotMap's own poses by
      default), whose **translation is in arbitrary units** and is multiplied by
      ``translation_scale``;
    * rotation is unitless and is used as-is.

    So ``p_cam_t = R_rel @ p_ext_cam_f + s * t_rel``. Scaling the whole result --
    the way the LingbotMap-depth path does -- would wrongly rescale metric depth.

    ``gate_on_lingbot_conf`` restricts external depth to the pixels LingbotMap's own
    confidence map keeps. The external model emits no confidence of its own, so
    without this it contributes a point at *every* pixel -- sky, thin vegetation,
    unreconstructable surfaces -- while the LingbotMap path is filtered. Both
    variants are reported: ungated is what the model can do standalone, gated is
    the matched-pixel-support comparison that isolates depth *quality*.

    Causal: only frames ``[anchor-history+1 .. anchor]`` are read.
    """
    if history < 1:
        raise ValueError("history must be >= 1")
    poses = seq.pred_pose_c2w if poses_c2w is None else poses_c2w
    start = max(frame_lo, anchor - history + 1)
    pose_t = np.eye(4)
    pose_t[:3, :4] = poses[anchor]

    chunks = []
    for f in range(start, anchor + 1):
        depth = np.asarray(external_depth[f], np.float32)
        conf = seq.pred_depth_conf[f].astype(np.float32)
        if cfg.pixel_stride > 1:
            depth = depth[::cfg.pixel_stride, ::cfg.pixel_stride]
            conf = conf[::cfg.pixel_stride, ::cfg.pixel_stride]
        K = (_intrinsics(seq, f, cfg.intrinsics) if K_override is None
             else np.asarray(K_override, float)).copy()
        if cfg.pixel_stride > 1:
            st = cfg.pixel_stride
            K[0, 0] /= st; K[1, 1] /= st
            K[0, 2] = (K[0, 2] - 0.5 * (st - 1)) / st
            K[1, 2] = (K[1, 2] - 0.5 * (st - 1)) / st
        # External depth is metric, so the metric thresholds apply directly.
        p_cam_f = camera_points_from_depth(
            depth, K,
            conf if gate_on_lingbot_conf else None,
            cfg.conf_threshold if gate_on_lingbot_conf else 0.0,
            cfg.min_depth, cfg.max_depth)
        if p_cam_f.shape[0] == 0:
            continue
        if f != anchor:
            pose_f = np.eye(4)
            pose_f[:3, :4] = poses[f]
            T = relative_c2w(pose_f, pose_t)
            T = T.copy()
            T[:3, 3] *= float(translation_scale)     # translation only
            p_cam_f = apply_transform(T, p_cam_f)
        chunks.append(p_cam_f)

    if not chunks:
        return np.zeros((0, 3))
    return np.concatenate(chunks, 0)
