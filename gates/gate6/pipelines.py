"""Per-benchmark, target-free construction of the semantic occupancy prediction.

One clip in, two labelled voxel sets out (``B-R`` raw and ``B-D`` dilated). The geometry
is the frozen one; the only new step is that every fused point now also carries the
teacher probability vector of the pixel it came from.

Inputs are RGB-derived only: the frozen LingBot cache, the frozen G51-B scale table, the
frozen Trident cache, and calibration. The prediction stage is executed inside
:class:`gate6.audit.Gate6Audit`, so touching a target, a LiDAR sweep or an oracle table
raises rather than silently succeeding.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from . import frames as F, grids, lifting, vocab
from .frames import KITTI360_SEQUENCE

CONF_THRESHOLD = 1.5          # frozen since Gate 0
MIN_DEPTH_M = 1.0             # frozen
MAX_DEPTH_M = 60.0            # frozen
DILATE_RADIUS_VOXELS = 2      # frozen `dilate_r2` == 0.4 m at 0.2 m voxels

SCALE_TABLE = {
    "semantickitti": "artifacts/gate5_1/scales_G51-B_kitti.csv",
    "occ3d": "artifacts/gate5_1/scales_G51-B_occ3d.csv",
    "kitti360": "artifacts/gate5_2/scales_B.csv",
}
GROUP_FIELD = {"semantickitti": "group", "occ3d": "group", "kitti360": "block"}


def load_scales(dataset: str, repo_root: str) -> Dict[str, Optional[float]]:
    """The frozen G51-B scalar per clip. ``None`` where the gauge declared failure."""
    path = os.path.join(repo_root, SCALE_TABLE[dataset])
    out: Dict[str, Optional[float]] = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            s = float(r["s_moge"]) if int(r["ok"]) else float("nan")
            out[r["clip_id"]] = s if np.isfinite(s) and s > 0 else None
    return out


def load_groups(dataset: str, repo_root: str) -> Dict[str, str]:
    path = os.path.join(repo_root, SCALE_TABLE[dataset])
    fld = GROUP_FIELD[dataset]
    with open(path) as fh:
        return {r["clip_id"]: str(r[fld]) for r in csv.DictReader(fh)}


@dataclass
class ClipGeometry:
    clip_id: str
    group: str
    dep: np.ndarray             # [T, H, W] float32, canonical LingBot depth
    conf: np.ndarray            # [T, H, W] float32
    K: np.ndarray               # [T, 3, 3] float64, processed lattice
    pose: np.ndarray            # [T, 4, 4] float64, predicted camera-to-world
    T_anchor_to_grid: np.ndarray  # 4x4, anchor camera -> the benchmark's grid frame
    frame_keys: Tuple[str, ...]
    scale: Optional[float]


def _as4x4(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, np.float64)
    return p if p.shape[-2:] == (4, 4) else np.concatenate(
        [p, np.tile(np.array([0., 0., 0., 1.]), (len(p), 1, 1))], axis=1)


def iter_clips(dataset: str, repo_root: str) -> Iterator[ClipGeometry]:
    """Yield the frozen geometry of every clip of a benchmark. Opens no target."""
    recs = F.read_manifest(dataset, repo_root)
    scales = load_scales(dataset, repo_root)
    groups = load_groups(dataset, repo_root)
    cam_to_velo = None
    if dataset == "semantickitti":
        from prompted_lingbot.occ_datasets import SemanticKittiOccSpec
        root = os.path.join(repo_root, F.SEMANTICKITTI_ROOT)
        cam_to_velo = {}

    for rec in recs:
        cid = rec.clip_id
        lp = F.lingbot_cache_path(dataset, cid, repo_root)
        if not os.path.exists(lp):
            continue
        d = np.load(lp, allow_pickle=False)
        dep = d["pred_depth"].astype(np.float32)
        conf = d["pred_depth_conf"].astype(np.float32)
        K = d["pred_K"].astype(np.float64)
        pose = _as4x4(d["pred_pose_c2w"].astype(np.float64))
        if dataset == "semantickitti":
            seq = rec.raw["sequence"]
            if seq not in cam_to_velo:
                cam_to_velo[seq] = SemanticKittiOccSpec.build(root, seq).cam_to_velo
            T = cam_to_velo[seq]
        elif dataset == "kitti360":
            T = d["rect_cam_to_velo"].astype(np.float64)
        else:
            T = d["T_camera_to_ego"][-1].astype(np.float64)
        yield ClipGeometry(clip_id=cid, group=groups.get(cid, "?"), dep=dep, conf=conf,
                           K=K, pose=pose, T_anchor_to_grid=T, frame_keys=rec.keys,
                           scale=scales.get(cid))


def load_semantics(dataset: str, keys: Tuple[str, ...], device) -> Optional[torch.Tensor]:
    """Stack the per-frame teacher caches into ``[T, C, H, W]`` on ``device``."""
    out = []
    for k in keys:
        p = F.semantic_cache_path(dataset, k)
        if not os.path.exists(p):
            return None
        with np.load(p) as z:
            out.append(torch.from_numpy(z["probs"].astype(np.float32)))
    return torch.stack(out, 0).to(device)


@dataclass
class ClipPrediction:
    clip_id: str
    group: str
    scale: float
    n_points: int
    raw_flat: np.ndarray        # eval-grid flat indices, B-R
    raw_channel: np.ndarray     # teacher channel (argmax), B-R
    dil_flat: np.ndarray        # eval-grid flat indices, B-D
    dil_channel: np.ndarray
    dil_support: np.ndarray     # 1 = reconstruction support, 0 = dilation-only


def predict_clip(dataset: str, clip: ClipGeometry, sem: torch.Tensor,
                 device) -> ClipPrediction:
    """Frozen geometry + frozen teacher -> the two labelled voxel sets. No target."""
    G = grids.PREDICTION_GRID[dataset]
    s = float(clip.scale)
    pts, fr, cf, dp, vp, up, _ = lifting.points_with_pixels(
        clip.dep, clip.conf, clip.K, clip.pose, s, CONF_THRESHOLD, MIN_DEPTH_M, MAX_DEPTH_M)
    R, t = clip.T_anchor_to_grid[:3, :3], clip.T_anchor_to_grid[:3, 3]
    pg = pts @ R.T + t if len(pts) else pts

    idx, keep = grids.voxelize(pg, G)
    flat = grids.flat_of(idx, G) if len(idx) else np.zeros((0,), np.int64)
    probs = lifting.sample_probs(sem, fr[keep], vp[keep], up[keep], device) \
        if len(idx) else torch.zeros((0, sem.shape[1]), device=device)
    raw_flat, raw_probs, _ = lifting.fuse_voxel_probs(flat, probs, int(np.prod(G.dims)),
                                                      device)

    # B-D: the frozen dilate_r2, and the frozen propagation of the probability vector.
    occ = torch.zeros(int(np.prod(G.dims)), dtype=torch.bool, device=device)
    if len(raw_flat):
        occ[torch.from_numpy(raw_flat).to(device)] = True
    from gates.voxel_gate.voxels import dilate
    dil = dilate(occ.view(G.dims), DILATE_RADIUS_VOXELS).reshape(-1)
    only = (dil & ~occ).nonzero(as_tuple=True)[0].cpu().numpy().astype(np.int64)
    only_probs, assigned = lifting.propagate_dilation(
        tuple(G.dims), raw_flat, raw_probs, only, DILATE_RADIUS_VOXELS, device)
    if len(only) and not assigned.all():
        raise AssertionError("dilation is extensive: every dilation-only voxel must have "
                             "a source inside the same radius")

    dil_flat = np.concatenate([raw_flat, only])
    dil_probs = torch.cat([raw_probs, only_probs], 0) if len(only) else raw_probs
    dil_support = np.concatenate([np.ones(len(raw_flat), np.uint8),
                                  np.zeros(len(only), np.uint8)])

    if grids.NEEDS_REDUCTION[dataset]:
        raw_flat_n, raw_probs_n = (grids.reduce_occ3d_probs(raw_flat, raw_probs)
                                   if len(raw_flat) else
                                   (np.zeros((0,), np.int64), raw_probs))
        dil_flat_n, dil_probs_n = (grids.reduce_occ3d_probs(dil_flat, dil_probs)
                                   if len(dil_flat) else
                                   (np.zeros((0,), np.int64), dil_probs))
        # a native voxel has reconstruction support iff any of its subvoxels is raw
        sup = np.isin(dil_flat_n, raw_flat_n).astype(np.uint8)
        raw_flat, raw_probs, dil_flat, dil_probs, dil_support = (
            raw_flat_n, raw_probs_n, dil_flat_n, dil_probs_n, sup)

    return ClipPrediction(
        clip_id=clip.clip_id, group=clip.group, scale=s, n_points=int(len(pts)),
        raw_flat=raw_flat.astype(np.int32),
        raw_channel=(raw_probs.argmax(1).cpu().numpy().astype(np.uint8)
                     if len(raw_flat) else np.zeros((0,), np.uint8)),
        dil_flat=dil_flat.astype(np.int32),
        dil_channel=(dil_probs.argmax(1).cpu().numpy().astype(np.uint8)
                     if len(dil_flat) else np.zeros((0,), np.uint8)),
        dil_support=dil_support)


__all__ = ["iter_clips", "load_semantics", "predict_clip", "load_scales", "load_groups",
           "ClipGeometry", "ClipPrediction", "CONF_THRESHOLD", "MIN_DEPTH_M",
           "MAX_DEPTH_M", "DILATE_RADIUS_VOXELS", "SCALE_TABLE"]
