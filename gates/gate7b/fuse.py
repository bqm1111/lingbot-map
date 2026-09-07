"""Materialise the causal map for one evaluation timestamp, in the benchmark's own grid.

The map is rebuilt from canonical observations at each official timestamp rather than
carried as one global metric volume. That is the brief's permitted rematerialisation, and
it is what makes the gauge honest: a scale that changes at time ``t`` transforms **every**
past observation, not only the new ones, so a running median can never leave two copies of
the same wall in the map. Its cost is measured and reported.

Nothing outside the causal prefix is read. The contributing set at time ``t`` is

    {f : f <= t}  intersect  {f : within the horizon}  intersect  {f : within reach}

where *reach* is a purely geometric prefilter -- a camera further from the evaluation box
than ``max_depth + box diagonal`` cannot place a voxel inside it -- and therefore removes
no frame that could have contributed.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # gates/<pkg>/ -> repo root
for _p in (_ROOT, os.path.join(_ROOT, "tools", "depth_gate")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from decompose_residual import scaled_relative_pose                      # noqa: E402

from . import evidence as EV, rays as RY
from .voxmap import EvidenceVolume, SRC_LINGBOT, SRC_MOGE, W_LINGBOT

HORIZONS = (1, 5, 20, 50, "all")


def horizon_start(t: int, horizon) -> int:
    """First frame index of the causal window ending at ``t`` (inclusive)."""
    if horizon == "all" or horizon is None:
        return 0
    return max(0, int(t) - int(horizon) + 1)


def reach_mask(poses_c2w: np.ndarray, t: int, scale: float, T_anchor_to_grid: np.ndarray,
               grid, max_depth_m: float) -> np.ndarray:
    """Which frames are geometrically capable of writing into the anchor's box."""
    n = len(poses_c2w)
    centres = np.zeros((n, 3))
    for f in range(n):
        T = scaled_relative_pose(poses_c2w, f, t, scale)
        centres[f] = T[:3, 3]
    c = centres @ T_anchor_to_grid[:3, :3].T + T_anchor_to_grid[:3, 3]
    lo = np.asarray(grid.origin, float)
    hi = lo + np.asarray(grid.dims, float) * float(grid.voxel_size)
    box_c = 0.5 * (lo + hi)
    box_r = 0.5 * float(np.linalg.norm(hi - lo))
    return np.linalg.norm(c - box_c, axis=1) <= (max_depth_m + box_r)


def fuse_window(vol: EvidenceVolume, stream: Dict[str, np.ndarray], frames: Sequence[int],
                anchor: int, scale_at, T_anchor_to_grid: np.ndarray, grid,
                variant: str, conf_threshold: float, device,
                moge_loader=None, sem_loader=None, veto_free: bool = False,
                max_depth_m: float = EV.MAX_DEPTH_M) -> Dict[str, float]:
    """Fuse ``frames`` (already causal and ordered) into ``vol``.

    ``scale_at(f)`` returns the gauge in force at frame ``f``. Because the whole window is
    rematerialised, the scale actually applied to *every* frame is the one in force at the
    evaluation timestamp -- ``scale_at(anchor)`` -- which is what keeps old and new
    observations in one consistent metric frame.
    """
    poses = stream["pred_pose_c2w"].astype(np.float64)
    Ks = stream["pred_K"].astype(np.float64)
    dep = stream["pred_depth"]
    cnf = stream["pred_depth_conf"]
    s = float(scale_at(anchor))
    origin = torch.as_tensor(np.asarray(grid.origin, np.float64), device=device)
    Tg = torch.as_tensor(np.asarray(T_anchor_to_grid, np.float64), device=device)
    eye = torch.eye(4, dtype=torch.float64, device=device)
    stats = {"n_frames": 0, "n_rays": 0, "n_occ_updates": 0, "n_free_updates": 0,
             "n_moge_rays": 0, "scale": s}
    dirs_cache: Dict[int, torch.Tensor] = {}

    for f in frames:
        d_can = torch.as_tensor(dep[f].astype(np.float32), device=device)
        conf = torch.as_tensor(cnf[f].astype(np.float32), device=device)
        mg = mgm = None
        if variant in ("S3", "S4") and moge_loader is not None:
            got = moge_loader(f)
            if got is not None:
                mg = torch.as_tensor(got[0], device=device)
                mgm = torch.as_tensor(got[1], device=device)
        acc = EV.accept(variant, d_can, conf, s, mg, mgm, conf_threshold=conf_threshold)

        key = f if Ks.ndim == 3 else 0
        if key not in dirs_cache:
            dirs_cache[key] = RY.pixel_rays(Ks[f], d_can.shape, device)
        dirs = dirs_cache[key]

        Trel = np.eye(4) if f == anchor else scaled_relative_pose(poses, f, anchor, s)
        Tcw = Tg @ torch.as_tensor(Trel, device=device)

        probs = w_sem = None
        if sem_loader is not None:
            p = sem_loader(f)
            if p is not None:
                probs = p.to(device)
                w_sem = EV.semantic_weight(variant, acc, conf)

        st = RY.cast_frame(vol, dirs, acc.depth_lingbot_m, acc.lingbot, Tcw, eye, origin,
                           grid.voxel_size, grid.dims, time_index=f, source=SRC_LINGBOT,
                           weight=W_LINGBOT, probs=probs, sem_weight=w_sem, carve=True,
                           max_depth_m=max_depth_m)
        stats["n_rays"] += st["n_rays"]
        stats["n_occ_updates"] += st["n_occ_updates"]
        stats["n_free_updates"] += st["n_free_updates"]

        if variant in ("S3", "S4") and acc.moge.any():
            keep = acc.moge
            if veto_free:
                keep = _veto(vol, dirs, acc.depth_moge_m, keep, Tcw, origin, grid, device)
            if keep.any():
                w = float(acc.moge_weight[keep].mean().item())
                st2 = RY.cast_frame(vol, dirs, acc.depth_moge_m, keep, Tcw, eye, origin,
                                    grid.voxel_size, grid.dims, time_index=f,
                                    source=SRC_MOGE, weight=w,
                                    probs=probs,
                                    sem_weight=(acc.moge_weight if probs is not None
                                                else None),
                                    carve=False, max_depth_m=max_depth_m)
                stats["n_moge_rays"] += st2["n_rays"]
                stats["n_occ_updates"] += st2["n_occ_updates"]
        stats["n_frames"] += 1
    return stats


def _veto(vol: EvidenceVolume, dirs: torch.Tensor, depth_m: torch.Tensor,
          keep: torch.Tensor, Tcw: torch.Tensor, origin, grid, device) -> torch.Tensor:
    """Refuse MoGe candidates landing where the causal map has already carved free."""
    sel = keep.reshape(-1).nonzero(as_tuple=True)[0]
    if sel.numel() == 0:
        return keep
    p = dirs.reshape(-1, 3)[sel] * depth_m.reshape(-1)[sel].unsqueeze(1)
    pw = p @ Tcw[:3, :3].T + Tcw[:3, 3]
    flat, ok = RY.to_grid(pw, torch.eye(4, dtype=torch.float64, device=device), origin,
                          grid.voxel_size, grid.dims)
    allowed = torch.zeros(sel.numel(), dtype=torch.bool, device=device)
    allowed[ok] = vol.logodds[flat[ok]] >= EV.S4_FREE_VETO_L
    out = torch.zeros_like(keep.reshape(-1))
    out[sel] = allowed
    return out.reshape(keep.shape)


__all__ = ["fuse_window", "horizon_start", "reach_mask", "HORIZONS"]
