"""Deployable causal geometry: a local-reset five-frame LingBot clip -> hard map channels (S, F).

Conventions, stated once:

* A **clip** is the frozen LingBot-Map output on the five images ``t-4..t`` alone (KV cache
  cleared, clip-local frame indices 0..4). Its depth ``pred_depth[q]`` is **canonical** (the
  model's own unit), its poses ``pred_pose_c2w[q]`` are canonical camera-to-world with the
  clip's first frame as origin, its ``pred_K[q]`` are intrinsics of the processed lattice.
* ``scale`` is the causal MoGe-2 clip scale ``exp(median_q log_s[q])`` (metres per canonical
  unit), computed from the clip's own five frames. It is applied **exactly once**, inside
  ``IncrementalMapper._integrate`` via ``scale_state.force(scale)``, to depth *and* to the
  pose translation, so the two stay mutually consistent. Nothing here pre-multiplies depth.
* The map's world frame is the **anchor camera** (frame ``t``): every pose is re-based as
  ``inv(P_4) @ P_q`` so the anchor is the identity. The benchmark grid is then placed with
  ``T_grid_to_world = inv(T_cam_to_grid)`` where ``T_cam_to_grid`` maps the anchor camera
  to the grid frame (velodyne / ego), in metres.
* Output: ``S`` = mapper occupied (+-0.2 m band, conf >= 1.5, 1-60 m), ``F`` = mapper free
  (carved to 0.2 m before the surface at stride 4), as flat boolean tensors on the grid.
  The network input is ``[S, 0, F]``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch

from core.mapping.mapper import FrameInput, IncrementalMapper
from tools.baseline_reconstruction import config as C


@dataclass(frozen=True)
class LocalClip:
    """One local-reset clip as saved by tools/local_registration_audit/run_local_lingbot.py."""
    anchor_index: int                 # stream index t of the anchor
    stream_indices: np.ndarray        # [5] = t-4..t
    keys: np.ndarray                  # [5] frame keys
    depth_canonical: np.ndarray       # [5, H, W] LingBot canonical depth (NOT metric)
    conf: np.ndarray                  # [5, H, W] LingBot confidence
    pose_c2w_canonical: np.ndarray    # [5, 4, 4] canonical camera-to-world, clip frame 0 = origin
    K: np.ndarray                     # [5, 3, 3] processed-lattice intrinsics
    scale_m_per_unit: float           # causal MoGe-2 clip scale, metres per canonical unit
    log_s_frames: np.ndarray          # [5] per-frame log-scale candidates (nan where MoGe missing)

    @staticmethod
    def load(path: str) -> "LocalClip":
        with np.load(path) as z:
            clip = LocalClip(int(z["t"]), z["stream_indices"].astype(np.int64), z["keys"],
                             z["pred_depth"].astype(np.float32), z["pred_depth_conf"].astype(np.float32),
                             z["pred_pose_c2w"].astype(np.float64), z["pred_K"].astype(np.float64),
                             float(z["scale"]), z["log_s_frames"].astype(np.float64))
        assert list(clip.stream_indices) == [clip.anchor_index + o for o in C.CLIP_OFFSETS], "clip is not t-4..t"
        assert np.isfinite(clip.scale_m_per_unit) and clip.scale_m_per_unit > 0, "clip has no causal scale"
        return clip


def frame_inputs(clip: LocalClip, device: torch.device) -> list:
    """The five mapper inputs, anchor camera = identity, depth still canonical (scaled by the mapper)."""
    P_anchor_inv = np.linalg.inv(clip.pose_c2w_canonical[-1])
    log_s = float(np.log(clip.scale_m_per_unit))
    return [FrameInput(index=q, depth_canonical=torch.from_numpy(clip.depth_canonical[q]).to(device),
                       conf=torch.from_numpy(clip.conf[q]).to(device), K=clip.K[q],
                       pose_c2w_canonical=P_anchor_inv @ clip.pose_c2w_canonical[q], log_scale_candidate=log_s)
            for q in range(len(C.CLIP_OFFSETS))]


def build_local_geometry(clip: LocalClip, grid, T_cam_to_grid: np.ndarray, device: torch.device
                         ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    """``(S, F, stats)`` on ``grid`` for one clip. ``T_cam_to_grid``: anchor camera -> grid frame, metres."""
    mapper = IncrementalMapper(device, sem_into=None, n_teacher=0)      # semantics off: the hard map H
    mapper.scale_state.force(clip.scale_m_per_unit)                      # the single place the scale enters
    n_rays = 0
    for f in frame_inputs(clip, device):
        n_rays += int(mapper.step(f).get("n_rays", 0))
    T_grid_to_world = np.linalg.inv(np.asarray(T_cam_to_grid, np.float64))
    q = mapper.query(grid, T_grid_to_world)
    S, F = q["occupied"].bool(), q["free"].bool()
    return S, F, {"n_rays": n_rays, "n_S": int(S.sum()), "n_F": int(F.sum()), "n_rows": len(mapper.table)}


def network_input(S: torch.Tensor, F: torch.Tensor, dims) -> torch.Tensor:
    """``[1, 3, X, Y, Z]`` float: channels [S, 0, F]."""
    X, Y, Z = (int(d) for d in dims)
    S3, F3 = S.reshape(X, Y, Z), F.reshape(X, Y, Z)
    return torch.stack([S3, torch.zeros_like(S3), F3]).float()[None]


__all__ = ["LocalClip", "frame_inputs", "build_local_geometry", "network_input"]
