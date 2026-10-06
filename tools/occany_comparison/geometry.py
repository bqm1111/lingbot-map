"""Hard map channels (S, F) from a five-frame clip whose target may be any of the five slots.

The frozen ``tools/baseline_reconstruction/current/geometry.py`` fixes the target at the last slot
(``inv(P_4) @ P_q``). OccAny's forward window puts the target first, so the only change here is
``anchor_slot``: poses are re-based as ``inv(P_anchor) @ P_q``. Frames are still integrated once each,
in temporal order, with the clip scale forced once inside the mapper. With ``anchor_slot = 4`` this
reproduces the frozen function bit for bit (``tests/occany_comparison``).

The mapper is built once per clip and queried once per benchmark placement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from core.mapping.mapper import FrameInput, IncrementalMapper


@dataclass(frozen=True)
class WindowClip:
    anchor_slot: int
    depth_canonical: np.ndarray       # [5, H, W]
    conf: np.ndarray                  # [5, H, W]
    pose_c2w_canonical: np.ndarray    # [5, 4, 4], clip frame 0 = origin
    K: np.ndarray                     # [5, 3, 3]
    scale_m_per_unit: float

    @staticmethod
    def load(path: str) -> "WindowClip":
        with np.load(path) as z:
            clip = WindowClip(int(z["anchor_slot"]), z["pred_depth"].astype(np.float32), z["pred_depth_conf"].astype(np.float32),
                              z["pred_pose_c2w"].astype(np.float64), z["pred_K"].astype(np.float64), float(z["scale"]))
        assert np.isfinite(clip.scale_m_per_unit) and clip.scale_m_per_unit > 0, f"{path}: clip has no scale"
        return clip

    @staticmethod
    def from_local(local) -> "WindowClip":
        """A frozen ``LocalClip`` (target = last slot)."""
        return WindowClip(len(local.K) - 1, local.depth_canonical, local.conf, local.pose_c2w_canonical, local.K, local.scale_m_per_unit)


def frame_inputs(clip: WindowClip, device: torch.device) -> list:
    P_anchor_inv = np.linalg.inv(clip.pose_c2w_canonical[clip.anchor_slot])
    log_s = float(np.log(clip.scale_m_per_unit))
    return [FrameInput(index=q, depth_canonical=torch.from_numpy(clip.depth_canonical[q]).to(device),
                       conf=torch.from_numpy(clip.conf[q]).to(device), K=clip.K[q],
                       pose_c2w_canonical=P_anchor_inv @ clip.pose_c2w_canonical[q], log_scale_candidate=log_s)
            for q in range(len(clip.K))]


def build_mapper(clip: WindowClip, device: torch.device) -> IncrementalMapper:
    mapper = IncrementalMapper(device, sem_into=None, n_teacher=0)      # semantics off: the hard map H
    mapper.scale_state.force(clip.scale_m_per_unit)                      # the single place the scale enters
    for f in frame_inputs(clip, device):
        mapper.step(f)
    return mapper


def hard_channels(mapper: IncrementalMapper, grid, T_cam_to_grid: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(S, F)`` flat booleans on ``grid``; ``T_cam_to_grid``: target camera -> grid frame, metres."""
    q = mapper.query(grid, np.linalg.inv(np.asarray(T_cam_to_grid, np.float64)))
    return q["occupied"].bool(), q["free"].bool()


__all__ = ["WindowClip", "frame_inputs", "build_mapper", "hard_channels"]
