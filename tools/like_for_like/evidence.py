"""Evidence lifting for a window whose target frame is not the last slot.

``tools/zero_shot_eval/prediction/evidence.lift_evidence`` is written for the causal clip: it asserts the target is slot 4
and ``Evidence.scorer_inputs`` reports ``current`` = "this voxel was seen in slot 4". Both are the same statement -- the
evidence was seen at prediction time -- expressed through the causal slot index. OccAny's forward clip puts the target in
slot 0, so the statement has to be read from the anchor slot instead.

Nothing else changes: the same ray endpoints, the same token sampling, the same count-weighted mean, the same frame-bit
construction, in the same order. With ``anchor_slot = 4`` this reproduces the frozen function exactly
(``tools/like_for_like/verify.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from core.mapping import depth as D7, rays as RY
from core.mapping.mapper import pack
from tools.occany_comparison.geometry import frame_inputs as window_frame_inputs
from tools.semantic_completion import config as SCC
from tools.semantic_completion.evidence.trident import load_token_map, sample_at_lattice
from tools.zero_shot_eval.prediction.evidence import TEACHER_DIM, Evidence
from tools.zero_shot_eval.prediction.grid import GridSpec, world_keys


@dataclass
class WindowEvidence(Evidence):
    """Evidence whose ``current`` flag is read from the window's anchor slot rather than the causal slot 4.

    ``n_slots`` is how many slots the window has (5 for either temporal protocol, 6 for the surround window), so
    ``n_frames`` counts over the slots that exist rather than a fixed five."""
    anchor_slot: int = 4
    n_slots: int = 5

    def scorer_inputs(self, trunk_flat: torch.Tensor, centres: torch.Tensor) -> dict:
        b = self.frame_bits.long()
        return {"feature": self.feature, "trunk": trunk_flat[:, self.flat].T, "xyz": centres, "count": self.count,
                "current": (b >> self.anchor_slot) & 1 == 1, "n_frames": sum((b >> q) & 1 for q in range(self.n_slots))}


@torch.no_grad()
def window_ray_endpoints(clip, device):
    """``tools/semantic_completion/evidence/lift.ray_endpoints`` with anchor-aware frame inputs.

    The Gate 15 version resolves poses against ``pose_c2w_canonical[-1]`` because its clips always end at the target. A
    forward clip holds the target in slot 0, so poses are resolved against ``clip.anchor_slot`` instead
    (``occany_comparison.geometry.frame_inputs``, the same function the geometry comparison maps with). For a causal clip
    the two are identical. Everything else -- acceptance rule, scale on translation, voxel keys -- is unchanged."""
    s = float(clip.scale_m_per_unit)
    out = []
    for f in window_frame_inputs(clip, device):
        d_m = s * f.depth_canonical.to(torch.float64)
        acc = (f.conf.to(torch.float64) >= D7.CONF_THRESHOLD) & torch.isfinite(d_m) & (d_m > D7.MIN_DEPTH_M) & (d_m < D7.MAX_DEPTH_M)
        T = np.asarray(f.pose_c2w_canonical, np.float64).copy(); T[:3, 3] *= s
        Tw = torch.as_tensor(T, device=device)
        sel = acc.reshape(-1).nonzero(as_tuple=True)[0]
        dirs = RY.pixel_rays(f.K, d_m.shape, device).reshape(-1, 3)[sel]
        p = dirs * d_m.reshape(-1)[sel].unsqueeze(1)
        pw = p @ Tw[:3, :3].T + Tw[:3, 3]
        out.append((pack(torch.floor(pw / SCC.VOXEL_SIZE_M).to(torch.int64)), sel))
    return out


@torch.no_grad()
def lift_window_evidence(clip, token_paths: Sequence[str], S_occupied: torch.Tensor, T_cam_to_grid: np.ndarray, grid: GridSpec,
                         device, anchor_slot: int) -> WindowEvidence:
    """``token_paths[q]`` are the tokens of clip slot q; ``anchor_slot`` is the slot holding the target frame.

    ``frame_bits`` is uint8, so a window may hold at most eight slots; both declared windows are well inside that."""
    n_slots = len(clip.K)
    assert getattr(clip, "anchor_slot", n_slots - 1) == anchor_slot and n_slots in (5, 6), "clip does not match a declared window"
    assert len(token_paths) == n_slots, f"{n_slots} clip slots but {len(token_paths)} token maps"
    lattice_hw = clip.depth_canonical.shape[1:]
    keys, feats, lengths = [], [], []
    for (k, pix), path in zip(window_ray_endpoints(clip, device), token_paths):
        keys.append(k); feats.append(sample_at_lattice(load_token_map(path, device), lattice_hw).reshape(-1, TEACHER_DIM)[pix]); lengths.append(len(k))
    keys_all, feats_all = torch.cat(keys), torch.cat(feats)
    uq, inv = torch.unique(keys_all, return_inverse=True)
    fsum = torch.zeros(len(uq), TEACHER_DIM, device=device).index_add_(0, inv, feats_all)
    cnt = torch.zeros(len(uq), dtype=torch.int32, device=device).index_add_(0, inv, torch.ones_like(inv, dtype=torch.int32))
    bits = torch.zeros(len(uq), dtype=torch.int32, device=device)
    o = 0
    for slot, n in enumerate(lengths):
        hit = torch.zeros(len(uq), dtype=torch.bool, device=device); hit[inv[o:o + n]] = True; o += n
        bits |= hit.to(torch.int32) << slot
    gk = world_keys(grid, np.linalg.inv(np.asarray(T_cam_to_grid, np.float64)), device)
    pos = torch.searchsorted(uq, gk).clamp_(max=len(uq) - 1)
    flat = ((uq[pos] == gk) & S_occupied.to(device)).nonzero(as_tuple=True)[0]
    row = pos[flat]
    return WindowEvidence(flat, torch.nn.functional.normalize(fsum[row], dim=-1), cnt[row], bits[row].to(torch.uint8),
                          anchor_slot=anchor_slot, n_slots=n_slots)


__all__ = ["WindowEvidence", "lift_window_evidence"]
