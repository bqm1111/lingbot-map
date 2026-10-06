"""Observed semantic evidence of one causal clip on any grid: the Gate 15 rule plus the pilot's frame provenance.

evidence voxel = grid voxel whose mapper world key received >= 1 accepted causal ray endpoint AND that the hard map marks
occupied (S); feature = L2-normalised count-weighted mean of the unit Trident-H token features sampled at the ray pixels;
frame_bits bit q = history slot q (t-4+q) contributed a ray. Identical operations and order to
``semantic_completion/evidence/lift.lift_clip_evidence`` and ``delayed_semantic/memory.EvidenceMemory.snapshot``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from tools.semantic_completion.evidence.lift import ray_endpoints
from tools.semantic_completion.evidence.trident import load_token_map, sample_at_lattice
from tools.zero_shot_eval.prediction.grid import GridSpec, world_keys

TEACHER_DIM = 1024
CURRENT_SLOT = 4


@dataclass
class Evidence:
    flat: torch.Tensor          # [N] int64 ascending
    feature: torch.Tensor       # [N, 1024] unit L2
    count: torch.Tensor         # [N] int32 rays
    frame_bits: torch.Tensor    # [N] uint8

    def dense(self, grid: GridSpec) -> torch.Tensor:
        m = torch.zeros(grid.n_voxels, dtype=torch.bool, device=self.flat.device); m[self.flat] = True
        return m

    def scorer_inputs(self, trunk_flat: torch.Tensor, centres: torch.Tensor) -> dict:
        """The evidence dictionary the association head reads (``delayed_semantic/model.EvidenceAssociation.forward``)."""
        b = self.frame_bits.long()
        return {"feature": self.feature, "trunk": trunk_flat[:, self.flat].T, "xyz": centres, "count": self.count,
                "current": (b >> CURRENT_SLOT) & 1 == 1, "n_frames": sum((b >> q) & 1 for q in range(5))}


@torch.no_grad()
def lift_evidence(clip, token_paths: Sequence[str], S_occupied: torch.Tensor, T_cam_to_grid: np.ndarray, grid: GridSpec, device) -> Evidence:
    """``clip``: five-frame causal clip with the target in the last slot; ``token_paths[q]``: Trident tokens of slot q."""
    assert getattr(clip, "anchor_slot", len(clip.K) - 1) == len(clip.K) - 1 == 4, "the causal clip must end at the target"
    lattice_hw = clip.depth_canonical.shape[1:]
    keys, feats, lengths = [], [], []
    for (k, pix), path in zip(ray_endpoints(clip, device), token_paths):
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
    return Evidence(flat, torch.nn.functional.normalize(fsum[row], dim=-1), cnt[row], bits[row].to(torch.uint8))


__all__ = ["Evidence", "lift_evidence"]
