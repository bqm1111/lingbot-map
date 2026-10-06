"""Candidate sets: the 16 observed evidence voxels a query voxel may draw semantics from (pilot rule, unchanged).

Dimensions: Q queries, K = 16 candidates, N evidence voxels, 18 classes, 1024-d teacher features.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from tools.delayed_semantic.gate16 import config as G
from tools.delayed_semantic.memory import ObservedEvidence
from tools.delayed_semantic.model import nearest_candidates, teacher_distribution, voxel_centres


@dataclass
class CandidateSet:
    query_flat: torch.Tensor      # [Q] int64 flat grid index of each query voxel
    rows: torch.Tensor            # [Q, K] int64 rows into the evidence arrays, ascending distance (row 0 = nearest)
    voxel_ids: torch.Tensor       # [Q, K] int64 flat grid index of each candidate
    distance_m: torch.Tensor      # [Q, K] float32, voxel-centre distance
    semantic_probs: torch.Tensor  # [Q, K, 18] softmax(40 cos(f_i, text)); each row sums to 1
    classes: torch.Tensor         # [Q, K] int64 in 1..18, argmax cosine readout (= Gate 15 classify)
    valid: torch.Tensor           # [Q, K] bool; False only if the anchor has fewer than K evidence voxels

    def __len__(self) -> int:
        return len(self.query_flat)

    def subset(self, mask: torch.Tensor) -> "CandidateSet":
        """The candidate records of the queries selected by ``mask`` [Q] bool (or an index tensor)."""
        return CandidateSet(*[getattr(self, f)[mask] for f in FIELDS])


FIELDS = ("query_flat", "rows", "voxel_ids", "distance_m", "semantic_probs", "classes", "valid")


def evidence_tensors(ev: ObservedEvidence, trunk: torch.Tensor) -> Dict[str, torch.Tensor]:
    """The scorer's evidence dictionary for one anchor (identical to the pilot's ``evaluate.head_predict``).
    trunk: [24, X, Y, Z] frozen CompletionUNet decoder feature."""
    bits = ev.frame_bits.long()
    return {"feature": ev.feature, "trunk": trunk.reshape(trunk.shape[0], -1)[:, ev.flat_index].T, "xyz": voxel_centres(ev.flat_index),
            "count": ev.count, "current": ev.current(), "n_frames": sum((bits >> q) & 1 for q in range(5)), "flat": ev.flat_index}


@torch.no_grad()
def build(query_flat: torch.Tensor, evd: Dict[str, torch.Tensor], text: torch.Tensor, allowed_rows: Optional[torch.Tensor] = None) -> CandidateSet:
    """Candidates among all evidence rows (evaluation, delayed queries) or among ``allowed_rows`` (training with hidden evidence)."""
    pool = allowed_rows if allowed_rows is not None else torch.arange(len(evd["xyz"]), device=query_flat.device)
    idx, dist = nearest_candidates(voxel_centres(query_flat), evd["xyz"][pool], G.K)
    rows = pool[idx]
    valid = torch.ones_like(rows, dtype=torch.bool)
    if rows.shape[1] < G.K:                                   # never happens on anchors used here; kept explicit
        pad = G.K - rows.shape[1]
        rows = torch.cat([rows, rows[:, :1].expand(-1, pad)], 1); dist = torch.cat([dist, dist[:, :1].expand(-1, pad)], 1)
        valid = torch.cat([valid, torch.zeros(len(rows), pad, dtype=torch.bool, device=rows.device)], 1)
    probs = teacher_distribution(evd["feature"][rows], text)
    return CandidateSet(query_flat, rows, evd["flat"][rows], dist.float(), probs, 1 + probs.argmax(-1), valid)


__all__ = ["CandidateSet", "evidence_tensors", "build"]
