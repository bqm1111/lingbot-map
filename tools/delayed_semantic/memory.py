"""Observed semantic evidence vs predicted semantics: two types that never mix.

``EvidenceMemory`` accumulates *observations* frame by frame (incremental interface): each call to
``observe`` adds the accepted causal LingBot ray endpoints of one history frame with their frozen
Trident-H features. ``snapshot`` applies the Gate 15 evidence rule unchanged (world key hit by >= 1
endpoint AND frozen hard map occupied; feature = normalised count-weighted mean of unit ray features)
and additionally records provenance per voxel: which history slots observed it and when.

``PredictedSemantics`` is what a completion method outputs. ``EvidenceMemory.observe`` rejects it, so a
prediction can never be re-inserted as observed teacher evidence.

Shapes: N evidence voxels; flat_index [N] int64 ascending; feature [N, 1024] unit L2; count [N] int32;
frame_bits [N] uint8 (bit q set iff history slot q = frame t-4+q hit the voxel); first/last_time_s [N].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

from tools.delayed_semantic import config as C
from tools.semantic_completion.evidence.lift import grid_world_keys


@dataclass
class ObservedEvidence:
    flat_index: torch.Tensor
    feature: Optional[torch.Tensor]
    count: torch.Tensor
    frame_bits: torch.Tensor
    first_time_s: torch.Tensor
    last_time_s: torch.Tensor
    t: int
    provenance: str = field(default="observed", init=False)

    def dense_mask(self) -> torch.Tensor:
        m = torch.zeros(C.N_VOXELS, dtype=torch.bool, device=self.flat_index.device); m[self.flat_index] = True
        return m

    def current(self) -> torch.Tensor:
        """[N] bool: observed in the prediction frame t itself."""
        return (self.frame_bits >> C.CURRENT_SLOT) & 1 == 1


@dataclass
class PredictedSemantics:
    flat_index: torch.Tensor      # [M] voxels the method assigns semantics to
    feature: torch.Tensor         # [M, 1024] unit L2, language-compatible
    t: int                        # prediction time (stream index); never changes after it is stored
    method: str
    provenance: str = field(default="predicted", init=False)


class EvidenceMemory:
    """Observed evidence of H_t, inserted one history frame at a time."""

    def __init__(self, t: int, device):
        self.t, self.device = t, device
        self._keys: List[torch.Tensor] = []
        self._feats: List[Optional[torch.Tensor]] = []
        self._slots: List[int] = []
        self._times: List[float] = []

    def observe(self, slot: int, stream_index: int, time_s: float, world_keys: torch.Tensor, features: Optional[torch.Tensor]) -> None:
        if isinstance(features, PredictedSemantics) or isinstance(world_keys, PredictedSemantics):
            raise TypeError("predicted semantics cannot be inserted into observed evidence memory")
        if stream_index > self.t:
            raise ValueError(f"frame {stream_index} is after the prediction time {self.t}")
        assert stream_index == self.t + C.HISTORY_OFFSETS[slot] and slot not in self._slots
        self._keys.append(world_keys); self._feats.append(features); self._slots.append(slot); self._times.append(float(time_s))

    def snapshot(self, S_occupied: torch.Tensor, T_cam_to_grid: np.ndarray) -> ObservedEvidence:
        with_feat = all(f is not None for f in self._feats)
        keys = torch.cat(self._keys)
        uq, inv = torch.unique(keys, return_inverse=True)
        cnt = torch.zeros(len(uq), dtype=torch.int32, device=self.device).index_add_(0, inv, torch.ones_like(inv, dtype=torch.int32))
        bits = torch.zeros(len(uq), dtype=torch.int32, device=self.device)
        first = torch.full((len(uq),), float("inf"), dtype=torch.float64, device=self.device); last = torch.full_like(first, -float("inf"))
        o = 0
        for k, slot, ts in zip(self._keys, self._slots, self._times):
            hit = torch.zeros(len(uq), dtype=torch.bool, device=self.device); hit[inv[o:o + len(k)]] = True; o += len(k)
            bits |= hit.to(torch.int32) << slot
            first = torch.where(hit, torch.minimum(first, torch.tensor(ts, dtype=torch.float64, device=self.device)), first)
            last = torch.where(hit, torch.maximum(last, torch.tensor(ts, dtype=torch.float64, device=self.device)), last)
        gk = grid_world_keys(np.linalg.inv(np.asarray(T_cam_to_grid, np.float64)), self.device)
        pos = torch.searchsorted(uq, gk).clamp_(max=len(uq) - 1)
        flat = ((uq[pos] == gk) & S_occupied.to(self.device)).nonzero(as_tuple=True)[0]
        row = pos[flat]
        feature = None
        if with_feat:
            feats = torch.cat(self._feats)
            fsum = torch.zeros(len(uq), C.TEACHER_DIM, device=self.device).index_add_(0, inv, feats)
            feature = torch.nn.functional.normalize(fsum[row], dim=-1)
        return ObservedEvidence(flat, feature, cnt[row], bits[row].to(torch.uint8), first[row], last[row], self.t)


def load_observed(drive: str, t: int, device, paths: C.Paths = C.Paths(), with_feature: bool = True) -> ObservedEvidence:
    """Gate 15 cached evidence (features) + this study's provenance cache (frame bits, times), index-checked."""
    from tools.semantic_completion.evidence.lift import load_evidence
    ev = load_evidence(paths.g15.evidence_path(drive, t), device)
    with np.load(paths.provenance_path(drive, t)) as z:
        flat = torch.from_numpy(z["flat_index"].astype(np.int64)).to(device)
        assert torch.equal(flat, ev.flat_index), f"provenance cache does not match the Gate 15 evidence of {drive}:{t}"
        bits = torch.from_numpy(z["frame_bits"]).to(device)
        slot_t = torch.from_numpy(z["slot_time_s"]).to(device)
    b = bits.to(torch.int64)
    has = torch.stack([(b >> q) & 1 == 1 for q in range(5)], 1)                       # [N, 5]
    tt = slot_t.expand(len(b), 5)
    first = torch.where(has, tt, torch.full_like(tt, float("inf"))).amin(1); last = torch.where(has, tt, torch.full_like(tt, -float("inf"))).amax(1)
    return ObservedEvidence(ev.flat_index, ev.feature if with_feature else None, ev.count, bits, first, last, t)


__all__ = ["ObservedEvidence", "PredictedSemantics", "EvidenceMemory", "load_observed"]
