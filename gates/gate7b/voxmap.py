"""A persistent occupancy / free-space / semantic evidence volume.

Four states are distinguished, and the distinction is the point of the gate:

* **occupied** -- accumulated log-odds above the threshold;
* **free**     -- accumulated log-odds below it, having actually been carved;
* **unknown**  -- never touched by any ray. Space *behind* a predicted surface is left
  here, never marked free;
* **provisional** -- occupied evidence exists but comes only from a low-reliability
  source and has not yet been confirmed by an independent later observation.

The update is a fixed, training-free log-odds accumulation with the published OctoMap
defaults (Hornung et al., 2013): ``l_occ = +0.85``, ``l_free = -0.40``, clamped to
``|L| <= 4.0``, occupied at ``L > 0``. They are stated in the precommit and are identical
on all three benchmarks; nothing here is fitted.

Occupancy evidence and semantic evidence are kept in **separate** accumulators, so a
geometry update never erases semantic history and vice versa.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

#: Published OctoMap log-odds defaults. Not tuned, not per-dataset.
L_OCC = 0.85
L_FREE = -0.40
L_CLAMP = 4.0
L_OCCUPIED_AT = 0.0

#: Reliability weight of an evidence source. MoGe is deliberately below LingBot so it can
#: propose geometry where LingBot is silent but cannot overwrite it where it is not.
W_LINGBOT = 1.0
W_MOGE = 0.5
#: A voxel whose occupied evidence is MoGe-only is provisional until this many independent
#: frames have supported it.
MOGE_CONFIRMATIONS = 2

SRC_LINGBOT, SRC_MOGE = 0, 1


@dataclass
class EvidenceVolume:
    """Dense evidence over one benchmark evaluation grid, on the GPU.

    Dense rather than hashed because the evaluation volume is small (2.1 M voxels at
    0.2 m, 0.64 M at 0.4 m) and every quantity the report needs is a reduction over it.
    ``tools/gate7b/map_scaling.py`` measures a hashed *global* map separately for the
    memory-versus-sequence-length figure.
    """
    dims: Tuple[int, int, int]
    device: object
    n_classes: int

    def __post_init__(self):
        n = int(np.prod(self.dims))
        z = lambda dt: torch.zeros(n, dtype=dt, device=self.device)
        self.logodds = z(torch.float32)
        self.w_occ = z(torch.float32)
        self.w_free = z(torch.float32)
        self.n_obs = z(torch.int32)
        self.n_occ_lingbot = z(torch.int32)
        self.n_occ_moge = z(torch.int32)
        self.last_time = torch.full((n,), -1, dtype=torch.int32, device=self.device)
        self.first_time = torch.full((n,), -1, dtype=torch.int32, device=self.device)
        self.touched = z(torch.bool)
        self.sem = torch.zeros(n, self.n_classes, dtype=torch.float32,
                               device=self.device)
        self.sem_w = z(torch.float32)

    def reset(self) -> None:
        """Zero every accumulator in place. Reusing one volume across evaluation
        timestamps avoids re-allocating (and re-zeroing) a semantic tensor of hundreds of
        megabytes at every anchor; the result is identical."""
        for t in (self.logodds, self.w_occ, self.w_free, self.n_obs, self.n_occ_lingbot,
                  self.n_occ_moge, self.sem, self.sem_w):
            t.zero_()
        self.touched.zero_()
        self.last_time.fill_(-1)
        self.first_time.fill_(-1)

    # ------------------------------------------------------------------ updates
    def add_free(self, flat: torch.Tensor, weight: float = W_LINGBOT) -> None:
        if flat.numel() == 0:
            return
        w = torch.full_like(flat, float(weight), dtype=torch.float32)
        self.logodds.index_add_(0, flat, w * L_FREE)
        self.w_free.index_add_(0, flat, w)
        self.touched[flat] = True

    def add_occupied(self, flat: torch.Tensor, time_index: int, source: int = SRC_LINGBOT,
                     weight: float = W_LINGBOT) -> None:
        if flat.numel() == 0:
            return
        w = torch.full_like(flat, float(weight), dtype=torch.float32)
        self.logodds.index_add_(0, flat, w * L_OCC)
        self.w_occ.index_add_(0, flat, w)
        self.n_obs.index_add_(0, flat, torch.ones_like(flat, dtype=torch.int32))
        tgt = self.n_occ_moge if source == SRC_MOGE else self.n_occ_lingbot
        tgt.index_add_(0, flat, torch.ones_like(flat, dtype=torch.int32))
        self.touched[flat] = True
        t = torch.full_like(flat, int(time_index), dtype=torch.int32)
        self.last_time[flat] = torch.maximum(self.last_time[flat], t)
        first = self.first_time[flat]
        self.first_time[flat] = torch.where(first < 0, t, torch.minimum(first, t))

    def add_semantics(self, flat: torch.Tensor, probs: torch.Tensor,
                      weight: torch.Tensor) -> None:
        """Semantic evidence, kept apart from occupancy evidence."""
        if flat.numel() == 0:
            return
        self.sem.index_add_(0, flat, probs * weight.unsqueeze(1))
        self.sem_w.index_add_(0, flat, weight)

    def clamp(self) -> None:
        self.logodds.clamp_(-L_CLAMP, L_CLAMP)

    # ------------------------------------------------------------------ readout
    def occupied(self, allow_provisional: bool = False) -> torch.Tensor:
        """Occupied voxels under the fixed threshold, with the provisional rule applied."""
        occ = self.logodds > L_OCCUPIED_AT
        if allow_provisional:
            return occ
        moge_only = (self.n_occ_lingbot == 0) & (self.n_occ_moge > 0)
        unconfirmed = moge_only & (self.n_occ_moge < MOGE_CONFIRMATIONS)
        return occ & ~unconfirmed

    def provisional(self) -> torch.Tensor:
        occ = self.logodds > L_OCCUPIED_AT
        moge_only = (self.n_occ_lingbot == 0) & (self.n_occ_moge > 0)
        return occ & moge_only & (self.n_occ_moge < MOGE_CONFIRMATIONS)

    def free(self) -> torch.Tensor:
        return self.touched & (self.logodds <= L_OCCUPIED_AT) & (self.w_free > 0)

    def unknown(self) -> torch.Tensor:
        return ~self.touched

    def semantic_channel(self) -> torch.Tensor:
        """Argmax of the fused Trident probability vector; ``-1`` where none was fused."""
        ch = self.sem.argmax(dim=1).to(torch.int32)
        return torch.where(self.sem_w > 0, ch, torch.full_like(ch, -1))

    def volumes(self) -> Dict[str, int]:
        return {"occupied": int(self.occupied().sum()),
                "provisional": int(self.provisional().sum()),
                "free": int(self.free().sum()),
                "unknown": int(self.unknown().sum())}

    def memory_bytes(self) -> int:
        tot = 0
        for t in (self.logodds, self.w_occ, self.w_free, self.n_obs, self.n_occ_lingbot,
                  self.n_occ_moge, self.last_time, self.first_time, self.touched,
                  self.sem, self.sem_w):
            tot += t.numel() * t.element_size()
        return int(tot)


__all__ = ["EvidenceVolume", "L_OCC", "L_FREE", "L_CLAMP", "L_OCCUPIED_AT", "W_LINGBOT",
           "W_MOGE", "MOGE_CONFIRMATIONS", "SRC_LINGBOT", "SRC_MOGE"]
