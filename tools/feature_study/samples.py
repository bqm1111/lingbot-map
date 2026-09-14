"""The gate's sample reader, carrying the feature target alongside the class target.

Everything about crop sampling, batching and tensor layout is
``core.model.data.DriveSamples`` -- the adopted Gate 8C-1 reader -- so A, B and C see the
same files in the same order at the same crops for a given seed. The batch dict gains two
entries and loses none:

``aux_index``  flat positions of the auxiliary-supervised voxels inside the batched crop
``aux_target`` their unit-norm 64-d feature targets

Both are sparse. A 128x128x32 crop holds of order 7 000 supervised voxels out of 524 288, so
a dense feature volume would be 99 % zeros and 70x the memory.

Pairing across conditions is not assumed, it is recorded: :meth:`StudySamples.batch` appends
every ``(file index, crop origin)`` it produces to ``self.trace``, and the trainer writes the
trace out. Two runs of the same seed with different objectives must produce byte-identical
traces.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from core.model.data import DriveSamples
from tools.feature_study import study as ST
from tools.feature_study import feature_targets as FT

_STEM = re.compile(r"/(?P<drive>[^/]+)/(?P<idx>\d+)\.npz$")


class StudySamples(DriveSamples):
    """``DriveSamples`` plus the feature sidecar. Consumes no extra randomness."""

    def __init__(self, *a, want_features: bool = True, **kw):
        super().__init__(*a, **kw)
        self.want_features = bool(want_features)
        self.trace: List[List[int]] = []
        self.n_missing_feature_target = 0

    def aux_path(self, i: int) -> Optional[str]:
        m = _STEM.search(self.files[i].replace("\\", "/"))
        if not m:
            return None
        return ST.feature_target_path(m.group("drive"), int(m.group("idx")))

    def load_aux(self, i: int) -> Optional[Dict[str, np.ndarray]]:
        if not self.want_features:
            return None
        p = self.aux_path(i)
        if p is None or not os.path.exists(p):
            self.n_missing_feature_target += 1
            return None
        return FT.load(p)

    def batch(self, idxs: Sequence[int], record: bool = True) -> Dict[str, torch.Tensor]:
        keys = ("input", "gt_occ", "gt_valid", "observed", "fut_p", "fut_valid",
                "base_logodds")
        out = {k: [] for k in keys}
        aux_idx, aux_tgt = [], []
        cx, cy, cz = self.crop
        for n, i in enumerate(idxs):
            d = self.load(i)
            x0, y0, z0 = self.crop_of(d)                 # the gate's sampler, same RNG draw
            if record:
                self.trace.append([int(i), int(x0), int(y0), int(z0)])
            sl = (slice(x0, x0 + cx), slice(y0, y0 + cy), slice(z0, z0 + cz))
            out["input"].append(d["input"][(slice(None),) + sl])
            for k in ("gt_occ", "gt_valid", "observed", "fut_valid", "base_logodds"):
                out[k].append(d[k][sl])
            out["fut_p"].append(d["fut_p"][sl].permute(3, 0, 1, 2))
            a = self.load_aux(i)
            if a is None or len(a["feat_rows"]) == 0:
                continue
            X, Y, Z = d["gt_occ"].shape
            r = a["feat_rows"]
            x = r // (Y * Z); y = (r // Z) % Y; z = r % Z
            keep = ((x >= x0) & (x < x0 + cx) & (y >= y0) & (y < y0 + cy)
                    & (z >= z0) & (z < z0 + cz))
            if not keep.any():
                continue
            lx, ly, lz = x[keep] - x0, y[keep] - y0, z[keep] - z0
            local = ((lx * cy) + ly) * cz + lz                 # inside one crop
            aux_idx.append(torch.from_numpy(local + n * cx * cy * cz).to(self.device))
            aux_tgt.append(torch.from_numpy(a["feat"][keep]).to(self.device))
        b = {k: torch.stack(v) for k, v in out.items()}
        b["aux_index"] = (torch.cat(aux_idx) if aux_idx
                          else torch.zeros(0, dtype=torch.long, device=self.device))
        b["aux_target"] = (torch.cat(aux_tgt) if aux_tgt
                           else torch.zeros(0, ST.FEATURE_DIM, device=self.device))
        return b


__all__ = ["StudySamples"]
