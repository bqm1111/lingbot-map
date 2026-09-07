"""Gate 8D: the training sample loader.

Gate 8C-1's loader, with the two surface-distance fields carried through the crop so the
Phase 4 auxiliary head can be supervised. The causal input, the occupancy target and the
semantic target are read exactly as before -- this class adds fields, it changes none.
"""
from __future__ import annotations

import numpy as np
import torch

from gates.gate8 import targets as TG
from gates.gate8c1.data import DriveSamples as _Base


class DriveSamples(_Base):
    """``gate8c1.data.DriveSamples`` plus ``tudf_m`` / ``tudf_valid``."""

    def load(self, i):
        with np.load(self.files[i]) as z:
            d = TG.unpack_sample(z, self.device)
            n = d["gt_occ"].numel()
            if "tudf_m" in z:
                t = torch.from_numpy(np.asarray(z["tudf_m"], np.float32)).to(self.device)
                v = torch.from_numpy(np.unpackbits(z["tudf_valid"])[:n]
                                     .astype(np.bool_)).to(self.device)
                d["tudf_m"] = t.reshape(d["gt_occ"].shape)
                d["tudf_valid"] = v.reshape(d["gt_occ"].shape)
            else:                   # built before Phase 4: no distance supervision at all
                d["tudf_m"] = torch.zeros_like(d["gt_occ"], dtype=torch.float32)
                d["tudf_valid"] = torch.zeros_like(d["gt_occ"], dtype=torch.bool)
        return d

    def batch(self, idxs):
        out = {k: [] for k in ("input", "gt_occ", "gt_valid", "observed", "fut_p",
                               "fut_valid", "base_logodds", "tudf_m", "tudf_valid")}
        for i in idxs:
            d = self.load(i)
            x0, y0, z0 = self.crop_of(d)
            cx, cy, cz = self.crop
            sl = (slice(x0, x0 + cx), slice(y0, y0 + cy), slice(z0, z0 + cz))
            out["input"].append(d["input"][(slice(None),) + sl])
            for k in ("gt_occ", "gt_valid", "observed", "fut_valid", "base_logodds",
                      "tudf_m", "tudf_valid"):
                out[k].append(d[k][sl])
            out["fut_p"].append(d["fut_p"][sl].permute(3, 0, 1, 2))
        return {k: torch.stack(v) for k, v in out.items()}


__all__ = ["DriveSamples"]
