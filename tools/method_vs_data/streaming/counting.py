"""Per-region confusion counting on the GPU.

Identical arithmetic to the canonical CPU path -- same pooling call
(``gates.gate8b.pooling.geometry_dilation``, applied to the PREDICTION only, after load and
before the counts), same ``label != 255`` valid rule, same undilated ground truth and region
masks. Only the device changes, because this phase counts 22 arms x 2 conventions x many
regions per anchor over 4,219 anchors and the boolean reductions, not the network, dominate.

``tests`` for this live in the callers: both scorers assert that the regional counts
reconstruct the global TP/FP/FN exactly, which no device difference would survive.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch

from gates.gate8b import pooling as PL


class RegionCounter:
    """Upload one anchor's ground truth and masks once, then count any number of arms."""

    def __init__(self, device, dims):
        self.dev = torch.device(device)
        self.dims = tuple(int(d) for d in dims)
        self._gt: torch.Tensor | None = None
        self._masks: Dict[str, torch.Tensor] = {}

    def set_anchor(self, gt_occupied: np.ndarray, masks: Dict[str, np.ndarray]) -> None:
        self._gt = torch.from_numpy(np.ascontiguousarray(gt_occupied)).to(self.dev)
        self._masks = {k: torch.from_numpy(np.ascontiguousarray(v)).to(self.dev)
                       for k, v in masks.items()}

    def support(self, keys: Sequence[str]) -> Dict[str, tuple]:
        """``(valid_voxels, gt_occupied)`` per region, for the composition table."""
        return {k: (int(self._masks[k].sum()), int((self._masks[k] & self._gt).sum()))
                for k in keys}

    def counts(self, occupied: np.ndarray, post: str,
               keys: Sequence[str]) -> Dict[str, np.ndarray]:
        occ = torch.from_numpy(np.ascontiguousarray(occupied)).to(self.dev)
        if post == "pooled":
            occ = PL.geometry_dilation(occ.reshape(self.dims)).reshape(-1)
        elif post != "raw":
            raise KeyError(post)
        gt = self._gt
        tp_all, fp_all, fn_all = occ & gt, occ & ~gt, ~occ & gt
        out = {}
        for k in keys:
            m = self._masks[k]
            out[k] = np.array([int((tp_all & m).sum()), int((fp_all & m).sum()),
                               int((fn_all & m).sum())], np.int64)
        return out


__all__ = ["RegionCounter"]
