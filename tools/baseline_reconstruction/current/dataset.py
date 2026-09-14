"""Training / validation data for the current baseline: packed H inputs + raw-LiDAR targets.

The volumes come from the frozen Gate 12 caches (``config.CurrentPaths``): ``S_H``/``F_H``
were built by ``build_local_geometry``-equivalent code from the local-reset clips, the
targets by ``core/supervision/rawtarget.py``. Everything is kept packed in RAM and
densified per batch. Crops are uniform ``128x128x32`` windows of the ``256x256x32`` grid.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from core.model import sampler as SMP
from tools.baseline_reconstruction import config as C

DIMS = C.GRID_DIMS
N_VOXELS = int(np.prod(DIMS))


def anchor_ids(drive: str, paths: C.CurrentPaths, every: int = 1) -> List[int]:
    """Anchors with both an input and a target on disk, ascending, optionally decimated."""
    ids = sorted(int(f[:-4]) for f in os.listdir(os.path.dirname(paths.input_path(drive, 0))) if f.endswith(".npz"))
    return [t for t in ids if os.path.exists(paths.target_path(drive, t))][::every]


def _dense(packed: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(np.unpackbits(packed)[:N_VOXELS].reshape(DIMS)).to(device).bool()


@dataclass
class Sample:
    drive: str
    anchor: int
    S_packed: np.ndarray
    F_packed: np.ndarray
    valid_packed: np.ndarray
    occ_packed: np.ndarray


class HInputDataset:
    """Packed volumes in RAM; ``batch`` returns the dict ``core.run._gate8a_train.step_loss`` expects."""

    def __init__(self, drive_ids: Sequence[Tuple[str, Sequence[int]]], paths: C.CurrentPaths, device, seed: int):
        self.device, self.rng, self.paths = device, random.Random(seed), paths
        self.items: List[Sample] = []
        for drive, ids in drive_ids:
            for t in ids:
                zi, zt = np.load(paths.input_path(drive, t)), np.load(paths.target_path(drive, t))
                self.items.append(Sample(drive, t, zi["S_H"], zi["F_H"], zt["valid_packed"], zt["occ_packed"]))

    def __len__(self) -> int:
        return len(self.items)

    def dense(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(input [3,X,Y,Z], gt_occ, gt_valid, observed)``; observed = S | F drives the loss weights."""
        s = self.items[i]
        S, F = _dense(s.S_packed, self.device), _dense(s.F_packed, self.device)
        x = torch.stack([S, torch.zeros_like(S), F]).float()
        return x, _dense(s.occ_packed, self.device).float(), _dense(s.valid_packed, self.device), S | F

    def batch(self, idxs: Sequence[int], crop: Sequence[int], draw_digest=None, target_digest=None) -> Dict[str, torch.Tensor]:
        out = {k: [] for k in ("input", "gt_occ", "gt_valid", "observed")}
        for i in idxs:
            x, occ, val, obs = self.dense(i)
            x0, y0, z0 = SMP.uniform_origin(self.rng, DIMS, crop)
            sl = (slice(x0, x0 + crop[0]), slice(y0, y0 + crop[1]), slice(z0, z0 + crop[2]))
            out["input"].append(x[(slice(None),) + sl]); out["gt_occ"].append(occ[sl])
            out["gt_valid"].append(val[sl]); out["observed"].append(obs[sl])
            if draw_digest is not None:
                s = self.items[i]
                draw_digest.update(f"{s.drive[17:21]}:{s.anchor}:{x0},{y0},{z0};".encode())
                target_digest.update(occ[sl].to(torch.uint8).cpu().numpy().tobytes())
        b = {k: torch.stack(v) for k, v in out.items()}
        B, (X, Y, Z) = len(idxs), crop
        # base log-odds is zero everywhere: the residual lock never engages and the network output is the map
        b["base_logodds"] = torch.zeros((B, X, Y, Z), device=self.device)
        b["fut_p"] = torch.zeros((B, 1, X, Y, Z), device=self.device)                       # no semantic target
        b["fut_valid"] = torch.zeros((B, X, Y, Z), dtype=torch.bool, device=self.device)
        return b


__all__ = ["DIMS", "N_VOXELS", "anchor_ids", "Sample", "HInputDataset"]
