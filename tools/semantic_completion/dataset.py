"""Training batches: frozen trunk on the full grid, then a uniform crop with its sparse evidence.

For every sample of a batch:
    x           [1, 3, 256, 256, 32]   frozen hard map [S, 0, F] (baseline input cache)
    trunk       [24, cx, cy, cz]       frozen decoder feature, cropped AFTER the full-grid forward so the
                                       head sees exactly what it will see at evaluation
    evidence    sparse (index into the crop, [N, 1024] unit-norm feature)
    hidden      [N] bool               evidence voxels in randomly hidden (8, 8, 4) blocks, removed from input
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch

from tools.semantic_completion import config as C
from tools.semantic_completion.evidence.lift import VoxelEvidence, load_evidence
from tools.semantic_completion.frozen_geometry import geometry_input, load_hard_map, occupancy_and_trunk


@dataclass
class CropSample:
    trunk: torch.Tensor          # [24, cx, cy, cz]
    local_index: torch.Tensor    # [N] flat C-order index inside the crop
    feature: torch.Tensor        # [N, 1024] unit norm
    hidden: torch.Tensor         # [N] bool


def crop_evidence(ev: VoxelEvidence, origin: Tuple[int, int, int], crop: Tuple[int, int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evidence voxels inside the crop: ``(local flat index [N], feature [N, 1024])``."""
    X, Y, Z = C.GRID_DIMS
    ix = ev.flat_index // (Y * Z); iy = (ev.flat_index // Z) % Y; iz = ev.flat_index % Z
    lo = [int(o) for o in origin]
    inside = (ix >= lo[0]) & (ix < lo[0] + crop[0]) & (iy >= lo[1]) & (iy < lo[1] + crop[1]) & (iz >= lo[2]) & (iz < lo[2] + crop[2])
    lx, ly, lz = ix[inside] - lo[0], iy[inside] - lo[1], iz[inside] - lo[2]
    return (lx * crop[1] + ly) * crop[2] + lz, ev.feature[inside]


def hidden_block_mask(local_index: torch.Tensor, crop: Tuple[int, int, int], block: Tuple[int, int, int], prob: float,
                      generator: torch.Generator) -> torch.Tensor:
    """``[N] bool``: evidence voxel lies in a block (block-aligned to the crop origin) drawn hidden with ``prob``."""
    nb = [(c + b - 1) // b for c, b in zip(crop, block)]
    hidden_blocks = torch.rand(nb[0] * nb[1] * nb[2], generator=generator) < prob
    lx = local_index // (crop[1] * crop[2]); ly = (local_index // crop[2]) % crop[1]; lz = local_index % crop[2]
    bidx = ((lx // block[0]) * nb[1] + ly // block[1]) * nb[2] + lz // block[2]
    return hidden_blocks.to(local_index.device)[bidx]


class EvidenceDataset:
    def __init__(self, drive_ids: Sequence[Tuple[str, Sequence[int]]], device, seed: int, paths: C.Paths = C.Paths()):
        self.items: List[Tuple[str, int]] = [(d, t) for d, ids in drive_ids for t in ids]
        self.device, self.paths = device, paths
        self.rng = random.Random(seed)
        self.gen = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return len(self.items)

    def batch(self, net, cfg: C.TrainConfig, digest=None) -> List[CropSample]:
        picks = [self.items[self.rng.randrange(len(self.items))] for _ in range(cfg.batch_size)]
        xs = torch.cat([geometry_input(*load_hard_map(d, t, self.device, self.paths)) for d, t in picks])
        _, trunk = occupancy_and_trunk(net, xs)
        out = []
        for b, (d, t) in enumerate(picks):
            origin = tuple(self.rng.randint(0, g - c) for g, c in zip(C.GRID_DIMS, cfg.crop))
            ev = load_evidence(self.paths.evidence_path(d, t), self.device)
            idx, feat = crop_evidence(ev, origin, cfg.crop)
            hid = hidden_block_mask(idx, cfg.crop, cfg.hide_block_vox, cfg.hide_block_prob, self.gen)
            sl = tuple(slice(o, o + c) for o, c in zip(origin, cfg.crop))
            out.append(CropSample(trunk[b][(slice(None),) + sl], idx, feat, hid))
            if digest is not None:
                digest.update(f"{d[17:21]}:{t}:{origin}:{int(hid.sum())};".encode())
        return out


__all__ = ["CropSample", "crop_evidence", "hidden_block_mask", "EvidenceDataset"]
