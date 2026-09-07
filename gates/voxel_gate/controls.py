"""Deterministic, non-learned voxel controls.

These answer the only question that makes a learned corrector interesting: is the gain
more than indiscriminate densification? Each control sees exactly the inputs the learned
model sees (frozen C3 occupancy and the correction region), is evaluated through the same
frozen voxeliser and evaluation mask, and has its parameters frozen on source data before
sequence 08 is touched.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .voxels import dilate


def neighbour_count(occ: torch.Tensor, radius: int) -> torch.Tensor:
    """Number of occupied voxels in the (2r+1)^3 neighbourhood, excluding the centre."""
    x = (occ > 0).float()[None, None]
    k = 2 * radius + 1
    tot = F.avg_pool3d(x, kernel_size=k, stride=1, padding=radius,
                       divisor_override=1)[0, 0]
    return tot - (occ > 0).float()


def control(occ: torch.Tensor, region: torch.Tensor, kind: str,
            radius: int = 1, min_neighbors: int = 0) -> torch.Tensor:
    """Apply one deterministic control inside ``region``.

    ``dilate``   -- binary dilation by ``radius``.
    ``close``    -- dilation followed by erosion (hole filling, no net growth outside).
    ``fill``     -- add empty voxels with at least ``min_neighbors`` occupied neighbours
                    in the (2*radius+1)^3 window; this is the count-tunable family.
    """
    o = occ > 0
    if kind == "dilate":
        out = dilate(o, radius)
    elif kind == "close":
        d = dilate(o, radius)
        out = ~dilate(~d, radius)
    elif kind == "fill":
        out = o | ((neighbour_count(o, radius) >= float(min_neighbors)) & region)
    else:
        raise ValueError(f"unknown control {kind!r}")
    return (out & region) | (o & ~region)


CONTROL_GRID = "see configs/voxel_gate/visible_correction.yaml: controls.*"


def enumerate_controls(cfg) -> Dict[str, dict]:
    """Every candidate control, keyed by a stable name."""
    out: Dict[str, dict] = {}
    for r in cfg.controls.candidate_radii:
        out[f"dilate_r{r}"] = dict(kind="dilate", radius=int(r))
        out[f"close_r{r}"] = dict(kind="close", radius=int(r))
        for k in cfg.controls.candidate_min_neighbors:
            out[f"fill_r{r}_k{k}"] = dict(kind="fill", radius=int(r),
                                          min_neighbors=int(k))
    return out
