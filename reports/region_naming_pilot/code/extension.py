"""One bounded extension rule: seeded geodesic growth of region membership over the FIXED occupancy.

Computed once per anchor, before and independently of any name, so P and Q extend through exactly the
same membership map. It changes which name a voxel receives, never whether the voxel is occupied.

Rules, all deterministic:
  * **seeds** are the owned evidence voxels (each already belongs to exactly one region);
  * growth moves to 6-connected neighbours that the frozen occupancy marks occupied and that the frozen
    hard map does **not** mark observed free -- observed free space is a wall, so a mask is never
    extruded through space the clip saw to be empty;
  * another region's seed is also a wall, so two objects that merely touch do not merge;
  * growth stops after ``max_hops`` voxels of geodesic distance (``0.2 m`` each): a finite, documented
    limit that keeps a road region from flooding the whole road-connected scene;
  * a voxel reached by several regions takes the lowest region id, and is flagged **contested**;
  * contested voxels and everything never reached stay unassigned and fall back to nearest evidence.

No appearance is invented for a hidden voxel: growth uses occupancy, observed free space and adjacency
only. The frozen trunk features are not consulted -- they are geometry cues, not instance descriptors.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch

import legacy as L

SETTINGS = {
    "connectivity": 6,
    "traversable": "frozen occupancy O, minus observed free F, minus other regions' seeds",
    "max_hops": 8,                    # 1.6 m; selected on the calibration anchors only
    "conflict": "lowest region id at equal geodesic distance, and the voxel is flagged contested",
    "unassigned_fallback": "the matching nearest-evidence prediction (P_N for P_R, Q_N for Q_R)",
    "max_hops_candidates": [2, 4, 6, 8, 12, 16],
}

_SHIFTS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def _shift(v: torch.Tensor, d: Tuple[int, int, int], fill: int) -> torch.Tensor:
    """``v`` shifted so that element ``x`` sees its neighbour at ``x + d``; the border takes ``fill``."""
    out = torch.full_like(v, fill)
    sl_dst, sl_src = [], []
    for k in range(3):
        if d[k] > 0:
            sl_dst.append(slice(0, v.shape[k] - d[k])); sl_src.append(slice(d[k], v.shape[k]))
        elif d[k] < 0:
            sl_dst.append(slice(-d[k], v.shape[k])); sl_src.append(slice(0, v.shape[k] + d[k]))
        else:
            sl_dst.append(slice(None)); sl_src.append(slice(None))
    out[tuple(sl_dst)] = v[tuple(sl_src)]
    return out


def grow(voxel_region: torch.Tensor, O: torch.Tensor, F_: torch.Tensor, max_hops: int = None) -> Dict:
    """``{member [N] int32 region id or -1, hops [N] int16, contested [N] bool, stats}``."""
    hops_max = SETTINGS["max_hops"] if max_hops is None else int(max_hops)
    dims = L.GRID_DIMS
    owner = voxel_region.reshape(dims).to(torch.int32).clone()
    seeds = owner >= 0
    hops = torch.where(seeds, torch.zeros_like(owner, dtype=torch.int16), torch.full(dims, -1, dtype=torch.int16, device=owner.device))
    contested = torch.zeros(dims, dtype=torch.bool, device=owner.device)
    allowed = O.reshape(dims) & ~F_.reshape(dims) & ~seeds
    BIG = torch.iinfo(torch.int32).max
    grown_per_hop = []
    for h in range(1, hops_max + 1):
        if not bool(allowed.any()):
            break
        lo = torch.full(dims, BIG, dtype=torch.int32, device=owner.device)
        hi = torch.full(dims, -1, dtype=torch.int32, device=owner.device)
        for d in _SHIFTS:
            s = _shift(owner, d, -1)
            lo = torch.minimum(lo, torch.where(s >= 0, s, torch.full_like(s, BIG)))
            hi = torch.maximum(hi, s)
        new = allowed & (hi >= 0)
        if not bool(new.any()):
            break
        owner = torch.where(new, lo, owner)
        contested |= new & (lo != hi)
        hops = torch.where(new, torch.full_like(hops, h), hops)
        allowed = allowed & ~new
        grown_per_hop.append(int(new.sum()))
    member = owner.reshape(-1)
    grown = (member >= 0) & ~voxel_region.reshape(-1).ge(0)
    stats = {"max_hops": hops_max, "n_seeds": int(seeds.sum()), "n_grown": int(grown.sum()),
             "n_contested": int(contested.sum()), "grown_per_hop": grown_per_hop,
             "n_occupied": int(O.sum()), "n_occupied_assigned": int((member >= 0)[O].sum()),
             "occupied_coverage": float((member >= 0)[O].sum() / max(int(O.sum()), 1))}
    return {"member": member, "hops": hops.reshape(-1), "contested": contested.reshape(-1), "stats": stats}
