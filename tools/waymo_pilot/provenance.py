"""Per-target provenance, kept lightweight and kept apart by source.

Every privileged target voxel records where it came from: which supervision source produced
it, when (seconds relative to the anchor), which camera, which geometry carried it, how many
observations supported it, and whether those observations agreed. Nothing here converts that
into a probability that the target is correct -- agreement is consistency between views of a
frozen teacher, which is not correctness, and the report says so wherever it is quoted.

Overlaps are retained rather than resolved: a voxel supported by two sources appears in both
source counts and in the pairwise overlap table, so "targets added" can never be read as a
sum over sources.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from tools.waymo_pilot import config as C

#: geometry that carried an observation into the grid
GEOM_LIDAR = "lidar_return"
GEOM_DEPTH = "depth_surface"


def summarize_single_source(sem: Dict, U: torch.Tensor, source: str) -> Dict:
    """Provenance aggregate when one source produced every target (the existing recipe)."""
    keep = U[sem["flat"]]          # rows of the cache that fall in the population
    n = int(keep.sum())
    nv = sem["n_views"].float()
    dt = sem["first_dt_s"].float()
    return {
        "sources": {source: n},
        "overlaps": {},
        "n_union": n,
        "geometry_source": GEOM_LIDAR,
        "camera": C.WAYMO_SELECTED_CAMERA if source == C.SOURCE_OTHER_CAMERA else "selected",
        "observation_count": _dist(nv[keep]),
        "first_observation_dt_s": _dist(dt[keep]),
        "tier": _tier_counts(sem, U),
    }


def combine_sources(per_source: Dict[str, torch.Tensor]) -> Dict:
    """Counts, pairwise overlaps and the union for a multi-source target set.

    ``per_source`` maps a source name to the boolean voxel mask it supports, already
    restricted to the population of interest.
    """
    names = [s for s in C.SOURCES if s in per_source]
    counts = {s: int(per_source[s].sum()) for s in names}
    overlaps: Dict[str, int] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlaps[f"{a}&{b}"] = int((per_source[a] & per_source[b]).sum())
    union = None
    for s in names:
        union = per_source[s] if union is None else (union | per_source[s])
    exclusive = {s: int((per_source[s] & ~_others(per_source, names, s)).sum())
                 for s in names}
    return {"sources": counts, "overlaps": overlaps, "exclusive_to_source": exclusive,
            "n_union": int(union.sum()) if union is not None else 0,
            "sum_over_sources": int(sum(counts.values()))}


def _others(per_source: Dict[str, torch.Tensor], names: Sequence[str],
            skip: str) -> torch.Tensor:
    out = None
    for s in names:
        if s == skip:
            continue
        out = per_source[s] if out is None else (out | per_source[s])
    if out is None:
        key = next(iter(per_source))
        return torch.zeros_like(per_source[key])
    return out


def agreement(sem: Dict, U: torch.Tensor) -> Dict:
    """Agreement between the first two privileged views of the same voxel.

    This measures whether the frozen teacher says the same thing from two different
    viewpoints. It is **consistency, not correctness**: two views can agree on a wrong class.
    """
    keep = U[sem["flat"]]
    nv = sem["n_views"][keep]
    v0 = sem["view0_top1"][keep]
    v1 = sem["view1_top1"][keep]
    two = nv >= 2
    n_two = int(two.sum())
    same = int(((v0 == v1) & two).sum())
    prob = sem["prob"][keep]
    top = prob.max(1).values if prob.numel() else prob.new_zeros(0)
    return {"n_with_two_views": n_two, "n_two_views_agree": same,
            "two_view_agreement": (same / n_two) if n_two else None,
            "mean_top_probability": float(top.mean()) if top.numel() else None,
            "note": "consistency between views of a frozen teacher, not semantic correctness"}


def repeat_vs_new(sem: Dict, U: torch.Tensor) -> Dict:
    """Separate voxels the causal cameras already measured from genuinely new support.

    ``hist_seen`` is privileged: it says a camera inside the causal window did have a LiDAR
    return at that voxel, so the student *missed* it rather than never having had a chance.
    A target there is weaker evidence of new coverage than one the causal window never saw.
    """
    keep = U[sem["flat"]]
    hs = sem["hist_seen"][keep]
    nv = sem["n_views"][keep]
    n = int(keep.sum())
    return {"n_targets": n,
            "n_causal_camera_had_a_return": int(hs.sum()),
            "n_newly_supported": int((~hs).sum()),
            "n_single_observation": int((nv <= 1).sum()),
            "n_repeated_observation": int((nv >= 2).sum())}


def _tier_counts(sem: Dict, U: torch.Tensor) -> Dict[str, int]:
    keep = U[sem["flat"]]
    tier = sem["tier"][keep]
    return {"A_occupied_at_t": int((tier == 1).sum()), "B_occupancy_uncertain": int((tier == 2).sum())}


def _dist(v: torch.Tensor) -> Dict:
    if v.numel() == 0:
        return {"n": 0}
    x = v.detach().float().cpu().numpy()
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
            "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90))}


__all__ = ["GEOM_LIDAR", "GEOM_DEPTH", "summarize_single_source", "combine_sources",
           "agreement", "repeat_vs_new"]
