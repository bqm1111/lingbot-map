"""Reproducible scene/anchor selection, fixed BEFORE any supervision yield is examined.

The rule is the same for both arms and has no free parameter beyond the caps in
``config``: order the scenes deterministically, take the first ``MAX_SEGMENTS``, then inside
each scene lay anchors on a fixed grid that spreads them over the whole scene and keeps
consecutive picks at least ``MIN_ANCHOR_SEPARATION_FRAMES`` stream frames apart, so no two
anchors of a scene share a causal clip or a privileged window. Nothing here reads a target,
an evidence file or a statistic -- only the list of eligible anchors.

Reported alongside every selection: the temporal overlap actually achieved, because
neighbouring frames are not independent scenes and a cap on anchors is not a guarantee of
independence.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from tools.waymo_pilot import config as C


@dataclass(frozen=True)
class Anchor:
    scene: str
    index: int           # stream index of the anchor inside its scene
    arm: str             # "k360" or "waymo"

    @property
    def key(self) -> str:
        return f"{self.scene}:{self.index:05d}"


def spread(eligible: Sequence[int], k: int,
           min_sep: int = C.MIN_ANCHOR_SEPARATION_FRAMES) -> List[int]:
    """``k`` indices spread over ``eligible``, at least ``min_sep`` apart where possible.

    Deterministic: walk the eligible list at an even stride and keep a candidate only if it
    clears ``min_sep`` from the last kept one. If the scene is too short to give ``k``
    separated anchors, fewer are returned -- the cap is never met by packing neighbours.
    """
    e = sorted(int(x) for x in eligible)
    if not e or k <= 0:
        return []
    if k == 1:
        return [e[len(e) // 2]]
    out: List[int] = []
    for pos in np.linspace(0, len(e) - 1, num=k * 4):
        cand = e[int(round(float(pos)))]
        if out and cand - out[-1] < min_sep:
            continue
        out.append(cand)
        if len(out) == k:
            break
    return out


def independence_ceiling(eligible: Sequence[int],
                         min_sep: int = C.MIN_ANCHOR_SEPARATION_FRAMES) -> int:
    """How many anchors of this scene can be temporally independent at all.

    ``min_sep`` is chosen so two anchors share no frame whatsoever: the causal clip reaches
    4 frames back and the privileged window 20 forward, so 25 would already suffice, and 40
    leaves a margin of one further clip. This ceiling is reported because a cap on anchors is
    not the same as having that many independent examples -- if the ceiling is below the cap,
    the scene pool is the limit, not the budget.
    """
    e = sorted(int(x) for x in eligible)
    if not e:
        return 0
    n, last = 1, e[0]
    for x in e[1:]:
        if x - last >= min_sep:
            n += 1
            last = x
    return n


def select(eligible_by_scene: Dict[str, Sequence[int]], arm: str, max_scenes: int,
           max_anchors: int, per_scene: int) -> Tuple[List[Anchor], Dict]:
    """Apply the rule. Returns the anchors and the selection record for the report."""
    scenes = sorted(eligible_by_scene)[:max_scenes]
    picked: List[Anchor] = []
    per_scene_counts: Dict[str, int] = {}
    for s in scenes:
        if len(picked) >= max_anchors:
            break
        want = min(per_scene, max_anchors - len(picked))
        idx = spread(eligible_by_scene[s], want)
        if not idx:                 # a scene that supplies nothing is not a scene used
            continue
        per_scene_counts[s] = len(idx)
        picked += [Anchor(scene=s, index=i, arm=arm) for i in idx]
    gaps = _gaps(picked)
    rec = {
        "arm": arm, "rule": ("deterministic scene order, first N scenes, anchors spread "
                             f"with min separation {C.MIN_ANCHOR_SEPARATION_FRAMES} frames"),
        "sample_seed": C.SAMPLE_SEED,
        "caps": {"max_scenes": max_scenes, "max_anchors": max_anchors,
                 "per_scene": per_scene,
                 "min_separation_frames": C.MIN_ANCHOR_SEPARATION_FRAMES},
        "n_scenes_available": len(eligible_by_scene),
        "n_scenes_used": len(per_scene_counts),
        "scenes_with_no_eligible_anchor": sorted(s for s in scenes
                                                 if s not in per_scene_counts),
        "independent_anchor_ceiling": {s: independence_ceiling(v)
                                       for s, v in eligible_by_scene.items()},
        "independent_anchor_ceiling_total": sum(independence_ceiling(v)
                                               for v in eligible_by_scene.values()),
        "eligible_anchors_total": sum(len(v) for v in eligible_by_scene.values()),
        "n_anchors": len(picked), "anchors_per_scene": per_scene_counts,
        "within_scene_gap_frames": gaps,
        "anchors": [a.key for a in picked],
        "digest": digest(picked),
    }
    return picked, rec


def _gaps(anchors: Sequence[Anchor]) -> Dict[str, float]:
    by: Dict[str, List[int]] = {}
    for a in anchors:
        by.setdefault(a.scene, []).append(a.index)
    g = [b - a for v in by.values() for a, b in zip(sorted(v), sorted(v)[1:])]
    if not g:
        return {"n": 0}
    return {"n": len(g), "min": int(min(g)), "median": float(np.median(g)),
            "max": int(max(g))}


def digest(anchors: Sequence[Anchor]) -> str:
    """Content hash of a selection, so a later run can prove it used the same anchors."""
    payload = json.dumps([a.key for a in anchors], sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def assert_dev_drive_absent(anchors: Sequence[Anchor]) -> None:
    """KITTI-360 drive 0006 is development data and may never be sampled for training."""
    bad = sorted({a.scene for a in anchors if C.K360_DEV_DRIVE in a.scene})
    if bad:
        raise AssertionError(f"development drive sampled into a training arm: {bad}")


__all__ = ["Anchor", "spread", "select", "digest", "assert_dev_drive_absent",
           "independence_ceiling"]
