"""Deterministic, NESTED anchor manifests for the controlled experiments E1 and E2.

The brief requires every E1 anchor to also be an E2 anchor. ``waymo_pilot.sample.spread``
alone cannot give that: called with k=2 and k=13 it returns two sets that need not nest. So
each segment's eligible anchors are put in one **priority order** built by calling ``spread``
with increasing k and appending whatever is new, which makes every prefix of that order both
nested and about as spread as ``spread`` would have made it on its own. An experiment of
``n`` anchors per segment is then the first ``n`` of the order.

Independence is unchanged by any of this and is reported separately: a 20 s Waymo segment
(39 stream frames) admits exactly ONE anchor whose five-frame clip and 20-frame privileged
window share nothing with another's, so the independent count is the SEGMENT count regardless
of how many anchors are packed into each.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Sequence, Tuple

from tools.waymo_arm import config as A
from tools.waymo_pilot import sample as SA

#: anchors per segment for each experiment. E1 ~1,500 and E2 ~10,000 over ~798 segments.
EXPERIMENTS: Dict[str, int] = {"E1": 2, "E2": 13}
#: the packing already on disk from the first clip build, kept as a prefix where possible
PREBUILT_PER_SEGMENT = 5
PREBUILT_MIN_SEP = 3


def nested_priority(eligible: Sequence[int]) -> List[int]:
    """Eligible anchors in one order whose every prefix is nested and spread."""
    e = sorted(int(x) for x in eligible)
    order: List[int] = []
    for k in range(1, len(e) + 1):
        sep = PREBUILT_MIN_SEP if k > PREBUILT_PER_SEGMENT else max(1, len(e) // max(k, 1))
        for i in SA.spread(e, k, sep):
            if i not in order:
                order.append(i)
        if len(order) >= len(e):
            break
    order += [i for i in e if i not in order]          # anything spread never reached
    return order


def manifest(n_per_segment: int, segments: Sequence[str],
             eligible_by_segment: Dict[str, Sequence[int]]) -> Dict:
    picked: Dict[str, List[int]] = {}
    for s in segments:
        order = nested_priority(eligible_by_segment[s])
        if order:
            picked[s] = sorted(order[:n_per_segment])
    anchors = [f"{s}:{t:05d}" for s in sorted(picked) for t in picked[s]]
    independent = len(picked)          # one per segment, by the measured segment length
    return {"n_per_segment": n_per_segment, "n_segments": len(picked),
            "n_anchors": len(anchors),
            "independent_anchors": independent,
            "independence_rule_frames": A.MIN_ANCHOR_SEPARATION_FRAMES,
            "independence_note": ("one anchor per 20 s segment has a non-overlapping clip and "
                                  "privileged window; n_anchors is a training volume, never "
                                  "an independent count"),
            "anchors_per_segment": {s: len(v) for s, v in picked.items()},
            "picked": picked,
            "digest": hashlib.sha256("\n".join(anchors).encode()).hexdigest()}


def build_all(paths: A.Paths = None) -> Dict:
    from tools.waymo_pilot import waymo_geometry as WG
    p = paths or A.Paths()
    segs = A.segments(p)
    eligible = {s: WG.WaymoSegment(s).anchors() for s in segs}
    sp = A.split(segs)
    out = {"version": A.VERSION, "n_segments_available": len(segs),
           "split": {"n_train": len(sp["train"]), "n_val": len(sp["val"]),
                     "val_segments": sp["val"], "disjoint": True},
           "eligible_per_segment": {"min": min(map(len, eligible.values())),
                                    "max": max(map(len, eligible.values())),
                                    "total": sum(map(len, eligible.values()))},
           "experiments": {}}
    for name, n in EXPERIMENTS.items():
        tr = manifest(n, sp["train"], eligible)
        va = manifest(n, sp["val"], eligible)
        out["experiments"][name] = {"train": tr, "val": va,
                                    "n_anchors_total": tr["n_anchors"] + va["n_anchors"]}
    # nesting assertion, per split, over every segment
    for a, b in (("E1", "E2"),):
        for part in ("train", "val"):
            pa = out["experiments"][a][part]["picked"]
            pb = out["experiments"][b][part]["picked"]
            for s, v in pa.items():
                assert set(v).issubset(set(pb.get(s, []))), f"{a} not nested in {b} at {s}"
    out["nested_E1_in_E2"] = True
    return out


def write(out: Dict, paths: A.Paths = None) -> str:
    p = paths or A.Paths()
    d = os.path.join(A.ARTIFACT_DIR, "manifests")
    os.makedirs(d, exist_ok=True)
    for name, rec in out["experiments"].items():
        for part in ("train", "val"):
            with open(os.path.join(d, f"{name}_{part}.json"), "w") as fh:
                json.dump(rec[part], fh, indent=1)
    path = os.path.join(d, "manifests.json")
    with open(path, "w") as fh:
        json.dump({k: v for k, v in out.items() if k != "experiments"} |
                  {"experiments": {n: {q: {kk: vv for kk, vv in r[q].items() if kk != "picked"}
                                       for q in ("train", "val")}
                                   for n, r in out["experiments"].items()}}, fh, indent=1)
    return path


__all__ = ["EXPERIMENTS", "nested_priority", "manifest", "build_all", "write"]
