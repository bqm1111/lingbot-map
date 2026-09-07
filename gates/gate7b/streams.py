"""The chronological, deduplicated frame stream of each benchmark.

Gate 6 and Gate 7A consumed **overlapping five-frame clips**. Those are not independent
streams: consecutive KITTI-360 clips share four of five frames, and SemanticKITTI clips
tile one sequence at stride five. Replaying them as separate streams would reset LingBot's
state 163/1182/1753 times and would feed the same frame to the model many times over.

This module produces, per benchmark, the ordered set of **unique** frames with a genuine
boundary marked only where one exists:

* SemanticKITTI: sequence 08, one stream, frames ordered by odometry frame id;
* SSCBench-KITTI-360: drive ``2013_05_28_drive_0006_sync``, one stream, ordered by
  native frame index;
* Occ3D-nuScenes: **150 official validation scenes**, one stream each, ordered by sample
  timestamp. A scene change is a genuine boundary and is the only place the state resets.

It also records, for every official evaluation clip, the index of its anchor frame in the
stream, so a target timestamp can be evaluated against exactly the causal prefix
``[0, anchor]``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from gates.gate6 import frames as G6F

#: How each benchmark's stream is segmented, and why.
BOUNDARY_RULE = {
    "semantickitti": ("one stream: SemanticKITTI odometry sequence 08. The Gate-6 clip "
                      "set tiles the sequence at stride 5, so the stream is the 815 "
                      "unique frames in odometry order."),
    "kitti360": ("one stream: KITTI-360 drive 2013_05_28_drive_0006_sync. The Gate-5.2 "
                 "clip set slides by one frame, so the stream is the 1777 unique "
                 "rectified frames in native index order."),
    "occ3d": ("150 streams, one per official nuScenes validation scene. A scene change "
              "is a genuine boundary; within a scene the frames are the unique CAM_FRONT "
              "samples in timestamp order."),
}


@dataclass
class Frame:
    """One element of a stream."""
    key: str                    # the Gate-6 frame key; indexes the Trident cache
    path: str                   # absolute image path
    order: int                  # sort key inside the segment
    index: int = -1             # position in the segment, filled by :func:`build`


@dataclass
class Segment:
    """A contiguous run of frames over which LingBot's state is never reset."""
    dataset: str
    name: str                   # sequence id or scene token
    frames: List[Frame] = field(default_factory=list)
    #: ``clip_id -> index of its anchor frame in this segment``
    anchors: Dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def keys(self) -> Tuple[str, ...]:
        return tuple(f.key for f in self.frames)


def _order_of(dataset: str, rec, i: int) -> int:
    """The chronological sort key of frame ``i`` of a clip record."""
    raw = rec.raw
    if dataset == "semantickitti":
        return int(raw["frame_ids"][i])
    if dataset == "kitti360":
        return int(raw["native_frames"][i])
    return int(raw["timestamps_ns"][i])


def _segment_of(dataset: str, rec) -> str:
    if dataset == "semantickitti":
        return str(rec.raw["sequence"])
    if dataset == "kitti360":
        return str(rec.raw["sequence"])
    return str(rec.raw["scene"])


def build(dataset: str, repo_root: str) -> List[Segment]:
    """The benchmark's streams, deduplicated and chronologically ordered.

    Every frame appears exactly once. Every official evaluation clip contributes its
    **anchor** -- the last frame of the clip, which is the frame its target is defined at
    -- so the evaluator can address the causal prefix that ends there.
    """
    root = G6F.image_root(dataset, repo_root)
    segs: Dict[str, Dict[str, Frame]] = {}
    anchors: Dict[str, Dict[str, int]] = {}
    order_of_key: Dict[str, Dict[str, int]] = {}

    for rec in G6F.read_manifest(dataset, repo_root):
        seg = _segment_of(dataset, rec)
        segs.setdefault(seg, {})
        order_of_key.setdefault(seg, {})
        for i, (rel, key) in enumerate(zip(rec.rel_images, rec.keys)):
            o = _order_of(dataset, rec, i)
            prev = order_of_key[seg].get(key)
            if prev is not None and prev != o:
                raise ValueError(f"{dataset}/{seg}: frame {key} has two different "
                                 f"chronological positions ({prev} and {o})")
            order_of_key[seg][key] = o
            segs[seg].setdefault(key, Frame(key=key, path=os.path.join(root, rel),
                                            order=o))
        anchors.setdefault(seg, {})[rec.clip_id] = rec.keys[-1]

    out: List[Segment] = []
    for name in sorted(segs):
        frames = sorted(segs[name].values(), key=lambda f: (f.order, f.key))
        for i, f in enumerate(frames):
            f.index = i
        pos = {f.key: f.index for f in frames}
        seg = Segment(dataset=dataset, name=name, frames=frames,
                      anchors={cid: pos[k] for cid, k in anchors[name].items()})
        out.append(seg)
    return out


def summarize(dataset: str, repo_root: str) -> dict:
    segs = build(dataset, repo_root)
    n = [len(s) for s in segs]
    return {"dataset": dataset, "n_segments": len(segs), "n_frames": int(sum(n)),
            "frames_per_segment": {"min": min(n), "median": sorted(n)[len(n) // 2],
                                   "max": max(n)},
            "n_anchor_clips": int(sum(len(s.anchors) for s in segs)),
            "boundary_rule": BOUNDARY_RULE[dataset]}


__all__ = ["Frame", "Segment", "build", "summarize", "BOUNDARY_RULE"]
