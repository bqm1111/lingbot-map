"""The physical streams of each benchmark, and the manifest written before the main run.

A *segment* is a run of frames over which the reconstruction state is never reset. There is
exactly one genuine boundary rule per benchmark and it is not a tuning knob:

* **SemanticKITTI** -- one segment: odometry sequence 08. Our frozen frame stream is the
  815 unique stride-5 frames (raw ids 0, 5, ... 4070) in odometry order, which is the set
  the canonical five-frame causal windows are addressed in. One stream, so SemanticKITTI
  contributes **one** independent stream unit, and every interval on it says so.
* **Occ3D-nuScenes** -- 150 segments, one per official validation scene, frames being the
  scene's CAM_FRONT keyframe samples in timestamp order. A scene change is the only reset.

The anchors are not re-derived here: they are exactly ``tools/zero_shot_eval.samples``'
canonical population (the samples OccAny scored whose causal window t-4..t exists), so the
streaming modes are scored on the same anchors as every historical table.

Every canonical anchor sits at stream position >= 4 by construction -- the causal window
needs four earlier frames -- and the mode-B scale is initialized from positions 0..4. The
first scoreable anchor is therefore position 4, whose own window *is* the initialization
window, and **no anchor is excluded by the initialization requirement**. The manifest
asserts this rather than assuming it.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from tools.method_vs_data.streaming import config as SV
from tools.occany_comparison import config as OC
from tools.zero_shot_eval import samples as ZS

CAUSAL = OC.PROTOCOLS["causal"]

BOUNDARY_RULE = {
    "semantickitti": ("one segment: SemanticKITTI odometry sequence 08, the 815 unique "
                      "stride-5 frames in odometry order. One physical stream."),
    "occ3d": ("150 segments, one per official nuScenes validation scene; a scene change is "
              "the only reset. Frames are the scene's CAM_FRONT keyframe samples in "
              "timestamp order."),
}


@dataclass
class Segment:
    dataset: str
    name: str
    frame_keys: Tuple[str, ...]
    frame_paths: Tuple[str, ...]
    times_s: np.ndarray                      # seconds, relative to the segment start
    anchors: List[Tuple[str, int]] = field(default_factory=list)   # (sample_id, position)

    def __len__(self) -> int:
        return len(self.frame_keys)

    @property
    def init_positions(self) -> List[int]:
        return list(range(min(SV.N_INIT_FRAMES, len(self))))

    @property
    def first_scoreable_position(self) -> int:
        """The earliest position at which the mode-B gauge exists AND a causal window fits."""
        return max(SV.N_INIT_FRAMES - 1, -min(CAUSAL.offsets))

    def gaps_s(self) -> np.ndarray:
        return np.diff(self.times_s) if len(self) > 1 else np.zeros(0)


def _sk_times(repo_root: str) -> np.ndarray:
    p = os.path.join(repo_root, "data", "kitti", "dataset", "sequences", "08", "times.txt")
    return np.loadtxt(p, dtype=np.float64)


def build(dataset: str, repo_root: str = SV.REPO_ROOT) -> List[Segment]:
    samples = ZS.evaluated_samples(dataset)
    assert samples, f"{dataset}: no canonical samples"
    by_seg: Dict[str, List] = {}
    for s in samples:
        by_seg.setdefault(s.segment, []).append(s)

    if dataset == "semantickitti":
        raw_times = _sk_times(repo_root)

    out: List[Segment] = []
    for name in sorted(by_seg):
        ss = sorted(by_seg[name], key=lambda s: s.target_pos)
        keys, paths = ss[0].frame_keys, ss[0].frame_paths
        for s in ss:                       # one ordered frame list per segment, asserted
            assert s.frame_keys == keys and s.frame_paths == paths, f"{name}/{s.sample_id}"
        if dataset == "semantickitti":
            ids = np.asarray([int(k.split("_")[1]) for k in keys], np.int64)
            assert np.array_equal(ids, 5 * np.arange(len(keys))), "stride-5 stream broken"
            t = raw_times[ids] - raw_times[ids[0]]
        else:
            from core.datasets import nuscenes as NU, sources as S
            ann = _ann(S.OCC3D_ROOT)
            frs = NU.scene_frames(ann, name, "CAM_FRONT", OC.NUSCENES_ROOT)
            fk = tuple(os.path.splitext(os.path.basename(f.image_path))[0] for f in frs)
            assert fk == tuple(keys), name
            ts = np.asarray([f.timestamp_ns for f in frs], np.float64)
            t = (ts - ts[0]) / 1e9
        seg = Segment(dataset, name, tuple(keys), tuple(paths), t,
                      [(s.sample_id, s.target_pos) for s in ss])
        lo = seg.first_scoreable_position
        assert min(p for _, p in seg.anchors) >= lo, \
            f"{name}: an anchor sits before the initialization window ends"
        out.append(seg)
    n = sum(len(s.anchors) for s in out)
    assert n == len(samples), f"{dataset}: {n} anchors placed, {len(samples)} canonical"
    return out


_ANN: Dict[str, dict] = {}


def _ann(root: str) -> dict:
    if root not in _ANN:
        from core.datasets import nuscenes as NU
        _ANN[root] = NU.load_annotations(root)
    return _ANN[root]


def manifest(dataset: str, repo_root: str = SV.REPO_ROOT) -> dict:
    """The protocol manifest, written before the main run and never regenerated after it."""
    from gates.gate7b import replay as RP
    segs = build(dataset, repo_root)
    rows, all_gaps = [], []
    for s in segs:
        k = RP.keyframe_interval_for(len(s))
        g = s.gaps_s()
        all_gaps.append(g)
        rows.append({
            "segment": s.name,
            "n_frames_processed": len(s),
            "stream_start": {"position": 0, "key": s.frame_keys[0], "time_s": 0.0},
            "end_condition": {"position": len(s) - 1, "key": s.frame_keys[-1],
                              "time_s": float(s.times_s[-1]),
                              "rule": "last frame of the physical segment"},
            "duration_s": float(s.times_s[-1]),
            "initialization_frames": [{"position": p, "key": s.frame_keys[p],
                                       "time_s": float(s.times_s[p])}
                                      for p in s.init_positions],
            "n_scored_anchors": len(s.anchors),
            "first_scored_position": int(min(p for _, p in s.anchors)),
            "last_scored_position": int(max(p for _, p in s.anchors)),
            "timestamp_gap_s": {"min": float(g.min()) if g.size else None,
                                "median": float(np.median(g)) if g.size else None,
                                "p95": float(np.percentile(g, 95)) if g.size else None,
                                "max": float(g.max()) if g.size else None},
            "lingbot_keyframe_interval": int(k),
            "rope_slots_used": int(RP.rope_slots(len(s), k)),
            "rope_table_size": int(RP.MAX_FRAME_NUM),
            "resets": 1,
            "reset_positions": [0],
        })
    gaps = np.concatenate(all_gaps) if all_gaps else np.zeros(0)
    n_anchor = sum(r["n_scored_anchors"] for r in rows)
    return {
        "dataset": dataset,
        "boundary_rule": BOUNDARY_RULE[dataset],
        "n_segments": len(segs),
        "n_independent_streams": len(segs),
        "n_frames_processed_total": int(sum(r["n_frames_processed"] for r in rows)),
        "n_scored_anchors_total": int(n_anchor),
        "canonical_population": int(len(ZS.evaluated_samples(dataset))),
        "exclusions": [],
        "exclusion_reason": ("none: every canonical anchor sits at stream position >= 4, the "
                             "causal window needs four earlier frames, and the mode-B gauge "
                             "is complete after position 4, so the initialization requirement "
                             "excludes no anchor"),
        "modes": {m: SV.MODE_NAMES[m] for m in SV.MODES},
        "mode_A": {"reset": "once per evaluation anchor (per five-frame clip)",
                   "scale": "estimated from that clip's own five frames",
                   "n_resets": int(n_anchor),
                   "source": "frozen clip cache tools/occany_comparison/cli/build_clips.py"},
        "mode_B": {"reset": "once per physical segment",
                   "scale": ("one scalar from the segment's first five frames, frozen for "
                             "the rest of the segment; no later MoGe call, no refresh"),
                   "n_resets": len(segs),
                   "mapper_window": ("the same five causal frames the reference uses for "
                                     "that anchor; the explicit occupancy-map window is NOT "
                                     "widened, only LingBot's own state persists")},
        "n_init_frames": SV.N_INIT_FRAMES,
        "causal_offsets": list(CAUSAL.offsets),
        "anchor_slot": CAUSAL.anchor_slot,
        "timestamp_gap_s_overall": {"min": float(gaps.min()), "median": float(np.median(gaps)),
                                    "p95": float(np.percentile(gaps, 95)),
                                    "max": float(gaps.max())},
        "segments": rows,
    }


def write_manifest(dataset: str) -> dict:
    SV.ensure_dirs()
    m = manifest(dataset)
    json.dump(m, open(SV.manifest_path(dataset), "w"), indent=1, default=float)
    return m


__all__ = ["Segment", "build", "manifest", "write_manifest", "BOUNDARY_RULE", "CAUSAL"]
