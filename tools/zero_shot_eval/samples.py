"""Evaluated samples and the target frames the frozen causal protocol reads.

Samples: every sample the released OccAny model scored (SemanticKITTI 08: 805; Occ3D-nuScenes val: 4,819) whose
causal five-frame window t-4..t exists in the ordered sequence (801 / 4,219). Sample lists, frame order, the
target-camera-to-grid transform and the label references come from ``tools/occany_comparison/samples.py``
(verified there against OccAny's outputs and the official labels).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

from tools.occany_comparison import config as OC, samples as OS
from tools.zero_shot_eval import config as C

CAUSAL = OC.PROTOCOLS["causal"]


def evaluated_samples(dataset: str) -> List[OS.Sample]:
    return [s for s in OS.build_samples(dataset) if s.protocol_ok(CAUSAL)]


def history_frames(sample: OS.Sample) -> List[Dict[str, str]]:
    """The five frames t-4..t, oldest first."""
    return [{"key": sample.frame_keys[j], "path": sample.frame_paths[j]} for j in sample.window(CAUSAL)]


def write_frame_list(dataset: str) -> Dict:
    seen, frames = set(), []
    samples = evaluated_samples(dataset)
    for s in samples:
        for f in history_frames(s):
            if f["key"] not in seen:
                seen.add(f["key"]); frames.append(f)
    rec = {"dataset": dataset, "n_samples": len(samples), "n_frames": len(frames), "frames": frames}
    os.makedirs(C.DATA_ROOT, exist_ok=True)
    json.dump(rec, open(C.FRAME_LIST.format(dataset=dataset), "w"))
    return rec


if __name__ == "__main__":
    for d in C.DATASETS:
        r = write_frame_list(d); print(d, r["n_samples"], "samples", r["n_frames"], "frames", r["frames"][0])
