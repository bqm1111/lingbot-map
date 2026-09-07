#!/usr/bin/env python
"""Gate 5.2 step 1 — the five-frame clip manifest, with timestamps.

Chronological ``[t-4, ..., t]`` SSCBench anchor clips over validation sequence 0006.
Eligibility is purely temporal and was declared before any result was seen; no target,
LiDAR or prediction file is opened here.

    python tools/gate5_2/build_manifest.py
"""
from __future__ import annotations

import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, load_config, write_json
from sscbench_kitti360.adapter import (ANCHOR_STRIDE, OFFICIAL_SPLIT, SEQUENCE,
                                       anchor_indices, build_clips, load_timestamps,
                                       parse_calibration, pose_frames,
                                       sscbench_to_native)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/gate5_2/kitti360_transfer.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    d = cfg.dataset
    assert d.sequence == SEQUENCE and OFFICIAL_SPLIT["val"] == [SEQUENCE], \
        "sequence 0006 is the official SSCBench-KITTI-360 validation split"

    frames = pose_frames(os.path.join(d.root, "data_poses", SEQUENCE, "poses.txt"))
    ts = load_timestamps(os.path.join(d.kitti360_root, "data_2d_raw", SEQUENCE,
                                      d.camera, "timestamps.txt"))
    anchors = anchor_indices(os.path.join(d.root, "data_2d_raw", SEQUENCE, "voxels"))
    calib = parse_calibration(os.path.join(d.root, "calibration"))
    clips, excluded = build_clips(anchors, frames, ts, int(d.clip_length),
                                  int(cfg.eval.block_size), int(d.anchor_stride))

    out_dir = os.path.join(REPO_ROOT, "manifests", "gate5_2")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "val.jsonl")
    with open(path, "w") as fh:
        for c in clips:
            fh.write(json.dumps(c.to_dict(), sort_keys=True) + "\n")

    spans = np.array([c.span_s for c in clips])
    reasons = {}
    for e in excluded:
        reasons[e["reason"]] = reasons.get(e["reason"], 0) + 1
    summary = {
        "sequence": SEQUENCE, "official_role": "validation",
        "n_anchors_official": len(anchors),
        "anchor_first": anchors[0], "anchor_last": anchors[-1],
        "anchor_stride": int(d.anchor_stride),
        "n_pose_frames": int(len(frames)),
        "index_mapping": "kitti360_frame = pose_frames[sscbench_index + 1]",
        "n_clips_eligible": len(clips), "n_excluded": len(excluded),
        "exclusion_reasons": reasons,
        "excluded": excluded[:50],
        "span_s": {"median": float(np.median(spans)), "min": float(spans.min()),
                   "max": float(spans.max()), "mean": float(spans.mean())},
        "n_blocks": int(max(c.block for c in clips) + 1),
        "block_size": int(cfg.eval.block_size),
        "native_hw": list(calib.native_hw), "K_native": calib.K.tolist(),
        "manifest": os.path.relpath(path, REPO_ROOT),
    }
    write_json(os.path.join(REPO_ROOT, cfg.experiment.output_dir, "manifest_summary.json"),
               summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "excluded"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
