#!/usr/bin/env python
"""Convert Waymo ``.tfrecord`` segments into the pilot's per-frame cache. REQUIRES REAL DATA.

This is the only step that needs TensorFlow and the Waymo SDK, and it is deliberately the
only place they are imported, so everything downstream is numpy/torch. It downloads nothing,
accepts no dataset licence, and fabricates nothing: without real ``.tfrecord`` files it
prints the exact missing prerequisite and exits non-zero.

    WAYMO_ROOT=/path/to/tfrecords conda run -n waymo python \\
        tools/waymo_pilot/cli/waymo_extract.py --segments 2 --frames-per-segment 40

Per frame it writes one npz: the five camera JPEGs, their per-image vehicle-to-world poses
and timestamps, the vehicle-frame LiDAR first return, and the SDK's own camera projection of
each point. Human annotations are never read.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.waymo_pilot import config as C, waymo as W                       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=C.WAYMO_ROOT)
    ap.add_argument("--segments", type=int, default=C.SMOKE_SEGMENTS,
                    help=f"segments to convert (cap {C.MAX_SEGMENTS})")
    ap.add_argument("--frames-per-segment", type=int, default=0,
                    help="native frames per segment; 0 (the default) means the whole "
                         "segment. An anchor needs frames [f-back, f+forward] around it, so "
                         "a short extraction supports NO anchors -- see --explain")
    ap.add_argument("--stride", type=int, default=1,
                    help="frame decimation at extraction; keep 1 so the interleaved "
                         "temporal source stays available (the stream stride is applied later)")
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--explain", action="store_true",
                    help="print the per-anchor frame budget and exit")
    a = ap.parse_args()

    budget = C.anchor_frame_budget()
    if a.explain:
        print(json.dumps({"stream_stride": C.WAYMO_STREAM_STRIDE,
                          "stream_dt_s": C.WAYMO_STREAM_STRIDE / C.WAYMO_NOMINAL_HZ,
                          **budget,
                          "anchors_supported_by_a_199_frame_segment":
                              C.anchors_supported(199)}, indent=1))
        return 0

    av = W.probe(a.root)
    print(json.dumps(av.as_dict(), indent=1))
    if a.probe_only:
        return 0 if av.ok else 2
    if not av.ok:
        print("\nBLOCKED -- the Waymo arm cannot run:", file=sys.stderr)
        for m in av.missing():
            print(f"  - {m}", file=sys.stderr)
        print("\n" + W.acquisition_instructions(a.root), file=sys.stderr)
        return 2
    if a.segments > C.MAX_SEGMENTS:
        print(f"refusing {a.segments} segments: the pilot cap is {C.MAX_SEGMENTS}",
              file=sys.stderr)
        return 2

    C.ensure_dirs()
    p = C.Paths()
    segs = sorted(av.segments)[:a.segments]
    t0, n_frames, n_bytes = time.time(), 0, 0
    manifest = []
    for seg in segs:
        src = os.path.join(a.root, seg)
        if not os.path.isfile(src):
            matches = [q for q in
                       (os.path.join(dp, seg) for dp, _, fn in os.walk(a.root) if seg in fn)]
            src = matches[0] if matches else src
        want = a.frames_per_segment or None
        if want is not None and C.anchors_supported(want) == 0:
            print(f"refusing --frames-per-segment {want}: an anchor needs "
                  f"{budget['span']} native frames around it "
                  f"({budget['frames_back']} back, {budget['frames_forward']} forward), so "
                  f"this extraction would support 0 anchors. Use 0 for the whole segment.",
                  file=sys.stderr)
            return 2
        rows = []
        for idx, fr in W.read_segment(src, frame_stride=a.stride, max_frames=want):
            dst = p.waymo_frame_path(seg, idx)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            sync = W.sync_error_s(fr["image_timing"], fr["frame_timestamp_s"])
            payload = {
                "segment": fr["segment"], "frame_index": np.int32(idx),
                "frame_timestamp_s": np.float64(fr["frame_timestamp_s"]),
                "vehicle_to_world": fr["vehicle_to_world"],
                "points_vehicle": fr["points_vehicle"], "cp_points": fr["cp_points"],
                "calibration_json": json.dumps({k: v.to_dict()
                                                for k, v in fr["cameras"].items()}),
                "timing_json": json.dumps(fr["image_timing"]),
                "sync_error_json": json.dumps(sync),
            }
            for nm, jpg in fr["images"].items():
                payload[f"image_{nm}"] = np.frombuffer(jpg, np.uint8)
                payload[f"pose_{nm}"] = fr["image_pose"][nm]
            tmp = dst + ".tmp.npz"
            np.savez_compressed(tmp, **payload)
            os.replace(tmp, dst)
            sz = os.path.getsize(dst)
            n_bytes += sz
            n_frames += 1
            rows.append({"frame_index": int(idx), "bytes": int(sz),
                         "max_sync_error_s": max(sync.values()) if sync else None,
                         "n_points": int(len(fr["points_vehicle"]))})
            if n_bytes / 2 ** 30 > C.MAX_DERIVED_GIB:
                print(f"STOP: derived cache reached the {C.MAX_DERIVED_GIB} GiB cap "
                      f"after {n_frames} frames; reporting actual coverage", file=sys.stderr)
                break
        manifest.append({"segment": seg, "n_frames": len(rows), "frames": rows})
        if n_bytes / 2 ** 30 > C.MAX_DERIVED_GIB:
            break
    el = time.time() - t0
    rec = {"version": C.VERSION, "root": a.root, "stride": a.stride,
           "anchor_frame_budget": budget,
           "anchors_supported_per_segment": {m["segment"]: C.anchors_supported(m["n_frames"])
                                             for m in manifest},
           "n_segments": len(manifest), "n_frames": n_frames,
           "gib": n_bytes / 2 ** 30, "seconds": el,
           "seconds_per_frame": el / max(n_frames, 1),
           "caps": {"max_derived_gib": C.MAX_DERIVED_GIB,
                    "max_preprocess_gpu_hours": C.MAX_PREPROCESS_GPU_HOURS},
           "segments": manifest}
    out = os.path.join(C.Paths().logs, "waymo_extract.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=2)
    print(f"[waymo_extract] {n_frames} frames from {len(manifest)} segments, "
          f"{n_bytes / 2 ** 30:.2f} GiB, {el / 60:.1f} min -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
