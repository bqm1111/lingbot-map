#!/usr/bin/env python
"""Build the frozen causal clips (LingBot + MoGe-2 scale) for Waymo anchors. GPU3 only.

The deployable half of the Waymo arm: five images of the selected camera ending at the
anchor, LingBot replayed on them alone with a cleared KV cache, and the causal MoGe-2 clip
scale. No LiDAR, no other camera, no future frame, no Waymo pose enters this -- the clip's
poses are LingBot's own predictions.

    CUDA_VISIBLE_DEVICES=3 python tools/waymo_pilot/cli/build_waymo_clips.py \\
        --segments 2 --anchors-per-segment 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from core.datasets.audit import FileAudit                                      # noqa: E402
from tools.waymo_pilot import config as C, sample as SA                        # noqa: E402
from tools.waymo_pilot import waymo_geometry as WG                             # noqa: E402


def clip_path(segment: str, t: int) -> str:
    return os.path.join(C.Paths().data_root, "clips", segment, f"{t:05d}.npz")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0",
                    help="index WITHIN CUDA_VISIBLE_DEVICES; pin the card with that env var")
    ap.add_argument("--segments", type=int, default=2)
    ap.add_argument("--anchors-per-segment", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-separation", type=int, default=C.MIN_ANCHOR_SEPARATION_FRAMES,
                    help="stream frames between two anchors of one scene. The default is the "
                         "pilot's independence rule (40), which a 39-frame Waymo segment can "
                         "satisfy exactly once. The training arm lowers it to pack more "
                         "anchors per scene; those anchors share privileged frames and the "
                         "independent count stays one per segment.")
    ap.add_argument("--log-tag", default="waymo_clips")
    ap.add_argument("--skip-segments", type=int, default=0,
                    help="skip this many segments before taking --segments (sharding)")
    a = ap.parse_args()

    C.ensure_dirs()
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    name = torch.cuda.get_device_name(dev)
    print(f"[clips] device {dev} ({name}); CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")

    root = C.Paths().waymo_frames
    segs = sorted(d for d in os.listdir(root)
                  if len(os.listdir(os.path.join(root, d))) >= 190)[a.skip_segments:a.skip_segments + a.segments]
    if not segs:
        print("no fully extracted segments yet", file=sys.stderr)
        return 2

    eligible = {}
    seg_objs = {}
    for s in segs:
        so = WG.WaymoSegment(s)
        seg_objs[s] = so
        eligible[s] = so.anchors()
    anchors, selection = SA.select(eligible, "waymo", len(segs),
                                   a.limit or len(segs) * a.anchors_per_segment,
                                   a.anchors_per_segment, min_sep=a.min_separation)
    selection["min_separation_frames_used"] = a.min_separation
    print(f"[clips] {selection['n_anchors']} anchors from {selection['n_scenes_used']} "
          f"segments (digest {selection['digest']})")

    from gates.gate7b import replay                                            # noqa: PLC0415
    from gates.scale_gate.config import load_config                            # noqa: PLC0415
    from moge_gauge.calibrated import CalibratedMoGe                           # noqa: PLC0415

    t_load = time.time()
    model, info = replay.build_model(dev)
    g5 = load_config("configs/gate5/moge_metric_gauge.yaml")
    moge = CalibratedMoGe(dev, moge_src=g5.moge.src_dir, hf_repo=g5.moge.hf_repo,
                          revision=g5.moge.hf_revision)
    prov = moge.provenance()
    assert prov["hf_revision"] == "39c4d5e957afe587e04eec59dc2bcc3be5ecd968", \
        "MoGe-2 revision differs from the frozen gauge"
    print(f"[clips] models loaded in {time.time() - t_load:.0f}s")

    cache = {}

    def moge_fn(path: str, fov: float):
        if path not in cache:
            rgb = replay.load_images([path])[0].numpy()
            o = moge.infer_calibrated(rgb, fov_x_deg=float(fov))
            cache[path] = (np.asarray(o.depth_z, np.float32), np.asarray(o.mask, bool))
            if len(cache) > 64:
                cache.pop(next(iter(cache)))
        return cache[path]

    rows, t0 = [], time.time()
    # the causal half may not read LiDAR, another camera, a future frame or a Waymo pose
    forbidden = ("points_vehicle", "cp_points", "/targets/", "delayed_targets")
    with FileAudit(patterns=C.BENCHMARK_FORBIDDEN, strict=True) as audit:
        for i, an in enumerate(anchors):
            out = clip_path(an.scene, an.index)
            if os.path.exists(out):
                rows.append({"segment": an.scene, "t": an.index, "skipped": True})
                continue
            seg = seg_objs[an.scene]
            cam = seg.calibration()
            ts = time.time()
            rec = WG.build_clip(seg, an.index, model, moge_fn, cam, dev)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            tmp = out + ".tmp.npz"
            np.savez_compressed(tmp, **rec)
            os.replace(tmp, out)
            rows.append({"segment": an.scene, "t": an.index,
                         "scale_m_per_unit": float(rec["scale"]),
                         "n_moge_ok": int(np.isfinite(rec["log_s_frames"]).sum()),
                         "seconds": time.time() - ts,
                         "bytes": os.path.getsize(out)})
            if (i + 1) % 5 == 0:
                print(f"  {i + 1}/{len(anchors)} {(time.time() - t0) / (i + 1):.1f}s each",
                      flush=True)

    built = [r for r in rows if not r.get("skipped")]
    scales = [r["scale_m_per_unit"] for r in built if np.isfinite(r["scale_m_per_unit"])]
    rec = {"version": C.VERSION, "device": name, "selection": selection,
           "n_built": len(built), "n_skipped": len(rows) - len(built),
           "seconds": time.time() - t0,
           "seconds_per_anchor": (time.time() - t0) / max(len(built), 1),
           "bytes_per_anchor": float(np.mean([r["bytes"] for r in built])) if built else 0,
           "scale_m_per_unit": {"n": len(scales),
                                "median": float(np.median(scales)) if scales else None,
                                "min": float(np.min(scales)) if scales else None,
                                "max": float(np.max(scales)) if scales else None},
           "n_anchors_all_five_moge": sum(1 for r in built if r.get("n_moge_ok") == 5),
           "firewall": {"n_opened": len(audit.opened), "violations": audit.violations},
           "rows": built[:50]}
    path = os.path.join(C.Paths().logs, f"{a.log_tag}.json")
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=2, default=float)
    print(f"\n[clips] {len(built)} built in {rec['seconds'] / 60:.1f} min "
          f"({rec['seconds_per_anchor']:.1f}s each, {rec['bytes_per_anchor'] / 2**20:.0f} MiB each)")
    print(f"[clips] clip scale median {rec['scale_m_per_unit']['median']}, "
          f"range [{rec['scale_m_per_unit']['min']}, {rec['scale_m_per_unit']['max']}]")
    print(f"[clips] anchors with all five MoGe candidates: "
          f"{rec['n_anchors_all_five_moge']}/{len(built)}  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
