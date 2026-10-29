#!/usr/bin/env python
"""What supervision the three trained checkpoints ACTUALLY saw, read off their own caches.

This is a provenance audit, not a statistics probe. It does not rebuild targets with a
chosen policy and report what that policy would give; it opens the target files the three
trainings consumed and reports what is recorded in them -- frame counts, sweep counts,
native indices, timestamps where the corpus publishes them, temporal direction, and the
occupancy statistics the builder itself wrote into ``stats_json``.

The distinction matters because ``once_supervision_validation.md`` §5 measured KITTI-360
under a *re-built* 5-sweep window to decide whether a predeclared band applied. That probe
never trained anything. The KITTI-360 checkpoint that every comparison uses as the
reference was trained on the frozen cache, and the only way to know what window that cache
holds is to read it.

    PYTHONPATH=$PWD python tools/method_vs_data/supervision_provenance.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.datasets import kitti360_splits as KS, kitti360_transforms as TF              # noqa: E402

#: the target caches the three trainings read, and the drives/segments/sequences each used
ARMS = {
    "kitti360": {
        "targets": "/media/SSD1/MINH_DATASETS/lingbot_k360_7drives/targets",
        "scenes": list(KS.TRAIN_DRIVES),
        "native_hz_nominal": 10.0,
        "lasers": 1,
    },
    "waymo": {
        "targets": "/media/minh/MINH1/lingbot_waymo_arm/targets",
        "scenes": None,                      # every segment present
        "native_hz_nominal": 10.0,
        "lasers": 5,
    },
    "once": {
        "targets": "/media/minh/MINH1/lingbot_once_arm/targets",
        "scenes": None,
        "native_hz_nominal": 2.0,
        "lasers": 1,
    },
}


def _k360_timestamps():
    """velodyne timestamps per train drive, seconds since the drive's first entry."""
    out = {}
    for d in KS.TRAIN_DRIVES:
        geo = TF.DriveGeometry(d, KS.SSCBENCH_ROOT, KS.KITTI360_ROOT)
        for kind in ("velodyne", "image"):
            try:
                out[d] = (kind, np.asarray(geo.timestamps(kind), np.float64))
                break
            except Exception:
                continue
    return out


def audit(arm: str, per_scene: int = 8, max_scenes: int = 0) -> dict:
    cfg = ARMS[arm]
    root = cfg["targets"]
    scenes = cfg["scenes"] or sorted(d for d in os.listdir(root)
                                     if os.path.isdir(os.path.join(root, d)))
    n_scenes_total = len(scenes)
    if max_scenes and len(scenes) > max_scenes:                 # even stride, no RNG
        scenes = scenes[::max(1, len(scenes) // max_scenes)][:max_scenes]
    ts = _k360_timestamps() if arm == "kitti360" else {}
    rows, spans_s, n_frames, n_sweeps, prev, validf, confl, occn = [], [], [], [], [], [], [], []
    direction = set()
    for sc in scenes:
        ps = sorted(glob.glob(os.path.join(root, sc, "*.npz")))
        step = max(1, len(ps) // per_scene)
        for p in ps[::step][:per_scene]:
            z = np.load(p, allow_pickle=True)
            fu = np.asarray(z["frames_used"]).ravel()
            anchor = int(z["native_frame"])
            fut = np.asarray(z["future_natives"]).ravel()
            n_frames.append(int(len(fu)))
            st = json.loads(str(z["stats_json"])) if "stats_json" in z.files else {}
            n_sweeps.append(int(st["n_sweeps"]) if "n_sweeps" in st
                            else (int(z["n_sweeps"]) if "n_sweeps" in z.files
                                  else int(len(fu)) * cfg["lasers"]))
            # occupancy statistics recomputed from the stored volumes, so the three arms are
            # summarised the same way whether or not the builder wrote a stats_json
            n = int(np.prod(np.asarray(z["dims"]).ravel()))
            val = np.unpackbits(z["valid_packed"])[:n].astype(bool)
            occ = np.unpackbits(z["occ_packed"])[:n].astype(bool)
            nv = int(val.sum())
            prev.append(float((occ & val).sum()) / max(nv, 1))
            validf.append(nv / max(n, 1))
            confl.append(float(st.get("conflict_fraction_of_touched", np.nan)))
            occn.append(int((occ & val).sum()))
            # temporal direction, from the native indices relative to the anchor
            rel = fut - anchor
            direction.add("future_only" if (rel > 0).all() else
                          ("past_only" if (rel < 0).all() else "mixed"))
            # span in seconds
            if "supervision_timestamps_s" in z.files:
                t = np.asarray(z["supervision_timestamps_s"], np.float64)
                spans_s.append(float(t.max() - t.min()))
                src = "published per-sweep timestamps in the target file"
            elif arm == "kitti360" and sc in ts:
                kind, tt = ts[sc]
                idx = np.concatenate([[anchor], fut]).astype(int)
                idx = idx[idx < len(tt)]
                spans_s.append(float(tt[idx].max() - tt[idx].min()))
                src = f"KITTI-360 {kind} timestamps.txt"
            else:
                spans_s.append(float((fut.max() - anchor) / cfg["native_hz_nominal"]))
                src = "native index span / nominal rate (no published timestamps read)"
            rows.append({"scene": sc, "file": os.path.basename(p), "anchor_native": anchor,
                         "n_frames_used": int(len(fu)), "n_sweeps": n_sweeps[-1],
                         "native_span": int(fut.max() - anchor), "span_s": spans_s[-1]})
    q = lambda a: {"min": float(np.nanmin(a)), "p50": float(np.nanmedian(a)),             # noqa: E731
                   "max": float(np.nanmax(a))}
    return {
        "arm": arm, "targets_root": root, "n_scenes_in_cache": n_scenes_total,
        "n_scenes_sampled": len(scenes),
        "anchors_sampled": len(rows), "per_scene": per_scene,
        "span_source": src,
        "lasers_per_frame": cfg["lasers"],
        "temporal_direction": sorted(direction),
        "n_frames_used": q(n_frames), "n_sweeps": q(n_sweeps),
        "supervision_span_s": q(spans_s),
        "sweep_spacing_s_implied": {"p50": float(np.nanmedian(spans_s)) /
                                    max(float(np.nanmedian(n_frames)) - 1.0, 1.0)},
        "prevalence_in_valid": q(prev), "valid_fraction": q(validf),
        "conflict_fraction_of_touched": q(confl), "n_occupied": q(occn),
        "samples": rows[:12],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-scene", type=int, default=40)
    ap.add_argument("--max-scenes", type=int, default=60,
                    help="cap on scenes sampled per arm (even stride); 0 = all")
    ap.add_argument("--out", default="artifacts/method_vs_data/waymo_canonical_rescore/"
                                    "supervision_provenance.json")
    a = ap.parse_args()
    out = {k: audit(k, a.per_scene, a.max_scenes) for k in ARMS}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    hdr = f"{'arm':10s} {'scenes':>7s} {'frames':>7s} {'sweeps':>7s} {'span_s':>8s} " \
          f"{'spacing':>8s} {'prev':>8s} {'direction':>12s}"
    print(hdr); print("-" * len(hdr))
    for k, v in out.items():
        print(f"{k:10s} {v['n_scenes_sampled']:7d} {v['n_frames_used']['p50']:7.0f} "
              f"{v['n_sweeps']['p50']:7.0f} {v['supervision_span_s']['p50']:8.2f} "
              f"{v['sweep_spacing_s_implied']['p50']:8.3f} "
              f"{v['prevalence_in_valid']['p50']:8.4f} "
              f"{','.join(v['temporal_direction']):>12s}")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
