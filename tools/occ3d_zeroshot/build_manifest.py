#!/usr/bin/env python
"""Gate 4 step 1 — build the frozen Occ3D-nuScenes val clip manifest. Reads no label.

Five official 2 Hz keyframes, one camera (CAM_FRONT), one scene, non-overlapping, anchored
at the LAST frame -- the anchor convention the Gate 0-3.1 fusion is frozen to. Ground-truth
occupancy is only checked for *file presence*; it is never opened.

Also measures the SemanticKITTI five-frame temporal span from that dataset's own
timestamps and reports whether the nuScenes clips match it, so the two protocols carry
comparable temporal evidence rather than merely equal frame counts.

    python tools/occ3d_zeroshot/build_manifest.py
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, load_config, write_json
from gates.scale_gate.kitti import read_manifest
from occ3d_zeroshot.nuscenes_adapter import build_clips, load_annotations, scene_frames, val_scenes


def kitti_span_s(scfg) -> dict:
    """Actual temporal span of the frozen SemanticKITTI five-frame protocol."""
    root = os.path.join(REPO_ROOT, scfg.dataset.root)
    recs = read_manifest(os.path.join(REPO_ROOT, scfg.experiment.output_dir,
                                      "manifests", "val.jsonl"))
    times = {}
    spans, gaps = [], []
    for r in recs:
        seq = r["sequence"]
        if seq not in times:
            p = os.path.join(root, "sequences", seq, "times.txt")
            times[seq] = np.loadtxt(p) if os.path.exists(p) else None
        t = times[seq]
        if t is None:
            continue
        ids = list(r["frame_ids"])
        if max(ids) >= len(t):
            continue
        spans.append(float(t[ids[-1]] - t[ids[0]]))
        gaps += [float(t[b] - t[a]) for a, b in zip(ids, ids[1:])]
    return {"n_clips": len(spans), "span_s_mean": float(np.mean(spans)),
            "span_s_median": float(np.median(spans)), "span_s_std": float(np.std(spans)),
            "gap_s_mean": float(np.mean(gaps)), "gap_s_median": float(np.median(gaps)),
            "source": "data/kitti/dataset/sequences/<seq>/times.txt"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/occ3d_zeroshot/frozen_transfer.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    scfg = load_config(cfg.frozen.gate0_config)
    t0 = time.time()

    ns_root = cfg.data.nuscenes_root
    occ3d = cfg.data.occ3d_root
    cam = cfg.data.camera
    ann = load_annotations(occ3d)
    scenes = val_scenes(ann)
    print(f"official val scenes: {len(scenes)}  camera: {cam}")

    all_clips, rejects, per_scene = [], {}, {}
    for s in scenes:
        fr = scene_frames(ann, s, cam, ns_root)
        clips, rej = build_clips(fr, int(cfg.clips.n_frames),
                                 int(cfg.clips.keyframe_stride),
                                 int(cfg.clips.clip_stride), occ3d)
        for k, v in rej.items():
            rejects[k] = rejects.get(k, 0) + v
        per_scene[s] = {"n_keyframes": len(fr), "n_clips": len(clips)}
        all_clips += clips
    print(f"clips: {len(all_clips)} from {sum(1 for v in per_scene.values() if v['n_clips'])} scenes")

    spacings = np.array([g for c in all_clips for g in c.spacings_s()])
    durations = np.array([c.duration_s for c in all_clips])
    kitti = kitti_span_s(scfg)

    out_path = os.path.join(REPO_ROOT, "manifests", "occ3d_zeroshot", "val.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        for c in sorted(all_clips, key=lambda x: (x.scene, x.frames[0].timestamp_ns)):
            fh.write(json.dumps({
                "clip_id": c.clip_id, "scene": c.scene, "camera": c.camera,
                "sample_tokens": [f.token for f in c.frames],
                "timestamps_ns": [f.timestamp_ns for f in c.frames],
                "frame_indices": [f.index_in_scene for f in c.frames],
                "image_paths": [os.path.relpath(f.image_path, ns_root) for f in c.frames],
                "anchor_token": c.anchor.token,
                "anchor_gt_path": c.anchor.gt_path,
                "duration_s": c.duration_s,
                "spacings_s": c.spacings_s(),
            }, sort_keys=True) + "\n")
    mhash = hashlib.sha256(open(out_path, "rb").read()).hexdigest()

    summary = {
        "config_sha256": hashlib.sha256(
            open(os.path.join(REPO_ROOT, a.config), "rb").read()).hexdigest(),
        "camera": cam, "split": cfg.data.split,
        "n_val_scenes_official": len(ann["val_split"]),
        "n_val_scenes_present": len(scenes),
        "n_scenes_with_clips": sum(1 for v in per_scene.values() if v["n_clips"]),
        "n_clips": len(all_clips),
        "clips_per_scene": {"min": int(min(v["n_clips"] for v in per_scene.values())),
                            "median": float(np.median([v["n_clips"] for v in per_scene.values()])),
                            "max": int(max(v["n_clips"] for v in per_scene.values()))},
        "keyframes_per_scene": {"min": int(min(v["n_keyframes"] for v in per_scene.values())),
                                "max": int(max(v["n_keyframes"] for v in per_scene.values()))},
        "rejected": rejects,
        "spacing_s": {"mean": float(spacings.mean()), "median": float(np.median(spacings)),
                      "min": float(spacings.min()), "max": float(spacings.max()),
                      "p05": float(np.quantile(spacings, .05)),
                      "p95": float(np.quantile(spacings, .95))},
        "duration_s": {"mean": float(durations.mean()),
                       "median": float(np.median(durations)),
                       "min": float(durations.min()), "max": float(durations.max())},
        "semantickitti_reference": kitti,
        "temporal_match": {
            "kitti_span_s_median": kitti["span_s_median"],
            "nuscenes_span_s_median": float(np.median(durations)),
            "abs_difference_s": abs(kitti["span_s_median"] - float(np.median(durations))),
            "comparable": bool(abs(kitti["span_s_median"] - float(np.median(durations))) < 0.2)},
        "manifest_path": os.path.relpath(out_path, REPO_ROOT),
        "manifest_sha256": mhash, "per_scene": per_scene,
        "elapsed_s": time.time() - t0,
    }
    write_json(os.path.join(REPO_ROOT, cfg.experiment.output_dir, "manifest_summary.json"),
               summary)

    print(f"\nspacing  s : median {summary['spacing_s']['median']:.3f} "
          f"[{summary['spacing_s']['min']:.3f}, {summary['spacing_s']['max']:.3f}]")
    print(f"duration s : median {summary['duration_s']['median']:.3f} "
          f"[{summary['duration_s']['min']:.3f}, {summary['duration_s']['max']:.3f}]")
    print(f"KITTI five-frame span: median {kitti['span_s_median']:.3f} s "
          f"(gap median {kitti['gap_s_median']:.3f} s, {kitti['n_clips']} clips)")
    print(f"temporal comparability: |diff| "
          f"{summary['temporal_match']['abs_difference_s']:.3f} s -> "
          f"{'COMPARABLE' if summary['temporal_match']['comparable'] else 'NOT COMPARABLE'}")
    print(f"rejected: {rejects}")
    print(f"\nwrote {summary['manifest_path']}  sha256 {mhash[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
