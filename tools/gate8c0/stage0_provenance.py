#!/usr/bin/env python
"""Gate 8C-0 Stage 0: join every KITTI-360 artifact by anchor and report every mismatch.

One row per labelled anchor of every drive Gate 8B touched, carrying the drive, the
SSCBench index, the native frame it resolves to, the image and velodyne timestamps, the
RGB and LiDAR paths, the pose row, the target file, the target's coordinate frame, the
causal input range the sample builder used and the future-target range.

Nothing is repaired here. Missing files, duplicate keys, non-monotonic sequences and
timestamp discrepancies are counted and listed.

    python tools/gate8c0/stage0_provenance.py
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, CFG, REPO_ROOT, cfg                                     # noqa: E402
from gates.scale_gate.config import write_json                                         # noqa: E402
from gates.gate8 import sources as S                                                   # noqa: E402
from gates.gate8b.sources import G8B_ROOT, K360_TRAIN_DRIVES, K360_VAL_DRIVE           # noqa: E402
from gates.gate8c0 import transforms as TF                                             # noqa: E402
from sscbench_kitti360 import adapter as K3                                      # noqa: E402


def sample_index(source: str) -> dict:
    """anchor stream index -> cached sample file, from the Gate 8B sample root."""
    import glob
    out = {}
    for p in glob.glob(f"{G8B_ROOT}/samples/{source}/*.npz"):
        stem = os.path.basename(p)[:-4]
        seg, t = stem.rsplit("_", 1)
        out[(seg, int(t))] = p
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="stage0_provenance")
    a = ap.parse_args()
    c = cfg()
    t0 = time.time()
    drives = list(K360_TRAIN_DRIVES) + [c["heldout_drive"]]
    partition = {d: ("train" if d in c["train_drives"] else
                     "source_validation" if d == c["val_drive"] else
                     "heldout" if d == c["heldout_drive"] else "train_pool") for d in drives}
    samples = {"k360_train": sample_index("k360_train"), "kitti360": sample_index("kitti360")}
    # stream caches, to resolve the causal input range per anchor
    streams = {}
    for src, segs in (("k360_train", K360_TRAIN_DRIVES), ("kitti360", (c["heldout_drive"],))):
        for seg in segs:
            p = S.stream_path(src, seg)
            if os.path.exists(p):
                with np.load(p) as z:
                    streams[(src, seg)] = {"keys": [str(k) for k in z["keys"]],
                                           "order": z["order"].astype(np.int64),
                                           "keyframe_interval": int(z["keyframe_interval"]),
                                           "proc_hw": tuple(int(x) for x in z["proc_hw"])}
    rows, issues = [], []
    per_drive = {}
    for d in drives:
        src = "kitti360" if d == c["heldout_drive"] else "k360_train"
        g = TF.DriveGeometry(d, c["sscbench_root"], c["kitti360_root"])
        ldir = os.path.join(c["sscbench_root"], "preprocess", "labels", d)
        anchors = sorted(int(f.split("_")[0]) for f in os.listdir(ldir)
                         if f.endswith("_1_1.npy")) if os.path.isdir(ldir) else []
        ts_img, ts_velo = g.timestamps("image"), g.timestamps("velodyne")
        st = streams.get((src, d))
        pos_of_key = {k: i for i, k in enumerate(st["keys"])} if st else {}
        if st and len(pos_of_key) != len(st["keys"]):
            issues.append({"drive": d, "kind": "duplicate_stream_key",
                           "detail": f"{len(st['keys']) - len(pos_of_key)} duplicate keys"})
        if st and np.any(np.diff(st["order"]) <= 0):
            issues.append({"drive": d, "kind": "non_monotonic_stream_order",
                           "detail": int((np.diff(st["order"]) <= 0).sum())})
        if np.any(np.diff(g.pose_frames) <= 0):
            issues.append({"drive": d, "kind": "non_monotonic_pose_frames", "detail": "poses.txt"})
        n_missing = {"image": 0, "velodyne": 0, "target": 0, "trident": 0, "sample": 0,
                     "pose": 0, "cam0_to_world": 0}
        for ai, anchor in enumerate(anchors):
            try:
                native = g.native(anchor)
            except IndexError:
                issues.append({"drive": d, "kind": "sscbench_index_out_of_range",
                               "anchor": anchor}); continue
            key = f"{d[-9:-5]}_{native:010d}" if src == "k360_train" else f"{native:010d}"
            img = os.path.join(c["kitti360_root"], "data_2d_raw", d, "image_00",
                              "data_rect", f"{native:010d}.png")
            velo = g.velodyne_path(native)
            tgt = os.path.join(ldir, f"{anchor:06d}_1_1.npy")
            tri = S.trident_path(src, key)
            idx = pos_of_key.get(key)
            smp = samples[src].get((d, idx)) if idx is not None else None
            for nm, p in (("image", img), ("velodyne", velo), ("target", tgt), ("trident", tri)):
                if not os.path.exists(p):
                    n_missing[nm] += 1
                    if n_missing[nm] <= 3:
                        issues.append({"drive": d, "kind": f"missing_{nm}", "anchor": anchor,
                                       "path": p})
            if native not in g.cam0_to_world:
                n_missing["cam0_to_world"] += 1
            if smp is None and idx is not None:
                n_missing["sample"] += 1
            in_range = out_range = None
            if smp is not None:
                with np.load(smp) as z:
                    inp, fut = z["input_frames"], z["target_frames"]
                    in_range = [int(inp.min()), int(inp.max())]
                    out_range = [int(fut.min()), int(fut.max())]
                    if int(z["t"]) != idx:
                        issues.append({"drive": d, "kind": "sample_t_mismatch",
                                       "anchor": anchor, "detail": [int(z["t"]), idx]})
                    if in_range[1] != idx or out_range[0] != idx + 1:
                        issues.append({"drive": d, "kind": "causal_range_mismatch",
                                       "anchor": anchor,
                                       "detail": {"input": in_range, "future": out_range,
                                                  "t": idx}})
            dt = float(ts_velo[native] - ts_img[native]) if native < min(len(ts_img), len(ts_velo)) else float("nan")
            if abs(dt) > 0.05:
                issues.append({"drive": d, "kind": "image_velodyne_timestamp_gap",
                               "anchor": anchor, "detail": round(dt, 6)})
            rows.append({
                "drive": d, "partition": partition[d], "sscbench_index": anchor,
                "native_frame": native, "stream_index": idx, "frame_key": key,
                "t_image_s": round(float(ts_img[native]), 6) if native < len(ts_img) else None,
                "t_velodyne_s": round(float(ts_velo[native]), 6) if native < len(ts_velo) else None,
                "dt_velo_minus_image_s": round(dt, 6),
                "rgb_path": os.path.relpath(img, c["kitti360_root"]),
                "lidar_path": os.path.relpath(velo, c["kitti360_root"]),
                "pose_row": int(np.searchsorted(g.pose_frames, native)),
                "pose_frame_matches_native": bool(native in g.cam0_to_world),
                "target_file": os.path.relpath(tgt, c["sscbench_root"]),
                "target_frame": f"velodyne_of_native_frame_{native}",
                "target_anchored_at": "t (last frame of the clip)",
                "trident_cache": os.path.exists(tri),
                "causal_input_range": in_range, "future_target_range": out_range,
                "sample_file": os.path.relpath(smp, G8B_ROOT) if smp else None})
        per_drive[d] = {"partition": partition[d], "n_labelled_anchors": len(anchors),
                        "n_stream_frames": len(st["keys"]) if st else 0,
                        "keyframe_interval": st["keyframe_interval"] if st else None,
                        "proc_hw": list(st["proc_hw"]) if st else None,
                        "n_pose_frames": int(len(g.pose_frames)),
                        "n_samples_cached": sum(1 for r in rows
                                                if r["drive"] == d and r["sample_file"]),
                        "missing": dict(n_missing),
                        "anchor_index_stride": int(np.median(np.diff(anchors))) if len(anchors) > 1 else None,
                        "native_frame_span": [int(g.native(anchors[0])), int(g.native(anchors[-1]))]
                        if anchors else None}
    # duplicate frame-key check ACROSS drives (cache-collision risk)
    from collections import Counter
    keyc = Counter((r["frame_key"], r["drive"]) for r in rows)
    bare = Counter(r["frame_key"] for r in rows)
    dup_across = {k: v for k, v in bare.items() if v > 1}
    if dup_across:
        issues.append({"kind": "frame_key_collision_across_drives",
                       "detail": {k: v for k, v in list(dup_across.items())[:10]},
                       "n": len(dup_across)})
    out = {"config": os.path.relpath(CFG, REPO_ROOT), "partition": partition,
           "per_drive": per_drive, "n_rows": len(rows), "n_issues": len(issues),
           "issues": issues[:200],
           "issue_kinds": dict(Counter(i["kind"] for i in issues)),
           "frame_key_collisions_across_drives": len(dup_across),
           "seconds": time.time() - t0}
    os.makedirs(ART, exist_ok=True)
    write_json(os.path.join(ART, f"{a.out}.json"), out)
    with open(os.path.join(ART, f"{a.out}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"stage 0: {len(rows)} anchor rows across {len(drives)} drives, "
          f"{len(issues)} issues -> {a.out}.{{json,csv}}")
    for d, v in per_drive.items():
        print(f"   {d} [{v['partition']:17s}] anchors {v['n_labelled_anchors']:4d} "
              f"stream {v['n_stream_frames']:4d} samples {v['n_samples_cached']:4d} "
              f"k={v['keyframe_interval']} missing {v['missing']}")
    for k, n in out["issue_kinds"].items():
        print(f"   ISSUE {k}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
