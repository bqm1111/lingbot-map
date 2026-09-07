#!/usr/bin/env python
"""Gate 4 diagnostic — the true metric scale of each clip, measured from LiDAR.

This exists to separate two very different explanations of a low transfer score:

    (a) the frozen KITTI-fitted scale does not transfer to nuScenes  -- a real result;
    (b) the coordinate chain is wrong                                -- a bug.

It projects the official LIDAR_TOP keyframe sweep into the anchor camera and computes the
Gate-0 robust log-ratio scale against LingBot's canonical depth. **This is a diagnostic
only.** Nothing it produces is fed back into the frozen stack, and no occupancy label,
mask or Occ3D array is opened.

    python tools/occ3d_zeroshot/lidar_scale_diagnostic.py --limit 120
"""
from __future__ import annotations

import argparse, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, load_config, write_json
from gates.scale_gate.kitti import Preprocess
from gates.scale_gate.scale import weighted_median
from occ3d_zeroshot.nuscenes_adapter import load_annotations, rt_to_T, scene_frames
from occ3d_zeroshot.pipeline import clip_scale, load_depth_head

INDEX = "nuscenes_lidar_index.json"


def build_lidar_index(ns_root: str, out_path: str) -> dict:
    """sample_token -> {filename, T_lidar_to_ego, T_ego_to_world} for LIDAR_TOP keyframes."""
    if os.path.exists(out_path):
        return json.load(open(out_path))
    t0 = time.time()
    meta = os.path.join(ns_root, "v1.0-trainval")
    sensor = {r["token"]: r for r in json.load(open(os.path.join(meta, "sensor.json")))}
    calib = {r["token"]: r for r in json.load(open(os.path.join(meta,
                                                               "calibrated_sensor.json")))}
    lidar_calib = {t for t, c in calib.items()
                   if sensor[c["sensor_token"]]["channel"] == "LIDAR_TOP"}
    print("  parsing sample_data.json (large) ...", flush=True)
    sd = json.load(open(os.path.join(meta, "sample_data.json")))
    ego = {r["token"]: r for r in json.load(open(os.path.join(meta, "ego_pose.json")))}
    idx = {}
    for r in sd:
        if not r["is_key_frame"] or r["calibrated_sensor_token"] not in lidar_calib:
            continue
        c = calib[r["calibrated_sensor_token"]]
        e = ego[r["ego_pose_token"]]
        idx[r["sample_token"]] = {
            "filename": r["filename"],
            "lidar_to_ego": {"translation": c["translation"], "rotation": c["rotation"]},
            "ego_to_world": {"translation": e["translation"], "rotation": e["rotation"]},
        }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(idx, open(out_path, "w"))
    print(f"  indexed {len(idx)} LiDAR keyframes in {time.time()-t0:.0f}s")
    return idx


def lidar_depth_on_lattice(ns_root: str, entry: dict, frame, pre: Preprocess):
    """Project the LiDAR sweep into the anchor camera's processed lattice (z-buffered)."""
    p = os.path.join(ns_root, entry["filename"])
    if not os.path.exists(p):
        return None, None
    pts = np.fromfile(p, dtype=np.float32).reshape(-1, 5)[:, :3].astype(np.float64)
    T_l2e = rt_to_T(entry["lidar_to_ego"]["translation"], entry["lidar_to_ego"]["rotation"])
    T_e2w = rt_to_T(entry["ego_to_world"]["translation"], entry["ego_to_world"]["rotation"])
    T_cam2w = frame.T_ego_cam_to_world @ frame.T_camera_to_ego_cam
    T = np.linalg.inv(T_cam2w) @ T_e2w @ T_l2e                       # lidar -> camera
    pc = pts @ T[:3, :3].T + T[:3, 3]
    z = pc[:, 2]
    keep = z > 1e-3
    pc, z = pc[keep], z[keep]
    uv = np.stack([pc[:, 0] / z * frame.K[0, 0] + frame.K[0, 2],
                   pc[:, 1] / z * frame.K[1, 1] + frame.K[1, 2]], -1)
    uvp = pre.map_pixels(uv)
    H, W = pre.proc_hw
    ui, vi = np.round(uvp[:, 0]).astype(int), np.round(uvp[:, 1]).astype(int)
    ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & (z > 1.0) & (z < 80.0)
    ui, vi, zz = ui[ok], vi[ok], z[ok]
    depth = np.zeros((H, W), np.float64)
    valid = np.zeros((H, W), bool)
    if zz.size:
        order = np.argsort(-zz)                       # nearest surface wins
        depth[vi[order], ui[order]] = zz[order]
        valid[vi[order], ui[order]] = True
    return depth, valid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/occ3d_zeroshot/frozen_transfer.yaml")
    ap.add_argument("--limit", type=int, default=150)
    a = ap.parse_args()
    cfg = load_config(a.config)
    dev = torch.device(cfg.lingbot.device if torch.cuda.is_available() else "cpu")
    fv = cfg.frozen_values
    ns_root, occ3d, cam = cfg.data.nuscenes_root, cfg.data.occ3d_root, cfg.data.camera
    art = os.path.join(REPO_ROOT, cfg.experiment.output_dir)

    idx = build_lidar_index(ns_root, os.path.join(art, INDEX))
    ann = load_annotations(occ3d)
    head, ck = load_depth_head(os.path.join(REPO_ROOT, cfg.frozen.depth_head), dev)
    pre = Preprocess.build((900, 1600), int(fv.lingbot_image_size), int(fv.lingbot_patch_size))
    print(f"preprocess: {pre.orig_hw} -> {pre.proc_hw} (sx {pre.sx:.4f}, sy {pre.sy:.4f})")

    recs = [json.loads(l) for l in
            open(os.path.join(REPO_ROOT, "manifests", "occ3d_zeroshot", "val.jsonl"))]
    rng = np.random.default_rng(0)
    sel = [recs[i] for i in sorted(rng.choice(len(recs), min(a.limit, len(recs)),
                                              replace=False))]

    fmap_cache, rows = {}, []
    for r in sel:
        cp = os.path.join(cfg.data.cache_root, r["clip_id"] + ".npz")
        tok = r["sample_tokens"][-1]
        if not os.path.exists(cp) or tok not in idx:
            continue
        if r["scene"] not in fmap_cache:
            fmap_cache[r["scene"]] = {f.token: f for f in
                                      scene_frames(ann, r["scene"], cam, ns_root)}
        frame = fmap_cache[r["scene"]][tok]
        d = np.load(cp)
        dep = d["pred_depth"].astype(np.float32)
        conf = d["pred_depth_conf"].astype(np.float32)
        a_clip, s_learned, support, sat = clip_scale(
            head, ck, dep, conf, float(fv.s0), float(fv.confidence_threshold),
            float(fv.min_depth_m), float(fv.max_depth_m), dev)

        gt, valid = lidar_depth_on_lattice(ns_root, idx[tok], frame, pre)
        if gt is None:
            continue
        anchor_dep = dep[-1].astype(np.float64)
        m = valid & (gt > 0) & np.isfinite(anchor_dep) & (anchor_dep > 0) \
            & (conf[-1] >= float(fv.confidence_threshold))
        if m.sum() < 200:
            continue
        lr = np.log(gt[m]) - np.log(anchor_dep[m])
        s_oracle = float(np.exp(weighted_median(lr, conf[-1][m].astype(np.float64))))
        rows.append({"clip_id": r["clip_id"], "scene": r["scene"],
                     "a_clip": a_clip, "s_learned": s_learned, "s0": float(fv.s0),
                     "s_oracle": s_oracle, "n_lidar_px": int(m.sum()),
                     "saturation": sat,
                     "ratio_learned": s_learned / s_oracle,
                     "ratio_const": float(fv.s0) / s_oracle})

    if not rows:
        print("no clips could be diagnosed"); return 1
    A = lambda k: np.array([r[k] for r in rows])
    out = {
        "n_clips": len(rows), "camera": cam,
        "note": ("LiDAR-derived diagnostic only. Never fed back into the frozen stack; "
                 "no Occ3D label, mask or occupancy array was opened."),
        "s_oracle": {"median": float(np.median(A("s_oracle"))),
                     "p05": float(np.quantile(A("s_oracle"), .05)),
                     "p95": float(np.quantile(A("s_oracle"), .95))},
        "s_learned": {"median": float(np.median(A("s_learned"))),
                      "p05": float(np.quantile(A("s_learned"), .05)),
                      "p95": float(np.quantile(A("s_learned"), .95))},
        "s0": float(fv.s0),
        "ratio_learned_over_oracle": {
            "median": float(np.median(A("ratio_learned"))),
            "p05": float(np.quantile(A("ratio_learned"), .05)),
            "p95": float(np.quantile(A("ratio_learned"), .95))},
        "ratio_const_over_oracle": {
            "median": float(np.median(A("ratio_const"))),
            "p05": float(np.quantile(A("ratio_const"), .05)),
            "p95": float(np.quantile(A("ratio_const"), .95))},
        "median_abs_log_error_learned": float(np.median(np.abs(np.log(A("ratio_learned"))))),
        "median_abs_log_error_const": float(np.median(np.abs(np.log(A("ratio_const"))))),
        "median_rel_error_learned": float(np.median(np.abs(A("ratio_learned") - 1))),
        "median_rel_error_const": float(np.median(np.abs(A("ratio_const") - 1))),
        "saturation_fraction_median": float(np.median(A("saturation"))),
        "a_clip": {"median": float(np.median(A("a_clip"))),
                   "p05": float(np.quantile(A("a_clip"), .05)),
                   "p95": float(np.quantile(A("a_clip"), .95))},
        "per_clip": rows,
    }
    write_json(os.path.join(art, "lidar_scale_diagnostic.json"), out)
    print(f"\n{len(rows)} clips diagnosed")
    print(f"  s_oracle (LiDAR)   median {out['s_oracle']['median']:7.3f} "
          f"[{out['s_oracle']['p05']:.2f}, {out['s_oracle']['p95']:.2f}]")
    print(f"  s_learned (frozen) median {out['s_learned']['median']:7.3f} "
          f"[{out['s_learned']['p05']:.2f}, {out['s_learned']['p95']:.2f}]")
    print(f"  s0 (KITTI constant)       {out['s0']:7.3f}")
    print(f"  learned/oracle     median {out['ratio_learned_over_oracle']['median']:7.3f} "
          f"[{out['ratio_learned_over_oracle']['p05']:.2f}, "
          f"{out['ratio_learned_over_oracle']['p95']:.2f}]")
    print(f"  const/oracle       median {out['ratio_const_over_oracle']['median']:7.3f}")
    print(f"  median |log err|: learned {out['median_abs_log_error_learned']:.4f}  "
          f"const {out['median_abs_log_error_const']:.4f}")
    print(f"  residual saturation fraction (median): {out['saturation_fraction_median']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
