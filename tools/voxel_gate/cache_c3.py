#!/usr/bin/env python
"""Gate 3 step 1 — reproduce frozen C3 and cache its voxel-space inputs and targets.

    python tools/voxel_gate/cache_c3.py --split val
    python tools/voxel_gate/cache_c3.py --split train

Per clip this writes, sparsely:

  inputs (inference-computable from frozen C3 alone)
    c3_flat, c3_count, c3_n_frames, c3_sum_conf, c3_sum_depth

  targets and evaluation masks (LiDAR-derived; NEVER model inputs)
    gt_flat      LiDAR-occupied voxels inside the evaluation mask
    vc_flat      Gate-1 configuration-A five-frame visible-reconstruction occupancy
    valid_bits   the frozen evaluation mask, bit-packed

Nothing here changes the grid, voxel size, voxeliser, calibration, pose convention,
confidence threshold, fusion support or evaluation mask.
"""
from __future__ import annotations

import argparse, glob, hashlib, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, collect_provenance, load_config, write_json
from gates.scale_gate.kitti import read_manifest
from gates.voxel_gate.c3 import (
    as4x4, c3_points, check_pose_scaling, clip_scale, load_head, visible_ceiling_points,
)
from gates.voxel_gate.voxels import flat_index, pack, sparse_voxel_features

from prompted_lingbot.occ_datasets import SemanticKittiOccSpec, apply_transform
from prompted_lingbot.occupancy import (
    SEMANTICKITTI_GRID as G, binary_occupancy_scores, occupancy_from_points, voxelize_points,
)

C3_TARGET_IOU, C3_TOL = 0.0778, 5e-4


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dir_sha256(pattern: str):
    paths = sorted(glob.glob(pattern))
    h = hashlib.sha256()
    for p in paths:
        h.update(os.path.basename(p).encode())
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
    return h.hexdigest(), len(paths)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/voxel_gate/visible_correction.yaml")
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    dcfg = load_config(cfg.data.depth_gate_config)
    scfg = load_config(dcfg.data.scale_gate_config)
    scfg["voxel"]["pixel_stride"] = 1                     # frozen Gate-1/2/3 setting
    dev = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    root = os.path.join(REPO_ROOT, scfg.dataset.root)
    cache = os.path.join(REPO_ROOT, scfg.cache.root)
    rgb_dir = os.path.join(REPO_ROOT, dcfg.data.rgb_cache)
    sg = os.path.join(REPO_ROOT, scfg.experiment.output_dir)
    run_dir = os.path.join(REPO_ROOT, dcfg.experiment.output_dir, "runs", cfg.c3.depth_run)
    out = os.path.join(REPO_ROOT, cfg.experiment.output_dir, "cache_c3")
    os.makedirs(out, exist_ok=True)

    model, ck = load_head(run_dir, dev)
    s0 = float(dcfg.scale.constant)
    assert abs(float(ck["metric_scale"]) - s0) < 1e-9, "checkpoint/config scale mismatch"
    conf_thr = float(scfg.lingbot.confidence_threshold)
    dmin, dmax = float(scfg.voxel.min_depth_m), float(scfg.voxel.max_depth_m)

    recs = read_manifest(os.path.join(sg, "manifests", f"{a.split}.jsonl"))
    if a.limit:
        recs = recs[: a.limit]

    specs, index, checked, t0 = {}, [], 0, time.time()
    for i, rec in enumerate(recs):
        cid, seq = rec["clip_id"], rec["sequence"]
        lp = os.path.join(cache, "lingbot", f"{cid}.npz")
        dp = os.path.join(cache, "lidar_depth", f"{cid}.npz")
        rp = os.path.join(rgb_dir, f"{cid}.npz")
        if not all(os.path.exists(x) for x in (lp, dp, rp)):
            continue
        if seq not in specs:
            specs[seq] = SemanticKittiOccSpec.build(root, seq)
        spec = specs[seq]
        anchor_frame = int(rec["frame_ids"][-1])
        target, valid_mask = spec.target(anchor_frame)
        if target is None:
            continue

        L = np.load(lp, allow_pickle=False)
        D = np.load(dp, allow_pickle=False)
        dep = L["pred_depth"].astype(np.float32)
        conf = L["pred_depth_conf"].astype(np.float32)
        K = L["pred_K"].astype(np.float64)
        pose_pred = as4x4(L["pred_pose_c2w"])
        rgb = (np.load(rp, allow_pickle=False)["rgb"].astype(np.float32) / 255.0
               if ck["use_rgb"] else None)

        # ---- frozen C3: scalar only, r_shape never formed --------------------- #
        a_clip, s_learned, support = clip_scale(model, ck, dep, conf, s0, conf_thr,
                                                dmin, dmax, rgb, dev)
        if checked < 5:
            check_pose_scaling(pose_pred, len(pose_pred) - 1, s_learned)
            checked += 1
        pts, fr, cf, pd, _ = c3_points(dep, conf, K, pose_pred, s_learned,
                                       conf_thr, dmin, dmax)
        pg = apply_transform(spec.cam_to_velo, pts) if len(pts) else pts
        sp = sparse_voxel_features(pg, fr, cf, pd)

        # ---- targets: LiDAR only, never inputs -------------------------------- #
        keep = (target != G.ignore_label) & valid_mask
        gt_occ = (target != G.empty_class) & keep
        vc_pts = visible_ceiling_points(D, L["gt_pose_c2w"])
        vc_g = apply_transform(spec.cam_to_velo, vc_pts) if len(vc_pts) else vc_pts
        vc_idx, _ = voxelize_points(vc_g, G)
        vc_flat = np.unique(flat_index(vc_idx)) if vc_idx.size else np.zeros((0,), np.int64)

        # equivalence check: cached sparse form must score identically to the volumes
        c3_occ = np.zeros(int(np.prod(G.dims)), bool); c3_occ[sp["flat"]] = True
        sc = binary_occupancy_scores(c3_occ.reshape(G.dims), target, G, valid=valid_mask)
        sc_ref = binary_occupancy_scores(
            occupancy_from_points(pg, G, scfg.voxel.min_points_per_voxel),
            target, G, valid=valid_mask)
        assert sc == sc_ref, f"{cid}: sparse feature voxelisation != frozen voxeliser"

        np.savez_compressed(
            os.path.join(out, f"{cid}.npz"),
            c3_flat=sp["flat"].astype(np.int32), c3_count=sp["count"].astype(np.float32),
            c3_n_frames=sp["n_frames"].astype(np.float32),
            c3_sum_conf=sp["sum_conf"].astype(np.float32),
            c3_sum_depth=sp["sum_depth"].astype(np.float32),
            gt_flat=np.flatnonzero(gt_occ.reshape(-1)).astype(np.int32),
            vc_flat=vc_flat.astype(np.int32),
            valid_bits=pack(keep), a_clip=np.float64(a_clip),
            s_learned=np.float64(s_learned), anchor_frame=np.int32(anchor_frame))

        index.append({"clip_id": cid, "sequence": seq, "anchor_frame": anchor_frame,
                      "a_clip": a_clip, "s_learned": s_learned,
                      "n_points": int(len(pts)), "n_support_px": int(support.sum()),
                      "n_c3_voxels": int(len(sp["flat"])),
                      "n_vc_voxels": int(len(vc_flat)),
                      "n_gt_voxels": int(gt_occ.sum()), "n_valid_voxels": int(keep.sum()),
                      **{k: sc[k] for k in ("iou", "precision", "recall", "tp", "fp", "fn")}})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(recs)}  {time.time()-t0:.0f}s", flush=True)

    iou = float(np.mean([r["iou"] for r in index]))
    print(f"\n{a.split}: {len(index)} clips, mean C3 IoU {iou:.4f} "
          f"({time.time()-t0:.0f}s, pose assertions on {checked} clips)")

    prov = {"provenance": collect_provenance(scfg, "voxel_gate.cache_c3"),
            "split": a.split, "n_clips": len(index), "mean_c3_iou": iou,
            "constant_scale_s0": s0, "depth_run": cfg.c3.depth_run,
            "depth_checkpoint_sha256": sha256(os.path.join(run_dir, "best.pt")),
            "lingbot_checkpoint_sha256": sha256(
                os.path.join(REPO_ROOT, scfg.lingbot.checkpoint)),
            "config_sha256": sha256(os.path.join(REPO_ROOT, a.config)),
            "manifest_sha256": sha256(os.path.join(sg, "manifests", f"{a.split}.jsonl")),
            "grid": {"dims": list(G.dims), "voxel_size": G.voxel_size,
                     "origin": list(G.origin)},
            "clips": index}
    for nm, pat in (("lingbot_cache", os.path.join(cache, "lingbot", "*.npz")),
                    ("lidar_depth_cache", os.path.join(cache, "lidar_depth", "*.npz")),
                    ("rgb_cache", os.path.join(rgb_dir, "*.npz"))):
        d, n = dir_sha256(pat)
        prov[f"{nm}_sha256"], prov[f"{nm}_n_files"] = d, n
    write_json(os.path.join(REPO_ROOT, cfg.experiment.output_dir, f"c3_{a.split}.json"), prov)

    if a.split == "val":
        if abs(iou - C3_TARGET_IOU) > C3_TOL:
            print(f"\nC3_REPRODUCTION_FAILED: {iou:.5f} vs {C3_TARGET_IOU} "
                  f"(|diff| {abs(iou-C3_TARGET_IOU):.5f} > {C3_TOL})")
            return 2
        print(f"C3 reproduced: {iou:.5f} vs {C3_TARGET_IOU} "
              f"(|diff| {abs(iou-C3_TARGET_IOU):.6f} <= {C3_TOL})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
