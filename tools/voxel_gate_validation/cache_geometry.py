#!/usr/bin/env python
"""Gate 3.1 step 1 — rebuild C0 and C3 voxel evidence from raw, with clean provenance.

    python tools/voxel_gate_validation/cache_geometry.py --geometry c3 --split val

Inputs are produced by :func:`voxel_gate_validation.geometry.geometry_evidence`, whose
signature contains **no** target, valid, label or oracle argument. Targets are written to
the same file but are never read by the inference path (see ``data.input_view``).

Both baselines are asserted on sequence 08 before anything downstream runs:

    C0  s0 * D_lingbot, translation s0            -> 0.0573
    C3  s_learned * D_lingbot, translation same   -> 0.0778
"""
from __future__ import annotations

import argparse, glob, hashlib, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, collect_provenance, load_config, write_json
from gates.scale_gate.kitti import read_manifest
from gates.voxel_gate.c3 import as4x4, check_pose_scaling, load_head, visible_ceiling_points
from gates.voxel_gate.voxels import flat_index, pack
from gates.voxel_gate_validation.geometry import clip_scales, geometry_evidence

from prompted_lingbot.occ_datasets import SemanticKittiOccSpec, apply_transform
from prompted_lingbot.occupancy import (
    SEMANTICKITTI_GRID as G, binary_occupancy_scores, occupancy_from_points, voxelize_points,
)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/voxel_gate_validation/clean_infill.yaml")
    ap.add_argument("--geometry", required=True, choices=["c0", "c3"])
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    dcfg = load_config(cfg.data.depth_gate_config)
    scfg = load_config(dcfg.data.scale_gate_config)
    scfg["voxel"]["pixel_stride"] = 1
    dev = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    root = os.path.join(REPO_ROOT, scfg.dataset.root)
    cache = os.path.join(REPO_ROOT, scfg.cache.root)
    rgb_dir = os.path.join(REPO_ROOT, dcfg.data.rgb_cache)
    sg = os.path.join(REPO_ROOT, scfg.experiment.output_dir)
    run_dir = os.path.join(REPO_ROOT, dcfg.experiment.output_dir, "runs", cfg.c3.depth_run)
    out = os.path.join(REPO_ROOT, cfg.experiment.output_dir, f"cache_{a.geometry}")
    os.makedirs(out, exist_ok=True)

    model, ck = load_head(run_dir, dev)
    s0 = float(dcfg.scale.constant)
    assert abs(float(ck["metric_scale"]) - s0) < 1e-9
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
        anchor = int(rec["frame_ids"][-1])
        target, valid_mask = spec.target(anchor)
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

        scales, a_clip, support = clip_scales(model, ck, dep, conf, s0, conf_thr,
                                              dmin, dmax, rgb, dev)
        s_use = scales[a.geometry]
        if checked < 5:
            check_pose_scaling(pose_pred, len(pose_pred) - 1, s_use)
            checked += 1

        # ---- inference evidence: no target-side argument reaches this call ---- #
        sp = geometry_evidence(dep, conf, K, pose_pred, spec.cam_to_velo, s_use,
                               conf_thr, dmin, dmax)

        # ---- targets, written but never read by the inference path ------------ #
        keep = (target != G.ignore_label) & valid_mask
        gt_occ = (target != G.empty_class) & keep
        vc_pts = visible_ceiling_points(D, L["gt_pose_c2w"])
        vc_g = apply_transform(spec.cam_to_velo, vc_pts) if len(vc_pts) else vc_pts
        vc_idx, _ = voxelize_points(vc_g, G)
        vc_flat = np.unique(flat_index(vc_idx)) if vc_idx.size else np.zeros((0,), np.int64)

        occ = np.zeros(int(np.prod(G.dims)), bool); occ[sp["flat"]] = True
        sc = binary_occupancy_scores(occ.reshape(G.dims), target, G, valid=valid_mask)

        np.savez_compressed(
            os.path.join(out, f"{cid}.npz"),
            c3_flat=sp["flat"].astype(np.int32), c3_count=sp["count"].astype(np.float32),
            c3_n_frames=sp["n_frames"].astype(np.float32),
            c3_sum_conf=sp["sum_conf"].astype(np.float32),
            c3_sum_depth=sp["sum_depth"].astype(np.float32),
            gt_flat=np.flatnonzero(gt_occ.reshape(-1)).astype(np.int32),
            vc_flat=vc_flat.astype(np.int32), valid_bits=pack(keep),
            a_clip=np.float64(a_clip), scale=np.float64(s_use),
            anchor_frame=np.int32(anchor))
        index.append({"clip_id": cid, "sequence": seq, "anchor_frame": anchor,
                      "a_clip": a_clip, "scale": s_use, "n_points": int(sp["n_points"]),
                      "n_support_px": int(support.sum()),
                      "n_voxels": int(len(sp["flat"])), "n_vc_voxels": int(len(vc_flat)),
                      "n_gt_voxels": int(gt_occ.sum()), "n_valid_voxels": int(keep.sum()),
                      **{k: sc[k] for k in ("iou", "precision", "recall", "tp", "fp", "fn")}})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(recs)}  {time.time()-t0:.0f}s", flush=True)

    iou = float(np.mean([r["iou"] for r in index]))
    print(f"\n{a.geometry} {a.split}: {len(index)} clips, mean IoU {iou:.5f} "
          f"({time.time()-t0:.0f}s, pose assertions on {checked} clips)")

    write_json(os.path.join(REPO_ROOT, cfg.experiment.output_dir,
                            f"{a.geometry}_{a.split}.json"),
               {"provenance": collect_provenance(scfg, "voxel_gate_validation.cache"),
                "geometry": a.geometry, "split": a.split, "n_clips": len(index),
                "mean_iou": iou, "constant_scale_s0": s0,
                "depth_checkpoint_sha256": sha256(os.path.join(run_dir, "best.pt")),
                "config_sha256": sha256(os.path.join(REPO_ROOT, a.config)),
                "manifest_sha256": sha256(os.path.join(sg, "manifests",
                                                       f"{a.split}.jsonl")),
                "grid": {"dims": list(G.dims), "voxel_size": G.voxel_size,
                         "origin": list(G.origin)},
                "clips": index})

    if a.split == "val":
        want = float(cfg.reference.c0_iou if a.geometry == "c0" else cfg.reference.c3_iou)
        tol = float(cfg.reference.tolerance)
        ok = abs(iou - want) <= tol
        print(f"{a.geometry.upper()} baseline {'reproduced' if ok else 'FAILED'}: "
              f"{iou:.5f} vs {want} (|diff| {abs(iou-want):.6f}, tol {tol})")
        if not ok:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
