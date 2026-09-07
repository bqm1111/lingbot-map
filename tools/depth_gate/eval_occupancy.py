#!/usr/bin/env python
"""Gate 2 — occupancy with refined depth, through the exact corrected Gate-1 fusion.

Three rows per model, all at the frozen Gate-1 settings (5-frame clips, stride 5,
pixel_stride 1, confidence 1.5, unchanged grid / voxeliser / evaluation mask):

    refined + GT poses                     diagnostic
    refined + LingBot poses + constant s   the actual deployable pipeline
    refined + LingBot poses + oracle s     diagnostic

Each is compared against its unrefined counterpart with a clip-level bootstrap on dIoU.
"""
from __future__ import annotations

import argparse, csv, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, collect_provenance, load_config, write_json
from gates.scale_gate.kitti import read_manifest
from gates.scale_gate.scale import bootstrap_ci
from gates.depth_gate.models import build_inputs, build_model, refine

from prompted_lingbot.occ_datasets import SemanticKittiOccSpec, apply_transform
from prompted_lingbot.occupancy import (
    SEMANTICKITTI_GRID as G, binary_occupancy_scores, occupancy_from_points,
)
sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "geometry_gate"))
from factorize import grid_fractions, relative_metric                      # noqa: E402

ROWS = {                       # name -> (pose source, scale policy)
    "gtpose":        ("gt",   "oracle"),
    "predpose_const": ("pred", "const"),
    "predpose_oracle": ("pred", "oracle"),
}


def fuse(scfg, L, D, depth_m: np.ndarray, mask: np.ndarray, pose_src: str,
         s_pose: float) -> np.ndarray:
    """Fuse already-metric per-frame depth into the last camera, Gate-1 style."""
    K = L["pred_K"].astype(np.float64)
    pose = (L["gt_pose_c2w"] if pose_src == "gt" else L["pred_pose_c2w"]).astype(np.float64)
    if pose.shape[-2] == 3:
        p4 = np.tile(np.eye(4), (len(pose), 1, 1)); p4[:, :3, :4] = pose; pose = p4
    ps = 1.0 if pose_src == "gt" else s_pose
    T, H, W = depth_m.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    chunks = []
    for f in range(T):
        m = mask[f]
        if not m.any():
            continue
        d = depth_m[f][m]
        p = np.stack([(u[m] - K[f][0, 2]) * d / K[f][0, 0],
                      (v[m] - K[f][1, 2]) * d / K[f][1, 1], d], axis=-1)
        if f != T - 1:
            p = apply_transform(relative_metric(pose, f, T - 1, ps), p)
        chunks.append(p)
    return np.concatenate(chunks, 0) if chunks else np.zeros((0, 3))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/depth_gate/refine.yaml")
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    scfg = load_config(cfg.data.scale_gate_config)
    scfg["voxel"]["pixel_stride"] = 1                      # frozen Gate-1 setting
    dev = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    root = os.path.join(REPO_ROOT, scfg.dataset.root)
    cache = os.path.join(REPO_ROOT, scfg.cache.root)
    rgb_dir = os.path.join(REPO_ROOT, cfg.data.rgb_cache)
    sg = os.path.join(REPO_ROOT, scfg.experiment.output_dir)
    run_dir = os.path.join(REPO_ROOT, cfg.experiment.output_dir, "runs", a.run)

    ck_path = os.path.join(run_dir, "best.pt")
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model = build_model(ck["arch"], ck["in_ch"], ck["base_channels"], ck["max_log_residual"])
        model.load_state_dict(ck["state_dict"]); model.to(dev).eval()
        stats, use_rgb = ck["stats"], ck["use_rgb"]
    else:
        stats = json.load(open(os.path.join(run_dir, "train.json")))["stats"]
        use_rgb = False
        model = build_model("identity", 5).to(dev).eval()

    s_const = float(cfg.scale.constant)
    tgt = {r["clip_id"]: r for r in csv.DictReader(open(os.path.join(sg, "scale_targets_val.csv")))}
    recs = read_manifest(os.path.join(sg, "manifests", "val.jsonl"))
    specs, rows = {}, []

    for i, rec in enumerate(recs):
        cid, seq = rec["clip_id"], rec["sequence"]
        lp, dp = (os.path.join(cache, "lingbot", f"{cid}.npz"),
                  os.path.join(cache, "lidar_depth", f"{cid}.npz"))
        rp = os.path.join(rgb_dir, f"{cid}.npz")
        if not all(os.path.exists(x) for x in (lp, dp, rp)) or cid not in tgt:
            continue
        try:
            s_star = float(tgt[cid]["s_joint"])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(s_star) or s_star <= 0:
            continue
        if seq not in specs:
            specs[seq] = SemanticKittiOccSpec.build(root, seq)
        spec = specs[seq]
        target, valid_mask = spec.target(int(rec["frame_ids"][-1]))
        if target is None:
            continue
        L = np.load(lp, allow_pickle=False)
        rgb = np.load(rp, allow_pickle=False)["rgb"].astype(np.float32) / 255.0
        dep = L["pred_depth"].astype(np.float32)
        conf = L["pred_depth_conf"].astype(np.float32)

        for row, (pose_src, pol) in ROWS.items():
            s_use = s_const if pol == "const" else s_star
            base = dep * s_use
            b = {"rgb": torch.from_numpy(rgb).to(dev),
                 "base_depth": torch.from_numpy(base).unsqueeze(1).to(dev),
                 "lingbot_confidence": torch.from_numpy(conf).unsqueeze(1).to(dev),
                 "valid_lingbot_mask": torch.from_numpy(
                     np.isfinite(base) & (base > 0)).unsqueeze(1).to(dev)}
            with torch.no_grad():
                r = model(build_inputs(b, stats, use_rgb))
            for label, dm in (("base", base),
                              ("refined", (b["base_depth"] * torch.exp(r))[:, 0].cpu().numpy())):
                m = (conf >= scfg.lingbot.confidence_threshold)
                m &= np.isfinite(dm) & (dm > scfg.voxel.min_depth_m) & (dm < scfg.voxel.max_depth_m)
                pts = fuse(scfg, L, None, dm.astype(np.float64), m, pose_src, s_use)
                pg = apply_transform(spec.cam_to_velo, pts) if len(pts) else pts
                fin, fout = grid_fractions(pg)
                sc = binary_occupancy_scores(
                    occupancy_from_points(pg, G, scfg.voxel.min_points_per_voxel),
                    target, G, valid=valid_mask)
                rows.append({"clip_id": cid, "row": row, "depth": label,
                             "pose": pose_src, "scale_policy": pol, "scale": s_use,
                             "iou": sc["iou"], "precision": sc["precision"],
                             "recall": sc["recall"],
                             "n_pred_occupied": int(sc["n_pred_occupied"]),
                             "n_points": int(pts.shape[0]),
                             "in_grid_fraction": fin, "out_of_grid_fraction": fout})
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(recs)} clips", flush=True)

    out = os.path.join(REPO_ROOT, cfg.experiment.output_dir, "occupancy_eval")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"per_clip_{a.run}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    summary, mean = {}, lambda rs, k: float(np.mean([x[k] for x in rs]))
    print(f"\n{a.run}\n{'row':18s} {'depth':8s} {'IoU':>8} {'P':>7} {'R':>7} "
          f"{'occupied':>9} {'points':>9} {'in-grid':>8}")
    for row in ROWS:
        rs = {lab: [x for x in rows if x["row"] == row and x["depth"] == lab]
              for lab in ("base", "refined")}
        ids = sorted({x["clip_id"] for x in rs["base"]})
        ib = {x["clip_id"]: x["iou"] for x in rs["base"]}
        ir = {x["clip_id"]: x["iou"] for x in rs["refined"]}
        d = [ir[c] - ib[c] for c in ids if c in ir]
        ci = bootstrap_ci(d, cfg.eval.bootstrap_n, cfg.eval.bootstrap_seed)
        summary[row] = {lab: {k: mean(rs[lab], k) for k in
                              ("iou", "precision", "recall", "n_pred_occupied",
                               "n_points", "in_grid_fraction", "out_of_grid_fraction")}
                        for lab in ("base", "refined")}
        summary[row]["delta_iou"] = ci
        summary[row]["improved_clip_fraction"] = float(np.mean([x > 0 for x in d]))
        for lab in ("base", "refined"):
            s = summary[row][lab]
            print(f"{row:18s} {lab:8s} {s['iou']:8.4f} {s['precision']:7.3f} {s['recall']:7.3f} "
                  f"{s['n_pred_occupied']:9.0f} {s['n_points']:9.0f} {s['in_grid_fraction']:8.3f}")
        print(f"{'':18s} {'dIoU':8s} {ci['mean']:+8.4f} [{ci['lo']:+.4f},{ci['hi']:+.4f}]"
              f"{'*' if ci['excludes_zero'] else ''}  improved "
              f"{summary[row]['improved_clip_fraction']:.1%}")
    write_json(os.path.join(out, f"summary_{a.run}.json"),
               {"provenance": collect_provenance(scfg, "depth_gate.eval_occupancy"),
                "run": a.run, "pixel_stride": 1, "constant_scale": s_const,
                "n_clips": len({x['clip_id'] for x in rows}), "rows": summary})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
