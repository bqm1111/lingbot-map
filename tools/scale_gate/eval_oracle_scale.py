#!/usr/bin/env python
"""Oracle scale diagnosis + downstream occupancy for every scale method.

    python tools/scale_gate/eval_oracle_scale.py --config <cfg> --split val

Every method is scored through the **same** existing evaluator, grid, frame selection
and confidence filtering (``prompted_lingbot.occupancy``), so only the scalar differs.
Methods valid at deployment are marked ``deployable``; the rest are marked ``ORACLE``.
"""
from __future__ import annotations

import argparse, csv, os, sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, collect_provenance, load_config, write_json
from gates.scale_gate.kitti import parse_calibration, read_manifest
from gates.scale_gate.scale import bootstrap_ci

from prompted_lingbot.occ_datasets import (
    SemanticKittiOccSpec, apply_transform, camera_points_from_depth, load_semantickitti_target,
)
from prompted_lingbot.occupancy import (
    SEMANTICKITTI_GRID as G, binary_occupancy_scores, occupancy_from_points,
)

METHODS = {                       # name -> (column in the targets CSV, deployable?)
    "raw_canonical":       (None,       True),
    "global_median_train": ("__const__", True),
    "oracle_depth":        ("s_depth",  False),
    "oracle_pose":         ("s_pose",   False),
    "oracle_joint":        ("s_joint",  False),
}


def read_targets(path):
    with open(path) as fh:
        return {r["clip_id"]: r for r in csv.DictReader(fh)}


def fuse_clip(L, s: float, cfg) -> np.ndarray:
    """Fuse a clip's frames into the LAST frame's camera, scaled by ``s``.

    Depth and inter-frame translation receive the same scalar: frames are transported
    in canonical units and the fused cloud is multiplied by ``s`` once, exactly as
    ``occ_eval.points_in_anchor_camera`` does.
    """
    dep = L["pred_depth"].astype(np.float32)
    conf = L["pred_depth_conf"].astype(np.float32)
    K = L["pred_K"].astype(np.float64)
    c2w = L["pred_pose_c2w"].astype(np.float64)
    T = len(dep)
    st = cfg.voxel.pixel_stride
    anchor = np.eye(4); anchor[:3, :4] = c2w[T - 1][:3, :4]
    chunks = []
    for f in range(T):
        d, c = dep[f], conf[f]
        Kf = K[f].copy()
        if st > 1:
            d, c = d[::st, ::st], c[::st, ::st]
            Kf[0, 0] /= st; Kf[1, 1] /= st
            Kf[0, 2] = (Kf[0, 2] - 0.5 * (st - 1)) / st
            Kf[1, 2] = (Kf[1, 2] - 0.5 * (st - 1)) / st
        p = camera_points_from_depth(d, Kf, c, cfg.lingbot.confidence_threshold,
                                     cfg.voxel.min_depth_m / max(s, 1e-9),
                                     cfg.voxel.max_depth_m / max(s, 1e-9))
        if p.shape[0] == 0:
            continue
        if f != T - 1:
            pf = np.eye(4); pf[:3, :4] = c2w[f][:3, :4]
            p = apply_transform(np.linalg.inv(anchor) @ pf, p)
        chunks.append(p)
    if not chunks:
        return np.zeros((0, 3))
    return np.concatenate(chunks, 0) * float(s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    root = os.path.join(REPO_ROOT, cfg.dataset.root)
    cache_root = os.path.join(REPO_ROOT, cfg.cache.root)
    out_dir = os.path.join(REPO_ROOT, cfg.experiment.output_dir)

    tgt = read_targets(os.path.join(out_dir, f"scale_targets_{a.split}.csv"))
    train_tgt = read_targets(os.path.join(out_dir, "scale_targets_train.csv"))
    # Deployment-legal constant: median of the reliable SOURCE-TRAINING clips only.
    tr_logs = [np.log(float(r["s_joint"])) for r in train_tgt.values()
               if r.get("reliable") == "1" and r["s_joint"]]
    s_const = float(np.exp(np.median(tr_logs))) if tr_logs else float("nan")
    print(f"global_median_train = {s_const:.4f}  (from {len(tr_logs)} reliable train clips)")

    recs = read_manifest(os.path.join(out_dir, "manifests", f"{a.split}.jsonl"))
    if a.limit:
        recs = recs[: a.limit]
    specs, rows = {}, []
    for i, rec in enumerate(recs):
        cid, seq = rec["clip_id"], rec["sequence"]
        lp = os.path.join(cache_root, "lingbot", f"{cid}.npz")
        if cid not in tgt or not os.path.exists(lp):
            continue
        t = tgt[cid]
        if seq not in specs:
            specs[seq] = SemanticKittiOccSpec.build(root, seq)
        spec = specs[seq]
        anchor_frame = int(rec["frame_ids"][-1])
        target, valid = spec.target(anchor_frame)
        if target is None:
            continue
        L = np.load(lp, allow_pickle=False)

        for name, (col, deployable) in METHODS.items():
            if name == "raw_canonical":
                s = 1.0
            elif col == "__const__":
                s = s_const
            else:
                try:
                    s = float(t[col])
                except (KeyError, ValueError, TypeError):
                    continue
            if not np.isfinite(s) or s <= 0:
                continue
            pts = fuse_clip(L, s, cfg)
            pts_g = apply_transform(spec.cam_to_velo, pts) if pts.shape[0] else pts
            vol = occupancy_from_points(pts_g, G, cfg.voxel.min_points_per_voxel)
            sc = binary_occupancy_scores(vol, target, G, valid=valid)
            rows.append({
                "clip_id": cid, "sequence": seq, "anchor_frame": anchor_frame,
                "method": name, "deployable": int(deployable), "scale": s,
                "s_star_joint": float(t["s_joint"]) if t.get("s_joint") else float("nan"),
                "abs_log_scale_err": abs(np.log(s) - np.log(float(t["s_joint"])))
                if t.get("s_joint") and float(t["s_joint"]) > 0 else float("nan"),
                "iou": sc["iou"], "precision": sc["precision"], "recall": sc["recall"],
                "n_pred_occupied": int(sc["n_pred_occupied"]),
                "n_gt_occupied": int(sc["n_gt_occupied"]), "n_points": int(pts.shape[0]),
                "reliable_target": int(t.get("reliable") == "1"),
            })
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(recs)} clips", flush=True)

    st = cfg.voxel.pixel_stride
    p_csv = os.path.join(out_dir, f"oracle_metrics_per_clip_s{st}.csv")
    if rows:
        with open(p_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)

    # ---- aggregate: pooled (benchmark-style) and per-clip, plus bootstrap ---- #
    by = {}
    for r in rows:
        by.setdefault(r["method"], []).append(r)
    base = {r["clip_id"]: r["iou"] for r in by.get("raw_canonical", [])}
    summary = {}
    for name, rs in by.items():
        iou = np.array([r["iou"] for r in rs])
        ale = np.array([r["abs_log_scale_err"] for r in rs])
        delta = [r["iou"] - base[r["clip_id"]] for r in rs if r["clip_id"] in base]
        summary[name] = {
            "deployable": bool(rs[0]["deployable"]), "n_clips": len(rs),
            "iou_mean": float(iou.mean()), "iou_median": float(np.median(iou)),
            "precision_mean": float(np.mean([r["precision"] for r in rs])),
            "recall_mean": float(np.mean([r["recall"] for r in rs])),
            "abs_log_scale_err_median": float(np.nanmedian(ale)),
            "n_points_mean": float(np.mean([r["n_points"] for r in rs])),
            "delta_iou_vs_raw": bootstrap_ci(delta, cfg.bootstrap.n_boot,
                                             cfg.bootstrap.seed, cfg.bootstrap.alpha),
        }
    print(f"\n{'method':22s} {'dep':>4} {'IoU':>8} {'P':>7} {'R':>7} {'|log err|':>10} "
          f"{'dIoU vs raw [95% CI]':>30}")
    for name in METHODS:
        if name not in summary:
            continue
        s = summary[name]; d = s["delta_iou_vs_raw"]
        print(f"{name:22s} {'yes' if s['deployable'] else 'ORCL':>4} {s['iou_mean']:8.4f} "
              f"{s['precision_mean']:7.3f} {s['recall_mean']:7.3f} "
              f"{s['abs_log_scale_err_median']:10.4f} "
              f"{d['mean']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}]"
              + ("*" if d.get("excludes_zero") else ""))

    write_json(os.path.join(out_dir, f"oracle_summary_s{st}.json"),
               {"provenance": collect_provenance(cfg, "eval_oracle_scale"),
                "split": a.split, "pixel_stride": st, "global_median_train": s_const,
                "n_train_clips_for_median": len(tr_logs),
                "grid": {"dims": list(G.dims), "voxel_size": G.voxel_size},
                "methods": summary, "per_clip_csv": os.path.relpath(p_csv, REPO_ROOT)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
