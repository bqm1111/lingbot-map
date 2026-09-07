#!/usr/bin/env python
"""Evaluate a learned scale predictor end to end on held-out sequence 08.

    python tools/scale_gate/eval_scale.py --config <cfg> --checkpoint <best.pt>

Predicts ``s_hat`` without touching any target, applies it to depth and camera
translations, re-fuses, re-voxelises with the unchanged evaluator, and compares against
the deployable global-median baseline and the oracle.
"""
from __future__ import annotations

import argparse, csv, json, math, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, collect_provenance, load_config, write_json
from gates.scale_gate.features import build_matrix
from gates.scale_gate.kitti import read_manifest
from gates.scale_gate.scale import bootstrap_ci

from prompted_lingbot.occ_datasets import SemanticKittiOccSpec, apply_transform
from prompted_lingbot.occupancy import (
    SEMANTICKITTI_GRID as G, binary_occupancy_scores, occupancy_from_points,
)
sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "scale_gate"))
from eval_oracle_scale import fuse_clip                      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--pixel-stride", type=int, default=None,
                    help="override voxel.pixel_stride (Gate 1 uses 1, Gate 0 used 2)")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.pixel_stride is not None:
        cfg["voxel"]["pixel_stride"] = a.pixel_stride
    root = os.path.join(REPO_ROOT, cfg.dataset.root)
    cache = os.path.join(REPO_ROOT, cfg.cache.root)
    out_dir = os.path.join(REPO_ROOT, cfg.experiment.output_dir)
    dev = torch.device(cfg.lingbot.device if torch.cuda.is_available() else "cpu")

    ck = torch.load(os.path.join(REPO_ROOT, a.checkpoint) if not os.path.isabs(a.checkpoint)
                    else a.checkpoint, map_location="cpu", weights_only=False)
    from train_scale import ScaleMLP                          # noqa: PLC0415
    model = ScaleMLP(ck["d_in"], ck["hidden"], 0.0)
    model.load_state_dict(ck["state_dict"]); model.to(dev).eval()
    name = ck["model"]

    recs = read_manifest(os.path.join(out_dir, "manifests", f"{a.split}.jsonl"))
    ids = [r["clip_id"] for r in recs]
    X, kept = build_matrix(cache, ids, ck["blocks"])
    with torch.no_grad():
        xn = torch.from_numpy((X - ck["mu"]) / ck["sd"]).float().to(dev)
        log_s = model(xn).cpu().numpy().astype(np.float64)
    s_hat = {c: float(np.exp(v)) for c, v in zip(kept, log_s)}
    print(f"{name}: predicted {len(s_hat)} scales, "
          f"range {min(s_hat.values()):.2f}-{max(s_hat.values()):.2f}")
    if not all(np.isfinite(list(s_hat.values()))) or min(s_hat.values()) <= 0:
        raise SystemExit("predicted scales are not all positive and finite")

    tgt = {r["clip_id"]: r for r in
           csv.DictReader(open(os.path.join(out_dir, f"scale_targets_{a.split}.csv")))}
    per_path = os.path.join(out_dir, f"oracle_metrics_per_clip_s{cfg.voxel.pixel_stride}.csv")
    if not os.path.exists(per_path):
        raise SystemExit(f"missing {per_path}: run eval_oracle_scale.py at this pixel_stride "
                         "first, so the baseline and the learned model share a protocol")
    per = list(csv.DictReader(open(per_path)))
    base = {r["clip_id"]: float(r["iou"]) for r in per if r["method"] == "global_median_train"}
    orc = {r["clip_id"]: float(r["iou"]) for r in per if r["method"] == "oracle_joint"}

    specs, rows = {}, []
    for rec in recs:
        cid, seq = rec["clip_id"], rec["sequence"]
        lp = os.path.join(cache, "lingbot", f"{cid}.npz")
        if cid not in s_hat or cid not in base or not os.path.exists(lp):
            continue
        if seq not in specs:
            specs[seq] = SemanticKittiOccSpec.build(root, seq)
        spec = specs[seq]
        target, valid = spec.target(int(rec["frame_ids"][-1]))
        if target is None:
            continue
        L = np.load(lp, allow_pickle=False)
        pts = fuse_clip(L, s_hat[cid], cfg)
        pg = apply_transform(spec.cam_to_velo, pts) if pts.shape[0] else pts
        sc = binary_occupancy_scores(occupancy_from_points(pg, G, cfg.voxel.min_points_per_voxel),
                                     target, G, valid=valid)
        st = float(tgt[cid]["s_joint"]) if tgt.get(cid, {}).get("s_joint") else float("nan")
        rows.append({"clip_id": cid, "sequence": seq, "method": name,
                     "scale": s_hat[cid], "s_star_joint": st,
                     "abs_log_scale_err": abs(math.log(s_hat[cid]) - math.log(st))
                     if np.isfinite(st) and st > 0 else float("nan"),
                     "iou": sc["iou"], "precision": sc["precision"], "recall": sc["recall"],
                     "iou_baseline": base[cid], "iou_oracle": orc.get(cid, float("nan"))})

    iou = np.array([r["iou"] for r in rows])
    d_base = [r["iou"] - r["iou_baseline"] for r in rows]
    d_orc = [r["iou_oracle"] - r["iou_baseline"] for r in rows if np.isfinite(r["iou_oracle"])]
    ci = bootstrap_ci(d_base, cfg.bootstrap.n_boot, cfg.bootstrap.seed, cfg.bootstrap.alpha)
    gain, head = float(np.mean(d_base)), float(np.mean(d_orc)) if d_orc else float("nan")
    frac = gain / head if head and abs(head) > 1e-9 else float("nan")

    # Is the result carried by a handful of clips?
    dsort = np.sort(np.array(d_base))[::-1]
    top10 = float(dsort[:10].sum() / np.sum(dsort)) if np.sum(dsort) != 0 else float("nan")

    summary = {
        "provenance": collect_provenance(cfg, "eval_scale"),
        "model": name, "checkpoint": a.checkpoint, "split": a.split, "n_clips": len(rows),
        "n_params": sum(p.numel() for p in model.parameters()),
        "iou_mean": float(iou.mean()),
        "iou_baseline_mean": float(np.mean([r["iou_baseline"] for r in rows])),
        "iou_oracle_mean": float(np.nanmean([r["iou_oracle"] for r in rows])),
        "abs_log_scale_err_median": float(np.nanmedian([r["abs_log_scale_err"] for r in rows])),
        "delta_iou_vs_baseline": ci,
        "recovered_oracle_fraction": frac,
        "oracle_headroom": head,
        "win_rate_vs_baseline": float(np.mean([x > 0 for x in d_base])),
        "top10_share_of_gain": top10,
        "scale_min": min(s_hat.values()), "scale_max": max(s_hat.values()),
        "all_scales_positive_finite": True,
    }
    summary["pixel_stride"] = cfg.voxel.pixel_stride
    tag = f"{name}_s{cfg.voxel.pixel_stride}"
    p = os.path.join(out_dir, f"learned_metrics_{tag}.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    write_json(os.path.join(out_dir, f"learned_summary_{tag}.json"), summary)

    print(f"  IoU {summary['iou_mean']:.4f}  vs baseline {summary['iou_baseline_mean']:.4f}"
          f"  vs oracle {summary['iou_oracle_mean']:.4f}")
    print(f"  dIoU {ci['mean']:+.4f} [{ci['lo']:+.4f},{ci['hi']:+.4f}]"
          f"{'*' if ci['excludes_zero'] else ''}   win rate {summary['win_rate_vs_baseline']:.1%}")
    print(f"  |log scale err| median {summary['abs_log_scale_err_median']:.4f}"
          f"   recovered oracle fraction {frac:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
