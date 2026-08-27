#!/usr/bin/env python
"""Frozen-depth substitution: does better depth + LingbotMap poses rescue occupancy?

Configurations (fixed in advance; never selected on test results):

  A_lingbot          LingbotMap depth + LingbotMap poses, running depth-scale
                     anchor + focal correction  -- the previous best, a regression check
  B_external_single  external metric depth alone, single frame (history 1 only).
                     The external model provides NO pose, so this is what it can do
                     standalone. It is the primary baseline the hybrid must beat.
  C_hybrid           external depth + LingbotMap poses, translations scaled causally
                     by median(D_ext / D_lingbot)                 -- THE EXPERIMENT
  D_external_gtpose  external depth + DATASET poses               -- ORACLE DIAGNOSTIC
  E_hybrid_<intr>    C with predicted / dataset / focal-corrected intrinsics

Everything reuses the already-validated voxelizer, masks, grids, anchor frames and
metric. The benchmark protocol is not modified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompted_lingbot.anchors import BASELINES
from prompted_lingbot.conventions import Sim3
from prompted_lingbot.external_depth import causal_translation_scale
from prompted_lingbot.occ_datasets import (
    SemanticKittiOccSpec, load_occ3d_target, sparse_depth_from_velodyne,
)
from prompted_lingbot.occ_eval import (
    OccPointConfig, evaluate_anchor_frame, external_points_in_anchor_camera,
    points_in_anchor_camera, pose_errors,
)
from prompted_lingbot.occ_datasets import apply_transform
from prompted_lingbot.occupancy import (
    OCC3D_NUSCENES_GRID, SEMANTICKITTI_GRID, accumulate_scores,
    binary_occupancy_scores, occupancy_from_points,
)
from prompted_lingbot.prompts import Prompt, _frame_rng
from prompted_lingbot.runner import load_cached

# (name, depth source, pose source, intrinsics, group, gate external depth on
#  LingbotMap's confidence map). The external model emits no confidence, so the
# ungated variants let it place a point at every pixel while the LingbotMap path is
# filtered -- an unequal pixel support. Both are evaluated; neither is hidden.
CONFIGS = [
    ("A_lingbot",              "lingbot",  "lingbot", "focal_corrected", "lingbot_depth",       False),
    ("B_external_single",      "external", "none",    "dataset",         "external_standalone", False),
    ("B_external_single_conf", "external", "none",    "dataset",         "external_standalone", True),
    ("C_hybrid",               "external", "lingbot", "dataset",         "hybrid",              False),
    ("C_hybrid_conf",          "external", "lingbot", "dataset",         "hybrid",              True),
    ("D_external_gtpose",      "external", "dataset", "dataset",         "oracle_diagnostic",   False),
    ("D_external_gtpose_conf", "external", "dataset", "dataset",         "oracle_diagnostic",   True),
    ("E_hybrid_predicted_K",   "external", "lingbot", "predicted",       "hybrid_intrinsics",   True),
    ("E_hybrid_focal_corr_K",  "external", "lingbot", "focal_corrected", "hybrid_intrinsics",   True),
]


def lingbot_scale_stream(seq, spec, root, sequence, chunk_start, k, seed, n_samples):
    """Causal running depth-scale anchor driven by real LiDAR prompts (config A)."""
    Ht, Wt = seq.pred_depth.shape[1:]
    anchor = BASELINES["running_depth_scale"]()
    anchor.reset()
    corrections = []
    for t in range(seq.n_frames):
        p = Prompt(frame=t)
        if t % k == 0:
            d, v = sparse_depth_from_velodyne(root, sequence, chunk_start + t, spec.calib,
                                              seq.meta.get("source_image_hw", (370, 1226)),
                                              (Ht, Wt))
            if d is not None and v.any():
                rng = _frame_rng(seed, seq.name, t, "occ_depth")
                rows, cols = np.nonzero(v)
                take = min(n_samples, rows.size)
                pick = rng.choice(rows.size, take, replace=False)
                p.has_depth = True
                p.depth_rows, p.depth_cols = rows[pick], cols[pick]
                p.depth_values = d[p.depth_rows, p.depth_cols].astype(np.float64)
                p.depth_confidence = 1.0
        anchor.update(seq.prediction(t), p)
        corrections.append(anchor.correction)
    return corrections


def external_scale_stream(seq, ext, cfg):
    """Causal running estimate of the LingbotMap translation scale.

    Uses only frames <= t. Fused in log space with a forgetting factor, mirroring
    the validated depth-prompt filter. A future frame can never change an earlier
    value -- asserted by tests/prompted_lingbot/test_depth_substitution.py.
    """
    mu, prec, out = 0.0, 0.0, []
    for t in range(seq.n_frames):
        est = causal_translation_scale(ext[t], seq.pred_depth[t].astype(np.float32),
                                       seq.pred_depth_conf[t].astype(np.float32),
                                       conf_threshold=cfg.conf_threshold,
                                       max_depth=cfg.max_depth)
        if est is not None:
            z, disp, n = est
            sigma = max(disp / np.sqrt(max(n, 1)), 1e-3)
            p_obs = 1.0 / sigma ** 2
            prec *= 0.9
            mu = (mu * prec + z * p_obs) / (prec + p_obs)
            prec += p_obs
        out.append(float(np.exp(mu)) if prec > 0 else 1.0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["semantickitti", "occ3d_nuscenes"], required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--sequence", default="08")
    ap.add_argument("--lingbot-cache", required=True)
    ap.add_argument("--external-cache", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--history-lengths", type=int, nargs="+", default=[1, 5, 20, 100])
    ap.add_argument("--prompt-interval", type=int, default=5)
    ap.add_argument("--eval-stride", type=int, default=5)
    ap.add_argument("--max-anchors-per-chunk", type=int, default=20)
    ap.add_argument("--conf-threshold", type=float, default=1.5)
    ap.add_argument("--max-depth", type=float, default=60.0)
    ap.add_argument("--num-depth-samples", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-sequences", type=int, default=None)
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    is_kitti = args.dataset == "semantickitti"
    grid = SEMANTICKITTI_GRID if is_kitti else OCC3D_NUSCENES_GRID
    spec = SemanticKittiOccSpec.build(args.dataset_root, args.sequence) if is_kitti else None

    lb_man = json.load(open(os.path.join(args.lingbot_cache, "manifest.json")))
    ex_man = json.load(open(os.path.join(args.external_cache, "manifest.json")))
    names = sorted(set(lb_man["sequences"]) & set(ex_man["sequences"]))
    if args.max_sequences:
        names = names[: args.max_sequences]
    todo = [c for c in CONFIGS if not args.configs or c[0] in set(args.configs)]

    rows_path = os.path.join(args.output_dir, "per_frame_rows.json")
    seq_path = os.path.join(args.output_dir, "per_sequence.json")
    all_rows = [] if args.overwrite or not os.path.isfile(rows_path) else json.load(open(rows_path))
    seq_rows = [] if args.overwrite or not os.path.isfile(seq_path) else json.load(open(seq_path))
    done = {(r["config"], r["sequence"]) for r in seq_rows}

    for name in names:
        lb = lb_man["sequences"][name]
        seq = load_cached(os.path.join(args.lingbot_cache, f"{name}.npz"), lb)
        ext = np.load(os.path.join(args.external_cache, f"{name}.npz"))["external_depth"]
        start = int(lb.get("extra", {}).get("source_start", 0))

        if is_kitti:
            cam_to_grid = spec.cam_to_velo
            def target_fn(t, g=start):
                return spec.target(g + t)
            anchors = [t for t in range(seq.n_frames) if (start + t) % args.eval_stride == 0]
        else:
            cam_to_grid = np.asarray(lb["extra"]["sensor_to_ego"], float)
            gt_dirs = lb["extra"]["gt_dirs"]
            def target_fn(t, _d=gt_dirs):
                return (None, None) if t >= len(_d) else load_occ3d_target(
                    _d[t], grid, apply_camera_mask=True, apply_lidar_mask=False,
                    single_camera=True)
            anchors = list(range(seq.n_frames))
        if args.max_anchors_per_chunk:
            step = max(1, len(anchors) // args.max_anchors_per_chunk)
            anchors = anchors[::step][: args.max_anchors_per_chunk]

        # causal scale streams, computed once per sequence
        t0 = time.perf_counter()
        s_ext = external_scale_stream(seq, ext, argparse.Namespace(
            conf_threshold=args.conf_threshold, max_depth=args.max_depth))
        ext_scale_ms = 1000.0 * (time.perf_counter() - t0) / max(1, seq.n_frames)
        lb_corr = (lingbot_scale_stream(seq, spec, args.dataset_root, args.sequence, start,
                                        args.prompt_interval, args.seed, args.num_depth_samples)
                   if is_kitti else [Sim3.identity()] * seq.n_frames)

        for cname, depth_src, pose_src, intr, group, gate in todo:
            if (cname, name) in done:
                continue
            if cname == "A_lingbot" and not is_kitti:
                continue                       # no LiDAR depth prompts on nuScenes
            point_cfg = OccPointConfig(conf_threshold=args.conf_threshold,
                                       max_depth=args.max_depth, intrinsics=intr)
            hists = [1] if cname.startswith("B_external_single") else args.history_lengths
            poses = None
            if pose_src == "dataset":
                poses = seq.gt_pose_c2w
            rows = []
            for t in anchors:
                target, valid = target_fn(t)
                if target is None:
                    continue
                for h in hists:
                    t1 = time.perf_counter()
                    if depth_src == "lingbot":
                        r = evaluate_anchor_frame(seq, t, h, lb_corr[t], grid, cam_to_grid,
                                                  target, valid, point_cfg, frame_lo=0)
                        n_pts, sc = r.n_points, r.scores
                    else:
                        ts = 1.0 if pose_src == "dataset" else s_ext[t]
                        pts = external_points_in_anchor_camera(
                            seq, ext, t, h, ts, point_cfg, frame_lo=0, poses_c2w=poses,
                            gate_on_lingbot_conf=gate)
                        vol = occupancy_from_points(apply_transform(cam_to_grid, pts), grid,
                                                    min_points_per_voxel=point_cfg.min_points_per_voxel)
                        sc = binary_occupancy_scores(vol, target, grid, valid=valid)
                        n_pts = int(pts.shape[0])
                    rows.append({"sequence": name, "config": cname, "group": group,
                                 "frame": start + t, "history": h, "n_points": n_pts,
                                 "eval_ms": 1000.0 * (time.perf_counter() - t1), **sc})
            if not rows:
                continue
            corr = lb_corr if depth_src == "lingbot" else [
                Sim3(s, np.eye(3), np.zeros(3)) for s in s_ext]
            pe = pose_errors(seq, corr)
            all_rows.extend(rows)
            seq_rows.append({"config": cname, "group": group, "sequence": name,
                             "depth_source": depth_src, "pose_source": pose_src,
                             "intrinsics": intr, "conf_gated": bool(gate),
                             "n_anchor_frames": len(anchors),
                             "external_scale_ms_per_frame": ext_scale_ms,
                             "final_translation_scale": float(s_ext[-1]),
                             **pe,
                             **{f"h{h}_{k}": v for h in hists
                                for k, v in accumulate_scores(
                                    [r for r in rows if r["history"] == h]).items()}})
            json.dump(all_rows, open(rows_path, "w"))
            json.dump(seq_rows, open(seq_path, "w"), indent=1)
            print(f"{cname:24s} {name:28s} " + "  ".join(
                f"h{h}:{accumulate_scores([r for r in rows if r['history']==h])['iou']:.4f}"
                for h in hists), flush=True)

    print(f"\nwrote {rows_path}\n      {seq_path}")


if __name__ == "__main__":
    main()
