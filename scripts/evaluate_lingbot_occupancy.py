#!/usr/bin/env python
"""Geometry-only binary occupancy evaluation for frozen, metrically anchored LingbotMap.

    python scripts/evaluate_lingbot_occupancy.py \
      --dataset semantickitti --dataset-root data/kitti/dataset --sequence 08 \
      --cache-dir outputs/prompted_lingbot/cache_semkitti08 \
      --history-lengths 1 5 20 100 --prompt-intervals 5 30 100 \
      --output-dir outputs/prompted_lingbot/occupancy_geometry/semantickitti

LingbotMap is frozen and is never run here: this reads the cache written by
``scripts/cache_lingbot_predictions.py``.  Prompts contribute a scalar or a
transform, never points.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompted_lingbot.anchors import BASELINES, Prediction
from prompted_lingbot.conventions import Sim3, umeyama_sim3
from prompted_lingbot.occ_datasets import (SemanticKittiOccSpec, load_occ3d_target,
                                          sparse_depth_from_velodyne)
from prompted_lingbot.occ_eval import (
    OccPointConfig, evaluate_anchor_frame, pose_errors,
)
from prompted_lingbot.occupancy import (OCC3D_NUSCENES_GRID, SEMANTICKITTI_EVAL_STRIDE,
                                        SEMANTICKITTI_GRID, accumulate_scores)
from prompted_lingbot.prompts import Prompt, PromptConfig, _frame_rng
from prompted_lingbot.runner import load_cached

# --------------------------------------------------------------------------- #
# metric configurations (Phase 4)
# --------------------------------------------------------------------------- #
def metric_configs(prompt_intervals, depth_prompts_available: bool = True):
    """(name, anchor, prompt-spec) triples. 'group' separates camera-only from
    one-prompt from repeated-prompt methods so they are never ranked together."""
    cfgs = [("A_raw", "raw", dict(depth=None, pose=None), "camera_only")]
    if depth_prompts_available:
        cfgs.append(("B_first_depth_scale", "first_depth_scale",
                     dict(depth="first", pose=None), "one_prompt"))
    for k in prompt_intervals:
        if depth_prompts_available:
            cfgs.append((f"C_running_depth_scale_k{k}", "running_depth_scale",
                         dict(depth=k, pose=None), "repeated_prompts"))
            cfgs.append((f"E_depth_scale_pose_se3_k{k}", "depth_scale_pose_se3",
                         dict(depth=k, pose=k), "repeated_prompts"))
        cfgs.append((f"D_causal_pose_sim3_k{k}", "causal_pose_sim3",
                     dict(depth=None, pose=k), "repeated_prompts"))
    cfgs.append(("Z_oracle_trajectory_scale", "__oracle__", dict(depth=None, pose=None),
                 "diagnostic_non_causal"))
    return cfgs


def build_prompt_stream(seq, spec, root, sequence, chunk_start, prompt_spec, cfg_seed,
                        num_depth_samples, noise, depth_source="velodyne"):
    """Prompts for one chunk.

    On SemanticKITTI the depth prompt is a **real LiDAR return** projected into
    the cached lattice -- that is what a sparse metric depth sensor actually is
    here.  ``depth_source="none"`` disables depth prompts on platforms where no
    metric depth sensor has been wired up (Occ3D-nuScenes today).
    """
    Ht, Wt = seq.pred_depth.shape[1:]
    prompts = []
    dspec, pspec = prompt_spec["depth"], prompt_spec["pose"]
    for t in range(seq.n_frames):
        p = Prompt(frame=t)
        want_depth = depth_source != "none" and (
            (dspec == "first" and t == 0) or (isinstance(dspec, int) and t % dspec == 0))
        want_pose = isinstance(pspec, int) and t % pspec == 0
        if want_depth:
            d, v = sparse_depth_from_velodyne(root, sequence, chunk_start + t, spec.calib,
                                              seq.meta.get("source_image_hw", (370, 1226)),
                                              (Ht, Wt))
            if d is not None and v.any():
                rng = _frame_rng(cfg_seed, seq.name, t, "occ_depth")
                rows, cols = np.nonzero(v)
                take = min(num_depth_samples, rows.size)
                pick = rng.choice(rows.size, take, replace=False)
                rows, cols = rows[pick], cols[pick]
                vals = d[rows, cols].astype(np.float64)
                if noise > 0:
                    vals = vals * np.exp(rng.normal(0.0, noise, vals.size))
                p.has_depth = True
                p.depth_rows, p.depth_cols, p.depth_values = rows, cols, vals
                p.depth_confidence = float(np.exp(-3.0 * noise))
        if want_pose:
            p.has_pose = True
            p.pose_c2w = seq.gt_pose_c2w[t].astype(np.float64).copy()
            p.pose_confidence = 1.0
        prompts.append(p)
    return prompts


def run_chunk(seq, spec, args, point_cfg, prompt_spec, anchor_name, chunk_start,
              grid=SEMANTICKITTI_GRID, cam_to_grid=None, target_fn=None,
              anchor_frames=None, depth_source="velodyne"):
    """Drive one anchor causally over a chunk and evaluate every anchor frame."""
    prompts = build_prompt_stream(seq, spec, args.dataset_root, args.sequence, chunk_start,
                                  prompt_spec, args.seed, args.num_depth_samples,
                                  args.depth_prompt_noise, depth_source=depth_source)
    t0 = time.perf_counter()
    if anchor_name == "__oracle__":
        S = umeyama_sim3(seq.pred_pose_c2w[:, :3, 3], seq.gt_pose_c2w[:, :3, 3])
        corrections = [Sim3(S.s, np.eye(3), np.zeros(3))] * seq.n_frames
        anchor = None
    else:
        anchor = BASELINES[anchor_name]()
        anchor.reset()
        corrections = []
        for t in range(seq.n_frames):
            anchor.update(seq.prediction(t), prompts[t])
            corrections.append(anchor.correction)
    anchor_ms = 1000.0 * (time.perf_counter() - t0) / max(1, seq.n_frames)

    pe = pose_errors(seq, corrections)
    held = 0.0
    if anchor is not None:
        inner = anchor if hasattr(anchor, "rotation_held_fraction") else getattr(anchor, "_pose", None)
        held = float(inner.rotation_held_fraction) if inner is not None else 0.0

    rows = []
    if anchor_frames is None:
        stride = args.eval_stride
        local = [t for t in range(seq.n_frames) if (chunk_start + t) % stride == 0]
    else:
        local = list(anchor_frames)
    if args.max_anchors_per_chunk:
        step = max(1, len(local) // args.max_anchors_per_chunk)
        local = local[::step][: args.max_anchors_per_chunk]
    if cam_to_grid is None:
        cam_to_grid = spec.cam_to_velo
    if target_fn is None:
        target_fn = lambda t, gframe: spec.target(gframe)
    for t in local:
        gframe = chunk_start + t
        target, valid = target_fn(t, gframe)
        if target is None:
            continue
        for hist in args.history_lengths:
            t1 = time.perf_counter()
            r = evaluate_anchor_frame(seq, t, hist, corrections[t], grid,
                                      cam_to_grid, target, valid, point_cfg, frame_lo=0)
            rows.append({
                "sequence": seq.name, "frame": gframe, "history": hist,
                "n_points": r.n_points,
                "voxelize_ms": 1000.0 * (time.perf_counter() - t1),
                **r.scores,
            })
    return rows, {**pe, "anchor_ms_per_frame": anchor_ms, "rotation_held_fraction": held,
                  "n_depth_prompts": int(sum(p.has_depth for p in prompts)),
                  "n_pose_prompts": int(sum(p.has_pose for p in prompts))}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["semantickitti", "occ3d_nuscenes"],
                    default="semantickitti")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--sequence", default="08")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--history-lengths", type=int, nargs="+", default=[1, 5, 20, 100])
    ap.add_argument("--prompt-intervals", type=int, nargs="+", default=[5, 30, 100])
    ap.add_argument("--eval-stride", type=int, default=SEMANTICKITTI_EVAL_STRIDE)
    ap.add_argument("--max-anchors-per-chunk", type=int, default=20)
    ap.add_argument("--conf-threshold", type=float, default=1.5)
    ap.add_argument("--max-depth", type=float, default=60.0)
    ap.add_argument("--min-points-per-voxel", type=int, default=1)
    ap.add_argument("--num-depth-samples", type=int, default=512)
    ap.add_argument("--depth-prompt-noise", type=float, default=0.0)
    ap.add_argument("--intrinsics", nargs="+", default=["predicted"],
                    choices=["predicted", "dataset", "focal_corrected"])
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-sequences", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    is_kitti = args.dataset == "semantickitti"
    spec = SemanticKittiOccSpec.build(args.dataset_root, args.sequence) if is_kitti else None
    grid = SEMANTICKITTI_GRID if is_kitti else OCC3D_NUSCENES_GRID
    manifest = json.load(open(os.path.join(args.cache_dir, "manifest.json")))
    chunks = sorted(f for f in os.listdir(args.cache_dir) if f.endswith(".npz"))
    if args.max_sequences:
        chunks = chunks[: args.max_sequences]

    todo = metric_configs(args.prompt_intervals, depth_prompts_available=is_kitti)
    if args.configs:
        todo = [c for c in todo if c[0] in set(args.configs)]

    out_rows_path = os.path.join(args.output_dir, "per_frame_rows.json")
    out_seq_path = os.path.join(args.output_dir, "per_sequence.json")
    all_rows, seq_rows = [], []
    if not args.overwrite and os.path.isfile(out_rows_path):
        all_rows = json.load(open(out_rows_path))
        seq_rows = json.load(open(out_seq_path)) if os.path.isfile(out_seq_path) else []
    done = {(r["config"], r["intrinsics"], r["sequence"]) for r in seq_rows}

    for cname, aname, pspec, group in todo:
        for intr in args.intrinsics:
            point_cfg = OccPointConfig(conf_threshold=args.conf_threshold,
                                       max_depth=args.max_depth,
                                       min_points_per_voxel=args.min_points_per_voxel,
                                       intrinsics=intr)
            for f in chunks:
                name = f[:-4]
                if (cname, intr, name) in done:
                    continue
                meta_entry = manifest["sequences"].get(name, {})
                seq = load_cached(os.path.join(args.cache_dir, f), meta_entry)
                start = int(meta_entry.get("extra", {}).get("source_start", 0))
                if is_kitti:
                    rows, meta = run_chunk(seq, spec, args, point_cfg, pspec, aname, start,
                                           grid=grid)
                else:
                    s2e = np.asarray(meta_entry["extra"]["sensor_to_ego"], float)
                    gt_dirs = meta_entry["extra"]["gt_dirs"]

                    def target_fn(t, _g, _dirs=gt_dirs):
                        if t >= len(_dirs):
                            return None, None
                        return load_occ3d_target(_dirs[t], grid,
                                                 apply_camera_mask=True,
                                                 apply_lidar_mask=False,
                                                 single_camera=True)

                    rows, meta = run_chunk(seq, spec, args, point_cfg, pspec, aname, 0,
                                           grid=grid, cam_to_grid=s2e, target_fn=target_fn,
                                           anchor_frames=range(seq.n_frames),
                                           depth_source="none")
                for r in rows:
                    r.update(config=cname, anchor=aname, group=group, intrinsics=intr)
                all_rows.extend(rows)
                seq_rows.append({"config": cname, "anchor": aname, "group": group,
                                 "intrinsics": intr, "sequence": name,
                                 "chunk_start": start, "n_anchor_frames": len(rows) // max(1, len(args.history_lengths)),
                                 **meta,
                                 **{f"h{h}_{k}": v for h in args.history_lengths
                                    for k, v in accumulate_scores(
                                        [r for r in rows if r["history"] == h]).items()}})
                json.dump(all_rows, open(out_rows_path, "w"))
                json.dump(seq_rows, open(out_seq_path, "w"), indent=1)
                print(f"{cname:34s} {intr:16s} {name:20s} "
                      + "  ".join(f"h{h}:IoU {accumulate_scores([r for r in rows if r['history']==h])['iou']:.4f}"
                                  for h in args.history_lengths), flush=True)

    print(f"\nwrote {out_rows_path}\n      {out_seq_path}")


if __name__ == "__main__":
    main()

