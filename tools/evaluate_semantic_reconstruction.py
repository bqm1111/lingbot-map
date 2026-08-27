#!/usr/bin/env python
"""Evaluate observed-surface semantic reconstruction against SemanticKITTI.

    python tools/evaluate_semantic_reconstruction.py \
        --config configs/semantic_sidecar/feasibility.yaml \
        --maps output/semantic_sidecar/maps/mlp_consensus --output eval.json

Protocol — ``calibrated-projection`` (no semantic information in the correspondence):
every labelled LiDAR point is projected into its own camera with the dataset's true
``P2``/``Tr``; the prediction is read at the pixel it lands on, via the voxel that
pixel's predicted depth unprojects into.  Because each point is scored in the frame
that observed it, kilometre-scale monocular drift cannot contaminate the semantic
score.  Geometry quality is reported separately, never folded into mIoU.

Two mIoUs are always reported:
  * end-to-end  — a visible GT point in no populated voxel counts as *unknown*, i.e. wrong
  * matched     — restricted to covered points, isolating semantic quality
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

import _bootstrap  # noqa: F401

from semantic.eval_semantickitti import CLASS_NAMES, CLASS_PROMPTS, build_lut, parse_calib, project_to_image
from semantic_sidecar.config import collect_provenance, load_config, set_seed, write_json
from semantic_sidecar.datasets import expand_scenes
from semantic_sidecar.lingbot_features import FeatureCacheReader, camera_center, manifest_path, unproject_depth
from semantic_sidecar.metrics import Confusion, CoverageCounter, geometry_fscore, umeyama_sim3
from semantic_sidecar.semantic_map import SparseSemanticMap
from semantic_sidecar.teacher_features import SemanticProjection, build_teacher, encode_text_queries

logger = logging.getLogger("evaluate")

#: Fixed prompt template set, taken verbatim from ``semantic/eval_semantickitti.py``.
#: Never tuned against results.
PROMPT_SOURCE = "semantic/eval_semantickitti.py::CLASS_PROMPTS"


#: Deterministic palette for the 19 benchmark classes (index 0 = unknown/grey).
CLASS_COLOURS = np.array(
    [
        [128, 128, 128], [245, 150, 100], [245, 230, 100], [150, 60, 30], [180, 30, 80],
        [255, 0, 0], [30, 30, 255], [200, 40, 255], [90, 30, 150], [255, 0, 255],
        [255, 150, 255], [75, 0, 75], [75, 0, 175], [0, 200, 255], [50, 120, 255],
        [0, 175, 0], [0, 60, 135], [80, 240, 150], [150, 240, 255], [0, 0, 255],
    ],
    dtype=np.uint8,
)


def export_class_ply(smap, text: torch.Tensor, path: str, temperature: float, device) -> None:
    """PLY of the map coloured by predicted class, using the fixed palette above."""
    labels, _ = smap.classify(text.to(device), temperature=temperature, device=device)
    colours = CLASS_COLOURS[(labels.cpu().numpy() + 1) % len(CLASS_COLOURS)]
    smap.export_ply(path, colours=colours)


def export_similarity_ply(smap, query: torch.Tensor, path: str, device) -> None:
    """PLY coloured by cosine similarity to one query (percentile-stretched)."""
    sims = smap.similarity(query.reshape(1, -1), device).squeeze(-1).cpu().numpy()
    lo, hi = np.percentile(sims, 2), np.percentile(sims, 98)
    t = np.clip((sims - lo) / max(hi - lo, 1e-6), 0, 1)
    colours = np.stack([255 * t, 60 * np.ones_like(t), 255 * (1 - t)], axis=-1).astype(np.uint8)
    smap.export_ply(path, colours=colours)


def load_labelled_scan(kitti_root: str, seq: str, idx: int, lut: np.ndarray):
    base = os.path.join(kitti_root, "sequences", seq)
    pts = np.fromfile(os.path.join(base, "velodyne", f"{idx:06d}.bin"), dtype=np.float32).reshape(-1, 4)[:, :3]
    raw = np.fromfile(os.path.join(base, "labels", f"{idx:06d}.label"), dtype=np.uint32) & 0xFFFF
    return pts, lut[np.clip(raw, 0, len(lut) - 1)]


def scene_frame_to_source(reader, i: int) -> int:
    """Original dataset frame index for the ``i``-th cached frame of a scene chunk.

    Read from the cached image filename (``000123.png`` -> 123) rather than recomputed
    from ``start``/``stride``, so it stays correct even when unreadable frames were
    dropped during caching.
    """
    stem = os.path.splitext(os.path.basename(reader.image_path(i)))[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    if not digits:
        raise ValueError(f"cannot derive a frame index from {stem!r}")
    return int(digits)


def evaluate_scene(
    smap: SparseSemanticMap,
    reader: FeatureCacheReader,
    spec,
    text: torch.Tensor,
    cfg,
    device: torch.device,
    temperature: float,
    conf_e2e: Confusion,
    conf_matched: Confusion,
    coverage: CoverageCounter,
    depth_ratios: List[torch.Tensor],
) -> None:
    """Score one cached scene chunk against its SemanticKITTI labels."""
    lut = build_lut()
    p2, tr = parse_calib(os.path.join(cfg.evaluation.kitti_root, "sequences", cfg.evaluation.sequence, "calib.txt"))
    labels, _ = smap.classify(text.to(device), temperature=temperature, device=device)
    voxel_class = (labels + 1).to(device)  # class ids are 1..19; 0 is unknown
    keys = smap.keys.to(device)
    H, W = reader.image_hw

    for i in tqdm(range(0, reader.num_frames, cfg.evaluation.frame_stride), desc=f"eval[{spec.name}]", leave=False):
        src = scene_frame_to_source(reader, i)
        try:
            pts, gt_raw = load_labelled_scan(cfg.evaluation.kitti_root, cfg.evaluation.sequence, src, lut)
        except FileNotFoundError:
            continue
        u, v, lidar_z, valid = project_to_image(
            pts, p2, tr, cfg.evaluation.orig_width, cfg.evaluation.orig_height
        )
        if valid.sum() == 0:
            continue

        g = reader.geometry(i)
        depth = g["depth"].to(device).float()
        uu = torch.from_numpy(u[valid] * (W / cfg.evaluation.orig_width)).to(device).long().clamp(0, W - 1)
        vv = torch.from_numpy(v[valid] * (H / cfg.evaluation.orig_height)).to(device).long().clamp(0, H - 1)
        gt = torch.from_numpy(gt_raw[valid].astype(np.int64)).to(device)
        lz = torch.from_numpy(lidar_z[valid].astype(np.float32)).to(device)

        flat = vv * W + uu
        world = unproject_depth(depth, g["intrinsic"].to(device).float(), g["extrinsic"].to(device).float())
        world = world.reshape(-1, 3)
        smap_keys_local = _lookup(world[flat], smap.voxel_size, keys)
        pred_depth = depth.reshape(-1)[flat]

        covered = smap_keys_local >= 0
        if cfg.evaluation.conf_threshold:
            conf = g["depth_conf"].to(device).float().reshape(-1)
            covered &= conf[flat] >= cfg.evaluation.conf_threshold

        pred = torch.where(covered, voxel_class[smap_keys_local.clamp_min(0)], torch.zeros_like(covered, dtype=torch.long))
        labelled = gt > 0
        conf_e2e.update(gt[labelled], pred[labelled])
        sel = labelled & covered
        conf_matched.update(gt[sel], pred[sel])
        coverage.update(int(labelled.sum()), int(sel.sum()))

        good = sel & (pred_depth > 1e-6)
        if bool(good.any()):
            depth_ratios.append((lz[good] / pred_depth[good]).cpu())


def _lookup(points: torch.Tensor, voxel_size: float, keys: torch.Tensor) -> torch.Tensor:
    from semantic.feature_field import quantize

    if keys.numel() == 0:
        return torch.full((points.shape[0],), -1, dtype=torch.long, device=points.device)
    pk, valid = quantize(points, voxel_size)
    idx = torch.searchsorted(keys, pk).clamp(max=keys.numel() - 1)
    hit = valid & (keys[idx] == pk)
    return torch.where(hit, idx, torch.full_like(idx, -1))


def geometry_report(cfg, specs, cache_root: str, device: torch.device, max_points: int = 60_000) -> Dict[str, object]:
    """Model-independent geometry check: trajectory Sim(3) fit + surface F-score.

    Identical for every model by construction — the sidecar cannot change geometry —
    so it is computed once and reported once.
    """
    gt_poses = np.loadtxt(
        os.path.join(cfg.evaluation.kitti_root, "sequences", cfg.evaluation.sequence, "poses.txt")
    ).reshape(-1, 3, 4)
    out: List[Dict[str, object]] = []
    for spec in specs:
        reader = FeatureCacheReader(cache_root, spec.name)
        pred_c, gt_c = [], []
        for i in range(reader.num_frames):
            src = scene_frame_to_source(reader, i)
            if src >= len(gt_poses):
                continue
            pred_c.append(camera_center(reader.geometry(i)["extrinsic"].float()).numpy())
            gt_c.append(gt_poses[src][:3, 3])
        if len(pred_c) < 4:
            continue
        R, t, s, rmse = umeyama_sim3(np.stack(pred_c), np.stack(gt_c))

        # Surface F-score after the geometry-only alignment, at 10 % of the median
        # LiDAR range so the threshold is scene-appropriate rather than arbitrary.
        lut = build_lut()
        p2, tr = parse_calib(
            os.path.join(cfg.evaluation.kitti_root, "sequences", cfg.evaluation.sequence, "calib.txt")
        )
        pred_pts, gt_pts = [], []
        step = max(reader.num_frames // 8, 1)
        for i in range(0, reader.num_frames, step):
            src = scene_frame_to_source(reader, i)
            g = reader.geometry(i)
            depth = g["depth"].to(device).float()
            world = unproject_depth(depth, g["intrinsic"].to(device).float(), g["extrinsic"].to(device).float())
            conf = g["depth_conf"].to(device).float()
            keep = (depth > 1e-3) & (conf >= cfg.evaluation.conf_threshold)
            w = world[keep][::37].cpu().numpy()
            pred_pts.append(w)
            try:
                pts, _ = load_labelled_scan(cfg.evaluation.kitti_root, cfg.evaluation.sequence, src, lut)
            except FileNotFoundError:
                continue
            # velodyne -> cam0 (Tr) -> world (the GT pose is cam0-to-world).
            homo = np.concatenate([pts, np.ones((len(pts), 1), np.float32)], 1)
            cam = homo @ tr.T
            front = cam[:, 2] > 0
            cam_h = np.concatenate([cam[front, :3], np.ones((int(front.sum()), 1))], 1)
            gt_pts.append((cam_h @ _pose4(gt_poses[src]).T)[:, :3][::7])
        if not gt_pts:
            continue
        pred_all = np.concatenate(pred_pts)[:max_points]
        gt_all = np.concatenate(gt_pts)[:max_points]
        aligned = (s * (R @ pred_all.T).T + t)
        median_range = float(np.linalg.norm(gt_all - gt_all.mean(0), axis=1).mean())
        fs = geometry_fscore(
            torch.from_numpy(aligned).float().to(device),
            torch.from_numpy(gt_all).float().to(device),
            threshold=0.1 * median_range,
        )
        out.append(
            {
                "scene": spec.name, "alignment": "Sim(3) (Umeyama, geometry only)",
                "scale": s, "trajectory_rmse": rmse, "frames": len(pred_c), **fs,
            }
        )
    return {"alignment_type": "Sim(3)", "per_scene": out}


def _pose4(p: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :4] = p
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--maps", required=True, help="a map directory produced by infer_semantic_sidecar.py")
    ap.add_argument("--output", default=None)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--report_geometry", action="store_true")
    ap.add_argument("--export_class_ply", action="store_true",
                    help="write a class-coloured PLY (and two per-class similarity PLYs) per scene")
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.set)
    set_seed(cfg.seed)
    device = torch.device(cfg.train.device)
    cache_root = cfg.paths.cache_root

    projection = SemanticProjection.load(os.path.join(cache_root, "pca.safetensors"))
    teacher = build_teacher(cfg.teacher)
    text = encode_text_queries(teacher, CLASS_PROMPTS, projection)
    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    specs = [s for s in expand_scenes(cfg.scenes) if s.role == "eval"]
    specs = [s for s in specs if os.path.exists(manifest_path(cache_root, s.name))]
    specs = [s for s in specs if os.path.exists(os.path.join(args.maps, f"{s.name}.npz"))]
    if not specs:
        raise SystemExit(f"no eval scenes with maps under {args.maps}")

    n_classes = len(CLASS_NAMES) + 1
    conf_e2e = Confusion(n_classes, device)
    conf_matched = Confusion(n_classes, device)
    coverage = CoverageCounter()
    ratios: List[torch.Tensor] = []

    t0 = time.time()
    for spec in specs:
        smap = SparseSemanticMap.load(os.path.join(args.maps, f"{spec.name}.npz"))
        reader = FeatureCacheReader(cache_root, spec.name)
        evaluate_scene(
            smap, reader, spec, text, cfg, device, args.temperature,
            conf_e2e, conf_matched, coverage, ratios,
        )
        if args.export_class_ply:
            export_class_ply(
                smap, text, os.path.join(args.maps, f"{spec.name}_classes.ply"), args.temperature, device
            )
            for cls in ("car", "vegetation"):
                q = text[CLASS_NAMES.index(cls)]
                export_similarity_ply(
                    smap, q, os.path.join(args.maps, f"{spec.name}_sim_{cls}.ply"), device
                )

    inference_summary = _read_json(os.path.join(args.maps, "inference_summary.json")) or {}
    def _finite(key: str):
        return [s[key] for s in inference_summary.get("scenes", [])
                if isinstance(s.get(key), float) and not np.isnan(s[key])]

    cvc = _finite("cross_view_consistency")
    div = _finite("embedding_diversity")
    heldout = [s["cross_view_consistency"] for s in inference_summary.get("scenes", [])
               if s.get("role") == "qualitative"
               and isinstance(s.get("cross_view_consistency"), float)
               and not np.isnan(s["cross_view_consistency"])]

    results = {
        "maps": os.path.abspath(args.maps),
        "name": inference_summary.get("name", os.path.basename(os.path.normpath(args.maps))),
        "needs_teacher_at_inference": inference_summary.get("needs_teacher_at_inference"),
        "dataset": f"SemanticKITTI seq {cfg.evaluation.sequence}",
        "protocol": "observed-surface, calibrated-projection (no semantics in the correspondence)",
        "alignment": "calibrated-projection (per-frame P2/Tr); no Sim(3) fit is used for semantics",
        "prompt_source": PROMPT_SOURCE,
        "prompts": dict(zip(CLASS_NAMES, CLASS_PROMPTS)),
        "temperature": args.temperature,
        "scenes": [s.name for s in specs],
        "frames_scored": sum(len(range(0, FeatureCacheReader(cache_root, s.name).num_frames,
                                       cfg.evaluation.frame_stride)) for s in specs),
        "end_to_end_miou": 100 * conf_e2e.miou(),
        "matched_miou": 100 * conf_matched.miou(),
        "mean_accuracy": 100 * conf_matched.mean_accuracy(),
        "point_accuracy": 100 * conf_matched.accuracy(),
        "coverage": 100 * coverage.ratio,
        "cross_view_consistency": float(np.mean(cvc)) if cvc else float("nan"),
        "embedding_diversity": float(np.mean(div)) if div else float("nan"),
        "cross_view_consistency_heldout": float(np.mean(heldout)) if heldout else float("nan"),
        "geometry_fingerprint": inference_summary.get("geometry_fingerprint"),
        "per_class_iou_matched": conf_matched.per_class(CLASS_NAMES),
        "per_class_iou_end_to_end": conf_e2e.per_class(CLASS_NAMES),
        "depth_scale_median": float(torch.cat(ratios).median()) if ratios else float("nan"),
        "eval_seconds": time.time() - t0,
        "inference_summary": inference_summary,
        "provenance": collect_provenance(cfg, tool="evaluate_semantic_reconstruction"),
    }
    if args.report_geometry:
        results["geometry"] = geometry_report(cfg, specs, cache_root, device)

    logger.info(
        "%s | end-to-end mIoU %.2f | matched mIoU %.2f | mean acc %.2f | coverage %.2f%% | "
        "cross-view %.4f | diversity %.4f",
        results["name"], results["end_to_end_miou"], results["matched_miou"],
        results["mean_accuracy"], results["coverage"], results["cross_view_consistency"],
        results["embedding_diversity"],
    )
    out = args.output or os.path.join(args.maps, "eval.json")
    write_json(out, results)
    logger.info("wrote %s", out)


def _read_json(path: str):
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return None


if __name__ == "__main__":
    main()
