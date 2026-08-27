"""Grid evaluation: sequences x prompt configs x anchors -> tidy rows."""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .anchors import BASELINES, MetricAnchor
from .prompts import PromptConfig, standard_configs
from .runner import (CachedSequence, DepthEvalCache, build_prompts, evaluate_run,
                     gt_point_cloud, load_cached, run_anchor)

# Point-cloud metrics are ~100x the cost of everything else, so they are computed
# only for the configs that actually appear in the report's point-cloud table.
POINT_CLOUD_CONFIGS = {
    "rgb_only", "depth_first_only_clean", "depth_k30_clean", "pose_k30_clean",
    "both_k30_clean", "both_k30_moderate", "depth_k30_moderate",
}


def evaluate_sequence(
    cache_path: str,
    configs: Dict[str, PromptConfig],
    anchor_names: Sequence[str],
    extra_anchors: Optional[Dict[str, Callable[[], MetricAnchor]]] = None,
    point_cloud: bool = True,
    seed: int = 0,
    meta: Optional[dict] = None,
) -> List[dict]:
    """Every (config, anchor) pair for one cached sequence."""
    seq = load_cached(cache_path, meta)
    depth_cache = DepthEvalCache.build(seq, seed=seed) if seq.has_gt_depth else None
    gt_pts = gt_point_cloud(seq, seed) if (point_cloud and seq.has_gt_depth) else None
    factories = dict(BASELINES)
    if extra_anchors:
        factories.update(extra_anchors)

    rows: List[dict] = []
    for cfg_name, cfg in configs.items():
        prompts = build_prompts(seq, cfg)
        want_pc = point_cloud and seq.has_gt_depth and cfg_name in POINT_CLOUD_CONFIGS
        for a_name in anchor_names:
            if a_name not in factories:
                continue
            anchor = factories[a_name]()
            if a_name == "per_frame_oracle_scale" and not seq.has_gt_depth:
                continue
            run = run_anchor(seq, anchor, prompts, depth_cache=depth_cache)
            m = evaluate_run(seq, run, point_cloud=want_pc, seed=seed,
                             depth_cache=depth_cache, gt_points=gt_pts)
            rows.append({
                "sequence": seq.name,
                "dataset": seq.meta.get("dataset", "unknown"),
                "scene": seq.meta.get("scene", "unknown"),
                "config": cfg_name,
                "anchor": a_name,
                "causal": bool(anchor.is_causal),
                "n_frames": seq.n_frames,
                "lingbot_ms_per_frame": float(seq.meta.get("inference_s_per_frame", float("nan")) * 1000),
                **m,
            })
    return rows


def _worker(args):
    path, configs, anchors, point_cloud, seed, meta, learned_ckpt = args
    try:
        extra = None
        if learned_ckpt:
            from .learned_anchor import load_checkpoint
            factory, _ = load_checkpoint(learned_ckpt, device="cpu")
            extra = {"learned_corrector": factory}
        return evaluate_sequence(path, configs, anchors, extra_anchors=extra,
                                 point_cloud=point_cloud, seed=seed, meta=meta)
    except Exception as exc:  # keep one bad sequence from killing the grid
        import traceback
        traceback.print_exc()
        return [{"sequence": os.path.basename(path), "error": f"{type(exc).__name__}: {exc}"}]


def evaluate_cache_dir(
    cache_dir: str,
    configs: Optional[Dict[str, PromptConfig]] = None,
    anchor_names: Optional[Sequence[str]] = None,
    names: Optional[Sequence[str]] = None,
    workers: int = 8,
    point_cloud: bool = True,
    seed: int = 0,
    learned_checkpoint: Optional[str] = None,
) -> List[dict]:
    configs = configs if configs is not None else standard_configs()
    anchor_names = list(anchor_names if anchor_names is not None else BASELINES.keys())
    if learned_checkpoint and "learned_corrector" not in anchor_names:
        anchor_names.append("learned_corrector")

    manifest_path = os.path.join(cache_dir, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.isfile(manifest_path) else {"sequences": {}}
    paths = []
    for f in sorted(os.listdir(cache_dir)):
        if not f.endswith(".npz"):
            continue
        name = f[:-4]
        if names is not None and name not in names:
            continue
        paths.append((os.path.join(cache_dir, f), configs, anchor_names, point_cloud, seed,
                      manifest["sequences"].get(name, {}), learned_checkpoint))
    if not paths:
        return []

    if workers <= 1:
        rows = []
        for a in paths:
            rows.extend(_worker(a))
        return rows

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(workers, len(paths))) as pool:
        out = pool.map(_worker, paths)
    return [r for chunk in out for r in chunk]
