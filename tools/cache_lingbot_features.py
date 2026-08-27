#!/usr/bin/env python
"""Cache frozen LingBot-MAP tokens and geometry for every configured scene.

    python tools/cache_lingbot_features.py --config configs/semantic_sidecar/feasibility.yaml

Every LingBot parameter is frozen and inference runs under ``torch.no_grad``; the
intermediate tokens are read with a forward hook that cannot change the model's
outputs.  Scenes already present in the cache are skipped unless ``--overwrite``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time

import torch

import _bootstrap  # noqa: F401  (sys.path side effect)

from semantic_sidecar.config import collect_provenance, load_config, set_seed, write_json
from semantic_sidecar.datasets import expand_scenes
from semantic_sidecar.lingbot_features import FrozenLingBot, cache_scene, list_scene_images, scene_is_cached

logger = logging.getLogger("cache_lingbot")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="dotted config overrides, key=value")
    ap.add_argument("--roles", nargs="*", default=None, help="only cache scenes with these roles")
    ap.add_argument("--scenes", nargs="*", default=None, help="only cache these scene names")
    ap.add_argument("--overwrite", action="store_true", help="re-cache scenes that already exist")
    ap.add_argument("--dry_run", action="store_true", help="list what would be cached and exit")
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.set)
    set_seed(cfg.seed)
    logger.info("seed=%d config=%s", cfg.seed, args.config)

    scenes = expand_scenes(cfg.scenes)
    if args.roles:
        scenes = [s for s in scenes if s.role in args.roles]
    if args.scenes:
        wanted = set(args.scenes)
        scenes = [s for s in scenes if s.name in wanted or s.name.split("_c")[0] in wanted]
    if not scenes:
        raise SystemExit("no scenes selected")

    root = cfg.paths.cache_root
    todo = [s for s in scenes if args.overwrite or not scene_is_cached(root, s.name)]
    logger.info("%d scene chunks selected, %d to cache (cache root %s)", len(scenes), len(todo), root)
    for s in scenes:
        state = "cached" if scene_is_cached(root, s.name) else "missing"
        logger.info("  %-44s %-6s %s", s.name, state, s.image_folder)
    if args.dry_run or not todo:
        if not todo:
            logger.info("nothing to do (use --overwrite to force)")
        return

    provenance = collect_provenance(cfg, tool="cache_lingbot_features")
    model = FrozenLingBot(cfg.lingbot, capture_tokens=True)
    model.assert_frozen()
    params = model.parameter_report()
    logger.info("LingBot parameters: %(lingbot_total)s total, %(lingbot_trainable)s trainable", params)
    provenance["lingbot_parameters"] = params
    provenance["checkpoint_sha256"] = model.checkpoint_sha256

    summary = []
    for i, spec in enumerate(todo, 1):
        paths = list_scene_images(spec)
        logger.info("[%d/%d] %s: %d frames", i, len(todo), spec.name, len(paths))
        if len(paths) > cfg.lingbot.max_frame_num:
            raise SystemExit(
                f"{spec.name}: {len(paths)} frames exceeds RoPE limit {cfg.lingbot.max_frame_num}; "
                "set scenes[].chunk_size"
            )
        out_dir = os.path.join(root, spec.name)
        if args.overwrite and os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        t0 = time.time()
        manifest = cache_scene(model, spec, root, provenance=provenance, batch_paths=paths)
        dt = time.time() - t0
        size_gb = sum(
            os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir)
        ) / 1e9
        logger.info("    %d frames in %.1fs (%.2f GB)", manifest["num_frames"], dt, size_gb)
        summary.append(
            {"scene": spec.name, "frames": manifest["num_frames"], "seconds": dt, "gigabytes": size_gb}
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_json(
        os.path.join(root, "cache_summary.json"),
        {"provenance": provenance, "scenes": summary, "config": args.config},
    )
    logger.info("done: %d scenes, %.2f GB", len(summary), sum(s["gigabytes"] for s in summary))


if __name__ == "__main__":
    main()
