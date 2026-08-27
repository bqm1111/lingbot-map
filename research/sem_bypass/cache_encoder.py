#!/usr/bin/env python
"""Cache frozen LingBot **pre-GCT encoder** tokens + geometry for configured scenes.

    python -m research.sem_bypass.cache_encoder --config research/sem_bypass/configs/gate1.yaml --roles train eval

Reuses the existing scene expansion, shard writer and manifest schema; only the token
source differs (see :mod:`research.sem_bypass.encoder_features`).  Scenes already cached
are skipped unless ``--overwrite``.
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import torch

from semantic_sidecar.config import collect_provenance, load_config, set_seed, write_json
from semantic_sidecar.datasets import expand_scenes
from semantic_sidecar.lingbot_features import scene_is_cached

from research.sem_bypass.encoder_features import FrozenLingBotEncoder, cache_scene_encoder

logger = logging.getLogger("sem_bypass.cache")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--roles", nargs="*", default=None)
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--log_level", default="INFO")
    a = ap.parse_args()

    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.config, a.set)
    if a.device:
        cfg.lingbot.device = a.device
    set_seed(cfg.seed)

    scenes = expand_scenes(cfg.scenes)
    if a.roles:
        scenes = [s for s in scenes if s.role in a.roles]
    if a.scenes:
        scenes = [s for s in scenes if s.name in a.scenes]
    root = cfg.paths.cache_root
    todo = [s for s in scenes if a.overwrite or not scene_is_cached(root, s.name)]
    logger.info("%d scenes, %d to cache -> %s", len(scenes), len(todo), root)
    if not todo:
        return

    model = FrozenLingBotEncoder(cfg.lingbot, capture_tokens=True)
    model.assert_frozen()
    logger.info("frozen LingBot: %s", model.parameter_report())
    prov = collect_provenance(cfg, tool="sem_bypass.cache_encoder")

    frames = 0
    t0 = time.time()
    for s in todo:
        t1 = time.time()
        m = cache_scene_encoder(model, s, root, provenance=prov)
        frames += m["num_frames"]
        logger.info("%s: %d frames, %.1fs", s.name, m["num_frames"], time.time() - t1)
    dt = time.time() - t0
    logger.info("cached %d frames in %.1fs (%.1f ms/frame)", frames, dt, 1000 * dt / max(frames, 1))
    write_json(os.path.join(root, "encoder_cache_summary.json"),
               {"provenance": prov, "frames": frames, "seconds": dt,
                "scenes": [s.name for s in todo]})


if __name__ == "__main__":
    main()
