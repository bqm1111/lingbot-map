#!/usr/bin/env python
"""Gate 1: three-seed SemanticKITTI reproduction of SemBypass.

Stages, per seed in {0, 1, 2}:

``tracks``  build 3D tracks + 25 % teacher masks (the seed drives the teacher draw);
``train``   fit the sidecar on pre-GCT tokens (primary: pixel-only distillation);
``infer``   build the semantic voxel map on held-out sequence 08;
``eval``    score matched observed-surface mIoU against SemanticKITTI labels.

``teacher_direct`` is built once: it depends only on the frozen teacher and the frozen
geometry, both of which are seed-independent.

Every stage is the unmodified ``tools/`` implementation, reached through
:mod:`research.sem_bypass._tool_shim` so that only the token width differs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Sequence

from semantic_sidecar.config import load_config

from research.sem_bypass._tool_shim import REPO_ROOT, run_tool

logger = logging.getLogger("sem_bypass.gate1")

CONFIG = "research/sem_bypass/configs/gate1.yaml"
DENSITY = 0.25
TAG = "density025"

#: Primary = consensus-free feature distillation (audit SS3). Secondary = the exact
#: ``mlp_full`` recipe that produced the audited 10.04, i.e. "the existing 3D consensus".
VARIANTS = {
    "sem_bypass": {"lambda_pixel": 1.0, "lambda_consensus": 0.0, "lambda_mv": 0.0, "lambda_rel": 0.0},
    "sem_bypass_consensus": {"lambda_pixel": 0.0, "lambda_consensus": 1.0, "lambda_mv": 0.25, "lambda_rel": 0.1},
}


def seed_sets(seed: int, tracks_base: str) -> List[str]:
    """Vary the global seed, the training seed and the teacher draw together."""
    return [f"seed={seed}", f"train.seed={seed}", f"train.teacher_density_seed={seed}",
            f"paths.tracks_root={tracks_base}/seed{seed}"]


def loss_sets(losses: Dict[str, float]) -> List[str]:
    return [f"loss.{k}={v}" for k, v in losses.items()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--stages", nargs="*",
                    default=["tracks", "train", "infer", "eval", "teacher"])
    ap.add_argument("--steps", type=int, default=None, help="override train.steps (smoke)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--log_level", default="INFO")
    a = ap.parse_args()

    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    paths = load_config(a.config, []).paths
    cache_root = paths.cache_root
    ow = ["--overwrite"] if a.overwrite else []
    dev = [f"lingbot.device={a.device}", f"train.device={a.device}", f"teacher.device={a.device}"]
    extra = ([f"train.steps={a.steps}"] if a.steps else [])
    t0 = time.time()

    # -- tracks (per seed: the 25 % teacher mask depends on it) ------------- #
    if "tracks" in a.stages:
        for s in a.seeds:  # noqa: PLW2901
            logger.info("=== tracks seed %d ===", s)
            run_tool("build_semantic_tracks",
                     ["--config", a.config, "--teacher_density", str(DENSITY),
                      "--set", *seed_sets(s, paths.tracks_root), *dev, *ow],
                     cache_root=cache_root)

    # -- train -------------------------------------------------------------- #
    if "train" in a.stages:
        for s in a.seeds:
            for v in a.variants:
                name = f"{v}_seed{s}"
                logger.info("=== train %s ===", name)
                run_tool("train_semantic_sidecar",
                         ["--config", a.config, "--run_name", name,
                          "--teacher_density", str(DENSITY),
                          "--set", "model.arch=mlp", *loss_sets(VARIANTS[v]),
                          *seed_sets(s, paths.tracks_root), *dev, *extra, *ow],
                         cache_root=cache_root)

    # -- teacher baseline (seed-independent) -------------------------------- #
    if "teacher" in a.stages:
        for base in ("teacher_direct",):
            logger.info("=== infer %s ===", base)
            run_tool("infer_semantic_sidecar",
                     ["--config", a.config, "--baseline", base, "--roles", "eval",
                      "--tracks_tag", TAG,
                      "--set", *seed_sets(a.seeds[0], paths.tracks_root), *dev, *ow],
                     cache_root=cache_root)

    # -- inference on held-out seq 08 --------------------------------------- #
    if "infer" in a.stages:
        for s in a.seeds:
            for v in a.variants:
                name = f"{v}_seed{s}"
                run_dir = os.path.join(paths.runs_root, name)
                if not os.path.exists(os.path.join(REPO_ROOT, run_dir, "best.pt")):
                    logger.warning("%s has no checkpoint; skipping", name)
                    continue
                logger.info("=== infer %s ===", name)
                run_tool("infer_semantic_sidecar",
                         ["--config", a.config, "--run_dir", run_dir, "--roles", "eval",
                          "--tracks_tag", TAG, "--set", *seed_sets(s, paths.tracks_root), *dev, *ow],
                         cache_root=cache_root)

    # -- evaluation --------------------------------------------------------- #
    if "eval" in a.stages:
        names = (["teacher_direct"] if "teacher" in a.stages else []) + \
                [f"{v}_seed{s}" for s in a.seeds for v in a.variants]
        for name in names:
            maps = os.path.join(paths.maps_root, name)
            if not os.path.isdir(os.path.join(REPO_ROOT, maps)):
                logger.warning("no map for %s; skipping eval", name)
                continue
            logger.info("=== eval %s ===", name)
            run_tool("evaluate_semantic_reconstruction",
                     ["--config", a.config, "--maps", maps, "--set", *dev],
                     cache_root=cache_root)

    logger.info("gate1 stages %s finished in %.1f s", a.stages, time.time() - t0)


if __name__ == "__main__":
    main()
