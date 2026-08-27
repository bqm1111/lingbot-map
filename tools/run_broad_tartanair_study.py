#!/usr/bin/env python
"""Controlled corpus-swap study: broad TartanAir vs the original narrow corpus.

    python tools/run_broad_tartanair_study.py --config configs/semantic_sidecar/broad_tartanair_controlled.yaml

The single experimental factor is the training RGB corpus.  Architecture, losses, PCA
basis, teacher, schedule, tracks, evaluation scenes, prompts and geometry are taken
unchanged from the primary study, and §5 of the report verifies that programmatically.

Runs four configurations at three seeds each (12 runs):
``mlp_pixel_d100``, ``mlp_full_d100``, ``mlp_pixel_d025``, ``mlp_full_d025``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from typing import Dict, List

import _bootstrap  # noqa: F401

from semantic_sidecar.config import load_config, write_json

logger = logging.getLogger("broad_study")
PYTHON = sys.executable
TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TOOLS)
PREFIX = "broad_tartanair_controlled"

#: (base name, loss weights) — exactly the two target definitions under comparison.
CONFIGS = [
    ("mlp_pixel_d100", 1.0, {"lambda_pixel": 1.0, "lambda_consensus": 0.0, "lambda_mv": 0.0, "lambda_rel": 0.0}),
    ("mlp_full_d100", 1.0, {"lambda_pixel": 0.0, "lambda_consensus": 1.0, "lambda_mv": 0.25, "lambda_rel": 0.1}),
    ("mlp_pixel_d025", 0.25, {"lambda_pixel": 1.0, "lambda_consensus": 0.0, "lambda_mv": 0.0, "lambda_rel": 0.0}),
    ("mlp_full_d025", 0.25, {"lambda_pixel": 0.0, "lambda_consensus": 1.0, "lambda_mv": 0.25, "lambda_rel": 0.1}),
]
SEEDS = [0, 1, 2]


def run(cmd: List[str], dry: bool = False) -> None:
    logger.info("$ %s", " ".join(cmd))
    if dry:
        return
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO)
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    logger.info("  ... %.1fs", time.time() - t0)


def loss_flags(spec: Dict[str, float]) -> List[str]:
    full = {"lambda_pixel": 0.0, "lambda_consensus": 0.0, "lambda_mv": 0.0,
            "lambda_rel": 0.0, "lambda_center": 0.0}
    full.update(spec)
    return [f"loss.{k}={v}" for k, v in full.items()]


def run_name(base: str, seed: int) -> str:
    return f"{PREFIX}_{base}_s{seed}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stages", nargs="*",
                    default=["cache", "teacher", "tracks", "train", "infer", "eval"])
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    tool = lambda n: os.path.join(TOOLS, n)  # noqa: E731
    ow = ["--overwrite"] if args.overwrite else []
    t0 = time.time()

    if "cache" in args.stages:
        run([PYTHON, tool("cache_lingbot_features.py"), "--config", args.config,
             "--roles", "train", *ow], args.dry_run)
    if "teacher" in args.stages:
        run([PYTHON, tool("cache_teacher_features.py"), "--config", args.config,
             "--roles", "train", *ow], args.dry_run)
    if "tracks" in args.stages:
        # Targets depend on the corpus, so they are rebuilt; the qualitative scene's
        # tracks are needed for the cross-view metric.
        for density in ("1.0", "0.25"):
            run([PYTHON, tool("build_semantic_tracks.py"), "--config", args.config,
                 "--teacher_density", density, "--roles", "train", "qualitative", *ow], args.dry_run)

    jobs = [(base, dens, losses, seed)
            for base, dens, losses in CONFIGS for seed in args.seeds]

    if "train" in args.stages:
        for base, dens, losses, seed in jobs:
            run([PYTHON, tool("train_semantic_sidecar.py"), "--config", args.config,
                 "--run_name", run_name(base, seed), "--teacher_density", str(dens),
                 "--set", "model.arch=mlp", f"train.seed={seed}", *loss_flags(losses), *ow],
                args.dry_run)

    if "infer" in args.stages:
        for base, dens, losses, seed in jobs:
            name = run_name(base, seed)
            rd = os.path.join(cfg.paths.runs_root, name)
            if not args.dry_run and not os.path.exists(os.path.join(rd, "best.pt")):
                logger.warning("%s has no checkpoint; skipping", name)
                continue
            run([PYTHON, tool("infer_semantic_sidecar.py"), "--config", args.config,
                 "--run_dir", rd, "--roles", "eval", "qualitative",
                 "--tracks_tag", "density100", *ow], args.dry_run)

    if "eval" in args.stages:
        for base, dens, losses, seed in jobs:
            name = run_name(base, seed)
            md = os.path.join(cfg.paths.maps_root, name)
            if not args.dry_run and not os.path.exists(os.path.join(md, "inference_summary.json")):
                logger.warning("%s has no maps; skipping", name)
                continue
            run([PYTHON, tool("evaluate_semantic_reconstruction.py"), "--config", args.config,
                 "--maps", md], args.dry_run)

    logger.info("broad study stages %s finished in %.1f min", args.stages, (time.time() - t0) / 60)
    if not args.dry_run:
        write_json(os.path.join(cfg.paths.reports_root, "run_index.json"),
                   {"prefix": PREFIX, "seeds": args.seeds,
                    "runs": [run_name(b, s) for b, _, _, s in jobs],
                    "config": os.path.abspath(args.config)})


if __name__ == "__main__":
    main()
