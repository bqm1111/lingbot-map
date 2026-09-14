#!/usr/bin/env python
"""Phase 2a: train one seed of the current geometry-only baseline (frozen Gate 12/13 recipe).

    python tools/baseline_reconstruction/cli/train_current.py --seed 0 --device cuda:0

Writes checkpoints every 1000 steps to artifacts/baseline_reconstruction/checkpoints and a
run record to .../runs. Never opens LiDAR, GT poses, semantics or a target dataset.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.baseline_reconstruction import config as C                        # noqa: E402
from tools.baseline_reconstruction.current.recipe import write_training_config  # noqa: E402
from tools.baseline_reconstruction.current.train import train_one_seed       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True, choices=C.SEEDS)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=None, help="override for smoke tests only (writes with a 'smoke' tag)")
    a = ap.parse_args()
    C.ensure_dirs(); write_training_config()
    train_one_seed(a.seed, a.device, steps=a.steps, tag_prefix="geo" if a.steps is None else "smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
