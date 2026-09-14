#!/usr/bin/env python
"""Train the Gate 15 propagation head on top of one frozen geometry seed (KITTI-360 0003/0007/0010 only).

    python tools/semantic_completion/cli/train.py --seed 0 --device cuda:0
    python tools/semantic_completion/cli/train.py --seed 0 --device cuda:0 --steps 20     # smoke test, separate tag
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.semantic_completion import config as C              # noqa: E402
from tools.semantic_completion.train import train_head           # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True, choices=C.SEEDS); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()
    C.ensure_dirs()
    train_head(a.seed, a.device, a.steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
