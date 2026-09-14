#!/usr/bin/env python
"""Evaluate B0 / B1 (all three final heads) / teacher on KITTI-360 0006 -- one shard of the anchors.

    for s in 0 1 2 3; do python tools/semantic_completion/cli/evaluate_source.py --shard $s --shards 4 --device cuda:$s & done; wait

Refuses to run if the promotion rule or the measuring code changed since the rule was frozen.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.semantic_completion import config as C, rules as RL                 # noqa: E402
from tools.semantic_completion.evaluation.evaluate import evaluate_source      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()
    RL.check_promotion_rule()
    p = C.Paths()
    evaluate_source({s: p.head_checkpoint(s, C.TrainConfig().steps) for s in C.SEEDS}, a.device, stride=1, shard=a.shard, shards=a.shards)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
