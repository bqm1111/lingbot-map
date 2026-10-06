#!/usr/bin/env python
"""Step 2: five-frame LingBot clips for every OccAny sample, forward and causal (resumable, shardable).

    python tools/occany_comparison/cli/build_clips.py --dataset occ3d --device cuda:1 --shard 0 --shards 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.occany_comparison import config as C                      # noqa: E402
from tools.occany_comparison.clips import build_clips               # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=C.DATASETS)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="debug only")
    a = ap.parse_args()
    rec = build_clips(a.dataset, a.device, a.shard, a.shards, a.limit)
    os.makedirs(C.LOG_DIR, exist_ok=True)
    if a.limit is None:
        json.dump(rec, open(os.path.join(C.LOG_DIR, f"clips_{a.dataset}_shard{a.shard}of{a.shards}.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
