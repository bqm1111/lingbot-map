#!/usr/bin/env python
"""Step 3: score OccAny and the frozen baseline on every OccAny sample (refuses unless the source freeze is intact).

    python tools/occany_comparison/cli/evaluate.py --dataset occ3d --device cuda:0 --shard 0 --shards 4

Requires predeclaration.md to exist (the primary comparison is fixed before any score is computed).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.occany_comparison import config as C                      # noqa: E402
from tools.occany_comparison.evaluate import evaluate_shard         # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=C.DATASETS)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="debug only; written to a separate _limit file")
    a = ap.parse_args()
    if not os.path.exists(os.path.join(C.ARTIFACT_DIR, "predeclaration.md")):
        raise SystemExit("predeclaration.md is missing: declare the primary comparison before scoring")
    rec = evaluate_shard(a.dataset, a.device, a.shard, a.shards, a.limit)
    if a.limit is None:
        json.dump(rec, open(os.path.join(C.LOG_DIR, f"eval_{a.dataset}_shard{a.shard}of{a.shards}.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
