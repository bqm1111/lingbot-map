#!/usr/bin/env python
"""Phase 3a: local-reset LingBot clips on a target stream (opens target images -- requires the source freeze).

    python tools/baseline_reconstruction/cli/build_target_clips.py --dataset semantickitti --device cuda:0
    python tools/baseline_reconstruction/cli/build_target_clips.py --dataset occ3d --shard 0 --shards 4 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.baseline_reconstruction import config as C, freeze as FZ        # noqa: E402
from tools.baseline_reconstruction.targets.clips import build_clips         # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(C.TARGET_PROTOCOLS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    FZ.require_source_freeze()
    stats = build_clips(a.dataset, a.device, shard=a.shard, shards=a.shards, limit=a.limit)
    os.makedirs(os.path.join(C.ARTIFACT_DIR, "logs"), exist_ok=True)
    json.dump(stats, open(os.path.join(C.ARTIFACT_DIR, "logs", f"target_clips_{a.dataset}_s{a.shard}.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
