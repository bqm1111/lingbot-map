#!/usr/bin/env python
"""Cache the causal Trident-H evidence of every anchor of a KITTI-360 drive (no labels, LiDAR, GT poses or target data).

    python tools/semantic_completion/cli/build_evidence.py --drive 2013_05_28_drive_0006_sync --device cuda:0 [--shard 0 --shards 2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.semantic_completion import config as C                     # noqa: E402
from tools.semantic_completion.evidence.build import build_drive        # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive", required=True, choices=list(C.TRAIN_DRIVES) + [C.VAL_DRIVE])
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()
    C.ensure_dirs()
    rec = build_drive(a.drive, a.device, a.shard, a.shards)
    json.dump(rec, open(os.path.join(C.ARTIFACT_DIR, "logs", f"evidence_{a.drive[17:21]}_s{a.shard}.json"), "w"), indent=1)
    return 0 if not rec["firewall"]["violations"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
