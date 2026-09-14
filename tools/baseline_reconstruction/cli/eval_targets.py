#!/usr/bin/env python
"""Phase 3b: the frozen zero-shot evaluation on a target dataset (requires the source freeze; refuses on any hash mismatch).

    python tools/baseline_reconstruction/cli/eval_targets.py --dataset semantickitti --device cuda:0
    python tools/baseline_reconstruction/cli/eval_targets.py --dataset occ3d --device cuda:1

Evaluates all three frozen final checkpoints in one pass and writes zero_shot_eval/zero_shot_<dataset>.json.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.baseline_reconstruction import config as C                    # noqa: E402
from tools.baseline_reconstruction.targets.evaluate import evaluate_dataset  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(C.TARGET_PROTOCOLS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None, help="debug only; a limited run is never a result")
    a = ap.parse_args()
    evaluate_dataset(a.dataset, a.device, limit=a.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
