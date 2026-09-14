#!/usr/bin/env python
"""Phase 2b: evaluate the trained checkpoints on KITTI-360 0006 (source side only).

    python tools/baseline_reconstruction/cli/eval_source.py --device cuda:0            # final checkpoints, all 590 anchors
    python tools/baseline_reconstruction/cli/eval_source.py --device cuda:0 --curve    # every 1000-step checkpoint, stride 5

Then writes source_results.md and tables/source_per_seed.csv, tables/source_per_range.csv.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.baseline_reconstruction import config as C                                # noqa: E402
from tools.baseline_reconstruction.current.evaluate_source import evaluate_checkpoint  # noqa: E402
from tools.baseline_reconstruction.reports import source as REP                      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--curve", action="store_true", help="evaluate the intermediate checkpoints at anchor stride 5 (learning curve)")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(C.SEEDS))
    ap.add_argument("--no-report", action="store_true")
    a = ap.parse_args()
    paths = C.CurrentPaths(); C.ensure_dirs()
    steps = range(1000, 6001, 1000) if a.curve else [6000]
    for seed in a.seeds:
        for step in steps:
            ck = paths.checkpoint_path(seed, step)
            if os.path.exists(ck):
                evaluate_checkpoint(ck, a.device, paths, stride=5 if a.curve else 1)
    if not a.no_report:
        REP.write_source_results(paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
