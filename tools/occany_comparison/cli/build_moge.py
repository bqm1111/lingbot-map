#!/usr/bin/env python
"""Step 1: MoGe-B frames the frozen cache lacks (Occ3D frames past our 40-frame streams), into a separate root.

    python tools/occany_comparison/cli/build_moge.py --dataset occ3d --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.occany_comparison import config as C                      # noqa: E402
from tools.occany_comparison.moge_extra import build_missing        # noqa: E402
from tools.occany_comparison.samples import build_samples           # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=C.DATASETS)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    rec = build_missing(a.dataset, build_samples(a.dataset), a.device)
    print(json.dumps(rec, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
