#!/usr/bin/env python
"""Step 5: six-panel 3D maps, ours vs OccAny, raw and pooled, both datasets (needs analysis.json; GPU + Open3D offscreen).

    python tools/occany_comparison/cli/figures_3d.py --device cuda:0
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.occany_comparison.figures_3d import render_all          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    rec = render_all(a.device)
    for ds, r in rec.items():
        print(ds, "seed", r["seed"], r["figures"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
