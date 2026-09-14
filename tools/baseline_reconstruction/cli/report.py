#!/usr/bin/env python
"""Phase 4: render every deliverable from the recorded results (no computation on any dataset).

    python tools/baseline_reconstruction/cli/report.py --device cuda:0     # the device is used only for the scale-once re-derivation check
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.baseline_reconstruction import config as C                                  # noqa: E402
from tools.baseline_reconstruction.reports import comparison as CMP, final as FIN, source as SRC, verification as VER, zero_shot as ZS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0", help="'none' skips the GPU re-derivation check in verification.md")
    a = ap.parse_args()
    paths = C.CurrentPaths()
    source = json.load(open(os.path.join(paths.source_eval, "source_summary.json")))
    hist = CMP.load_historical()
    zs = ZS.write_zero_shot_results(paths, {ds: hist["results"][ds]["median_pct"] for ds in C.HISTORICAL_TARGETS})
    comp = CMP.write_comparability(zs["zero_shot"], sum(source["reproduction"]["ours_full_iou"].values()) / 3)
    ver = VER.write_verification(paths, None if a.device == "none" else a.device)
    out = FIN.write_report(source, zs["zero_shot"], hist, comp, zs["ci"], ver)
    print("wrote", out, f"({ver['n_pass']}/{ver['n_total']} verification checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
