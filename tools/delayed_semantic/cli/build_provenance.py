#!/usr/bin/env python
"""Step 1: per-anchor evidence provenance (frame bits, times), replayed frame by frame; asserts Gate 15 evidence equality.

    python tools/delayed_semantic/cli/build_provenance.py --drive 2013_05_28_drive_0006_sync --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.delayed_semantic import config as C          # noqa: E402
from tools.delayed_semantic.provenance import build_drive  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive", required=True); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args(); C.ensure_dirs()
    rec = build_drive(a.drive, a.device, a.shard, a.shards)
    os.makedirs(os.path.join(C.ARTIFACT_DIR, "logs"), exist_ok=True)
    json.dump(rec, open(os.path.join(C.ARTIFACT_DIR, "logs", f"provenance_{a.drive[17:21]}_s{a.shard}.json"), "w"), indent=1)
    print(a.drive, "anchors", rec["n"], "feature checks", [(c["t"], round(c["max_abs"], 5), c["class_agree"]) for c in rec["feature_checks"]][:5], "violations", rec["firewall"]["violations"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
