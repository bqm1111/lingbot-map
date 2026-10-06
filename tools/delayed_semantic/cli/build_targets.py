#!/usr/bin/env python
"""Step 2: delayed semantic targets (future LiDAR -> future camera -> Trident-H -> anchor grid). Requires the 'reliability' freeze.

    python tools/delayed_semantic/cli/build_targets.py --drive 2013_05_28_drive_0006_sync --device cuda:0 --shard 0 --shards 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.delayed_semantic import config as C, rules  # noqa: E402
from tools.delayed_semantic.targets import build_drive  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive", required=True); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--variant", default="", choices=["", "4s"], help="'4s' = addendum A sensitivity window (same code, 8 frames / 4 s)")
    a = ap.parse_args(); C.ensure_dirs(); rules.check("reliability")
    from tools.delayed_semantic.train_cache import VARIANT_CONFIG, target_paths
    rec = build_drive(a.drive, a.device, a.shard, a.shards, a.limit, cfg=VARIANT_CONFIG[a.variant], paths=target_paths(a.variant))
    json.dump(rec, open(os.path.join(C.ARTIFACT_DIR, "logs", f"targets{a.variant}_{a.drive[17:21]}_s{a.shard}of{a.shards}.json"), "w"), indent=1)
    print(a.drive, a.shard, "anchors", rec["n_anchors"], "written", len(rec["rows"]), f"{rec['seconds']/60:.1f} min", "violations", rec["firewall"]["violations"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
