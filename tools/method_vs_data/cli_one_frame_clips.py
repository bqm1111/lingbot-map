#!/usr/bin/env python
"""Build the Phase A one-frame causal clips. See ``tools/method_vs_data/one_frame.py``.

    PATH=<env>/bin:$PATH PYTHONPATH=$PWD python tools/method_vs_data/cli_one_frame_clips.py \
        --datasets semantickitti,occ3d --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.method_vs_data import one_frame as OF                                        # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="semantickitti,occ3d")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    out = {}
    for ds in a.datasets.split(","):
        out[ds] = OF.build(ds, a.device, a.shard, a.shards, a.limit)
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "artifacts", "method_vs_data", "logs")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"one_frame_clips_s{a.shard}of{a.shards}.json")
    json.dump(out, open(p, "w"), indent=1, default=str)
    print(f"-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
