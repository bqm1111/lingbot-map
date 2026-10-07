#!/usr/bin/env python
"""Run the pilot's four focused checks.

    python tools/waymo_pilot/cli/run_checks.py --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.waymo_pilot import checks as CK, config as C                       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--drive", default=None)
    ap.add_argument("--anchor", type=int, default=None)
    a = ap.parse_args()
    C.ensure_dirs()
    dev = torch.device(a.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    rec = CK.run_all(dev, a.drive, a.anchor)
    out = os.path.join(C.Paths().logs, "checks.json")
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=2, default=float)
    for c in rec["checks"]:
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['name']}")
        for k, v in c.items():
            if k not in ("name", "ok"):
                print(f"          {k}: {v}")
    print(f"\n{'ALL CHECKS PASS' if rec['all_ok'] else 'CHECK FAILURE'} -> {out}")
    return 0 if rec["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
