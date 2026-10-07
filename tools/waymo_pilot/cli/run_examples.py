#!/usr/bin/env python
"""Render the pilot's deterministic inspectable examples from a measured arm.

    python tools/waymo_pilot/cli/run_examples.py --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.semantic_completion.evidence import trident as TR                  # noqa: E402
from tools.waymo_pilot import config as C, examples as EX                      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--measurement", default=os.path.join(C.Paths().logs,
                                                          "k360_reference.json"))
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    C.ensure_dirs()
    dev = torch.device(a.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    with open(a.measurement) as fh:
        meas = json.load(fh)
    rows = meas["per_anchor"]
    chosen = EX.pick(rows, a.n)
    text, _p, _l = TR.load_text_embeddings()
    text = text.to(dev)
    out_dir = os.path.join(C.Paths().examples, meas["variant"])
    sides = [EX.render(r, dev, text, out_dir) for r in chosen]
    idx = os.path.join(out_dir, "index.json")
    with open(idx, "w") as fh:
        json.dump({"variant": meas["variant"], "selection_rule":
                   f"fixed quantiles {EX.QUANTILES} of per-anchor U/hidden yield, then "
                   f"unrepresented scenes", "examples": sides}, fh, indent=2, default=float)
    for s in sides:
        print(f"  {s['scene'][17:21]} anchor {s['anchor']:5d}  "
              f"yield {100 * s['yield_U_over_hidden']:5.1f} %  -> {s['figure']}")
    print(f"\n{len(sides)} examples -> {idx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
