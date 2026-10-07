#!/usr/bin/env python
"""ANNOTATION-ASSISTED: audit the privileged target's CORRECTNESS against human labels.

Separate from every other entry point on purpose. Human labels are read only here, and
nothing this prints feeds a target, a threshold or a model selection.

    python tools/waymo_pilot/cli/run_diagnostic.py --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.semantic_completion.evidence import trident as TR                  # noqa: E402
from tools.waymo_pilot import config as C, diagnostic as DG                    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--measurement", default=os.path.join(C.Paths().logs,
                                                          "k360_reference.json"))
    a = ap.parse_args()
    C.ensure_dirs()
    dev = torch.device(a.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    with open(a.measurement) as fh:
        meas = json.load(fh)
    text, _p, _l = TR.load_text_embeddings()
    text = text.to(dev)
    rows = [DG.teacher_correctness(r["scene"], r["anchor"], dev, text)
            for r in meas["per_anchor"]]
    summary = DG.summarize(rows)
    n_missing = sum(1 for r in rows if not r.get("available"))
    summary["n_anchors_without_labels"] = n_missing
    out = os.path.join(C.Paths().logs, f"teacher_correctness_{meas['variant']}.json")
    with open(out, "w") as fh:
        json.dump({"variant": meas["variant"], "summary": summary,
                   "per_anchor": [{k: v for k, v in r.items() if k != "confusion"}
                                  for r in rows]}, fh, indent=2, default=float)
    if not summary.get("available"):
        print(summary["note"])
        return 0
    print(f"  variant                         {meas['variant']}")
    print(f"  anchors scored / unlabelled     {summary['n_anchors']} / {n_missing}")
    print(f"  target voxels with a label      {summary['n_U_with_label']:,}")
    print(f"  top-1 CORRECTNESS               {100 * summary['annotated_top1_accuracy']:.1f} %")
    print(f"  target mIoU (18 classes)        {100 * summary['annotated_target_miou']:.2f}")
    print(f"  top-1 on 2-view-agreeing voxels {100 * summary['annotated_top1_two_view_agreeing']:.1f} %")
    for name in ("A", "B"):
        acc = summary.get(f"annotated_top1_tier{name}")
        n = summary.get(f"n_tier{name}", 0)
        print(f"  tier {name} top-1 / voxels          "
              f"{'n/a' if acc is None else f'{100 * acc:.1f} %'} / {n:,}")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
