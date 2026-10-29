#!/usr/bin/env python
"""Consolidate the four canonical rescore JSONs into the tables the report quotes.

Reads ``canonical_{dataset}_{post}.json`` and emits markdown tables plus a flat CSV, so no
number in the report is typed by hand.

    PYTHONPATH=$PWD python tools/method_vs_data/rescore_tables.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os

D = "artifacts/method_vs_data/waymo_canonical_rescore"
CELLS = [("semantickitti", "raw"), ("semantickitti", "pooled"),
         ("occ3d", "raw"), ("occ3d", "pooled")]
NICE = {"k360": "ours, KITTI-360", "waymo": "ours, Waymo", "once": "ours, ONCE",
        "occany_1f": "OccAny 1-frame causal [reproduced]",
        "occany_5f_fwd": "OccAny 5-frame forward [reproduced]"}


def load(root):
    out = {}
    for ds, post in CELLS:
        p = os.path.join(root, f"canonical_{ds}_{post}.json")
        if os.path.exists(p):
            out[(ds, post)] = json.load(open(p))
    return out


def fmt(x):
    s = "*" if x["significant_at_95"] else "ns"
    return f"{x['delta']:+.2f} [{x['ci_low']:+.2f}, {x['ci_high']:+.2f}] {s}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=D)
    a = ap.parse_args()
    R = load(a.root)
    rows = []
    md = []

    md.append("## Absolute SC IoU, `all` (full official valid volume)\n")
    md.append("| arm | SK raw | SK pooled | Occ3D raw | Occ3D pooled |")
    md.append("|---|---:|---:|---:|---:|")
    for arm in ["k360", "waymo", "once", "occany_1f", "occany_5f_fwd"]:
        c = []
        for k in CELLS:
            d = R.get(k)
            c.append(f"{d['iou']['all'][arm]['iou']:.2f}" if d and arm in d["arms"] else "—")
        md.append(f"| {NICE[arm]} | " + " | ".join(c) + " |")
    md.append("")

    md.append("## Precision / recall and confusion counts, `all`\n")
    for k in CELLS:
        d = R.get(k)
        if not d:
            continue
        md.append(f"**{k[0]} / {k[1]}** — {d['n_samples']} anchors\n")
        md.append("| arm | SC IoU | precision | recall | TP | FP | FN | evaluated (valid) voxels |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        nv = d["composition"]["all"]["valid_voxels"]
        for arm in d["arms"]:
            v = d["iou"]["all"][arm]
            md.append(f"| {NICE[arm]} | {v['iou']:.2f} | {v['precision']:.2f} | "
                      f"{v['recall']:.2f} | {v['tp']:,} | {v['fp']:,} | {v['fn']:,} | {nv:,} |")
        md.append("")

    md.append("## Regional SC IoU and paired differences\n")
    for k in CELLS:
        d = R.get(k)
        if not d:
            continue
        md.append(f"### {k[0]} / {k[1]}\n")
        md.append("| region | % valid | % GT occ | " +
                  " | ".join(NICE[a] for a in d["arms"]) + " |")
        md.append("|---|---:|---:|" + "---:|" * len(d["arms"]))
        for r, c in d["composition"].items():
            md.append(f"| {r} | {100 * c['share_of_valid']:.2f} | "
                      f"{100 * c['share_of_gt_occupied']:.2f} | " +
                      " | ".join(f"{d['iou'][r][a]['iou']:.2f}" for a in d["arms"]) + " |")
        md.append("")
        names = [n for n in d["contrasts"] if n.startswith("waymo")]
        md.append("| contrast | " + " | ".join(d["composition"]) + " |")
        md.append("|---|" + "---|" * len(d["composition"]))
        for n in names:
            md.append(f"| {n} | " +
                      " | ".join(fmt(d["contrasts"][n][r]) for r in d["composition"]) + " |")
        md.append("")

    for k, d in R.items():
        for r in d["composition"]:
            for arm in d["arms"]:
                v = d["iou"][r][arm]
                rows.append({"dataset": k[0], "postprocess": k[1], "region": r, "arm": arm,
                             "n_samples": d["n_samples"],
                             "valid_voxels": d["composition"][r]["valid_voxels"],
                             "gt_occupied": d["composition"][r]["gt_occupied"],
                             **{q: v[q] for q in ("iou", "precision", "recall",
                                                  "tp", "fp", "fn")}})
            for n, c in d["contrasts"].items():
                x = c[r]
                rows.append({"dataset": k[0], "postprocess": k[1], "region": r,
                             "arm": f"CONTRAST:{n}", "n_samples": d["n_samples"],
                             "valid_voxels": d["composition"][r]["valid_voxels"],
                             "gt_occupied": d["composition"][r]["gt_occupied"],
                             "iou": x["delta"], "precision": x["ci_low"],
                             "recall": x["ci_high"],
                             "tp": int(x["significant_at_95"]), "fp": "", "fn": ""})
    csv_path = os.path.join(a.root, "metrics.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    md_path = os.path.join(a.root, "tables.md")
    open(md_path, "w").write("\n".join(md))
    print("\n".join(md))
    print(f"\n-> {md_path}\n-> {csv_path}  ({len(rows)} rows; for CONTRAST rows the "
          "iou/precision/recall columns hold delta/ci_low/ci_high and tp holds significance)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
