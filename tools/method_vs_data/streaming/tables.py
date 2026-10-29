#!/usr/bin/env python
"""Render every table of the streaming reports from the score JSONs, so nothing is hand-typed.

Writes ``tables.md`` and ``metrics.csv`` next to the score files. Per-seed rows and the
three-seed mean/spread are emitted as SEPARATE blocks and never merged with an
evaluation-sample confidence interval, because they are different quantities: the interval
describes resampling anchors at one seed, the spread describes retraining.

    PYTHONPATH=$PWD python tools/method_vs_data/streaming/tables.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.method_vs_data.streaming import config as SV                                 # noqa: E402

SIG = lambda c: " *" if c["significant_at_95"] else " ns"                       # noqa: E731
F = lambda x: ("%+.2f" % x) if x is not None else "--"                          # noqa: E731


def cell(c: Dict) -> str:
    return f"{c['delta']:+.2f} [{c['ci_low']:+.2f}, {c['ci_high']:+.2f}]{SIG(c)}"


def load(dataset: str) -> Optional[dict]:
    p = os.path.join(SV.SCORE_DIR, f"streaming_{dataset}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def headline(r: dict) -> List[str]:
    out = [f"### {r['dataset']} — global SC IoU, {r['n_anchors']} anchors "
           f"(canonical population {r['canonical_population']}, exclusions "
           f"{len(r['anchors_dropped'])})", ""]
    arms = r["arms"]
    ours = [a for a in arms if ":" in a]
    occ = [a for a in arms if ":" not in a]
    out += ["| arm | raw IoU | raw P | raw R | pooled IoU | pooled P | pooled R |",
            "|---|---|---|---|---|---|---|"]
    for a in ours + occ:
        x, y = r["iou"]["raw"]["all"][a], r["iou"]["pooled"]["all"][a]
        name = a.replace("A:", "clip-reset / ").replace("B:", "streaming / ")
        out.append(f"| {name} | {x['iou']:.2f} | {x['precision']:.2f} | {x['recall']:.2f} "
                   f"| {y['iou']:.2f} | {y['precision']:.2f} | {y['recall']:.2f} |")
    return out + [""]


def ab_table(r: dict) -> List[str]:
    out = ["### Streaming minus clip-reset, within each source and seed", "",
           "Paired block bootstrap, 2,000 resamples, seed 0. `*` = interval excludes zero; "
           "`ns` = no difference detected, which is **not** demonstrated equivalence.", "",
           "| source | seed | raw Δ IoU [95% CI] | pooled Δ IoU [95% CI] |",
           "|---|---|---|---|"]
    for src in r["sources"] + ["map_only"]:
        seeds = [None] if src == "map_only" else r["seeds"]
        for s in seeds:
            key = (f"streaming_minus_clipreset__map_only" if src == "map_only"
                   else f"streaming_minus_clipreset__{src}_s{s}")
            if key not in r["contrasts"]:
                continue
            c = r["contrasts"][key]
            lab = "map-only (no head)" if src == "map_only" else src
            out.append(f"| {lab} | {'—' if s is None else s} | "
                       f"{cell(c['raw']['all'])} | {cell(c['pooled']['all'])} |")
    return out + [""]


def seed_block(r: dict) -> List[str]:
    out = ["### Per-seed IoU and the three-seed spread (a different quantity from the CI)", "",
           "| source | mode | seed 0 | seed 1 | seed 2 | mean | max−min |",
           "|---|---|---|---|---|---|---|"]
    for post in ("raw", "pooled"):
        out.append(f"| **{post}** | | | | | | |")
        for src in r["sources"]:
            for m in r["modes"]:
                v = [r["iou"][post]["all"].get(f"{m}:{src}_s{s}", {}).get("iou")
                     for s in r["seeds"]]
                v = [x for x in v if x is not None]
                if not v:
                    continue
                name = "clip-reset" if m == "A" else "streaming"
                out.append(f"| {src} | {name} | " + " | ".join(f"{x:.2f}" for x in v)
                           + f" | {np.mean(v):.2f} | {max(v)-min(v):.2f} |")
    return out + [""]


def source_table(r: dict) -> List[str]:
    out = ["### Source differences, held at one fixed inference mode", "",
           "| contrast | mode | seed | raw Δ IoU [95% CI] | pooled Δ IoU [95% CI] |",
           "|---|---|---|---|---|"]
    for a, b in (("k360", "waymo"), ("k360", "once"), ("waymo", "once")):
        for m in r["modes"]:
            for s in r["seeds"]:
                k = f"source__{a}_vs_{b}__mode{m}_s{s}"
                if k not in r["contrasts"]:
                    continue
                c = r["contrasts"][k]
                out.append(f"| {a} − {b} | {'clip-reset' if m=='A' else 'streaming'} | {s} "
                           f"| {cell(c['raw']['all'])} | {cell(c['pooled']['all'])} |")
    return out + [""]


def maponly_table(r: dict) -> List[str]:
    out = ["### Completion minus map-only, under each mode", "",
           "| source | seed | mode | raw Δ IoU [95% CI] | pooled Δ IoU [95% CI] |",
           "|---|---|---|---|---|"]
    for src in r["sources"]:
        for s in r["seeds"]:
            for m in r["modes"]:
                k = f"completion_minus_maponly__{src}_s{s}__mode{m}"
                if k not in r["contrasts"]:
                    continue
                c = r["contrasts"][k]
                out.append(f"| {src} | {s} | {'clip-reset' if m=='A' else 'streaming'} "
                           f"| {cell(c['raw']['all'])} | {cell(c['pooled']['all'])} |")
    return out + [""]


def occany_table(r: dict) -> List[str]:
    out = ["### Against the fixed OccAny comparators (seed 0)", "",
           "OccAny's forward configuration consumes **four future frames (~4 s)**; both of our "
           "modes are causal and consume none. These rows are not temporally matched and are "
           "never described as such.", "",
           "| comparator | source | mode | raw Δ IoU [95% CI] | pooled Δ IoU [95% CI] |",
           "|---|---|---|---|---|"]
    for oc in ("occany_1f", "occany_5f_fwd"):
        for src in r["sources"]:
            for m in r["modes"]:
                k = f"vs_{oc}__{src}_s0__mode{m}"
                if k not in r["contrasts"]:
                    continue
                c = r["contrasts"][k]
                lab = "causal 1-frame" if oc.endswith("1f") else "forward 5-frame"
                out.append(f"| {lab} | {src} | {'clip-reset' if m=='A' else 'streaming'} "
                           f"| {cell(c['raw']['all'])} | {cell(c['pooled']['all'])} |")
    return out + [""]


def frustum_table(r: dict) -> List[str]:
    out = ["### Frustum regions (global map-evidence-free cut)", "",
           "| region | share of valid | share of GT occupied |", "|---|---|---|"]
    for k in ("all", "in_frustum", "out_frustum"):
        c = r["composition"][k]
        out.append(f"| {k} | {100*c['share_of_valid']:.2f}% | "
                   f"{100*c['share_of_gt_occupied']:.2f}% |")
    out += ["", "| arm | in-frustum raw | out-frustum raw | in-frustum pooled | out-frustum pooled |",
            "|---|---|---|---|---|"]
    for a in r["arms"]:
        name = a.replace("A:", "clip-reset / ").replace("B:", "streaming / ")
        out.append(f"| {name} | " + " | ".join(
            f"{r['iou'][p][k][a]['iou']:.2f}"
            for p in ("raw", "pooled") for k in ("in_frustum", "out_frustum")) + " |")
    return out + [""]


def sensitivity_table(r: dict) -> List[str]:
    if "sk_stream_sensitivity" not in r:
        return []
    s = r["sk_stream_sensitivity"]["pooled_all_region"]
    out = ["### SemanticKITTI block-length sensitivity (one physical stream)", "",
           "SemanticKITTI is a single stream, so no block length turns these into "
           "between-stream intervals. The widening shows how much the canonical 25-anchor "
           "block understates within-stream correlation; the 801 row is one block and its "
           "interval is degenerate.", "",
           "| contrast | " + " | ".join(f"L={L}" for L in ("25", "50", "100", "200", "400", "801"))
           + " |", "|---|" + "---|" * 6]
    for name, rows in s.items():
        out.append(f"| {name.replace('streaming_minus_clipreset__','')} | " + " | ".join(
            f"{rows[L]['ci_low']:+.2f},{rows[L]['ci_high']:+.2f}{'*' if rows[L]['significant_at_95'] else ''}"
            for L in ("25", "50", "100", "200", "400", "801")) + " |")
    return out + [""]


def write_csv(path: str, recs: Dict[str, dict]) -> int:
    n = 0
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "kind", "arm_or_contrast", "mode", "source", "seed",
                    "postprocess", "region", "iou", "precision", "recall", "tp", "fp", "fn",
                    "delta", "ci_low", "ci_high", "significant_at_95"])
        for ds, r in recs.items():
            for post in ("raw", "pooled"):
                for k in r["iou"][post]:
                    for a, v in r["iou"][post][k].items():
                        m = a.split(":")[0] if ":" in a else ""
                        rest = a.split(":")[1] if ":" in a else a
                        src = rest.split("_s")[0]
                        sd = rest.split("_s")[1] if "_s" in rest else ""
                        w.writerow([ds, "iou", a, m, src, sd, post, k, f"{v['iou']:.6f}",
                                    f"{v['precision']:.6f}", f"{v['recall']:.6f}",
                                    v["tp"], v["fp"], v["fn"], "", "", "", ""])
                        n += 1
            for name, c in r["contrasts"].items():
                for post in ("raw", "pooled"):
                    for k, v in c[post].items():
                        w.writerow([ds, "contrast", name, "", "", "", post, k, "", "", "",
                                    "", "", "", f"{v['delta']:.6f}", f"{v['ci_low']:.6f}",
                                    f"{v['ci_high']:.6f}", int(v["significant_at_95"])])
                        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=list(SV.DATASETS))
    a = ap.parse_args()
    recs = {d: load(d) for d in a.datasets}
    recs = {d: r for d, r in recs.items() if r}
    assert recs, "no score file found"
    lines = ["# Streaming-validation accuracy tables (generated — do not hand-edit)", "",
             f"Source: `{SV.SCORE_DIR}`", ""]
    for d, r in recs.items():
        lines += [f"## {d}", ""]
        lines += headline(r) + ab_table(r) + seed_block(r) + source_table(r)
        lines += maponly_table(r) + occany_table(r) + frustum_table(r) + sensitivity_table(r)
    p = os.path.join(SV.SCORE_DIR, "tables.md")
    open(p, "w").write("\n".join(lines) + "\n")
    c = os.path.join(SV.SCORE_DIR, "metrics.csv")
    n = write_csv(c, recs)
    print(f"-> {p}\n-> {c} ({n} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
