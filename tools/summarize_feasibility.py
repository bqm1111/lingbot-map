#!/usr/bin/env python
"""Derive the go/no-go comparisons from a finished feasibility run.

    python tools/summarize_feasibility.py --results output/semantic_sidecar/reports/feasibility_results.json

Prints (and optionally writes) the four contrasts the decision actually rests on:
consensus vs. pixelwise targets, MLP vs. linear, sparse vs. full teacher, and how
close the teacher-free sidecar gets to the teacher it was distilled from.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np

import _bootstrap  # noqa: F401

from semantic_sidecar.metrics import format_table

METRICS = ("end_to_end_miou", "matched_miou", "mean_accuracy", "coverage",
           "cross_view_consistency", "cross_view_consistency_heldout", "embedding_diversity")


def _get(raw: Dict[str, dict], name: str, metric: str) -> float:
    entry = raw.get(name)
    if entry is None:
        return float("nan")
    return float(entry.get(metric, float("nan")))


def _delta_rows(raw: Dict[str, dict], pairs: List[tuple], label: str) -> List[Dict[str, object]]:
    rows = []
    for better, worse, note in pairs:
        if better not in raw or worse not in raw:
            continue
        row: Dict[str, object] = {"contrast": f"{better} - {worse}", "what it tests": note}
        for metric in ("matched_miou", "end_to_end_miou", "cross_view_consistency_heldout"):
            row[metric] = _get(raw, better, metric) - _get(raw, worse, metric)
        rows.append(row)
    _ = label
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--output", default=None, help="write the markdown to this path")
    args = ap.parse_args()

    with open(args.results) as fh:
        payload = json.load(fh)
    raw: Dict[str, dict] = payload["raw"]

    sections: List[str] = []

    # -- headline table ---------------------------------------------------- #
    headline = []
    for name, res in raw.items():
        headline.append({
            "model": name,
            "teacher@infer": "yes" if res.get("needs_teacher_at_inference") else "no",
            **{m: float(res.get(m, float("nan"))) for m in METRICS},
        })
    headline.sort(key=lambda r: (-r["matched_miou"] if not np.isnan(r["matched_miou"]) else 0))
    sections.append("### All models\n\n" + format_table(headline))

    # -- the four decision contrasts --------------------------------------- #
    contrasts = _delta_rows(raw, [
        ("linear_consensus_d100", "linear_pixel_d100", "3D consensus vs pixelwise (linear)"),
        ("mlp_consensus_d100", "mlp_pixel_d100", "3D consensus vs pixelwise (MLP)"),
        ("mlp_consensus_d100", "linear_consensus_d100", "MLP capacity over a linear bridge"),
        ("mlp_consensus_mv_d100", "mlp_consensus_d100", "adding cross-view consistency"),
        ("mlp_full_d100", "mlp_consensus_mv_d100", "adding relational preservation"),
        ("mlp_full_center_d100", "mlp_full_d100", "adding the mean-removed cosine"),
        ("mlp_consensus_d025", "mlp_consensus_d100", "25% teacher vs 100% teacher"),
        ("linear_consensus_d100", "lingbot_raw", "does the bridge add text alignment at all"),
    ], "contrasts")
    sections.append("### Decision contrasts (positive = the first model wins)\n\n" + format_table(contrasts))

    # -- fraction of the teacher recovered --------------------------------- #
    ceiling_rows = []
    for ceiling in ("teacher_direct", "teacher_fullres"):
        base = _get(raw, ceiling, "matched_miou")
        if np.isnan(base) or base <= 0:
            continue
        for name in raw:
            if name.startswith("teacher_") or name == "lingbot_raw":
                continue
            ceiling_rows.append({
                "model": name,
                "ceiling": ceiling,
                "matched mIoU": _get(raw, name, "matched_miou"),
                "ceiling mIoU": base,
                "% of ceiling": 100.0 * _get(raw, name, "matched_miou") / base,
            })
    sections.append("### Fraction of the frozen teacher recovered without it at inference\n\n"
                    + format_table(ceiling_rows))

    # -- geometry preservation --------------------------------------------- #
    fingerprints = payload.get("geometry_fingerprints", {})
    unique = {f for f in fingerprints.values() if f}
    sections.append(
        "### Geometry preservation\n\n"
        f"- identical across all models: **{payload.get('geometry_identical_across_models')}**\n"
        f"- distinct fingerprints observed: {len(unique)}\n"
        f"- fingerprint: `{next(iter(unique), 'n/a')[:32]}`\n"
    )

    text = "\n\n".join(sections)
    print(text)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as fh:
            fh.write(text + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
