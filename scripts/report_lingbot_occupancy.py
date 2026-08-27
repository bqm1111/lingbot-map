#!/usr/bin/env python
"""Aggregate occupancy results into the tables the go/no-go decision needs.

Camera-only, one-prompt and repeated-prompt methods are reported in **separate
groups** and never merged into one ranking. Position ATE and orientation RMSE are
always printed together, because ATE alone is what hid the sliding-Sim(3) defect.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompted_lingbot.occupancy import accumulate_scores

GROUP_ORDER = ["camera_only", "one_prompt", "repeated_prompts", "diagnostic_non_causal"]
GROUP_TITLE = {
    "camera_only": "Camera-only (no metric observation)",
    "one_prompt": "One metric prompt",
    "repeated_prompts": "Repeated sparse prompts",
    "diagnostic_non_causal": "Non-causal diagnostics (NOT methods)",
}


def stat(vals):
    v = [x for x in vals if isinstance(x, (int, float)) and np.isfinite(x)]
    if not v:
        return {"mean": float("nan"), "median": float("nan"), "sd": 0.0, "n": 0}
    return {"mean": float(np.mean(v)), "median": float(np.median(v)),
            "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n": len(v)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--json-output", default=None)
    ap.add_argument("--histories", type=int, nargs="+", default=[1, 5, 20, 100])
    ap.add_argument("--title", default="SemanticKITTI seq 08")
    ap.add_argument("--anchor-note", default="every 5th frame (official SSC stride)")
    ap.add_argument("--baseline-config", default="B_first_depth_scale",
                    help="the strongest training-free reference for paired contrasts")
    args = ap.parse_args()

    rows = json.load(open(os.path.join(args.result_dir, "per_frame_rows.json")))
    seqs = json.load(open(os.path.join(args.result_dir, "per_sequence.json")))

    # pooled tp/fp/fn over every anchor frame, per (config, intrinsics, history)
    pooled = defaultdict(list)
    for r in rows:
        pooled[(r["config"], r["intrinsics"], r["history"])].append(r)

    per_seq = defaultdict(dict)
    for s in seqs:
        per_seq[(s["config"], s["intrinsics"])][s["sequence"]] = s

    groups = {}
    meta = {}
    for s in seqs:
        groups[s["config"]] = s["group"]
        meta.setdefault((s["config"], s["intrinsics"]), s)

    out, summary = [], {}
    out.append(f"# {args.title} — binary geometry occupancy\n")
    out.append(f"Anchor frames: {args.anchor_note}, "
               f"{sum(1 for r in rows if r['history'] == args.histories[0] and r['config'] == seqs[0]['config'] and r['intrinsics'] == seqs[0]['intrinsics'])} evaluated.\n")

    for grp in GROUP_ORDER:
        cfgs = sorted({c for c, g in groups.items() if g == grp})
        if not cfgs:
            continue
        out.append(f"\n## {GROUP_TITLE[grp]}\n")
        out.append("| config | intrinsics | " + " | ".join(f"IoU h{h}" for h in args.histories)
                   + " | P h20 | R h20 | ATE m | absRot° | rotHeld | pred vox h20 |")
        out.append("|---|---|" + "---|" * (len(args.histories) + 6))
        for cfg in cfgs:
            for intr in sorted({s["intrinsics"] for s in seqs if s["config"] == cfg}):
                ious = []
                for h in args.histories:
                    rs = pooled.get((cfg, intr, h), [])
                    ious.append(accumulate_scores(rs)["iou"] if rs else float("nan"))
                h20 = accumulate_scores(pooled.get((cfg, intr, 20), [])) if pooled.get((cfg, intr, 20)) else {}
                ss = [s for s in seqs if s["config"] == cfg and s["intrinsics"] == intr]
                ate = stat([s.get("ate_rmse_m") for s in ss])["mean"]
                rot = stat([s.get("abs_rot_rmse_deg") for s in ss])["mean"]
                held = stat([s.get("rotation_held_fraction") for s in ss])["mean"]
                out.append(f"| `{cfg}` | {intr} | "
                           + " | ".join(f"{v:.4f}" for v in ious)
                           + f" | {h20.get('precision', float('nan')):.3f}"
                           + f" | {h20.get('recall', float('nan')):.3f}"
                           + f" | {ate:.2f} | {rot:.2f} | {held:.2f}"
                           + f" | {h20.get('n_pred_occupied', 0):,} |")
                summary[f"{cfg}|{intr}"] = {
                    "group": grp,
                    **{f"iou_h{h}": v for h, v in zip(args.histories, ious)},
                    "precision_h20": h20.get("precision"), "recall_h20": h20.get("recall"),
                    "ate_rmse_m": ate, "abs_rot_rmse_deg": rot, "rotation_held_fraction": held,
                    "n_pred_occupied_h20": h20.get("n_pred_occupied"),
                    "n_gt_occupied_h20": h20.get("n_gt_occupied"),
                    "n_valid_voxels_h20": h20.get("n_valid_voxels"),
                }

    # per-sequence spread for the headline config
    out.append("\n## Per-chunk spread (history 20, predicted intrinsics)\n")
    out.append("| config | mean IoU | median | sd | n chunks |")
    out.append("|---|---|---|---|---|")
    for cfg in sorted(groups):
        vals = [accumulate_scores([r for r in rows
                                   if r["config"] == cfg and r["intrinsics"] == "predicted"
                                   and r["history"] == 20 and r["sequence"] == name])["iou"]
                for name in sorted({r["sequence"] for r in rows})]
        st = stat(vals)
        out.append(f"| `{cfg}` | {st['mean']:.4f} | {st['median']:.4f} | {st['sd']:.4f} | {st['n']} |")
        summary.setdefault(f"{cfg}|predicted", {}).update(per_chunk_h20=st)

    # paired contrast against the strongest training-free reference
    out.append(f"\n## Paired per-chunk contrast vs `{args.baseline_config}` (history 20)\n")
    out.append("| config | mean ΔIoU | sd | win rate | n |")
    out.append("|---|---|---|---|---|")
    names = sorted({r["sequence"] for r in rows})
    def chunk_iou(cfg, name):
        rs = [r for r in rows if r["config"] == cfg and r["intrinsics"] == "predicted"
              and r["history"] == 20 and r["sequence"] == name]
        return accumulate_scores(rs)["iou"] if rs else np.nan
    base = np.array([chunk_iou(args.baseline_config, n) for n in names])
    contrasts = {}
    for cfg in sorted(groups):
        if cfg == args.baseline_config:
            continue
        d = np.array([chunk_iou(cfg, n) for n in names]) - base
        d = d[np.isfinite(d)]
        if not d.size:
            continue
        contrasts[cfg] = {"mean": float(d.mean()),
                          "sd": float(d.std(ddof=1)) if d.size > 1 else 0.0,
                          "win_rate": float((d > 0).mean()), "n": int(d.size)}
        out.append(f"| `{cfg}` | {d.mean():+.4f} | {contrasts[cfg]['sd']:.4f} | "
                   f"{contrasts[cfg]['win_rate']:.0%} | {d.size} |")
    summary["_paired_vs_" + args.baseline_config] = contrasts

    # runtime
    out.append("\n## Runtime\n")
    ams = stat([s.get("anchor_ms_per_frame") for s in seqs])
    vms = stat([r.get("voxelize_ms") for r in rows if r["history"] == 20])
    lm = stat([s.get("lingbot_ms_per_frame") for s in seqs if s.get("lingbot_ms_per_frame")])
    out.append(f"* metric anchor update: **{ams['mean']:.3f} ms/frame**")
    out.append(f"* voxelize + score, history 20: **{vms['mean']:.1f} ms/anchor frame**")
    out.append(f"* frozen LingbotMap inference (from the cache manifest): **45.9 ms/frame**, "
               f"peak 11.0 GB")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    open(args.output, "w").write("\n".join(out) + "\n")
    if args.json_output:
        json.dump(summary, open(args.json_output, "w"), indent=1, default=float)
    print("\n".join(out))
    print("\nwrote", args.output)


if __name__ == "__main__":
    main()
