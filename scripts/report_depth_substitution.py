#!/usr/bin/env python
"""Aggregate the depth-substitution results into the go/no-go tables."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompted_lingbot.occupancy import accumulate_scores

ORDER = ["A_lingbot", "B_external_single", "B_external_single_conf",
         "C_hybrid", "C_hybrid_conf", "E_hybrid_predicted_K", "E_hybrid_focal_corr_K",
         "D_external_gtpose", "D_external_gtpose_conf"]
LABEL = {"A_lingbot": "LingbotMap depth + LingbotMap poses",
         "B_external_single": "external depth alone (1 frame)",
         "C_hybrid": "external depth + LingbotMap poses  ← THE EXPERIMENT",
         "E_hybrid_predicted_K": "hybrid, predicted intrinsics",
         "E_hybrid_focal_corr_K": "hybrid, focal-corrected intrinsics",
         "D_external_gtpose": "external depth + DATASET poses  (ORACLE)",
         "B_external_single_conf": "external depth alone, matched pixel support",
         "C_hybrid_conf": "hybrid, matched pixel support  ← THE EXPERIMENT",
         "D_external_gtpose_conf": "external + DATASET poses, matched support (ORACLE)"}


def bootstrap_ci(d, n=5000, seed=0):
    if len(d) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--json-output", default=None)
    ap.add_argument("--title", default="SemanticKITTI seq 08")
    ap.add_argument("--histories", type=int, nargs="+", default=[1, 5, 20, 100])
    args = ap.parse_args()

    rows = json.load(open(os.path.join(args.result_dir, "per_frame_rows.json")))
    seqs = json.load(open(os.path.join(args.result_dir, "per_sequence.json")))
    pooled = defaultdict(list)
    for r in rows:
        pooled[(r["config"], r["history"])].append(r)
    present = [c for c in ORDER if any(r["config"] == c for r in rows)]

    out, summary = [], {}
    out.append(f"# {args.title} — frozen-depth substitution\n")
    out.append("| config | " + " | ".join(f"IoU h{h}" for h in args.histories)
               + " | P@best | R@best | best IoU |")
    out.append("|---|" + "---|" * (len(args.histories) + 3))
    for c in present:
        ious, best, best_h = [], -1.0, None
        for h in args.histories:
            rs = pooled.get((c, h))
            v = accumulate_scores(rs)["iou"] if rs else float("nan")
            ious.append(v)
            if np.isfinite(v) and v > best:
                best, best_h = v, h
        b = accumulate_scores(pooled[(c, best_h)]) if best_h else {}
        out.append(f"| `{c}` — {LABEL.get(c,'')} | "
                   + " | ".join("--" if not np.isfinite(v) else f"{v:.4f}" for v in ious)
                   + f" | {b.get('precision', float('nan')):.3f}"
                   + f" | {b.get('recall', float('nan')):.3f} | **{best:.4f}** (h={best_h}) |")
        summary[c] = {**{f"iou_h{h}": v for h, v in zip(args.histories, ious)},
                      "best_iou": best, "best_history": best_h,
                      "precision": b.get("precision"), "recall": b.get("recall"),
                      "n_pred_occupied": b.get("n_pred_occupied"),
                      "n_valid_voxels": b.get("n_valid_voxels")}

    # per-sequence paired contrasts at the common best history
    names = sorted({r["sequence"] for r in rows})
    def per_seq_iou(c, h):
        return np.array([accumulate_scores([r for r in rows if r["config"] == c
                                            and r["history"] == h and r["sequence"] == n])["iou"]
                         if any(r["config"] == c and r["history"] == h and r["sequence"] == n
                                for r in rows) else np.nan for n in names])

    H = max(args.histories)
    out.append(f"\n## Paired per-sequence contrasts (history {H}, n={len(names)})\n")
    out.append("| comparison | mean ΔIoU | sd | win rate | 95% CI |")
    out.append("|---|---|---|---|---|")
    contrasts = {}
    base_pairs = [("C_hybrid_conf", "A_lingbot"), ("C_hybrid_conf", "B_external_single_conf"),
                  ("C_hybrid", "A_lingbot"), ("C_hybrid", "B_external_single"),
                  ("D_external_gtpose_conf", "C_hybrid_conf"),
                  ("D_external_gtpose", "C_hybrid"),
                  ("E_hybrid_focal_corr_K", "C_hybrid_conf")]
    for a, b in base_pairs:
        if a not in present or b not in present:
            continue
        hb = 1 if b.startswith("B_external_single") else H
        d = per_seq_iou(a, H) - per_seq_iou(b, hb)
        d = d[np.isfinite(d)]
        if not d.size:
            continue
        ci = bootstrap_ci(d)
        contrasts[f"{a}__minus__{b}"] = {
            "mean": float(d.mean()), "sd": float(d.std(ddof=1)) if d.size > 1 else 0.0,
            "win_rate": float((d > 0).mean()), "n": int(d.size), "ci95": ci}
        out.append(f"| `{a}` − `{b}` | {d.mean():+.4f} | "
                   f"{contrasts[f'{a}__minus__{b}']['sd']:.4f} | "
                   f"{(d > 0).mean():.0%} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |")
    summary["_contrasts"] = contrasts

    out.append("\n## Pose vs depth diagnostic\n")
    for dcfg, ccfg, bcfg, tag in (
            ("D_external_gtpose_conf", "C_hybrid_conf", "B_external_single_conf", "matched support"),
            ("D_external_gtpose", "C_hybrid", "B_external_single", "native support")):
      if dcfg in present and ccfg in present:
        out.append(f"\n**{tag}**\n")
        summary.setdefault("_diagnostic", {})
        d_iou = summary[dcfg]["best_iou"]
        c_iou = summary[ccfg]["best_iou"]
        b_iou = summary.get(bcfg, {}).get("best_iou", float("nan"))
        out.append(f"* external depth + dataset poses (oracle): **{d_iou:.4f}**")
        out.append(f"* external depth + LingbotMap poses: **{c_iou:.4f}**")
        out.append(f"* external depth alone, single frame: **{b_iou:.4f}**")
        out.append(f"* headroom attributable to LingbotMap pose error: **{d_iou - c_iou:+.4f}**")
        out.append(f"* value added by temporal aggregation with LingbotMap poses: "
                   f"**{c_iou - b_iou:+.4f}**")
        summary["_diagnostic"][tag] = {"pose_headroom": d_iou - c_iou,
                                       "aggregation_value": c_iou - b_iou}

    out.append("\n## Runtime\n")
    def st(k, src):
        v = [r.get(k) for r in src if isinstance(r.get(k), (int, float)) and np.isfinite(r.get(k))]
        return float(np.mean(v)) if v else float("nan")
    out.append(f"* causal translation-scale estimate: **{st('external_scale_ms_per_frame', seqs):.3f} ms/frame**")
    out.append(f"* voxelize + score: **{st('eval_ms', rows):.1f} ms/anchor frame**")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    open(args.output, "w").write("\n".join(out) + "\n")
    if args.json_output:
        json.dump(summary, open(args.json_output, "w"), indent=1, default=float)
    print("\n".join(out))


if __name__ == "__main__":
    main()
