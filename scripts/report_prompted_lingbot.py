#!/usr/bin/env python
"""Turn evaluation rows into the go/no-go tables in summary.md."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CAUSAL_ORDER = ["raw", "first_depth_scale", "running_depth_scale", "causal_pose_sim3",
                "sliding_pose_sim3", "depth_scale_pose_se3", "learned_corrector"]
ORACLES = ["offline_oracle_sim3", "per_frame_oracle_scale"]


def load(paths):
    rows = []
    for p in paths:
        rows.extend(json.load(open(p)))
    return [r for r in rows if "error" not in r]


def cell(rows, key):
    v = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and np.isfinite(r[key])]
    return float(np.mean(v)) if v else float("nan")


def sd(rows, key):
    v = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and np.isfinite(r[key])]
    return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def table(rows, configs, metrics, anchors, dataset=None, fmt="{:.3f}"):
    sel = [r for r in rows if (dataset is None or r["dataset"] == dataset)]
    g = defaultdict(list)
    for r in sel:
        g[(r["config"], r["anchor"])].append(r)
    have = {a for (_, a) in g}
    anchors = [a for a in anchors if a in have]
    lines = ["| config | anchor | " + " | ".join(metrics) + " |",
             "|---|---|" + "---|" * len(metrics)]
    for c in configs:
        for a in anchors:
            rs = g.get((c, a))
            if not rs:
                continue
            mark = " *" if a in ORACLES else ""
            vals = []
            for m in metrics:
                v = cell(rs, m)
                vals.append("--" if not np.isfinite(v) else fmt.format(v))
            lines.append(f"| {c} | `{a}`{mark} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def paired_delta(rows, config, a_name, b_name, key):
    """Per-sequence paired difference b - a (negative = b better)."""
    by = defaultdict(dict)
    for r in rows:
        if r["config"] == config and r["anchor"] in (a_name, b_name):
            by[r["sequence"]][r["anchor"]] = r.get(key)
    d = [by[s][b_name] - by[s][a_name] for s in by
         if a_name in by[s] and b_name in by[s]
         and isinstance(by[s].get(a_name), (int, float))
         and isinstance(by[s].get(b_name), (int, float))
         and np.isfinite(by[s][a_name]) and np.isfinite(by[s][b_name])]
    if not d:
        return None
    d = np.asarray(d, float)
    return {"n": len(d), "mean": float(d.mean()),
            "sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
            "win_rate": float((d < 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--json-output", default=None)
    args = ap.parse_args()
    rows = load(args.rows)
    out = {}

    order = CAUSAL_ORDER + ORACLES
    key_configs = ["rgb_only", "depth_first_only_clean", "depth_k30_clean", "pose_k30_clean",
                   "both_k30_clean", "both_k30_moderate", "both_k30_heavy"]
    out["main_table"] = table(
        rows, key_configs,
        ["ate_rmse_m", "ate_se3_rmse_m", "ate_sim3_rmse_m", "abs_rel", "rmse",
         "rpe_trans_pct", "update_ms_per_frame"], order)

    out["interval_table"] = table(
        rows, [f"{m}_k{k}_clean" for m in ("depth", "pose", "both") for k in (1, 5, 10, 30, 100)],
        ["ate_rmse_m", "abs_rel"], CAUSAL_ORDER)

    out["point_cloud_table"] = table(
        rows, ["rgb_only", "depth_k30_clean", "both_k30_clean", "both_k30_moderate"],
        ["acc_mean_m", "comp_mean_m", "f1_at_0.25", "f1_at_0.5"], order)

    # head-to-head: learned vs the strongest causal training-free anchor
    contrasts = {}
    for cfg in key_configs + ["both_k100_clean", "both_k5_clean"]:
        for metric in ("ate_rmse_m", "abs_rel"):
            d = paired_delta(rows, cfg, "depth_scale_pose_se3", "learned_corrector", metric)
            if d:
                contrasts[f"{cfg}::{metric}"] = d
    out["learned_vs_best_baseline"] = contrasts

    firstonly = {}
    for metric in ("ate_rmse_m", "abs_rel"):
        for cfg in ["depth_k30_clean", "depth_k5_clean", "both_k30_clean"]:
            d = paired_delta(rows, cfg, "first_depth_scale", "running_depth_scale", metric)
            if d:
                firstonly[f"{cfg}::{metric}"] = d
    out["running_vs_first_depth_scale"] = firstonly

    text = ["## Main table\n", out["main_table"],
            "\n\n(`*` = non-causal diagnostic, not a method)\n",
            "\n## Error vs prompt interval\n", out["interval_table"],
            "\n\n## Point cloud (no scale alignment)\n", out["point_cloud_table"],
            "\n\n## Learned corrector vs `depth_scale_pose_se3` (paired, per sequence)\n",
            "| comparison | n | mean delta | sd | win rate |", "|---|---|---|---|---|"]
    for k, v in contrasts.items():
        text.append(f"| {k} | {v['n']} | {v['mean']:+.4f} | {v['sd']:.4f} | {v['win_rate']:.0%} |")
    text += ["\n## Running vs first-only depth scale (paired)\n",
             "| comparison | n | mean delta | sd | win rate |", "|---|---|---|---|---|"]
    for k, v in firstonly.items():
        text.append(f"| {k} | {v['n']} | {v['mean']:+.4f} | {v['sd']:.4f} | {v['win_rate']:.0%} |")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    open(args.output, "w").write("\n".join(text) + "\n")
    if args.json_output:
        json.dump(out, open(args.json_output, "w"), indent=1)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
