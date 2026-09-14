"""Render the source-side deliverables: source_results.md, tables/source_per_seed.csv, tables/source_per_range.csv, figures."""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from tools.baseline_reconstruction import config as C
from tools.baseline_reconstruction.current.evaluate_source import REGIONS
from tools.baseline_reconstruction.metrics.occupancy import seed_statistics
from tools.baseline_reconstruction.reports.tables import markdown_table, pct, write_csv


def load_final_evals(paths: C.CurrentPaths) -> Dict[int, Dict]:
    out = {}
    for s in C.SEEDS:
        p = os.path.join(paths.source_eval, f"geo_seed{s}_step06000.json")
        if os.path.exists(p):
            out[s] = json.load(open(p))
    return out


def load_curve_evals(paths: C.CurrentPaths) -> Dict[int, Dict[int, float]]:
    """seed -> step -> full IoU at stride 5."""
    out: Dict[int, Dict[int, float]] = {}
    for p in sorted(glob.glob(os.path.join(paths.source_eval, "geo_seed*_step*_stride5.json"))):
        d = json.load(open(p)); out.setdefault(d["seed"], {})[d["step"]] = d["summary"]["full"]["iou"]
    return out


def reproduction_verdict(finals: Dict[int, Dict]) -> Dict:
    """Predeclared check against Gate 13 M (config.GATE13_M_FULL_IOU, config.SOURCE_REPRODUCTION_TOLERANCE)."""
    ours = {s: finals[s]["summary"]["full"]["iou"] for s in finals}
    ref = C.GATE13_M_FULL_IOU; tol = C.SOURCE_REPRODUCTION_TOLERANCE
    per_seed = {s: abs(ours[s] - ref[s]) for s in ours}
    mean_diff = abs(np.mean(list(ours.values())) - np.mean([ref[s] for s in ours]))
    return {"ours_full_iou": ours, "gate13_M_full_iou": {s: ref[s] for s in ours}, "per_seed_abs_diff": per_seed,
            "mean_abs_diff": float(mean_diff), "tolerance": tol,
            "pass": bool(mean_diff <= tol["mean_full_iou_abs"] and all(d <= tol["per_seed_full_iou_abs"] for d in per_seed.values()))}


def training_curves(paths: C.CurrentPaths) -> Dict[int, List[Tuple[int, float, float]]]:
    """seed -> [(step, train loss (50-step mean), val loss)] from the run records."""
    out = {}
    for s in C.SEEDS:
        p = os.path.join(paths.runs, f"train_geo_seed{s}.json")
        if not os.path.exists(p):
            continue
        log = json.load(open(p))["log"]
        train = [(m["step"], m["loss"]) for m in log if "val" not in m]
        val = {m["val"]["step"]: m["val"]["loss"] for m in log if "val" in m}
        rows = []
        for step in sorted(val):
            recent = [l for st, l in train if step - 50 < st <= step]
            rows.append((step, float(np.mean(recent)) if recent else float("nan"), val[step]))
        out[s] = rows
    return out


def write_figures(paths: C.CurrentPaths, curve: Dict[int, Dict[int, float]], tcurves) -> List[str]:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = os.path.join(C.ARTIFACT_DIR, "figures"); os.makedirs(fig_dir, exist_ok=True); out = []
    if curve:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        for s, pts in sorted(curve.items()):
            xs = sorted(pts); ax.plot(xs, [100 * pts[x] for x in xs], marker="o", label=f"seed {s}")
        ax.axhline(100 * np.mean(list(C.GATE13_M_FULL_IOU.values())), color="k", ls="--", lw=1, label="Gate 13 M (mean, full 590)")
        ax.set_xlabel("training step"); ax.set_ylabel("0006 full-region SC IoU % (stride-5 anchors)"); ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); p = os.path.join(fig_dir, "source_learning_curve_iou.png"); fig.savefig(p, dpi=150); plt.close(fig); out.append(p)
    if tcurves:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        for s, rows in sorted(tcurves.items()):
            ax.plot([r[0] for r in rows], [r[1] for r in rows], label=f"train seed {s}")
            ax.plot([r[0] for r in rows], [r[2] for r in rows], ls="--", label=f"val seed {s}")
        ax.set_xlabel("training step"); ax.set_ylabel("focal + Dice loss"); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3)
        fig.tight_layout(); p = os.path.join(fig_dir, "source_training_loss.png"); fig.savefig(p, dpi=150); plt.close(fig); out.append(p)
    return out


def write_source_results(paths: C.CurrentPaths = C.CurrentPaths()) -> Dict:
    finals = load_final_evals(paths); curve = load_curve_evals(paths); tcurves = training_curves(paths)
    assert finals, "no final evaluations found"
    verdict = reproduction_verdict(finals)
    tables_dir = os.path.join(C.ARTIFACT_DIR, "tables")
    # ---- per seed ------------------------------------------------------------------
    per_seed_rows = []
    for s, d in sorted(finals.items()):
        for r in REGIONS:
            x = d["summary"][r]
            per_seed_rows.append({"seed": s, "region": r, "iou": x["iou"], "precision": x["precision"], "recall": x["recall"],
                                  "tp": x["tp"], "fp": x["fp"], "fn": x["fn"], "n_gt": x["n_gt"], "n_anchors": d["n_anchors"],
                                  "checkpoint": d["checkpoint"], "checkpoint_sha256": d["checkpoint_sha256"]})
        m = d["map_only"]["full"]
        per_seed_rows.append({"seed": s, "region": "map_only_full", "iou": m["iou"], "precision": m["precision"], "recall": m["recall"],
                              "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "n_gt": m["n_gt"], "n_anchors": d["n_anchors"], "checkpoint": "", "checkpoint_sha256": ""})
    write_csv(os.path.join(tables_dir, "source_per_seed.csv"), per_seed_rows)
    # ---- per range -----------------------------------------------------------------
    range_rows = []
    for r in REGIONS:
        for name in list(C.RANGE_NAMES) + ["all"]:
            vals = {s: (finals[s]["summary"][r]["by_range"][name] if name != "all" else finals[s]["summary"][r]) for s in finals}
            st = seed_statistics([v["iou"] for v in vals.values()])
            range_rows.append({"region": r, "range": name, "iou_mean": st["mean"], "iou_median": st["median"], "iou_sd": st["sd"],
                               "precision_mean": float(np.mean([v["precision"] for v in vals.values()])),
                               "recall_mean": float(np.mean([v["recall"] for v in vals.values()])),
                               "iou_per_seed": " / ".join(f"{vals[s]['iou']:.4f}" for s in sorted(vals)), "n_gt": vals[min(vals)]["n_gt"]})
    write_csv(os.path.join(tables_dir, "source_per_range.csv"), range_rows)
    figs = write_figures(paths, curve, tcurves)
    json.dump({"reproduction": verdict, "learning_curve_full_iou_stride5": curve}, open(os.path.join(paths.source_eval, "source_summary.json"), "w"), indent=1)
    # ---- markdown ------------------------------------------------------------------
    full = {s: finals[s]["summary"]["full"] for s in finals}; st = seed_statistics([v["iou"] for v in full.values()])
    md = [f"# Source results -- {C.CURRENT_BASELINE_NAME}\n",
          f"KITTI-360 0006, {finals[min(finals)]['n_anchors']} anchors, final-step (6000) checkpoints, occupied := final log-odds > {C.OCCUPANCY_THRESHOLD}. "
          "Regions and range bins as in Gates 12/13.\n",
          "## Headline (full region)\n",
          markdown_table(["seed", "SC IoU %", "precision %", "recall %", "map-only IoU %", "checkpoint sha256"],
                         [[s, pct(full[s]["iou"]), pct(full[s]["precision"]), pct(full[s]["recall"]), pct(finals[s]["map_only"]["full"]["iou"]),
                           finals[s]["checkpoint_sha256"][:12]] for s in sorted(full)] +
                         [["**mean**", f"**{100*st['mean']:.2f}**", pct(np.mean([v['precision'] for v in full.values()])), pct(np.mean([v['recall'] for v in full.values()])), "", ""],
                          ["median", f"{100*st['median']:.2f}", "", "", "", ""], ["sd", f"{100*st['sd']:.2f}", "", "", "", ""]]),
          "\n## Reproduction of Gate 13 M (predeclared tolerance)\n",
          markdown_table(["seed", "this run", "Gate 13 M", "abs diff", "tolerance"],
                         [[s, f"{verdict['ours_full_iou'][s]:.4f}", f"{verdict['gate13_M_full_iou'][s]:.4f}", f"{verdict['per_seed_abs_diff'][s]:.4f}",
                           verdict["tolerance"]["per_seed_full_iou_abs"]] for s in sorted(verdict["ours_full_iou"])] +
                         [["mean", f"{np.mean(list(verdict['ours_full_iou'].values())):.4f}", f"{np.mean(list(verdict['gate13_M_full_iou'].values())):.4f}",
                           f"{verdict['mean_abs_diff']:.4f}", verdict["tolerance"]["mean_full_iou_abs"]]]),
          f"\n**Verdict: {'PASS -- the current pipeline reproduces the Gate 13 M regime' if verdict['pass'] else 'FAIL -- STOP: diagnose before any target evaluation'}.**\n",
          "## By region and range (mean over seeds; per-seed in tables/source_per_range.csv)\n",
          markdown_table(["region", "range", "IoU % mean", "median", "sd", "P % mean", "R % mean", "GT occupied voxels"],
                         [[r["region"], r["range"], pct(r["iou_mean"]), pct(r["iou_median"]), pct(r["iou_sd"]), pct(r["precision_mean"]), pct(r["recall_mean"]), f"{r['n_gt']:,}"]
                          for r in range_rows if r["range"] == "all" or r["region"] in ("full", "past5", "unobs5_later")]),
          "\n`past5` = observed by the five causal sweeps; `unobs5_later` = unobserved by them, resolved by the future sweeps (the completion region).\n",
          "## Learning curves\n",
          markdown_table(["step"] + [f"seed {s} IoU %" for s in sorted(curve)] + ["mean"],
                         [[step] + [pct(curve[s].get(step, float('nan'))) for s in sorted(curve)] + [pct(np.mean([curve[s][step] for s in curve if step in curve[s]]))]
                          for step in sorted({st_ for s in curve for st_ in curve[s]})]) if curve else "(no curve evaluations)\n",
          "\nFigures: " + ", ".join(os.path.relpath(f, C.REPO_ROOT) for f in figs) + "\n"]
    out = os.path.join(C.ARTIFACT_DIR, "source_results.md"); open(out, "w").write("\n".join(md))
    return {"path": out, "verdict": verdict}


__all__ = ["load_final_evals", "load_curve_evals", "reproduction_verdict", "write_source_results"]
