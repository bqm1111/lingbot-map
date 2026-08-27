#!/usr/bin/env python
"""Collect Gate 1 results into metrics.json, per_class.csv and the gate decision."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from typing import Dict, List

from semantic_sidecar.config import write_json
from semantic.eval_semantickitti import CLASS_PROMPTS

MAPS = "research/sem_bypass/outputs/maps"
RUNS = "research/sem_bypass/outputs/runs"
CLASSES = ["car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist",
           "motorcyclist", "road", "parking", "sidewalk", "other-ground", "building",
           "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign"]
#: Classes that dominate a driving benchmark; used to check the mean is not carried by them.
DOMINANT = {"road", "building", "car", "vegetation", "sidewalk"}
#: Pre-registered Gate 1 thresholds.
RETENTION_MIN, ABS_GAP_MAX, STD_MAX = 0.90, 1.5, 0.5


def load(name: str) -> Dict:
    p = os.path.join(MAPS, name, "eval.json")
    return json.load(open(p)) if os.path.exists(p) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--output-dir", default="research/sem_bypass/outputs/gate1")
    a = ap.parse_args()

    teacher = load("teacher_direct")
    if teacher is None:
        raise SystemExit("teacher_direct eval missing")
    t_miou = teacher["matched_miou"]

    variants: Dict[str, List[Dict]] = {}
    for v in ("sem_bypass", "sem_bypass_consensus"):
        rows = []
        for s in a.seeds:
            d = load(f"{v}_seed{s}")
            if d is None:
                continue
            summ = os.path.join(RUNS, f"{v}_seed{s}", "summary.json")
            tr = json.load(open(summ)) if os.path.exists(summ) else {}
            rows.append({"seed": s, "eval": d, "train": tr})
        if rows:
            variants[v] = rows

    def stats(rows, key="matched_miou"):
        vals = [r["eval"][key] for r in rows]
        return {"values": vals, "mean": statistics.fmean(vals),
                "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                "min": min(vals), "max": max(vals), "n": len(vals)}

    summary: Dict[str, Dict] = {"teacher_direct": {
        "matched_miou": t_miou, "end_to_end_miou": teacher["end_to_end_miou"],
        "coverage": teacher["coverage"], "needs_teacher_at_inference": True,
        "per_class_iou_matched": teacher["per_class_iou_matched"]}}

    for v, rows in variants.items():
        st = stats(rows)
        cov = {r["eval"]["coverage"] for r in rows}
        summary[v] = {
            "matched_miou": st,
            "end_to_end_miou": stats(rows, "end_to_end_miou"),
            "mean_accuracy": stats(rows, "mean_accuracy"),
            "cross_view_consistency": stats(rows, "cross_view_consistency"),
            "embedding_diversity": stats(rows, "embedding_diversity"),
            "coverage": sorted(cov),
            "coverage_identical_to_teacher": all(abs(c - teacher["coverage"]) < 1e-9 for c in cov),
            "retention_vs_teacher": st["mean"] / t_miou,
            "abs_gap_to_teacher": t_miou - st["mean"],
            "needs_teacher_at_inference": [r["eval"]["needs_teacher_at_inference"] for r in rows],
            "teacher_frames": [r["train"].get("teacher_frames") for r in rows],
            "train_seconds": [r["train"].get("seconds") or r["train"].get("train_seconds") for r in rows],
            "peak_vram_gb": [r["train"].get("peak_vram_gb") for r in rows],
            "trainable_params": [r["train"].get("parameter_report", {}).get("sidecar_trainable") for r in rows],
        }

    # -- per-class CSV ------------------------------------------------------ #
    os.makedirs(a.output_dir, exist_ok=True)
    csv_path = os.path.join(a.output_dir, "per_class.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "seed", "class", "prompt", "iou_matched", "iou_end_to_end"])
        def rows_for(model, seed, d):
            m, e = d["per_class_iou_matched"], d["per_class_iou_end_to_end"]
            for i, c in enumerate(CLASSES):
                w.writerow([model, seed, c, CLASS_PROMPTS[i], m.get(c), e.get(c)])
        rows_for("teacher_direct", "-", teacher)
        for v, rs in variants.items():
            for r in rs:
                rows_for(v, r["seed"], r["eval"])

    # -- rare vs dominant --------------------------------------------------- #
    def split_miou(per_class):
        """Per-class IoU is keyed by class name; absent classes score 0."""
        dom = [per_class.get(c, 0.0) for c in CLASSES if c in DOMINANT]
        rare = [per_class.get(c, 0.0) for c in CLASSES if c not in DOMINANT]
        return {"dominant_mean_iou": statistics.fmean(dom),
                "rare_mean_iou": statistics.fmean(rare),
                "n_rare_nonzero": sum(1 for x in rare if x > 0.1)}
    summary["teacher_direct"].update(split_miou(teacher["per_class_iou_matched"]))
    for v, rows in variants.items():
        per = [split_miou(r["eval"]["per_class_iou_matched"]) for r in rows]
        summary[v]["class_balance"] = {k: statistics.fmean([p[k] for p in per]) for k in per[0]}

    # -- gate decision ------------------------------------------------------ #
    gate: Dict[str, object] = {"thresholds": {
        "retention_min": RETENTION_MIN, "abs_gap_max": ABS_GAP_MAX, "std_max": STD_MAX}}
    if "sem_bypass" in summary:
        p = summary["sem_bypass"]
        checks = {
            "retains_90pct_of_teacher": p["retention_vs_teacher"] >= RETENTION_MIN,
            "within_1.5_abs_miou": p["abs_gap_to_teacher"] <= ABS_GAP_MAX,
            "seed_std_at_most_0.5": p["matched_miou"]["std"] <= STD_MAX,
            "no_coverage_advantage": p["coverage_identical_to_teacher"],
            "three_seeds_present": p["matched_miou"]["n"] >= 3,
        }
        gate["checks"] = checks
        gate["GATE1_PASS"] = all(checks.values())
        gate["verdict"] = "PASS" if gate["GATE1_PASS"] else "FAIL_REPRODUCIBILITY"
        gate["observed"] = {"teacher_matched_miou": t_miou,
                            "sem_bypass_mean": p["matched_miou"]["mean"],
                            "sem_bypass_std": p["matched_miou"]["std"],
                            "retention": p["retention_vs_teacher"],
                            "abs_gap": p["abs_gap_to_teacher"]}

    # -- consensus stability (secondary, reported separately) --------------- #
    if "sem_bypass" in summary and "sem_bypass_consensus" in summary:
        base = summary["sem_bypass"]["matched_miou"]["values"]
        cons = summary["sem_bypass_consensus"]["matched_miou"]["values"]
        deltas = [c - b for b, c in zip(base, cons)]
        gate["consensus"] = {
            "per_seed_delta": deltas,
            "mean_delta": statistics.fmean(deltas),
            "std_delta": statistics.pstdev(deltas) if len(deltas) > 1 else 0.0,
            "positive_on_all_seeds": all(d > 0 for d in deltas),
            "adopt_into_method": all(d > 0 for d in deltas) and (
                statistics.pstdev(deltas) if len(deltas) > 1 else 0.0) <= STD_MAX,
        }

    payload = {"summary": summary, "gate": gate,
               "audited_reference": {
                   "note": "prior single-seed numbers used mid+final GCT tokens [11, 23]",
                   "mlp_full_d025_matched_miou": 10.0437,
                   "mlp_pixel_d025_matched_miou": 9.0239,
                   "mlp_pixel_d100_matched_miou": 8.9829,
                   "teacher_direct_matched_miou": 10.5027}}
    write_json(os.path.join(a.output_dir, "metrics.json"), payload)
    print(json.dumps(gate, indent=2))
    print(f"wrote {a.output_dir}/metrics.json and {csv_path}")


if __name__ == "__main__":
    main()
