#!/usr/bin/env python
"""Collect, aggregate and contrast the broad-corpus study results.

    python tools/collect_broad_study_results.py --config configs/semantic_sidecar/broad_tartanair_controlled.yaml

Writes per-seed rows, mean/std aggregates, the four paired contrasts, per-class IoU and
the three-corpus comparison, as both CSV and JSON.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import _bootstrap  # noqa: F401

from semantic_sidecar.config import load_config, write_json
from semantic_sidecar.metrics import format_table

BASES = ["mlp_pixel_d100", "mlp_full_d100", "mlp_pixel_d025", "mlp_full_d025"]
PREFIX = "broad_tartanair_controlled"
FOCUS_CLASSES = ["car", "road", "building", "vegetation", "sidewalk"]
METRICS = ["end_to_end_miou", "matched_miou", "mean_accuracy", "coverage",
           "cross_view_consistency", "embedding_diversity",
           "teacher_cosine", "teacher_cosine_centered",
           "training_seconds", "teacher_frames", "training_observations"]


def teacher_fidelity(cfg, name: str, scenes: Sequence[str]) -> Dict[str, float]:
    """Ordinary and mean-removed cosine between a model's tokens and the teacher.

    Measured on the evaluation scenes, in the shared 64-d output space.
    """
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cache_teacher_features import load_teacher_features
    from semantic_sidecar.teacher_features import SemanticProjection

    proj = SemanticProjection.load(os.path.join(cfg.paths.cache_root, "pca.safetensors"))
    tea_root = os.path.join(cfg.paths.cache_root, "teacher")
    preds, teas = [], []
    for s in scenes:
        p = os.path.join(cfg.paths.maps_root, name, f"{s}_token_embeddings.npy")
        if not os.path.exists(p):
            continue
        preds.append(torch.from_numpy(np.load(p)).float().reshape(-1, 64))
        teas.append(proj.project(load_teacher_features(tea_root, s).float().reshape(-1, 512)))
    if not preds:
        return {"teacher_cosine": float("nan"), "teacher_cosine_centered": float("nan")}
    P, T = torch.cat(preds), torch.cat(teas)
    P, T = F.normalize(P, dim=-1), F.normalize(T, dim=-1)
    mu = F.normalize(T.mean(0), dim=-1)
    res = lambda X: F.normalize(X - (X @ mu).unsqueeze(1) * mu, dim=-1)  # noqa: E731
    return {"teacher_cosine": float((P * T).sum(-1).mean()),
            "teacher_cosine_centered": float((res(P) * res(T)).sum(-1).mean())}


def load_run(cfg, base: str, seed: int, eval_scenes: Sequence[str]) -> Optional[Dict[str, Any]]:
    name = f"{PREFIX}_{base}_s{seed}"
    ev = os.path.join(cfg.paths.maps_root, name, "eval.json")
    if not os.path.exists(ev):
        return None
    with open(ev) as fh:
        res = json.load(fh)
    summ_path = os.path.join(cfg.paths.runs_root, name, "summary.json")
    summ = json.load(open(summ_path)) if os.path.exists(summ_path) else {}
    tracks_tag = "density100" if base.endswith("d100") else "density025"
    tsum = os.path.join(cfg.paths.tracks_root, tracks_tag, "tracks_summary.json")
    teacher_frames = train_obs = float("nan")
    if os.path.exists(tsum):
        with open(tsum) as fh:
            scenes = json.load(fh)["scenes"]
        train_only = [s for s in scenes if str(s["scene"]).startswith("broad_")]
        teacher_frames = sum(int(s.get("teacher_frames", 0)) for s in train_only)
        train_obs = sum(int(s.get("observations", 0)) for s in train_only)

    row: Dict[str, Any] = {
        "run": name, "base": base, "seed": seed,
        "density": 1.0 if base.endswith("d100") else 0.25,
        "targets": "pixel" if "pixel" in base else "consensus(full)",
        "end_to_end_miou": res["end_to_end_miou"], "matched_miou": res["matched_miou"],
        "mean_accuracy": res["mean_accuracy"], "coverage": res["coverage"],
        "cross_view_consistency": res.get("cross_view_consistency", float("nan")),
        "embedding_diversity": res.get("embedding_diversity", float("nan")),
        "training_seconds": summ.get("training_seconds", float("nan")),
        "steps": summ.get("steps"),
        "trainable_params": summ.get("parameter_report", {}).get("sidecar_trainable"),
        "geometry_fingerprint": res.get("geometry_fingerprint"),
        "teacher_frames": teacher_frames, "training_observations": train_obs,
    }
    for cls in FOCUS_CLASSES:
        row[f"iou_{cls}"] = res.get("per_class_iou_matched", {}).get(cls, float("nan"))
    row.update(teacher_fidelity(cfg, name, eval_scenes))
    return row


def agg(rows: List[Dict[str, Any]], base: str, key: str) -> Dict[str, float]:
    vals = [r[key] for r in rows if r["base"] == base and isinstance(r.get(key), (int, float))
            and not np.isnan(float(r[key]))]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n": len(vals), "values": [float(v) for v in vals]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    from semantic_sidecar.datasets import expand_scenes

    eval_scenes = [s.name for s in expand_scenes(cfg.scenes) if s.role == "eval"]

    rows: List[Dict[str, Any]] = []
    for base in BASES:
        for seed in args.seeds:
            r = load_run(cfg, base, seed, eval_scenes)
            if r:
                rows.append(r)
    if not rows:
        raise SystemExit("no completed runs found")

    aggregates = {b: {m: agg(rows, b, m) for m in METRICS + [f"iou_{c}" for c in FOCUS_CLASSES]}
                  for b in BASES}

    def paired(a: str, b: str) -> Dict[str, Any]:
        """Seed-paired difference a - b."""
        out: Dict[str, Any] = {"contrast": f"{a} - {b}"}
        for m in ("matched_miou", "end_to_end_miou", "embedding_diversity", "teacher_cosine_centered"):
            diffs = []
            for seed in args.seeds:
                ra = next((r for r in rows if r["base"] == a and r["seed"] == seed), None)
                rb = next((r for r in rows if r["base"] == b and r["seed"] == seed), None)
                if ra and rb and not np.isnan(float(ra[m])) and not np.isnan(float(rb[m])):
                    diffs.append(float(ra[m]) - float(rb[m]))
            out[m] = {"mean": float(np.mean(diffs)) if diffs else float("nan"),
                      "std": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
                      "per_seed": diffs,
                      "consistent_sign": bool(diffs) and (all(d > 0 for d in diffs) or all(d < 0 for d in diffs))}
        return out

    contrasts = [paired("mlp_full_d025", "mlp_pixel_d025"),
                 paired("mlp_full_d100", "mlp_pixel_d100"),
                 paired("mlp_pixel_d025", "mlp_pixel_d100"),
                 paired("mlp_full_d025", "mlp_full_d100")]

    fields = list(rows[0].keys())
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    fps = {r["run"]: r["geometry_fingerprint"] for r in rows}
    payload = {
        "prefix": PREFIX, "seeds": args.seeds, "per_seed": rows,
        "aggregates": aggregates, "contrasts": contrasts,
        "geometry_fingerprints": fps,
        "geometry_identical": len({v for v in fps.values() if v}) == 1,
        "all_steps_20000": all(r["steps"] == 20000 for r in rows),
        "all_params_9112896": all(r["trainable_params"] == 9_112_896 for r in rows),
        "coverage_spread": max(r["coverage"] for r in rows) - min(r["coverage"] for r in rows),
    }
    write_json(args.out_json, payload)

    table = [{"model": b, "seeds": aggregates[b]["matched_miou"]["n"],
              "matched mIoU": aggregates[b]["matched_miou"]["mean"],
              "±sd": aggregates[b]["matched_miou"]["std"],
              "3D mIoU": aggregates[b]["end_to_end_miou"]["mean"],
              "mean acc": aggregates[b]["mean_accuracy"]["mean"],
              "diversity": aggregates[b]["embedding_diversity"]["mean"],
              "cos_ctr": aggregates[b]["teacher_cosine_centered"]["mean"]} for b in BASES]
    print(format_table(table))
    print(f"\nwrote {args.out_json}\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
