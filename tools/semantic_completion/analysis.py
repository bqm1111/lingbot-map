"""Aggregate the source evaluation, apply the frozen promotion rule, write tables and results_table.md.

Every number comes from the saved per-anchor confusion matrices (``source_eval/source_eval_stride1_shard*.npz``).
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List

import numpy as np

from tools.semantic_completion import config as C, rules as RL
from tools.semantic_completion.evaluation.regions import PRIMARY, SEMANTIC_REGIONS, SSC_REGIONS
from tools.semantic_completion.evaluation.semantic_metrics import summarize
from tools.semantic_completion.evidence.trident import load_text_embeddings

TABLES = os.path.join(C.ARTIFACT_DIR, "tables")


def load_shards(paths: C.Paths = C.Paths(), stride: int = 1) -> Dict[str, np.ndarray]:
    files = sorted(glob.glob(os.path.join(paths.source_eval, f"source_eval_stride{stride}_shard*.npz")))
    assert files, "no source evaluation shards"
    parts = [dict(np.load(f)) for f in files]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.argsort(out["ids"])
    return {k: v[order] for k, v in out.items()}


def is_semantic(region: str) -> bool:
    return region in SEMANTIC_REGIONS


def miou_of(conf_sum: np.ndarray, region: str) -> float:
    return summarize(conf_sum, is_semantic(region))["miou"]


def bootstrap_diff(a: List[np.ndarray], b: List[np.ndarray], region: str, blocks: int = 20, resamples: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Paired clip-block bootstrap of mean-over-seeds [mIoU(a) - mIoU(b)]; a/b: per seed [anchors, 19, 19]."""
    rng = np.random.default_rng(seed); n = len(a[0]); starts = np.arange(0, n, blocks)
    diffs = []
    for _ in range(resamples):
        idx = np.concatenate([np.arange(s, min(s + blocks, n)) for s in rng.choice(starts, len(starts))])
        diffs.append(np.mean([miou_of(x[idx].sum(0), region) - miou_of(y[idx].sum(0), region) for x, y in zip(a, b)]))
    return {"lo": float(np.percentile(diffs, 2.5)), "hi": float(np.percentile(diffs, 97.5))}


def analyse(paths: C.Paths = C.Paths()) -> Dict:
    RL.check_promotion_rule()
    z = load_shards(paths); _, phrases, _ = load_text_embeddings(paths.text_embeddings)
    names = {c: str(phrases[c - 1]) for c in range(1, C.N_CLASSES + 1)}
    seeds = [s for s in C.SEEDS if f"conf__B1_seed{s}__{PRIMARY}" in z]
    res: Dict = {"n_anchors": int(len(z["ids"])), "seeds": seeds, "regions": {}, "teacher_observed": summarize(z["teacher_obs"].sum(0), True)}
    for r in SEMANTIC_REGIONS + SSC_REGIONS:
        res["regions"][r] = {m: {s: summarize(z[f"conf__{m}_seed{s}__{r}"].sum(0), is_semantic(r)) for s in seeds} for m in ("B0", "B1")}
    # distance bins (primary region)
    res["distance"] = {m: {s: [summarize(z[f"confdist__{m}_seed{s}"][:, b].sum(0), True) for b in range(len(C.DISTANCE_NAMES))] for s in seeds} for m in ("B0", "B1")}
    # held-out evidence cosine (label-free)
    ho = {s: z[f"heldout_seed{s}"].sum(0) for s in seeds}
    res["heldout"] = {s: {"n_hidden": int(v[0]), "cos_B1": v[1] / v[0], "cos_B0_nearest_visible": v[2] / v[0], "agree_B1": v[3] / v[0], "agree_B0": v[4] / v[0]} for s, v in ho.items()}
    # occupancy identity vs the frozen baseline
    occ = {}
    for s in seeds:
        base = json.load(open(os.path.join(paths.geometry_source_eval, f"geo_seed{s}_step06000.json")))["summary"]["full"]
        ours = z[f"occ_seed{s}"].sum(0)
        occ[s] = {"ours": ours.tolist(), "baseline": [base["tp"], base["fp"], base["fn"]], "identical": ours.tolist() == [base["tp"], base["fp"], base["fn"]],
                  "iou": float(ours[0] / ours.sum())}
    res["occupancy"] = occ
    train_runs = {s: json.load(open(os.path.join(paths.runs, f"train_sem_seed{s}.json"))) for s in seeds}
    # ---- promotion rule ----------------------------------------------------------------------------
    rule = RL.RULE; prim = res["regions"][PRIMARY]
    d1 = {s: prim["B1"][s]["miou"] - prim["B0"][s]["miou"] for s in seeds}
    ci1 = bootstrap_diff([z[f"conf__B1_seed{s}__{PRIMARY}"] for s in seeds], [z[f"conf__B0_seed{s}__{PRIMARY}"] for s in seeds], PRIMARY)
    P1 = np.mean(list(d1.values())) >= rule["P1_unobserved_gain"]["mean_over_seeds_at_least"] and all(v > 0 for v in d1.values()) and ci1["lo"] > 0
    d2 = {s: res["regions"]["ssc_full"]["B1"][s]["miou"] - res["regions"]["ssc_full"]["B0"][s]["miou"] for s in seeds}
    ci2 = bootstrap_diff([z[f"conf__B1_seed{s}__ssc_full"] for s in seeds], [z[f"conf__B0_seed{s}__ssc_full"] for s in seeds], "ssc_full")
    P2 = np.mean(list(d2.values())) > 0 and all(v > 0 for v in d2.values())
    gains = {c: float(np.mean([(prim["B1"][s]["per_class_iou"][c] or 0) - (prim["B0"][s]["per_class_iou"][c] or 0) for s in seeds]))
             for c in range(1, C.N_CLASSES + 1) if prim["B0"][seeds[0]]["per_class_iou"][c] is not None}
    n_gain = int(sum(v > 0 for v in gains.values()))
    dom = float(np.mean([prim["B1"][s]["dominant_pred_fraction"] for s in seeds])); dom_gt = prim["B1"][seeds[0]]["dominant_gt_fraction"]
    dom_limit = max(0.60, dom_gt + 0.10); ncls1 = float(np.mean([prim["B1"][s]["n_classes_ge_1pct_of_predictions"] for s in seeds]))
    collapse_ok = dom <= dom_limit and ncls1 >= 5
    P3 = n_gain >= 5 and collapse_ok
    P4 = all(o["identical"] for o in occ.values()) and all(r["geometry_state_digest_before"] == r["geometry_state_digest_after"] for r in train_runs.values())
    teach = res["teacher_observed"]; D = teach["overall_accuracy"] < 0.40 or teach["miou"] < 0.15
    decision = "INVALID" if not P4 else "D" if D else "A" if (P1 and P2 and P3) else "C" if (not P1 and not collapse_ok) else "B"
    res["promotion"] = {"P1": {"pass": bool(P1), "diff_per_seed": d1, "mean": float(np.mean(list(d1.values()))), "ci95": ci1},
                        "P2": {"pass": bool(P2), "diff_per_seed": d2, "mean": float(np.mean(list(d2.values()))), "ci95": ci2},
                        "P3": {"pass": bool(P3), "classes_with_gain": n_gain, "per_class_gain": {names[c]: v for c, v in gains.items()},
                               "b1_dominant_fraction": dom, "dominant_limit": dom_limit, "b1_classes_ge_1pct": ncls1, "collapse_clause_ok": bool(collapse_ok)},
                        "P4": {"pass": bool(P4)}, "D_teacher_poor": bool(D), "decision": decision}
    res["class_names"] = names
    json.dump(res, open(os.path.join(C.ARTIFACT_DIR, "analysis.json"), "w"), indent=1, default=float)
    write_tables(res, z)
    return res


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.2f}"


def write_tables(res: Dict, z: Dict[str, np.ndarray]) -> None:
    from tools.baseline_reconstruction.reports.tables import write_csv
    seeds, names = res["seeds"], res["class_names"]
    rows = []
    for r in SEMANTIC_REGIONS + SSC_REGIONS:
        for m in ("B0", "B1"):
            for s in seeds:
                x = res["regions"][r][m][s]
                rows.append({"region": r, "method": m, "seed": s, "miou": x["miou"], "mean_class_accuracy": x["mean_class_accuracy"], "overall_accuracy": x["overall_accuracy"],
                             "class_coverage": x["class_coverage"], "n_classes_ge_1pct": x["n_classes_ge_1pct_of_predictions"], "dominant_pred_class": names[x["dominant_pred_class"]],
                             "dominant_pred_fraction": x["dominant_pred_fraction"], "pred_entropy_bits": x["pred_entropy_bits"], "n_voxels": x["n_voxels"]})
    t = res["teacher_observed"]
    rows.append({"region": "oracle_observed", "method": "teacher_readout", "seed": "", "miou": t["miou"], "mean_class_accuracy": t["mean_class_accuracy"], "overall_accuracy": t["overall_accuracy"],
                 "class_coverage": t["class_coverage"], "n_classes_ge_1pct": t["n_classes_ge_1pct_of_predictions"], "dominant_pred_class": names[t["dominant_pred_class"]],
                 "dominant_pred_fraction": t["dominant_pred_fraction"], "pred_entropy_bits": t["pred_entropy_bits"], "n_voxels": t["n_voxels"]})
    write_csv(os.path.join(TABLES, "per_region.csv"), rows)
    pc = []
    for r in ("oracle_observed", PRIMARY, "ssc_full"):
        for c in range(1, C.N_CLASSES + 1):
            row = {"region": r, "class_id": c, "class": names[c]}
            for m in ("B0", "B1"):
                v = [res["regions"][r][m][s]["per_class_iou"][c] for s in seeds]
                row[f"{m}_iou_mean"] = None if v[0] is None else float(np.mean(v))
                row[f"{m}_iou_per_seed"] = "" if v[0] is None else " / ".join(f"{x:.4f}" for x in v)
            if r == "oracle_observed":
                row["teacher_iou"] = res["teacher_observed"]["per_class_iou"][c]
            pc.append(row)
    write_csv(os.path.join(TABLES, "per_class.csv"), pc)
    dist = []
    for b, name in enumerate(C.DISTANCE_NAMES):
        row = {"distance": name, "n_voxels_seed0": res["distance"]["B1"][seeds[0]][b]["n_voxels"]}
        for m in ("B0", "B1"):
            row[f"{m}_miou_mean"] = float(np.mean([res["distance"][m][s][b]["miou"] or 0 for s in seeds]))
            row[f"{m}_accuracy_mean"] = float(np.mean([res["distance"][m][s][b]["overall_accuracy"] for s in seeds]))
        dist.append(row)
    write_csv(os.path.join(TABLES, "per_distance.csv"), dist)
    freq = []
    for c in range(1, C.N_CLASSES + 1):
        p = res["regions"][PRIMARY]
        freq.append({"class_id": c, "class": names[c], "gt_frequency": p["B1"][seeds[0]]["gt_frequency"][c - 1],
                     "B0_pred_frequency": float(np.mean([p["B0"][s]["pred_frequency"][c - 1] for s in seeds])),
                     "B1_pred_frequency": float(np.mean([p["B1"][s]["pred_frequency"][c - 1] for s in seeds])),
                     "teacher_observed_pred_frequency": res["teacher_observed"]["pred_frequency"][c - 1], "teacher_observed_gt_frequency": res["teacher_observed"]["gt_frequency"][c - 1]})
    write_csv(os.path.join(TABLES, "class_frequency.csv"), freq)


def _int_keys(summary: Dict) -> Dict:
    return {**summary, "per_class_iou": {int(c): v for c, v in summary["per_class_iou"].items()},
            "per_class_accuracy": {int(c): v for c, v in summary["per_class_accuracy"].items()}}


def load_analysis(path: str = None) -> Dict:
    """analysis.json with the integer keys JSON turned into strings restored (seeds, class ids)."""
    res = json.load(open(path or os.path.join(C.ARTIFACT_DIR, "analysis.json")))
    res["seeds"] = [int(s) for s in res["seeds"]]
    res["class_names"] = {int(k): v for k, v in res["class_names"].items()}
    res["teacher_observed"] = _int_keys(res["teacher_observed"])
    res["regions"] = {r: {m: {int(s): _int_keys(v) for s, v in d.items()} for m, d in rr.items()} for r, rr in res["regions"].items()}
    res["distance"] = {m: {int(s): [_int_keys(x) for x in v] for s, v in d.items()} for m, d in res["distance"].items()}
    res["occupancy"] = {int(s): v for s, v in res["occupancy"].items()}
    res["heldout"] = {int(s): v for s, v in res["heldout"].items()}
    for k in ("P1", "P2"):
        res["promotion"][k]["diff_per_seed"] = {int(s): v for s, v in res["promotion"][k]["diff_per_seed"].items()}
    return res


__all__ = ["load_shards", "bootstrap_diff", "analyse", "write_tables", "load_analysis"]
