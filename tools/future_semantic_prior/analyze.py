"""Gate 21's tables and its decision.

Everything is computed from the per-anchor confusion matrices written by ``evaluate.py``, so every number
in the report comes from the frozen metric (``semantic_metrics.summarize(conf, semantic_only=True)``:
classes 1..18 with GT support in the region, class 0 excluded) applied to summed confusions. mIoU is a
ratio of sums, so it is always recomputed from a summed matrix and never averaged across anchors.

The decision follows ``decision_rule.md`` exactly and reads nothing else. ``validate.py`` has already
checked that the document still hashes to what was registered before training.

    PYTHONPATH=$PWD python -m tools.future_semantic_prior.analyze
"""
from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from tools.future_semantic_prior import config as G, evaluate as EV
from tools.semantic_completion.evaluation import semantic_metrics as SM

TH = G.Thresholds()
NB = len(G.DISTANCE_NAMES)
THING = set(G.THING_LABELS)


def _summ(conf: np.ndarray, semantic_only: bool = True) -> Dict:
    """mIoU, top-1, thing/stuff mIoU and support from one summed confusion, all in percent."""
    s = SM.summarize(conf, semantic_only=semantic_only)
    iou = s["per_class_iou"]
    sup = [c for c in range(1, 19) if iou[c] is not None]
    th = [iou[c] for c in sup if c in THING]
    st = [iou[c] for c in sup if c not in THING]
    return {"miou": 100.0 * s["miou"] if s["miou"] is not None else float("nan"),
            "top1": 100.0 * s["overall_accuracy"],
            "thing_miou": 100.0 * float(np.mean(th)) if th else float("nan"),
            "stuff_miou": 100.0 * float(np.mean(st)) if st else float("nan"),
            "n": int(s["n_voxels"]), "n_classes": len(sup),
            # collapse diagnostics: mIoU averages over supported classes, so a model that predicts only
            # the few dominant ones can gain accuracy and LOSE mIoU. Section 16 asks whether a gain comes
            # from one or two easy classes; the same numbers answer the reverse question for a loss.
            "class_coverage": 100.0 * s["class_coverage"] if s["class_coverage"] is not None else float("nan"),
            "n_classes_predicted": int(s["gt_classes_predicted"]),
            "n_classes_ge_1pct": int(s["n_classes_ge_1pct_of_predictions"]),
            "dominant_pred_class": int(s["dominant_pred_class"]),
            "dominant_pred_fraction": 100.0 * float(s["dominant_pred_fraction"]),
            "pred_entropy_bits": float(s["pred_entropy_bits"]),
            "per_class_iou": {c: (100.0 * iou[c] if iou[c] is not None else None) for c in range(1, 19)}}


@lru_cache(maxsize=32)
def load_eval(control: str, seed: int, step: Optional[int] = None, stride: int = 1) -> Dict[str, np.ndarray]:
    """Cached: the tables below read each evaluation file a few hundred times and they are tens of MB."""
    step = step or G.TrainConfig().steps
    with np.load(EV.eval_path(control, seed, step, stride)) as z:
        return {k: z[k] for k in z.files}


def available(step: Optional[int] = None, stride: int = 1) -> Dict[str, List[int]]:
    step = step or G.TrainConfig().steps
    return {c: [s for s in G.SEEDS if os.path.exists(EV.eval_path(c, s, step, stride))]
            for c in G.LEARNED_CONTROLS}


def block_bootstrap(model_confs: Sequence[np.ndarray], b0_conf: np.ndarray, block: int = G.BOOTSTRAP_BLOCK,
                    resamples: int = G.BOOTSTRAP_RESAMPLES, seed: int = G.BOOTSTRAP_SEED) -> Dict:
    """95 % interval on (seed-mean model mIoU) - (B0 mIoU), resampling contiguous anchor blocks.

    ``model_confs`` is one ``[A, 19, 19]`` stack per seed, ``b0_conf`` the matching B0 stack. Per resample
    the blocks' confusions are SUMMED and mIoU recomputed from the sum, which is the only correct way to
    bootstrap a ratio of sums."""
    a = b0_conf.shape[0]
    nb = (a + block - 1) // block
    starts = np.arange(nb) * block
    rng = np.random.default_rng(seed)
    out = np.empty(resamples)
    for i in range(resamples):
        pick = rng.integers(0, nb, nb)
        idx = np.concatenate([np.arange(starts[p], min(starts[p] + block, a)) for p in pick])
        b0 = _summ(b0_conf[idx].sum(0))["miou"]
        ms = [_summ(m[idx].sum(0))["miou"] for m in model_confs]
        out[i] = float(np.mean(ms)) - b0
    lo, hi = (float(x) for x in np.percentile(out, [2.5, 97.5]))
    return {"mean": float(out.mean()), "ci95": [lo, hi], "resamples": resamples, "block_anchors": block,
            "seed": seed, "fraction_positive": float((out > 0).mean())}


def analyse(step: Optional[int] = None, stride: int = 1) -> Dict:
    step = step or G.TrainConfig().steps
    have = available(step, stride)
    assert any(have.values()), "no evaluation output found; run tools.future_semantic_prior.evaluate first"
    ref_control = next(c for c in G.LEARNED_CONTROLS if have[c])
    ref = load_eval(ref_control, have[ref_control][0], step, stride)
    n_anchors = len(ref["ids"])

    # B0 is read from the frozen cache by every run, so every run's B0 arrays must be byte-identical
    b0_identical = True
    for c, seeds in have.items():
        for s in seeds:
            e = load_eval(c, s, step, stride)
            for k in [f"conf__B0__{p}" for p in EV.POPULATIONS] + ["confdist__B0", "confthing__B0", "confstuff__B0"]:
                b0_identical &= bool(np.array_equal(e[k], ref[k]))

    out: Dict = {"version": G.VERSION, "step": step, "n_dev_anchors": n_anchors,
                 "seeds_available": have, "b0_identical_across_runs": b0_identical,
                 "all_geometry_identical": True, "populations": {}, "per_class_E2": {},
                 "distance_E2": {}, "thing_stuff_E2": {}, "ssc": {}, "class_names": list(G.CLASS_NAMES)}

    for c, seeds in have.items():
        for s in seeds:
            m = json.load(open(EV.eval_path(c, s, step, stride)[:-4] + ".json"))
            out["all_geometry_identical"] &= bool(m["all_geometry_identical"])

    # ---- the central table ------------------------------------------------------------------------
    for p in EV.POPULATIONS:
        b0 = _summ(ref[f"conf__B0__{p}"].sum(0))
        keep = ("miou", "top1", "thing_miou", "stuff_miou", "n", "n_classes", "class_coverage",
                "n_classes_predicted", "n_classes_ge_1pct", "dominant_pred_class",
                "dominant_pred_fraction", "pred_entropy_bits")
        row = {"B0": {k: b0[k] for k in keep}}
        for c, seeds in have.items():
            if not seeds:
                continue
            per = [_summ(load_eval(c, s, step, stride)[f"conf__model__{p}"].sum(0)) for s in seeds]
            row[c] = {"seeds": seeds,
                      "miou_per_seed": [x["miou"] for x in per],
                      "miou_mean": float(np.mean([x["miou"] for x in per])),
                      "miou_std": float(np.std([x["miou"] for x in per])),
                      "top1_mean": float(np.mean([x["top1"] for x in per])),
                      "top1_std": float(np.std([x["top1"] for x in per])),
                      "thing_miou_mean": float(np.nanmean([x["thing_miou"] for x in per])),
                      "stuff_miou_mean": float(np.nanmean([x["stuff_miou"] for x in per]))}
            row[c]["delta_miou_mean"] = row[c]["miou_mean"] - b0["miou"]
            row[c]["delta_miou_per_seed"] = [x - b0["miou"] for x in row[c]["miou_per_seed"]]
            row[c]["delta_top1_mean"] = row[c]["top1_mean"] - b0["top1"]
            for k in ("class_coverage", "n_classes_predicted", "n_classes_ge_1pct",
                      "dominant_pred_fraction", "pred_entropy_bits"):
                row[c][k] = float(np.mean([x[k] for x in per]))
            row[c]["dominant_pred_class"] = int(np.median([x["dominant_pred_class"] for x in per]))
            row[c]["delta_thing_miou"] = row[c]["thing_miou_mean"] - b0["thing_miou"]
            row[c]["delta_stuff_miou"] = row[c]["stuff_miou_mean"] - b0["stuff_miou"]
        out["populations"][p] = row

    # ---- bootstrap on the primary metric ----------------------------------------------------------
    out["bootstrap_E2"] = {}
    for c, seeds in have.items():
        if not seeds:
            continue
        confs = [load_eval(c, s, step, stride)["conf__model__E2"] for s in seeds]
        out["bootstrap_E2"][c] = block_bootstrap(confs, ref["conf__B0__E2"])

    # ---- per class on E2 ---------------------------------------------------------------------------
    b0e2 = _summ(ref["conf__B0__E2"].sum(0))
    gt_support = ref["conf__B0__E2"].sum(0).sum(1)
    for ci in range(1, 19):
        e = {"class": G.CLASS_NAMES[ci - 1], "support": int(gt_support[ci]),
             "thing": ci in THING, "B0": b0e2["per_class_iou"][ci]}
        for c, seeds in have.items():
            if not seeds:
                continue
            v = [_summ(load_eval(c, s, step, stride)["conf__model__E2"].sum(0))["per_class_iou"][ci]
                 for s in seeds]
            v = [x for x in v if x is not None]
            e[c] = float(np.mean(v)) if v else None
            e[f"delta_{c}"] = (e[c] - e["B0"]) if (e[c] is not None and e["B0"] is not None) else None
        out["per_class_E2"][str(ci)] = e

    # ---- distance to causal evidence, on E2 --------------------------------------------------------
    for b in range(NB):
        d = {"bin": G.DISTANCE_NAMES[b]}
        b0b = _summ(ref["confdist__B0"][:, b].sum(0))
        d["B0"] = {"miou": b0b["miou"], "top1": b0b["top1"], "n": b0b["n"], "n_classes": b0b["n_classes"]}
        for c, seeds in have.items():
            if not seeds:
                continue
            per = [_summ(load_eval(c, s, step, stride)["confdist__model"][:, b].sum(0)) for s in seeds]
            d[c] = {"miou_mean": float(np.mean([x["miou"] for x in per])),
                    "miou_std": float(np.std([x["miou"] for x in per])),
                    "delta": float(np.mean([x["miou"] for x in per])) - b0b["miou"]}
        out["distance_E2"][G.DISTANCE_NAMES[b]] = d

    # ---- thing / stuff split of E2 (as populations, not as class averages) ------------------------
    for key, tag in (("confthing", "thing"), ("confstuff", "stuff")):
        b0t = _summ(ref[f"{key}__B0"].sum(0))
        row = {"B0": {k: b0t[k] for k in ("miou", "top1", "n", "n_classes")}}
        for c, seeds in have.items():
            if not seeds:
                continue
            per = [_summ(load_eval(c, s, step, stride)[f"{key}__model"].sum(0)) for s in seeds]
            row[c] = {"miou_mean": float(np.mean([x["miou"] for x in per])),
                      "miou_std": float(np.std([x["miou"] for x in per])),
                      "delta": float(np.mean([x["miou"] for x in per])) - b0t["miou"]}
        out["thing_stuff_E2"][tag] = row

    # ---- composite SSC: reported, never decisive --------------------------------------------------
    if "confssc__B0" in ref:
        b0s = _summ(ref["confssc__B0"].sum(0), semantic_only=False)
        out["ssc"]["B0"] = {"miou": b0s["miou"], "n": b0s["n"]}
        for c, seeds in have.items():
            per = [_summ(load_eval(c, s, step, stride)["confssc__model"].sum(0), semantic_only=False)
                   for s in seeds if "confssc__model" in load_eval(c, s, step, stride)]
            if per:
                out["ssc"][c] = {"miou_mean": float(np.mean([x["miou"] for x in per])),
                                 "delta": float(np.mean([x["miou"] for x in per])) - b0s["miou"]}

    out["b0_overlap"] = b0_overlap(out)
    out["decision"] = decide(out)
    return out


def b0_overlap(a: Dict) -> Dict:
    """What the model does to B0's answers, split by whether B0 was already right.

    `P1` is defined as "B0's class is wrong", so `E*_P1` and its complement inside `E*` partition the
    population by B0's correctness, and the model's accuracy on each half follows exactly from the two
    top-1 numbers and their voxel counts -- no extra pass over the data. It is the one number that says
    whether a top-1 loss is "the model is uniformly worse" or "the model trades B0's correct answers for a
    smaller number of fixes", which are different findings with different consequences."""
    out: Dict = {}
    for base in ("E1", "E2", "E0"):
        pw = f"{base}_P1"
        if pw not in a["populations"]:
            continue
        tot = a["populations"][base]
        wrong = a["populations"][pw]
        n_t, n_w = tot["B0"]["n"], wrong["B0"]["n"]
        n_r = n_t - n_w                                   # voxels B0 already gets right
        if n_r <= 0:
            continue
        row = {"n": n_t, "n_b0_wrong": n_w, "n_b0_right": n_r,
               "b0_top1": tot["B0"]["top1"]}
        for c in G.LEARNED_CONTROLS:
            if c not in tot or not tot[c].get("seeds"):
                continue
            hit_t = tot[c]["top1_mean"] / 100.0 * n_t
            hit_w = wrong[c]["top1_mean"] / 100.0 * n_w
            kept = (hit_t - hit_w) / n_r                  # fraction of B0's correct answers preserved
            row[c] = {"keeps_b0_correct_pct": 100.0 * kept,
                      "fixes_b0_wrong_pct": wrong[c]["top1_mean"],
                      "n_broken": (1.0 - kept) * n_r,
                      "n_fixed": hit_w,
                      "net_voxels": hit_w - (1.0 - kept) * n_r}
        out[base] = row
    return out


def decide(a: Dict) -> Dict:
    """Apply ``decision_rule.md`` §4 to the primary control. No other input."""
    c = G.PRIMARY_CONTROL
    e2 = a["populations"]["E2"]
    if c not in e2 or not e2[c]["seeds"]:
        return {"outcome": "INCOMPLETE", "reason": f"{c} has no evaluated seed"}
    d = e2[c]["delta_miou_mean"]
    per = e2[c]["delta_miou_per_seed"]
    bs = a["bootstrap_E2"][c]
    integrity = {"geometry_identical": bool(a["all_geometry_identical"]),
                 "b0_identical_across_runs": bool(a["b0_identical_across_runs"])}
    A = {"A1_delta_ge_1.0": d >= TH.a_delta_e2_min,
         "A2_all_seeds_positive": all(x > 0 for x in per),
         "A3_ci_lower_above_0": bs["ci95"][0] > TH.a_ci_lower_above}
    Cc = {"C1_delta_le_0.25": d <= TH.c_delta_e2_max,
          "C2_a_seed_negative_and_below_bar": any(x < 0 for x in per) and d < TH.a_delta_e2_min}
    if not all(integrity.values()):
        outcome = "VOID"
    elif all(A.values()):
        outcome = "A"
    elif any(Cc.values()):
        outcome = "C"
    else:
        outcome = "B"
    return {"outcome": outcome, "control": c, "delta_E2_miou": d, "delta_E2_per_seed": per,
            "b0_E2_miou": e2["B0"]["miou"], "model_E2_miou": e2[c]["miou_mean"],
            "model_E2_miou_std": e2[c]["miou_std"], "bootstrap": bs, "A": A, "C": Cc,
            "integrity": integrity, "n_seeds": len(e2[c]["seeds"]),
            "thresholds": {"a_delta_e2_min": TH.a_delta_e2_min, "c_delta_e2_max": TH.c_delta_e2_max,
                           "a_ci_lower_above": TH.a_ci_lower_above}}


def write_artifacts(r: Dict) -> List[str]:
    """Split ``analysis.json`` into the files section 19 asks for, plus one per seed."""
    out: List[str] = []
    def dump(name: str, obj) -> None:
        p = os.path.join(G.ARTIFACT_DIR, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(obj, open(p, "w"), indent=1)
        out.append(p)
    dump("analysis.json", r)
    dump("metrics.json", {"version": r["version"], "step": r["step"], "n_dev_anchors": r["n_dev_anchors"],
                          "populations": r["populations"], "thing_stuff_E2": r["thing_stuff_E2"],
                          "bootstrap_E2": r["bootstrap_E2"], "ssc": r["ssc"],
                          "decision": r["decision"]})
    dump("per_class.json", {"population": "E2", "class_names": r["class_names"],
                            "note": "every one of the 18 classes is listed, including those never predicted",
                            "rows": r["per_class_E2"]})
    dump("distance_bins.json", {"population": "E2", "edges_m": list(G.DISTANCE_EDGES_M),
                                "measure": "distance from the voxel centre to the nearest causal "
                                           "semantic evidence voxel centre (B0's own EDT)",
                                "rows": r["distance_E2"]})
    for seed in G.SEEDS:
        per = {}
        for c, seeds in r["seeds_available"].items():
            if seed not in seeds:
                continue
            i = seeds.index(seed)
            per[c] = {p: {"miou": r["populations"][p][c]["miou_per_seed"][i],
                          "delta_miou": r["populations"][p][c]["delta_miou_per_seed"][i],
                          "b0_miou": r["populations"][p]["B0"]["miou"]} for p in EV.POPULATIONS}
        if per:
            dump(os.path.join(f"seed_{seed}", "metrics.json"),
                 {"seed": seed, "step": r["step"], "controls": per})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int); ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    G.ensure_dirs()
    r = analyse(a.step, a.stride)
    for p in write_artifacts(r):
        print(f"-> {os.path.relpath(p, G.ARTIFACT_DIR)}")
    print(json.dumps(r["decision"], indent=1))
    for p in EV.POPULATIONS:
        row = r["populations"][p]
        cells = " ".join(f"{c.split('_')[0]} {row[c]['miou_mean']:6.2f}" for c in G.LEARNED_CONTROLS if c in row)
        print(f"{p:7s} n={row['B0']['n']:>9,}  B0 {row['B0']['miou']:6.2f} | {cells}")


if __name__ == "__main__":
    main()
