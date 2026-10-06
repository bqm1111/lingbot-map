"""Gate 16 analysis: selection (stage 1), semantic completion (stage 2), per class / distance / reveal, decision.

Seeds: B3 and B3geom have seeds 0-2; per-seed values and their mean are reported. Paired uncertainty uses 20-anchor block
resamples (2,000, seed 0) of confusions pooled over the three seeds against the single-seed controls (weighted ×3), so every
resample compares identical anchors. B1/B2 are pilot single-seed controls and carry no seed uncertainty.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Callable, Dict, List

import numpy as np

from tools.delayed_semantic import analysis as PA
from tools.delayed_semantic.gate16 import config as G, rules as R
from tools.delayed_semantic.gate16.candidate_metrics import FIELDS, accuracy, recovered_fraction, summarize_selection
from tools.delayed_semantic.gate16.evaluate import CAND_REGIONS, TOP_REGIONS
from tools.delayed_semantic.regions import PRIMARY, SEMANTIC_REGIONS, SSC_REGIONS
from tools.semantic_completion.evaluation import semantic_metrics as SM

NAMES = ["car", "bicycle", "motorcycle", "truck", "other vehicle", "person", "road", "parking", "sidewalk", "other ground", "building",
         "fence", "vegetation", "terrain", "pole", "traffic sign", "other structure", "other object"]
REGIONS = SEMANTIC_REGIONS + CAND_REGIONS + SSC_REGIONS


def load(paths_glob: str = os.path.join(G.EVAL, "eval_shard*of*.npz")) -> Dict:
    parts = [dict(np.load(f)) for f in sorted(glob.glob(paths_glob))]
    ids = np.concatenate([p["ids"] for p in parts]); order = np.argsort(ids); assert len(np.unique(ids)) == len(ids)
    per_anchor = [k for k in parts[0] if k.startswith(("conf__", "conftop__")) or k in ("occ", "evidence_path_equal", "n_query")]
    z = {k: np.concatenate([p[k] for p in parts])[order] for k in per_anchor}
    z["ids"] = ids[order]
    for k in parts[0]:
        if k.startswith(("dist_", "reveal_", "class_hit")) or k == "selection":
            z[k] = sum(p[k] for p in parts)
    z["methods"] = [str(m) for m in parts[0]["methods"]]
    assert all([str(m) for m in p["methods"]] == z["methods"] for p in parts)
    return z


def seeds_of(z, variant: str) -> List[str]:
    return [m for m in z["methods"] if m.startswith(variant + "_s")]


def bootstrap(metric: Callable[[np.ndarray], float], a: np.ndarray, b: np.ndarray, blk: np.ndarray, resamples: int = 2000, seed: int = 0) -> Dict:
    nb = int(blk.max()) + 1
    ba = np.stack([a[blk == i].sum(0) for i in range(nb)]); bb = np.stack([b[blk == i].sum(0) for i in range(nb)])
    idx = np.random.default_rng(seed).integers(nb, size=(resamples, nb))
    d = np.array([metric(ba[i].sum(0)) - metric(bb[i].sum(0)) for i in idx])
    return {"delta": metric(a.sum(0)) - metric(b.sum(0)), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def per_seed(z, seeds: List[str], key: Callable[[str], str], metric) -> List[float]:
    return [metric(z[key(s)].sum(0)) for s in seeds]


def analyse() -> Dict:
    R.check("predeclaration"); ceil_rec = R.check("ceiling"); R.check("code")
    thr = ceil_rec["payload"]; ceiling = json.load(open(os.path.join(G.ARTIFACT_DIR, "candidate_ceiling.json")))
    z = load(); ms = z["methods"]; blk = PA.blocks(len(z["ids"])); pilot = PA.load_eval()
    b3, b3g = seeds_of(z, "B3"), seeds_of(z, "B3geom")
    pooled3 = lambda seeds, key: sum(z[key(s)] for s in seeds)
    rec: Dict = {"n_anchors": len(z["ids"]), "methods": ms, "threshold": thr}

    # ---- reproduction of the pilot and of the frozen ceiling ----
    pos = np.searchsorted(pilot["ids"], z["ids"])
    rec["reproduction"] = {
        "ids_equal": bool(np.array_equal(pilot["ids"][pos], z["ids"])),
        "occupancy_equal_pilot": bool(np.array_equal(z["occ"], pilot["occ"][pos])),
        **{f"{m}_{r}_equal_pilot": bool(np.array_equal(z[f"conf__{m}__{r}"], pilot[f"conf__{m}__{r}"][pos]))
           for m in ("B0", "B1", "B2") for r in SEMANTIC_REGIONS + SSC_REGIONS},
        "C0_equal_ceiling": bool(np.array_equal(z[f"conf__C0__{PRIMARY}"].sum(0), np.asarray(ceiling["confusions"]["C0"]))),
        "oracle_equal_ceiling": bool(np.array_equal(z[f"conf__oracle__{PRIMARY}"].sum(0), np.asarray(ceiling["confusions"]["oracle"]))),
        "evidence_path_equal_all": bool(z["evidence_path_equal"].all())}

    # ---- stage 1: selection ----
    N, O = thr["N_nearest_candidate_acc"], thr["O_oracle_acc"]
    acc = lambda m, r=PRIMARY, top=False: accuracy(z[f"{'conftop' if top else 'conf'}__{m}__{r}"].sum(0))
    sel = {"N": N, "O": O, "H": thr["H"], "T": thr["T"], "B0_readout_acc": acc("B0"), "C0_acc": accuracy(z[f"conf__C0__{PRIMARY}"].sum(0)),
           "oracle_acc": accuracy(z[f"conf__oracle__{PRIMARY}"].sum(0))}
    for name, seeds in (("B3", b3), ("B3geom", b3g)):
        rows = {s: {"readout_acc": acc(s), "top_weight_acc": acc(s, top=True), "hit_acc": acc(s, "primary_candidate_hit"), "miss_acc": acc(s, "primary_candidate_miss"),
                    "hit_top_weight_acc": acc(s, "primary_candidate_hit", True)} for s in seeds}
        for s in seeds:
            rows[s]["R"] = recovered_fraction(rows[s]["readout_acc"], N, O); rows[s]["R_top_weight"] = recovered_fraction(rows[s]["top_weight_acc"], N, O)
        mean = {k: float(np.mean([v[k] for v in rows.values()])) for k in next(iter(rows.values()))}
        ga = bootstrap(accuracy, pooled3(seeds, lambda s: f"conf__{s}__{PRIMARY}"), 3 * z[f"conf__B0__{PRIMARY}"], blk)
        gh = bootstrap(accuracy, pooled3(seeds, lambda s: f"conf__{s}__primary_candidate_hit"), 3 * z["conf__B0__primary_candidate_hit"], blk)
        sel[name] = {"per_seed": rows, "mean": mean, "acc_gain_vs_B0": ga, "hit_acc_gain_vs_B0": gh}
    for m in ("B0", "B1", "B2"):
        sel[m] = {"readout_acc": acc(m), "top_weight_acc": acc(m, top=True), "hit_acc": acc(m, "primary_candidate_hit"), "miss_acc": acc(m, "primary_candidate_miss"),
                  "R": recovered_fraction(acc(m), N, O)}
    sel["label_free"] = {m: summarize_selection(z["selection"][i]) for i, m in enumerate(ms)}
    rec["selection"] = sel

    # ---- stage 2: semantic ----
    miou = lambda m, r: PA.fast_miou(z[f"conf__{m}__{r}"].sum(0))
    sem = {r: {m: miou(m, r) for m in ("B0", "B1", "B2")} for r in REGIONS}
    for r in REGIONS:
        for name, seeds in (("B3", b3), ("B3geom", b3g)):
            v = [miou(s, r) for s in seeds]; sem[r][name] = float(np.mean(v)); sem[r][name + "_per_seed"] = v
    rec["semantic_miou"] = sem
    keys = [PRIMARY, "completed_geo_observed", "completed_geo_unobserved", "completed_later_revealed", "completed_revealed_unseen", "primary_candidate_hit",
            "primary_candidate_miss", "ssc_full"]
    rec["semantic_deltas"] = {}
    for name, seeds in (("B3", b3), ("B3geom", b3g)):
        for ctrl in ("B0", "B1", "B2"):
            rec["semantic_deltas"][f"{name}-{ctrl}"] = {r: bootstrap(PA.fast_miou, pooled3(seeds, lambda s, r=r: f"conf__{s}__{r}"), 3 * z[f"conf__{ctrl}__{r}"], blk) for r in keys}
    rec["semantic_deltas"]["B3-B3geom"] = {r: bootstrap(PA.fast_miou, pooled3(b3, lambda s, r=r: f"conf__{s}__{r}"), pooled3(b3g, lambda s, r=r: f"conf__{s}__{r}"), blk) for r in keys}

    # ---- per class ----
    iou = lambda cm: [(cm[c, c] / (cm[c].sum() + cm[:, c].sum() - cm[c, c])) if cm[c].sum() else None for c in range(1, 19)]
    c0 = z[f"conf__C0__{PRIMARY}"].sum(0); ch = z["class_hit"]
    b3_iou = np.array([[x if x is not None else np.nan for x in iou(z[f"conf__{s}__{PRIMARY}"].sum(0))] for s in b3])
    b3_top = np.array([[z[f"conftop__{s}__{PRIMARY}"].sum(0)[c, c] / max(z[f"conftop__{s}__{PRIMARY}"].sum(0)[c].sum(), 1) for c in range(1, 19)] for s in b3])
    b0_iou = iou(z[f"conf__B0__{PRIMARY}"].sum(0))
    rec["per_class"] = {}
    for c in range(1, 19):
        n = int(ch[c, 0]); any16 = ch[c, 1] / max(n, 1); near = c0[c, c] / max(c0[c].sum(), 1); sel_c = float(np.mean(b3_top[:, c - 1]))
        rec["per_class"][NAMES[c - 1]] = {"gt_voxels": n, "nearest_hit": float(near), "any16_hit": float(any16), "B3_selected_correct": sel_c,
                                          "B3_iou": float(np.nanmean(b3_iou[:, c - 1])) if n else None, "B0_iou": b0_iou[c - 1],
                                          "absence_gap": float(1 - any16), "association_gap": float(any16 - sel_c),
                                          "limited_by": ("candidate absence" if (1 - any16) >= (any16 - sel_c) else "association") if n else None}
    # ---- distance and reveal ----
    def bins(prefix, hits, names):
        out = {}
        for b, nm in enumerate(names):
            out[nm] = {"voxels": int(hits[b, 0]), "candidate_hit_rate": hits[b, 1] / max(hits[b, 0], 1),
                       "B0_acc": accuracy(z[f"{prefix}_label__B0"][b]), "B0_miou": PA.fast_miou(z[f"{prefix}_label__B0"][b]),
                       "B3_top_weight_acc": float(np.mean([accuracy(z[f"{prefix}_top__{s}"][b]) for s in b3])),
                       "B3_readout_acc": float(np.mean([accuracy(z[f"{prefix}_label__{s}"][b]) for s in b3])),
                       "B3_miou": float(np.mean([PA.fast_miou(z[f"{prefix}_label__{s}"][b]) for s in b3])),
                       "B2_acc": accuracy(z[f"{prefix}_label__B2"][b])}
        return out
    from tools.delayed_semantic import config as PC
    rec["per_distance"] = bins("dist", z["dist_hit"], PC.DISTANCE_NAMES); rec["per_reveal"] = bins("reveal", z["reveal_hit"], PC.REVEAL_NAMES)

    # ---- criteria and decision ----
    s1 = sel["B3"]; g = s1["acc_gain_vs_B0"]; gh = s1["hit_acc_gain_vs_B0"]
    seed_gain = [s1["per_seed"][s]["readout_acc"] - sel["B0"]["readout_acc"] for s in b3]
    seed_hit_gain = [s1["per_seed"][s]["hit_acc"] - sel["B0"]["hit_acc"] for s in b3]
    S1 = s1["mean"]["readout_acc"] >= thr["T"] and all(x > 0 for x in seed_gain) and g["ci95"][0] > 0
    S2 = gh["delta"] >= 0.02 and all(x > 0 for x in seed_hit_gain) and gh["ci95"][0] > 0
    d0 = rec["semantic_deltas"]["B3-B0"][PRIMARY]
    seed_miou_gain = [x - sem[PRIMARY]["B0"] for x in sem[PRIMARY]["B3_per_seed"]]
    classes_up = sum(1 for c in NAMES if rec["per_class"][c]["gt_voxels"] and rec["per_class"][c]["B3_iou"] is not None and rec["per_class"][c]["B0_iou"] is not None
                     and rec["per_class"][c]["B3_iou"] > rec["per_class"][c]["B0_iou"])
    M1 = d0["delta"] >= 0.01 and all(x > 0 for x in seed_miou_gain) and d0["ci95"][0] > 0 and sem[PRIMARY]["B3"] > sem[PRIMARY]["B2"] and classes_up >= 5
    Occ = rec["reproduction"]["occupancy_equal_pilot"]
    lf = sel["label_free"]; top1_b3 = float(np.mean([lf[s]["top1"] for s in b3]))
    D = (top1_b3 - lf["B1"]["top1"] >= 0.10) and g["ci95"][1] < 0
    decision = "invalid" if not Occ else "D" if D else "A" if (S1 and S2 and M1) else "B" if S1 else "C"
    rec["criteria"] = {"S1": {"seed_mean_acc": s1["mean"]["readout_acc"], "T": thr["T"], "R": s1["mean"]["R"], "seed_gains": seed_gain, "ci95": g["ci95"], "pass": S1},
                       "S2": {"gain": gh["delta"], "seed_gains": seed_hit_gain, "ci95": gh["ci95"], "pass": S2},
                       "M1": {"gain": d0["delta"], "ci95": d0["ci95"], "seed_gains": seed_miou_gain, "B3_minus_B2": sem[PRIMARY]["B3"] - sem[PRIMARY]["B2"],
                              "classes_up": classes_up, "pass": M1},
                       "Occ": {"pass": Occ}, "D": {"B3_top1": top1_b3, "B1_top1": lf["B1"]["top1"], "acc_gain_ci_upper": g["ci95"][1], "pass": D}}
    rec["decision"] = decision
    json.dump(rec, open(os.path.join(G.ARTIFACT_DIR, "analysis.json"), "w"), indent=1, default=float)
    return rec


__all__ = ["load", "seeds_of", "bootstrap", "analyse"]
