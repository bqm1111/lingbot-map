"""Stage 4: scoring with the legacy repository's own scorer, on its own regions.

Nothing is reimplemented: ``regions.build_regions`` defines the populations, ``semantic_metrics``
defines the confusion and the mIoU (classes 1..18, GT-supported classes only, the empty class scored in
the SSC regions and excluded from the average). Confusions are pooled over anchors and the metric is
computed once from the pooled matrix -- never averaged per anchor.

Added here, and kept separate from the frozen regions:
  * ``completed_unobserved`` split by whether the clip geometrically observed the voxel at all
    (``S | F``). This is a geometry statement, not an occlusion statement: the frozen representation
    cannot tell "physically occluded in all five frames" from "never resolved", so physical-occlusion
    attribution is reported as unavailable.
  * object classes versus broad surface classes.
  * per-anchor (= per temporal block) paired deltas and a paired block bootstrap.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import torch

import legacy as L
import manifest as MF
import paths as P
import predict as PD

OBJECT_CLASSES = (1, 2, 3, 4, 5, 6, 15, 16, 18)      # car bicycle motorcycle truck other-vehicle person pole sign other-object
SURFACE_CLASSES = (7, 8, 9, 10, 11, 12, 13, 14, 17)  # road parking sidewalk other-ground building fence vegetation terrain other-structure
CLASS_NAMES = {}

EXTRA_REGIONS = ("completed_unobserved_geo_observed", "completed_unobserved_geo_unobserved",
                 "completed_unobserved_objects", "completed_unobserved_surfaces")
ALL_REGIONS = tuple(L.SEMANTIC_REGIONS) + tuple(L.SSC_REGIONS) + EXTRA_REGIONS
SEMANTIC_ONLY = set(L.SEMANTIC_REGIONS) | set(EXTRA_REGIONS)


def regions_of(sem_gt, E, O, S, F_) -> Dict[str, torch.Tensor]:
    reg = dict(L.build_regions(sem_gt, E, O))
    prim = reg[L.PRIMARY]
    obs = S | F_
    reg["completed_unobserved_geo_observed"] = prim & obs
    reg["completed_unobserved_geo_unobserved"] = prim & ~obs
    isobj = torch.zeros(19, dtype=torch.bool, device=sem_gt.device); isobj[list(OBJECT_CLASSES)] = True
    issrf = torch.zeros(19, dtype=torch.bool, device=sem_gt.device); issrf[list(SURFACE_CLASSES)] = True
    g = sem_gt.clamp(0, 18)
    reg["completed_unobserved_objects"] = prim & isobj[g]
    reg["completed_unobserved_surfaces"] = prim & issrf[g]
    return reg


def score_anchor(t: int, device) -> Dict:
    pr = PD.load(t, device)
    sem_gt = L.load_sem_gt(t, device)
    valid, gt_occ = L.load_valid_and_gt_occ(t, device)
    O, E, S, F_ = pr["O"], pr["E"], pr["S"], pr["F"]
    reg = regions_of(sem_gt, E, O, S, F_)
    prim = reg[L.PRIMARY]
    conf = {m: {} for m in PD.METHODS}
    for m in PD.METHODS:
        lab = pr["labels"][m]
        ssc = torch.where(O, lab, torch.zeros_like(lab))
        for r in ALL_REGIONS:
            conf[m][r] = L.SM.confusion(sem_gt, lab if r in SEMANTIC_ONLY else ssc, reg[r])
    # occupancy: one volume, shared -- proven here rather than assumed
    occ_counts = [int((O & gt_occ & valid).sum()), int((O & ~gt_occ & valid).sum()), int((~O & gt_occ & valid).sum())]
    # corrections and damage on the primary population, against B0
    right = {m: (pr["labels"][m] == sem_gt) for m in PD.METHODS}
    fixed, broken = {}, {}
    for m in PD.METHODS:
        if m == "B0":
            continue
        fixed[m] = int((prim & ~right["B0"] & right[m]).sum())
        broken[m] = int((prim & right["B0"] & ~right[m]).sum())
    for a, b in (("Q_N", "P_N"), ("P_R", "P_N"), ("Q_R", "Q_N"), ("Q_R", "P_R")):
        fixed[f"{a}|{b}"] = int((prim & ~right[b] & right[a]).sum())
        broken[f"{a}|{b}"] = int((prim & right[b] & ~right[a]).sum())
    # region-level recognition, measured after the predictions exist
    rec = region_recognition(t, pr, sem_gt, device)
    return {"t": t, "conf": conf, "occ": occ_counts, "fixed": fixed, "broken": broken,
            "recognition": rec, "n_primary": int(prim.sum()),
            "populations": {r: int(reg[r].sum()) for r in ALL_REGIONS},
            "stats": pr["stats"]}


def region_recognition(t: int, pr: Dict, sem_gt: torch.Tensor, device) -> Dict:
    """Per-region accuracy of P and Q against the modal GT class of the region's own evidence voxels.

    The modal class is defined over the region's **owned evidence voxels that carry a human label**
    (``sem_gt`` in 1..18). A region with no such voxel is *unknown*, never a wrong prediction.
    Purity = modal count / labelled count, computed after prediction.
    """
    import proposals as PR
    prop = PR.load(t)
    qrec = json.load(open(os.path.join(P.QWEN_CACHE, f"{t:05d}.json")))
    voxel_region = torch.from_numpy(prop["voxel_region"].astype(np.int64)).to(device)
    out = []
    labelled = (sem_gt >= 1) & (sem_gt <= 18)
    for r_str, q in qrec["regions"].items():
        r = int(r_str)
        sel = (voxel_region == r) & labelled
        n_lab = int(sel.sum())
        row = {"region_id": r, "n_evidence": q["n_evidence_voxels"], "n_labelled": n_lab,
               "n_frames": q["n_frames"], "outcome": q["parsed"]["outcome"],
               "p_label": int(pr["pooled"][r]), "q_label": int(pr["qwen"][r])}
        if n_lab:
            cnt = torch.bincount(sem_gt[sel], minlength=19)[1:]
            row["modal_gt"] = int(cnt.argmax()) + 1
            row["purity"] = float(cnt.max() / n_lab)
        else:
            row["modal_gt"], row["purity"] = None, None
        sv = q.get("single_view")
        row["single_view_label"] = sv["parsed"]["label"] if sv else None
        row["single_view_outcome"] = sv["parsed"]["outcome"] if sv else None
        out.append(row)
    return out


def pooled(rows: List[Dict], key: str) -> Dict:
    """Pool the confusions of ``rows`` and summarise every (method, region)."""
    res = {}
    for m in PD.METHODS:
        res[m] = {}
        for r in ALL_REGIONS:
            cm = np.sum([x["conf"][m][r] for x in rows], axis=0)
            res[m][r] = L.SM.summarize(cm, semantic_only=r in SEMANTIC_ONLY)
            res[m][r]["confusion_trace"] = int(np.trace(cm))
    occ = np.sum([x["occ"] for x in rows], axis=0)
    tp, fp, fn = (int(v) for v in occ)
    res["_occupancy"] = {"tp": tp, "fp": fp, "fn": fn, "sc_iou": tp / max(tp + fp + fn, 1),
                         "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}
    res["_set"] = key
    res["_n_anchors"] = len(rows)
    return res


def bootstrap_delta(rows: List[Dict], a: str, b: str, region: str, n: int = 2000, seed: int = 0) -> Dict:
    """Paired block bootstrap of ``mIoU(a) - mIoU(b)`` on ``region``; one block per anchor."""
    rng = np.random.default_rng(seed)
    ca = np.stack([x["conf"][a][region] for x in rows]).astype(np.float64)
    cb = np.stack([x["conf"][b][region] for x in rows]).astype(np.float64)
    so = region in SEMANTIC_ONLY

    def miou(c):
        s = L.SM.summarize(c, semantic_only=so)["miou"]
        return np.nan if s is None else s

    point = miou(ca.sum(0)) - miou(cb.sum(0))
    draws = np.empty(n)
    for i in range(n):
        k = rng.integers(0, len(rows), len(rows))
        draws[i] = miou(ca[k].sum(0)) - miou(cb[k].sum(0))
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return {"delta": point, "ci95": [float(lo), float(hi)], "n_blocks": len(rows),
            "p_gt_zero": float(np.nanmean(draws > 0))}


def main(device: str = "cuda:0") -> Dict:
    a = MF.select()
    sets = {"evaluation": a["evaluation"], "calibration": a["calibration"], "golden": [a["golden"]]}
    rows_by_set, per_anchor = {}, {}
    with L.FileAudit(patterns=L.EVAL_FORBIDDEN, strict=True) as audit:
        for name, ids in sets.items():
            rows = []
            for t in ids:
                if not os.path.exists(os.path.join(P.PREDICTIONS, f"{t:05d}.npz")):
                    continue
                r = score_anchor(t, device)
                rows.append(r); per_anchor[t] = r
                print(f"  scored t={t} ({name}) primary={r['n_primary']}", flush=True)
            rows_by_set[name] = rows
    out = {"sets": {k: pooled(v, k) for k, v in rows_by_set.items() if v},
           "per_anchor": {str(t): {"n_primary": r["n_primary"], "populations": r["populations"],
                                   "fixed": r["fixed"], "broken": r["broken"],
                                   "miou": {m: L.SM.summarize(r["conf"][m][L.PRIMARY], True)["miou"] for m in PD.METHODS},
                                   "stats": r["stats"]} for t, r in per_anchor.items()},
           "audit": {"opened": len(audit.opened), "violations": audit.violations}}
    ev = rows_by_set.get("evaluation") or []
    if ev:
        out["bootstrap"] = {f"{x}-{y} @ {r}": bootstrap_delta(ev, x, y, r)
                            for r in (L.PRIMARY, "ssc_no_evidence", "ssc_full")
                            for x, y in (("P_N", "B0"), ("Q_N", "P_N"), ("P_R", "P_N"), ("Q_R", "Q_N"), ("Q_R", "P_R"))}
    out["recognition"] = {str(t): r["recognition"] for t, r in per_anchor.items()}
    np.savez_compressed(P.under_experiment(os.path.join(P.SCORES, "confusions.npz")),
                        **{f"{t}__{m}__{r}": per_anchor[t]["conf"][m][r]
                           for t in per_anchor for m in PD.METHODS for r in ALL_REGIONS})
    with open(P.under_experiment(os.path.join(P.ARTIFACTS, "analysis.json")), "w") as f:
        json.dump(out, f, indent=1, default=float)
    return out


if __name__ == "__main__":
    import sys
    o = main(sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
    for name, s in o["sets"].items():
        print(f"\n== {name} ({s['_n_anchors']} anchors) SC IoU {s['_occupancy']['sc_iou']*100:.2f}")
        for r in (L.PRIMARY, "ssc_no_evidence", "ssc_full"):
            print(f"  {r:22s} " + "  ".join(f"{m} {100*s[m][r]['miou']:.2f}" for m in PD.METHODS))
