"""Aggregate the per-anchor Gate 20 records into the numbers the decision rule reads.

Every ratio is reported twice: **micro** (total revealed / total target voxels) and **macro** (mean of the
per-anchor ratios). The pre-registered thresholds are read on micro, over the training drives, at the
`visible` level, with the `mean_probability` readout.

mIoU is the frozen definition throughout: ``semantic_metrics.summarize(conf, semantic_only=True)``, which
averages IoU over classes 1..18 that have GT support and excludes class 0.

    PYTHONPATH=$PWD python -m tools.future_revealability.analyze
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from tools.future_revealability import config as G
from tools.semantic_completion.evaluation import semantic_metrics as SM

NH = len(G.HORIZONS_S)
LEVEL_KEY = {"geometric": "geo_first_dt", "in_frustum": "img_first_dt", "visible": "vis_first_dt",
             "deployable": "dep_first_dt"}
POPS = ("P0", "P1", "P0_correct")
READOUTS = ("first", "latest", "mean_probability")
DT_BINS = np.arange(0.0, 10.05, 0.05)                 # 201 edges -> 200 bins, plus a "never" bin
NVIEW_RANGES = ((1, 1), (2, 2), (3, 5), (6, 10 ** 9))
NVIEW_NAMES = ("1", "2", "3-5", ">5")


def _hist(dt: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """[201] -- 200 time bins plus a final 'never revealed' bin."""
    d = dt[sel]
    h = np.zeros(len(DT_BINS), np.int64)
    fin = np.isfinite(d)
    if fin.any():
        h[:-1] = np.histogram(d[fin], bins=DT_BINS)[0]
    h[-1] = int((~fin).sum())
    return h


def _percentile(hist: np.ndarray, q: float) -> float:
    """Percentile of the reveal time over REVEALED voxels only; nan if none."""
    rev = hist[:-1]
    n = rev.sum()
    if n == 0:
        return float("nan")
    c = np.cumsum(rev)
    i = int(np.searchsorted(c, q * n, side="left"))
    return float(0.5 * (DT_BINS[min(i, 199)] + DT_BINS[min(i, 199) + 1]))


def anchor_summary(z) -> Dict:
    sem = z["sem_gt"].astype(np.int64)
    b0ok = z["b0_correct"].astype(bool)
    dbin = z["dist_bin"].astype(np.int64)
    pops = {"P0": np.ones(len(sem), bool), "P1": ~b0ok, "P0_correct": b0ok}
    dts = {lv: z[k].astype(np.float64) for lv, k in LEVEL_KEY.items()}
    out: Dict = {"n_pop": {p: int(pops[p].sum()) for p in POPS}}

    # revealed counts per level x population x horizon
    rev = {}
    for lv, dt in dts.items():
        for p in POPS:
            rev[f"{lv}|{p}"] = np.asarray([int(((dt <= h) & pops[p]).sum()) for h in G.HORIZONS_S], np.int64)
    out["rev"] = rev

    # distance bins (P0 and P1), per level, per horizon
    dist = {}
    for lv, dt in dts.items():
        for p in ("P0", "P1"):
            m = np.zeros((len(G.DISTANCE_NAMES), NH), np.int64)
            tot = np.zeros(len(G.DISTANCE_NAMES), np.int64)
            for b in range(len(G.DISTANCE_NAMES)):
                s = pops[p] & (dbin == b)
                tot[b] = int(s.sum())
                m[b] = [int(((dt <= h) & s).sum()) for h in G.HORIZONS_S]
            dist[f"{lv}|{p}"] = m
            dist[f"tot|{p}"] = tot
    out["dist"] = dist

    # classes (visible level, P0 and P1)
    cls = {}
    for p in ("P0", "P1"):
        tot = np.zeros(19, np.int64)
        got = np.zeros((19, NH), np.int64)
        for c in range(1, 19):
            s = pops[p] & (sem == c)
            tot[c] = int(s.sum())
            got[c] = [int(((dts["visible"] <= h) & s).sum()) for h in G.HORIZONS_S]
        cls[f"tot|{p}"] = tot
        cls[f"rev|{p}"] = got
    out["cls"] = cls

    # reveal-time histograms
    out["dt_hist"] = {f"{lv}|{p}": _hist(dts[lv], pops[p]) for lv in dts for p in POPS}
    out["dt_hist_dist"] = np.stack([_hist(dts["visible"], dbin == b) for b in range(len(G.DISTANCE_NAMES))])
    out["dt_hist_class"] = np.stack([_hist(dts["visible"], sem == c) for c in range(19)])

    # future-target quality on revealed P0, per readout per horizon
    top = {"first": z["first_top1"].astype(np.int64), "latest": z["last_top1"].astype(np.int64),
           "mean_probability": z["mean_top1"].astype(np.int64)}
    conf = {}
    for ro in READOUTS:
        cs = np.zeros((NH, 19, 19), np.int64)
        for hi, h in enumerate(G.HORIZONS_S):
            s = (dts["visible"] <= h)
            if not s.any():
                continue
            t = top[ro] if ro == "first" else top[ro][:, hi]
            lab = (t[s] + 1)                                   # teacher index 0 == sem label 1
            lab[t[s] == 255] = 0
            cs[hi] = np.bincount(sem[s] * 19 + lab, minlength=361).reshape(19, 19)
        conf[ro] = cs
    out["conf"] = conf
    # ... and stratified by how many future views the voxel got (mean_probability, eventual horizon)
    hi = NH - 1
    nv = z["vis_n_views"][:, hi].astype(np.int64)
    t = top["mean_probability"][:, hi]
    cs = np.zeros((len(NVIEW_RANGES), 19, 19), np.int64)
    for i, (lo, hiv) in enumerate(NVIEW_RANGES):
        s = (nv >= lo) & (nv <= hiv) & np.isfinite(dts["visible"])
        if s.any():
            lab = (t[s] + 1).astype(np.int64); lab[t[s] == 255] = 0
            cs[i] = np.bincount(sem[s] * 19 + lab, minlength=361).reshape(19, 19)
    out["conf_nviews"] = cs

    # the same teacher readout on the voxels B0 gets wrong, and B0's own confusion there (for context)
    s = (dts["visible"] <= G.EVENTUAL_S)
    lab = (top["mean_probability"][:, hi] + 1).astype(np.int64); lab[top["mean_probability"][:, hi] == 255] = 0
    out["conf_P1_future"] = (np.bincount(sem[s & ~b0ok] * 19 + lab[s & ~b0ok], minlength=361).reshape(19, 19)
                             if (s & ~b0ok).any() else np.zeros((19, 19), np.int64))
    out["conf_P0_b0"] = np.bincount(sem * 19 + z["b0_label"].astype(np.int64), minlength=361).reshape(19, 19)
    # the apples-to-apples comparison: B0 and the future target scored on the SAME revealed voxels
    b0l = z["b0_label"].astype(np.int64)
    out["conf_b0_on_revealed"] = (np.bincount(sem[s] * 19 + b0l[s], minlength=361).reshape(19, 19)
                                  if s.any() else np.zeros((19, 19), np.int64))
    out["conf_b0_on_revealed_P1"] = (np.bincount(sem[s & ~b0ok] * 19 + b0l[s & ~b0ok], minlength=361).reshape(19, 19)
                                     if (s & ~b0ok).any() else np.zeros((19, 19), np.int64))
    out["conf_b0_on_unrevealed"] = (np.bincount(sem[~s] * 19 + b0l[~s], minlength=361).reshape(19, 19)
                                    if (~s).any() else np.zeros((19, 19), np.int64))
    # two ceilings over ALL of P0: the future target where it exists and B0 elsewhere (what this
    # supervision can hand a learner), and GT where it exists and B0 elsewhere (a learner that fits the
    # supervision perfectly but learns nothing about the never-revealed part).
    fut = (top["mean_probability"][:, NH - 1] + 1).astype(np.int64)
    fut[top["mean_probability"][:, NH - 1] == 255] = 0
    hyb = np.where(s, fut, b0l)
    out["conf_hybrid_future"] = np.bincount(sem * 19 + hyb, minlength=361).reshape(19, 19)
    out["conf_hybrid_gt"] = np.bincount(sem * 19 + np.where(s, sem, b0l), minlength=361).reshape(19, 19)
    # and the same target restricted to the voxels the frozen label-free selector keeps -- its tier/static
    # rules are what a real pipeline would use to reject a moving surface
    sd = dts["deployable"] <= G.EVENTUAL_S
    out["conf_deployable_target"] = (np.bincount(sem[sd] * 19 + fut[sd], minlength=361).reshape(19, 19)
                                     if sd.any() else np.zeros((19, 19), np.int64))
    out["conf_deployable_b0"] = (np.bincount(sem[sd] * 19 + b0l[sd], minlength=361).reshape(19, 19)
                                 if sd.any() else np.zeros((19, 19), np.int64))

    # region analysis: GT connected components, by how much of their HIDDEN support is revealed
    cid = z["comp_id"].astype(np.int64)
    chid = z["comp_hidden"].astype(np.int64)
    ccls = z["comp_class"].astype(np.int64)
    nrev = np.bincount(cid[np.isfinite(dts["visible"])], minlength=len(chid)).astype(np.int64)
    elig = chid >= G.MIN_REGION_HIDDEN_VOX
    frac = np.divide(nrev, np.maximum(chid, 1), dtype=np.float64)
    thing = np.isin(ccls, G.THING_LABELS)
    reg = {"n_eligible": int(elig.sum()), "n_eligible_thing": int((elig & thing).sum()),
           "n_eligible_stuff": int((elig & ~thing).sum())}
    for f in G.REGION_FRACTIONS:
        reg[f"ge_{int(f*100)}"] = int((elig & (frac >= f)).sum())
        reg[f"ge_{int(f*100)}_thing"] = int((elig & thing & (frac >= f)).sum())
        reg[f"ge_{int(f*100)}_stuff"] = int((elig & ~thing & (frac >= f)).sum())
    out["region"] = reg
    return out


def load_drive(drive: str) -> List[Dict]:
    d = os.path.join(G.DATA_ROOT, "population", drive)
    rows = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".npz"):
            continue
        with np.load(os.path.join(d, f)) as z:
            if len(z["flat_index"]) == 0:
                continue
            s = anchor_summary(z)
        s["t"] = int(f[:-4])
        s["drive"] = drive
        rows.append(s)
    return rows


def _sum(rows: Sequence[Dict], path: Sequence) -> np.ndarray:
    acc = None
    for r in rows:
        v = r
        for k in path:
            v = v[k]
        v = np.asarray(v)
        acc = v.copy() if acc is None else acc + v
    return acc


def bootstrap_ratio(num: np.ndarray, den: np.ndarray, block: int = G.BOOTSTRAP_BLOCK,
                    n: int = G.BOOTSTRAP_RESAMPLES, seed: int = G.SEED) -> Tuple[float, float]:
    """95 % percentile interval of a pooled ratio under a temporal-block bootstrap over anchors."""
    m = len(num)
    if m == 0 or den.sum() == 0:
        return (float("nan"), float("nan"))
    starts = np.arange(0, m, block)
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    nb = len(starts)
    for i in range(n):
        pick = rng.integers(0, nb, nb)
        idx = np.concatenate([np.arange(starts[p], min(starts[p] + block, m)) for p in pick])
        d = den[idx].sum()
        out[i] = 100.0 * num[idx].sum() / d if d else np.nan
    return (float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5)))


def _miou(conf: np.ndarray, subset: Sequence[int] | None = None) -> Dict:
    s = SM.summarize(conf, semantic_only=True)
    if subset is None:
        return {"miou": None if s["miou"] is None else 100.0 * s["miou"], "top1": 100.0 * s["overall_accuracy"], "n": s["n_voxels"],
                "per_class_iou": {G.CLASS_NAMES[c - 1]: (None if s["per_class_iou"][c] is None else 100 * s["per_class_iou"][c])
                                  for c in range(1, 19)}}
    v = [s["per_class_iou"][c] for c in subset if s["per_class_iou"][c] is not None]
    gt = conf.sum(1)
    tp = np.diag(conf)
    den = gt[list(subset)].sum()
    return {"miou": 100.0 * float(np.mean(v)) if v else None,
            "top1": 100.0 * float(tp[list(subset)].sum() / den) if den else None, "n": int(den)}


def summarize(rows: Sequence[Dict], label: str) -> Dict:
    stuff = [c for c in range(1, 19) if c not in G.THING_LABELS]
    npop = {p: int(sum(r["n_pop"][p] for r in rows)) for p in POPS}
    out: Dict = {"label": label, "n_anchors": len(rows), "n_pop": npop, "horizons_s": list(G.HORIZONS_S)}

    curves = {}
    for lv in G.DIAG_LEVELS:
        for p in POPS:
            num = np.stack([r["rev"][f"{lv}|{p}"] for r in rows])          # [A, NH]
            den = np.asarray([r["n_pop"][p] for r in rows], np.int64)
            micro = 100.0 * num.sum(0) / max(den.sum(), 1)
            with np.errstate(invalid="ignore", divide="ignore"):
                per = np.where(den[:, None] > 0, 100.0 * num / np.maximum(den[:, None], 1), np.nan)
            ci = [bootstrap_ratio(num[:, h], den) for h in range(NH)]
            curves[f"{lv}|{p}"] = {"micro": micro.tolist(), "macro": np.nanmean(per, 0).tolist(),
                                   "macro_p25": np.nanpercentile(per, 25, axis=0).tolist(),
                                   "macro_p75": np.nanpercentile(per, 75, axis=0).tolist(),
                                   "ci95": ci, "n_revealed": num.sum(0).tolist(), "n_target": int(den.sum())}
    out["curves"] = curves

    dist = {}
    for lv in G.DIAG_LEVELS:
        for p in ("P0", "P1"):
            num = _sum(rows, ("dist", f"{lv}|{p}"))                        # [B, NH]
            den = _sum(rows, ("dist", f"tot|{p}"))
            far_n = np.stack([r["dist"][f"{lv}|{p}"][list(G.FAR_BINS)].sum(0) for r in rows])
            far_d = np.asarray([r["dist"][f"tot|{p}"][list(G.FAR_BINS)].sum() for r in rows], np.int64)
            vf_n = np.stack([r["dist"][f"{lv}|{p}"][list(G.VERY_FAR_BINS)].sum(0) for r in rows])
            vf_d = np.asarray([r["dist"][f"tot|{p}"][list(G.VERY_FAR_BINS)].sum() for r in rows], np.int64)
            dist[f"{lv}|{p}"] = {
                "n_target": den.tolist(),
                "revealability": (100.0 * num / np.maximum(den[:, None], 1)).tolist(),
                "far_micro": (100.0 * far_n.sum(0) / max(far_d.sum(), 1)).tolist(),
                "far_ci95": bootstrap_ratio(far_n[:, NH - 1], far_d), "far_n_target": int(far_d.sum()),
                "very_far_micro": (100.0 * vf_n.sum(0) / max(vf_d.sum(), 1)).tolist(),
                "very_far_ci95": bootstrap_ratio(vf_n[:, NH - 1], vf_d), "very_far_n_target": int(vf_d.sum())}
    out["distance"] = dist
    dh = _sum(rows, ("dt_hist_dist",))
    out["distance_reveal_time"] = {G.DISTANCE_NAMES[b]: {"median_s": _percentile(dh[b], 0.5),
                                                         "p90_s": _percentile(dh[b], 0.9)}
                                   for b in range(len(G.DISTANCE_NAMES))}

    ch = _sum(rows, ("dt_hist_class",))
    per_class = {}
    for p in ("P0", "P1"):
        tot = _sum(rows, ("cls", f"tot|{p}"))
        got = _sum(rows, ("cls", f"rev|{p}"))
        per_class[p] = {G.CLASS_NAMES[c - 1]: {"n_target": int(tot[c]),
                                               "n_revealed": int(got[c, NH - 1]),
                                               "revealability": float(100.0 * got[c, NH - 1] / tot[c]) if tot[c] else None,
                                               "median_reveal_s": _percentile(ch[c], 0.5),
                                               "thing": c in G.THING_LABELS}
                        for c in range(1, 19)}
    out["per_class"] = per_class
    tot = _sum(rows, ("cls", "tot|P0"))
    got = _sum(rows, ("cls", "rev|P0"))[:, NH - 1]
    out["thing_stuff"] = {
        "thing": {"n_target": int(tot[list(G.THING_LABELS)].sum()), "n_revealed": int(got[list(G.THING_LABELS)].sum()),
                  "revealability": float(100.0 * got[list(G.THING_LABELS)].sum() / max(tot[list(G.THING_LABELS)].sum(), 1))},
        "stuff": {"n_target": int(tot[stuff].sum()), "n_revealed": int(got[stuff].sum()),
                  "revealability": float(100.0 * got[stuff].sum() / max(tot[stuff].sum(), 1))}}
    out["n_classes_ge_20pct"] = int(sum(1 for c in range(1, 19) if tot[c] > 0 and 100.0 * got[c] / tot[c] >= 20.0))
    out["n_classes_with_support"] = int((tot[1:] > 0).sum())

    quality = {}
    for ro in READOUTS:
        cs = _sum(rows, ("conf", ro))
        quality[ro] = {"eventual": _miou(cs[NH - 1]),
                       "thing": _miou(cs[NH - 1], G.THING_LABELS), "stuff": _miou(cs[NH - 1], stuff),
                       "by_horizon": [{"horizon_s": G.HORIZONS_S[h], **{k: v for k, v in _miou(cs[h]).items() if k != "per_class_iou"}}
                                      for h in range(NH)]}
    out["future_target_quality"] = quality
    cn = _sum(rows, ("conf_nviews",))
    out["quality_by_n_views"] = {NVIEW_NAMES[i]: {k: v for k, v in _miou(cn[i]).items() if k != "per_class_iou"}
                                 for i in range(len(NVIEW_NAMES))}
    out["quality_on_P1"] = {k: v for k, v in _miou(_sum(rows, ("conf_P1_future",))).items() if k != "per_class_iou"}
    out["b0_on_P0"] = {k: v for k, v in _miou(_sum(rows, ("conf_P0_b0",))).items() if k != "per_class_iou"}
    stuff_ = [c for c in range(1, 19) if c not in G.THING_LABELS]
    cb = _sum(rows, ("conf_b0_on_revealed",))
    cf = _sum(rows, ("conf", "mean_probability"))[NH - 1]
    out["head_to_head_on_revealed"] = {
        "b0": {**{k: v for k, v in _miou(cb).items() if k != "per_class_iou"},
               "thing_miou": _miou(cb, G.THING_LABELS)["miou"], "stuff_miou": _miou(cb, stuff_)["miou"]},
        "future_target": {**{k: v for k, v in _miou(cf).items() if k != "per_class_iou"},
                          "thing_miou": _miou(cf, G.THING_LABELS)["miou"], "stuff_miou": _miou(cf, stuff_)["miou"]},
        "b0_on_P1_revealed": {k: v for k, v in _miou(_sum(rows, ("conf_b0_on_revealed_P1",))).items() if k != "per_class_iou"},
        "b0_on_unrevealed": {k: v for k, v in _miou(_sum(rows, ("conf_b0_on_unrevealed",))).items() if k != "per_class_iou"}}
    cd_ = _sum(rows, ("conf_deployable_target",)); cdb = _sum(rows, ("conf_deployable_b0",))
    out["deployable_level_quality"] = {
        "future_target": {**{k: v for k, v in _miou(cd_).items() if k != "per_class_iou"},
                          "thing_miou": _miou(cd_, G.THING_LABELS)["miou"], "stuff_miou": _miou(cd_, stuff_)["miou"]},
        "b0": {**{k: v for k, v in _miou(cdb).items() if k != "per_class_iou"},
               "thing_miou": _miou(cdb, G.THING_LABELS)["miou"], "stuff_miou": _miou(cdb, stuff_)["miou"]}}
    out["ceilings_on_all_P0"] = {
        "b0": out["b0_on_P0"],
        "future_where_revealed_else_b0": {k: v for k, v in _miou(_sum(rows, ("conf_hybrid_future",))).items() if k != "per_class_iou"},
        "gt_where_revealed_else_b0": {k: v for k, v in _miou(_sum(rows, ("conf_hybrid_gt",))).items() if k != "per_class_iou"}}

    for p in POPS:
        h = _sum(rows, ("dt_hist", f"visible|{p}"))
        out.setdefault("reveal_time", {})[p] = {
            "median_s": _percentile(h, 0.5), "p90_s": _percentile(h, 0.9),
            "never_revealed_frac": float(100.0 * h[-1] / max(h.sum(), 1)),
            "cdf": (np.cumsum(h[:-1]) / max(h.sum(), 1) * 100.0).tolist()}

    reg = {k: int(sum(r["region"][k] for r in rows)) for k in rows[0]["region"]}
    out["region"] = {**reg, **{f"frac_ge_{int(f*100)}": (100.0 * reg[f"ge_{int(f*100)}"] / reg["n_eligible"] if reg["n_eligible"] else None)
                               for f in G.REGION_FRACTIONS}}
    return out


def decide(train: Dict) -> Dict:
    T = G.Thresholds()
    lv, ro = G.DECISION_LEVEL, G.DECISION_READOUT
    p0 = train["curves"][f"{lv}|P0"]["micro"][NH - 1]
    p1 = train["curves"][f"{lv}|P1"]["micro"][NH - 1]
    far = train["distance"][f"{lv}|P0"]["far_micro"][NH - 1]
    q = train["future_target_quality"][ro]["eventual"]
    top1, miou = q["top1"], (q["miou"] or 0.0)                      # both already in percent
    nc = train["n_classes_ge_20pct"]
    A = {"A1": p0 >= T.a_p0_eventual, "A2": p1 >= T.a_p1_eventual, "A3": far >= T.a_far_eventual,
         "A4": top1 >= T.a_teacher_top1 and miou >= T.a_teacher_miou, "A5": nc >= T.a_min_classes_20pct}
    Cc = {"C1": p0 < T.c_p0_eventual, "C2": p1 < T.c_p1_eventual, "C3": far < T.c_far_eventual,
          "C4": top1 < T.c_teacher_top1 or miou < T.c_teacher_miou}
    outcome = "C" if any(Cc.values()) else ("A" if all(A.values()) else "B")
    return {"outcome": outcome, "A": A, "C": Cc, "values": {"p0_eventual": p0, "p1_eventual": p1,
            "far_eventual": far, "teacher_top1": top1, "teacher_miou": miou, "n_classes_ge_20pct": nc},
            "thresholds": T.__dict__, "level": lv, "readout": ro}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(G.ARTIFACT_DIR, "analysis.json"))
    a = ap.parse_args()
    by_drive = {d: load_drive(d) for d in list(G.TRAIN_DRIVES) + [G.DEV_DRIVE]}
    train_rows = [r for d in G.TRAIN_DRIVES for r in by_drive[d]]
    res = {"version": G.VERSION, "horizons_s": list(G.HORIZONS_S), "levels": list(G.LEVELS),
           "decision_level": G.DECISION_LEVEL, "decision_readout": G.DECISION_READOUT,
           "train": summarize(train_rows, "training drives (0003+0007+0010)"),
           "dev": summarize(by_drive[G.DEV_DRIVE], "0006 (held out from every threshold)"),
           "per_drive": {d: summarize(by_drive[d], d) for d in by_drive}}
    res["decision"] = decide(res["train"])
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps(res["decision"], indent=1))


if __name__ == "__main__":
    main()
