"""Gate 19 analysis: the geometry and semantic learning curves, and the outcome.

Both regimes are reported because they answer different questions:
  fixed_steps  -- the same optimisation budget; more data means more unique anchors seen per update
  fixed_epochs -- the same number of passes over the data; larger sets get proportionally more updates
A curve that rises only under fixed_epochs is an optimisation result, not a data result.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List

import numpy as np

from tools.data_scaling import config as G19

SCALES = list(G19.SCALES)
#: Three regimes, answering three different questions:
#:   fixed_steps   every scale gets 6,000 updates      -> data effect at a SMALL compute budget
#:   fixed_compute every scale gets 40,056 updates     -> data effect at a LARGE compute budget (the control)
#:   fixed_epochs  every scale gets 17.67 passes       -> CONFOUNDED: larger sets also get more updates
#: D_small under fixed_epochs is the same run as under fixed_steps (6,000 steps is already 17.67 epochs at 1,358
#: anchors), so it is aliased rather than retrained.
REGIMES = ("fixed_steps", "fixed_compute", "fixed_epochs")
CONFOUNDED = ("fixed_epochs",)
GEO_REGIONS = ("full", "past5", "unobs5_later", "H_evidence", "no_H_evidence")
SEM_REGIONS = ("completed_never_observed", "never_observed", "ssc_full", "ssc_never_observed")


def load_all() -> List[Dict]:
    out = []
    for f in sorted(glob.glob(os.path.join(G19.ARTIFACT_DIR, "eval", "*.json"))):
        if f.endswith("_geometry.json"):
            continue
        out.append(json.load(open(f)))
    return out


def alias_small(cells: Dict) -> Dict:
    """D_small's fixed_epochs cell IS its fixed_steps cell; without this the fixed_epochs curve has no left endpoint."""
    src = cells.get(("D_small", "fixed_steps"))
    if src is not None and ("D_small", "fixed_epochs") not in cells:
        cells[("D_small", "fixed_epochs")] = dict(src)
    return cells


def by_cell(recs: List[Dict]) -> Dict:
    """(scale, regime) -> per-seed values, plus mean and spread."""
    cells: Dict = {}
    for r in recs:
        key = (r["scale"], r["regime"])
        c = cells.setdefault(key, {"seeds": [], "n_anchors": r["n_anchors"], "steps": r["steps"],
                                   "epochs": r["epochs"], "gpu_hours": 0.0,
                                   "geo": {g: [] for g in GEO_REGIONS}, "sem": {s: [] for s in SEM_REGIONS}})
        c["seeds"].append(r["seed"]); c["gpu_hours"] += r["gpu_hours"]
        for g in GEO_REGIONS:
            c["geo"][g].append(100 * r["geometry"][g]["iou"])
        for s in SEM_REGIONS:
            v = r["semantic"][s]["miou"]
            if v is not None:
                c["sem"][s].append(v)
    for c in cells.values():
        c["geo_mean"] = {g: float(np.mean(v)) for g, v in c["geo"].items() if v}
        c["geo_sd"] = {g: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for g, v in c["geo"].items() if v}
        c["sem_mean"] = {s: float(np.mean(v)) for s, v in c["sem"].items() if v}
        c["sem_sd"] = {s: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for s, v in c["sem"].items() if v}
    return cells


def per_class_table(recs: List[Dict], regime: str, region: str = "completed_never_observed") -> Dict:
    out: Dict[str, Dict[str, float]] = {}
    for sc in SCALES:
        rs = [r for r in recs if r["scale"] == sc and r["regime"] == regime]
        if not rs:
            continue
        acc: Dict[str, List[float]] = {}
        for r in rs:
            for c, v in r["semantic"][region]["per_class_iou"].items():
                if v is not None:
                    acc.setdefault(c, []).append(v)
        out[sc] = {c: float(np.mean(v)) for c, v in acc.items()}
    return out


def convergence(regime: str) -> Dict:
    """Final train and validation loss per scale, to separate under- from over-fitting."""
    out: Dict[str, Dict] = {}
    for sc in SCALES:
        tr, va = [], []
        for f in glob.glob(os.path.join(G19.ARTIFACT_DIR, "runs", f"train_{sc}_{regime}_seed*.json")):
            log = json.load(open(f))["log"]
            steps = [e for e in log if "val" not in e]
            vals = [e["val"] for e in log if "val" in e]
            if steps:
                tr.append(float(np.mean([e["loss"] for e in steps[-200:]])))
            if vals:
                va.append(float(vals[-1]["loss"]))
        if tr:
            out[sc] = {"train_loss": float(np.mean(tr)), "val_loss": float(np.mean(va)) if va else None,
                       "gap": (float(np.mean(va)) - float(np.mean(tr))) if va else None}
    return out


def run() -> Dict:
    recs = load_all()
    cells = alias_small(by_cell(recs))
    res = {"version": G19.VERSION, "n_runs_evaluated": len(recs),
           "curve": {f"{sc}|{rg}": {"n_anchors": c["n_anchors"], "steps": c["steps"], "epochs": c["epochs"],
                                    "seeds": sorted(c["seeds"]), "gpu_hours": c["gpu_hours"],
                                    "geometry": c["geo_mean"], "geometry_sd": c["geo_sd"],
                                    "semantic": c["sem_mean"], "semantic_sd": c["sem_sd"]}
                     for (sc, rg), c in sorted(cells.items())},
           "per_class": {rg: per_class_table(recs, rg) for rg in REGIMES},
           "convergence": {rg: convergence(rg) for rg in REGIMES}}

    def get(sc, rg, kind, key):
        c = cells.get((sc, rg))
        return None if c is None else c[f"{kind}_mean"].get(key)

    deltas = {}
    for rg in REGIMES:
        a, b = get("D_small", rg, "geo", "full"), get("D_all", rg, "geo", "full")
        sa, sb = get("D_small", rg, "sem", "completed_never_observed"), get("D_all", rg, "sem", "completed_never_observed")
        if None not in (a, b, sa, sb):
            deltas[rg] = {"geometry_full_small_to_all": b - a, "semantic_completed_small_to_all": sb - sa,
                          "geometry_small": a, "geometry_all": b, "semantic_small": sa, "semantic_all": sb}
    res["deltas"] = deltas

    # The decomposition the control exists for. fixed_epochs gives the bigger pool BOTH more data and more updates;
    # these three numbers separate the two. Endpoints differ by regime (fixed_compute has no D_all run), so each is
    # labelled with the pools it actually compares.
    dec = {}
    s6, s40 = get("D_small", "fixed_steps", "geo", "full"), get("D_small", "fixed_compute", "geo", "full")
    l40 = get("D_large", "fixed_compute", "geo", "full")
    a40 = get("D_all", "fixed_epochs", "geo", "full")
    if None not in (s6, s40, l40, a40):
        dec["geometry_full"] = {
            "compute_only_D_small_6k_to_40k": s40 - s6,
            "data_only_at_40k_D_small_to_D_large": l40 - s40,
            "confounded_fixed_epochs_D_small_to_D_all": a40 - s6,
            "endpoints": {"D_small@6k": s6, "D_small@40k": s40, "D_large@40k": l40, "D_all@40k": a40}}
    ss6, ss40 = get("D_small", "fixed_steps", "sem", "completed_never_observed"), get("D_small", "fixed_compute", "sem", "completed_never_observed")
    sl40 = get("D_large", "fixed_compute", "sem", "completed_never_observed")
    if None not in (ss6, ss40, sl40):
        dec["semantic_completed"] = {
            "compute_only_D_small_6k_to_40k": ss40 - ss6,
            "data_only_at_40k_D_small_to_D_large": sl40 - ss40,
            "endpoints": {"D_small@6k": ss6, "D_small@40k": ss40, "D_large@40k": sl40}}
    res["decomposition"] = dec

    # Outcome. "Material" is read against the seed spread: a move smaller than twice the seed sd is not a result.
    # The read is taken on fixed_compute when it exists, because that is the only regime where the scales differ ONLY
    # in data. fixed_epochs is never used for the decision: it gives the larger pools more updates as well as more
    # data, so a rise there cannot be attributed.
    if dec.get("geometry_full") and dec.get("semantic_completed"):
        # read on the CONTROL: equal updates, 4.4x the data (D_small -> D_large at 40,056 steps)
        rg = "fixed_compute (D_small -> D_large at equal updates)"
        d = {"geometry_full_small_to_all": dec["geometry_full"]["data_only_at_40k_D_small_to_D_large"],
             "semantic_completed_small_to_all": dec["semantic_completed"]["data_only_at_40k_D_small_to_D_large"],
             "geometry_small": dec["geometry_full"]["endpoints"]["D_small@40k"],
             "geometry_all": dec["geometry_full"]["endpoints"]["D_large@40k"],
             "semantic_small": dec["semantic_completed"]["endpoints"]["D_small@40k"],
             "semantic_all": dec["semantic_completed"]["endpoints"]["D_large@40k"]}
    elif deltas:
        rg = "fixed_steps"
        d = deltas[rg]
    if deltas:
        sd_geo = float(np.mean([c["geo_sd"]["full"] for c in cells.values() if "full" in c["geo_sd"]]))
        sd_sem = float(np.mean([c["sem_sd"]["completed_never_observed"] for c in cells.values()
                                if "completed_never_observed" in c["sem_sd"]]))
        geo_up = d["geometry_full_small_to_all"] > max(2 * sd_geo, 1.0)
        sem_up = d["semantic_completed_small_to_all"] > max(2 * sd_sem, 0.5)
        if geo_up and sem_up:
            outcome, why = "A", "both geometry and semantics keep rising with training data"
        elif geo_up:
            outcome, why = "B", "geometry improves materially with data; semantic SSC does not follow"
        else:
            outcome, why = "C", "both curves flatten early: the formulation, not the data volume, is the limit"
        res["seed_sd"] = {"geometry_full": sd_geo, "semantic_completed": sd_sem}
        res["materiality_bars"] = {"geometry_pp": max(2 * sd_geo, 1.0), "semantic_pp": max(2 * sd_sem, 0.5)}
        res["exceeds_seed_noise"] = {"geometry": d["geometry_full_small_to_all"] > 2 * sd_geo,
                                     "semantic": d["semantic_completed_small_to_all"] > 2 * sd_sem}
        res["outcome"], res["outcome_reason"], res["outcome_regime"] = outcome, why, rg
    os.makedirs(G19.ARTIFACT_DIR, exist_ok=True)
    json.dump(res, open(os.path.join(G19.ARTIFACT_DIR, "analysis.json"), "w"), indent=1, default=float)
    return res


if __name__ == "__main__":
    r = run()
    print("Gate 19 learning curve — %d evaluated runs\n" % r["n_runs_evaluated"])
    for rg in REGIMES:
        rows = [(sc, r["curve"].get(f"{sc}|{rg}")) for sc in SCALES]
        rows = [(sc, v) for sc, v in rows if v]
        if not rows:
            continue
        print("== %s ==" % rg)
        print("%-10s %8s %8s %8s | %10s %10s %10s | %12s %12s" % (
            "pool", "anchors", "steps", "epochs", "geo full", "geo unobs", "geo noH", "sem compl", "sem sscfull"))
        for sc, v in rows:
            print("%-10s %8d %8d %8.2f | %10.2f %10.2f %10.2f | %12.2f %12.2f" % (
                sc, v["n_anchors"], v["steps"], v["epochs"], v["geometry"]["full"], v["geometry"]["unobs5_later"],
                v["geometry"]["no_H_evidence"], v["semantic"].get("completed_never_observed", float("nan")),
                v["semantic"].get("ssc_full", float("nan"))))
        print()
    if "outcome" in r:
        print("seed sd: geometry %.3f, semantic %.3f" % (r["seed_sd"]["geometry_full"], r["seed_sd"]["semantic_completed"]))
        for rg, d in r["deltas"].items():
            print("%s: geometry %+0.2f (%.2f -> %.2f), semantic %+0.2f (%.2f -> %.2f)" % (
                rg, d["geometry_full_small_to_all"], d["geometry_small"], d["geometry_all"],
                d["semantic_completed_small_to_all"], d["semantic_small"], d["semantic_all"]))
        print("\nOUTCOME %s — %s (read on %s)" % (r["outcome"], r["outcome_reason"], r["outcome_regime"]))
