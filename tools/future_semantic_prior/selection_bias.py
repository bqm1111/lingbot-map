"""How unlike the training population is the population the gate is decided on (section 12).

Gate 20 showed that future-revealed voxels are not a random subset of the hidden population: revealability
rises with distance from causal evidence and varies by class. Gate 21 trains on the revealed half and is
decided on the never-revealed half, so the shift between them has to be on the table before the result is
interpreted -- a failure caused by a 5x class-frequency shift is a different finding from a failure of the
mechanism itself.

Four cells are tabulated, not two, so the reveal selection can be separated from the drive change::

    U^R_train   revealed,       0003/0007/0010   <- what the loss is computed on
    U^N_train   never revealed, 0003/0007/0010
    U^R_dev     revealed,       0006             <- E1
    U^N_dev     never revealed, 0006             <- E2, the decision population

Over class, distance-to-evidence bin, height bin, range bin and thing/stuff, each marginal is reported with
its total-variation distance from ``U^R_train``. Nothing is corrected here: section 12 asks for the
measurement, and an inverse-frequency weighting would be a separate ablation, not a silent change.

    PYTHONPATH=$PWD python -m tools.future_semantic_prior.selection_bias
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from tools.delayed_semantic import config as C
from tools.future_revealability import config as G20
from tools.future_semantic_prior import config as G

X, Y, Z = (int(d) for d in C.GRID_DIMS)
OX, OY, OZ = C.GRID_ORIGIN_M
#: height bins in metres above the velodyne origin plane; the grid spans -1.9 .. 4.3 m
HEIGHT_EDGES_M: Tuple[float, ...] = (-1.0, 0.0, 1.0, 2.0)
HEIGHT_NAMES: Tuple[str, ...] = ("<-1 m", "-1-0 m", "0-1 m", "1-2 m", ">2 m")
#: horizontal range from the sensor, metres
RANGE_EDGES_M: Tuple[float, ...] = (10.0, 20.0, 30.0, 40.0)
RANGE_NAMES: Tuple[str, ...] = ("0-10 m", "10-20 m", "20-30 m", "30-40 m", ">40 m")
CELLS: Tuple[str, ...] = ("UR_train", "UN_train", "UR_dev", "UN_dev")


def geometry_of(flat: np.ndarray) -> Dict[str, np.ndarray]:
    """Voxel-centre height and horizontal range in metres from flat C-order indices."""
    f = flat.astype(np.int64)
    ix, iy, iz = f // (Y * Z), (f // Z) % Y, f % Z
    cx = (ix + 0.5) * C.VOXEL_SIZE_M + OX
    cy = (iy + 0.5) * C.VOXEL_SIZE_M + OY
    cz = (iz + 0.5) * C.VOXEL_SIZE_M + OZ
    return {"height_m": cz, "range_m": np.sqrt(cx * cx + cy * cy)}


def _accumulate() -> Dict[str, Dict[str, np.ndarray]]:
    """One pass over both caches; returns raw counts per cell."""
    out = {c: {"class": np.zeros(19, np.int64), "dist": np.zeros(len(G.DISTANCE_NAMES), np.int64),
               "height": np.zeros(len(HEIGHT_NAMES), np.int64), "range": np.zeros(len(RANGE_NAMES), np.int64),
               "thing": np.zeros(2, np.int64), "n": 0} for c in CELLS}
    thing = np.asarray(G.THING_LABELS)
    for drive in G.TRAIN_DRIVES + (G.DEV_DRIVE,):
        is_dev = drive == G.DEV_DRIVE
        root = os.path.join(G20.DATA_ROOT, "population", drive)
        for f in sorted(os.listdir(root)):
            if not f.endswith(".npz"):
                continue
            with np.load(os.path.join(root, f)) as z:
                flat, sem, db = z["flat_index"], z["sem_gt"].astype(np.int64), z["dist_bin"].astype(np.int64)
                rev = np.isfinite(z["vis_first_dt"])
            gm = geometry_of(flat)
            hb = np.digitize(gm["height_m"], np.asarray(HEIGHT_EDGES_M))
            rb = np.digitize(gm["range_m"], np.asarray(RANGE_EDGES_M))
            tb = np.isin(sem, thing).astype(np.int64)
            for sel, name in ((rev, "UR_dev" if is_dev else "UR_train"),
                              (~rev, "UN_dev" if is_dev else "UN_train")):
                o = out[name]
                o["class"] += np.bincount(sem[sel], minlength=19)
                o["dist"] += np.bincount(db[sel], minlength=len(G.DISTANCE_NAMES))
                o["height"] += np.bincount(hb[sel], minlength=len(HEIGHT_NAMES))
                o["range"] += np.bincount(rb[sel], minlength=len(RANGE_NAMES))
                o["thing"] += np.bincount(tb[sel], minlength=2)
                o["n"] += int(sel.sum())
    return out


def _frac(counts: np.ndarray) -> List[float]:
    n = counts.sum()
    return [float(x) for x in (counts / max(n, 1))]


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance between two count vectors, as a percentage."""
    a, b = p / max(p.sum(), 1), q / max(q.sum(), 1)
    return float(100.0 * 0.5 * np.abs(a - b).sum())


def analyse() -> Dict:
    raw = _accumulate()
    ref = raw["UR_train"]
    marg = {"class": ("class", [f"{i}:{n}" for i, n in enumerate(("empty",) + G.CLASS_NAMES)]),
            "dist": ("distance to causal evidence", list(G.DISTANCE_NAMES)),
            "height": ("height above the velodyne plane", list(HEIGHT_NAMES)),
            "range": ("horizontal range from the sensor", list(RANGE_NAMES)),
            "thing": ("thing / stuff", ["stuff", "thing"])}
    out: Dict = {"version": G.VERSION, "n_voxels": {c: raw[c]["n"] for c in CELLS},
                 "reference": "UR_train", "marginals": {}}
    for key, (label, names) in marg.items():
        out["marginals"][key] = {
            "label": label, "names": names,
            "fraction": {c: _frac(raw[c][key]) for c in CELLS},
            "tv_from_UR_train_pct": {c: round(tv_distance(ref[key], raw[c][key]), 3) for c in CELLS},
            "count": {c: [int(x) for x in raw[c][key]] for c in CELLS}}
    # the headline numbers the report quotes
    cf = out["marginals"]["class"]["fraction"]
    ratio = [(G.CLASS_NAMES[i - 1], round(cf["UN_dev"][i] / max(cf["UR_train"][i], 1e-12), 3),
              round(100 * cf["UR_train"][i], 3), round(100 * cf["UN_dev"][i], 3)) for i in range(1, 19)]
    ratio.sort(key=lambda r: -r[1])
    out["class_frequency_ratio_UN_dev_over_UR_train"] = [
        {"class": n, "ratio": r, "pct_UR_train": a, "pct_UN_dev": b} for n, r, a, b in ratio]
    out["summary"] = {
        "tv_class_pct": out["marginals"]["class"]["tv_from_UR_train_pct"]["UN_dev"],
        "tv_distance_pct": out["marginals"]["dist"]["tv_from_UR_train_pct"]["UN_dev"],
        "tv_height_pct": out["marginals"]["height"]["tv_from_UR_train_pct"]["UN_dev"],
        "tv_range_pct": out["marginals"]["range"]["tv_from_UR_train_pct"]["UN_dev"],
        "tv_thing_pct": out["marginals"]["thing"]["tv_from_UR_train_pct"]["UN_dev"],
        "thing_pct_UR_train": round(100 * out["marginals"]["thing"]["fraction"]["UR_train"][1], 3),
        "thing_pct_UN_dev": round(100 * out["marginals"]["thing"]["fraction"]["UN_dev"][1], 3),
        "n_classes_with_ratio_outside_half_to_two": sum(
            1 for r in out["class_frequency_ratio_UN_dev_over_UR_train"] if not 0.5 <= r["ratio"] <= 2.0)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    G.ensure_dirs()
    r = analyse()
    p = os.path.join(G.ARTIFACT_DIR, "selection_bias.json")
    json.dump(r, open(p, "w"), indent=1)
    print(json.dumps({"n_voxels": r["n_voxels"], "summary": r["summary"]}, indent=1))
    print(f"-> {p}")


if __name__ == "__main__":
    main()
