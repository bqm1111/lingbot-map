#!/usr/bin/env python
"""Gate 5.1 step 2 — occupancy for the four variants plus the frozen Gate-5 baselines.

Conditions: M0 (constant s0), M1 (learned C3), G51-A/B/C/D, OR (LiDAR oracle, diagnostic),
each with raw and the frozen 0.4 m `dilate_r2`. V3 is not run. Only the scalar differs.

    python tools/gate5_1/eval_gate5_1.py --dataset occ3d
"""
from __future__ import annotations

import argparse, csv, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "gate5")))

from gates.scale_gate.config import REPO_ROOT, load_config, write_json
from gates.scale_gate.scale import bootstrap_ci
from gates.voxel_gate.controls import control
from eval_gate5 import CORRECTORS, run_kitti, run_occ3d, scores               # noqa: E402

VARIANTS = ["G51-A", "G51-B", "G51-C", "G51-D"]
BASE = ["M0", "M1", "OR"]
ORACLE = ("OR",)
GATE5_REPRO = {"occ3d": {"M0-R": 0.0647, "M0-D": 0.1597, "M1-R": 0.0411, "M1-D": 0.1434,
                         "OR-R": 0.1027, "OR-D": 0.2292, "G51-A-R": 0.1045,
                         "G51-A-D": 0.2282},
               "kitti": {"M0-R": 0.0573, "M0-D": 0.1592, "M1-R": 0.0778, "M1-D": 0.1759,
                         "OR-R": 0.0769, "OR-D": 0.1757, "G51-A-R": 0.0390,
                         "G51-A-D": 0.1246}}
TOL = 5e-4


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/gate5_1/calibrated_gauge.yaml")
    ap.add_argument("--dataset", required=True, choices=["occ3d", "kitti"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    g5 = load_config(cfg.experiment.gate5_config)
    g4 = load_config(cfg.experiment.gate4_config)
    fv = g4.frozen_values
    dev = torch.device(g5.moge.device if torch.cuda.is_available() else "cpu")
    art = os.path.join(REPO_ROOT, cfg.experiment.output_dir)
    v1_kw = dict(fv.v1_control)
    assert v1_kw == {"kind": "dilate", "radius": 2}
    r_vox = int(round(float(fv.radius_m) / float(fv.canonical_voxel_size)))
    assert r_vox == 3 and abs(v1_kw["radius"] * float(fv.canonical_voxel_size) - 0.4) < 1e-12

    # per-variant scale tables; the Gate-5 builders supply M0 / M1 / OR themselves
    vs = {}
    for v in VARIANTS:
        p = os.path.join(art, f"scales_{v}_{a.dataset}.csv")
        vs[v] = {r["clip_id"]: (float(r["s_moge"]) if int(r["ok"]) else None)
                 for r in csv.DictReader(open(p))}
    builder, unit = (run_occ3d if a.dataset == "occ3d" else run_kitti)(
        g5, dev, {}, v1_kw, r_vox)

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    rows, band_rows, t0 = [], [], time.time()
    skipped = {c: 0 for c in BASE + VARIANTS}
    for i, (cid, group, S, build, labels, bands, to_native) in enumerate(builder()):
        if a.limit and i >= a.limit:
            break
        S = {k: S[k] for k in BASE}
        S.update({v: vs[v].get(cid) for v in VARIANTS})
        canon, meta = {}, {}
        for c in BASE + VARIANTS:
            s = S.get(c)
            if s is None or not np.isfinite(s) or s <= 0:
                skipped[c] += 1
                continue
            occ, R, npts, ingrid = build(float(s))
            canon[(c, "raw")] = occ
            canon[(c, "dilate_r2")] = control(occ, R, **v1_kw)
            meta[c] = {"n_points": npts, "in_grid": ingrid, "scale": float(s)}
        gt, keep = labels()                       # labels opened only now
        for (c, k), pc in canon.items():
            pn = to_native(pc)
            sc = scores(pn, gt, keep)
            rows.append({"clip_id": cid, "group": group, "condition": c, "corrector": k,
                         "is_oracle": int(c in ORACLE), "scale_used": meta[c]["scale"],
                         **sc, "canonical_occupied": int(pc.sum()),
                         "native_occupied": int(pn.sum()),
                         "frac_pred_in_eval_mask": (float((pn & keep).sum()) /
                                                    max(float(pn.sum()), 1.0)),
                         "n_fused_points": meta[c]["n_points"],
                         "frac_points_in_grid": meta[c]["in_grid"]})
            for bn, bm in bands.items():
                band_rows.append({"clip_id": cid, "condition": c, "corrector": k,
                                  "band": bn, **scores(pn & bm, gt & bm, keep)})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}  {time.time()-t0:.0f}s", flush=True)

    by = {}
    for x in rows:
        by.setdefault((x["condition"], x["corrector"]), {})[x["clip_id"]] = x
    name = lambda c, k: f"{c}-{'R' if k == 'raw' else 'D'}"
    mean = lambda key, f: float(np.mean([v[f] for v in by[key].values()]))
    agg = {name(c, k): {"condition": c, "corrector": k, "n_clips": len(by[(c, k)]),
                        "is_oracle": int(c in ORACLE),
                        **{f: mean((c, k), f) for f in
                           ("iou", "precision", "recall", "tp", "fp", "fn",
                            "n_pred_occupied", "canonical_occupied", "native_occupied",
                            "frac_pred_in_eval_mask", "n_fused_points",
                            "frac_points_in_grid", "scale_used")},
                        **{f"iou_{q}": float(np.quantile(
                            [v["iou"] for v in by[(c, k)].values()], p))
                           for q, p in (("p25", .25), ("median", .5), ("p75", .75))}}
           for (c, k) in by}

    repro, ok = {}, True
    for k, w in GATE5_REPRO[a.dataset].items():
        if k not in agg:
            continue
        got = agg[k]["iou"]; d = abs(got - w); ok &= d <= TOL
        repro[k] = {"gate5": w, "gate5_1": got, "abs_diff": d,
                    "within_tolerance": bool(d <= TOL)}
    full = (a.limit is None)
    for k, v in repro.items():
        m = ("OK " if v["within_tolerance"] else "FAIL") if full else "subset"
        print(f"  {m:6s} {k:9s} {v['gate5_1']:.5f} vs {v['gate5']:.5f}  "
              f"d={v['abs_diff']:.2e}")
    if full and not ok:
        write_json(os.path.join(art, f"reproduction_failure_{a.dataset}.json"), repro)
        print("\nGATE5_REPRODUCTION_FAILED"); return 2

    group_of = {x["clip_id"]: x["group"] for x in rows}
    if unit == "scene":
        units = sorted(set(group_of.values()))
        per_unit = {name(c, k): {u: float(np.mean(
            [v["iou"] for cid, v in by[(c, k)].items() if group_of[cid] == u]))
            for u in units} for (c, k) in by}
    else:
        ids = sorted({x["clip_id"] for x in rows})
        bs = int(cfg.eval.kitti_block_size)
        blocks = {f"block{j//bs:03d}": ids[j:j + bs] for j in range(0, len(ids), bs)}
        units = sorted(blocks)
        per_unit = {name(c, k): {u: float(np.mean(
            [by[(c, k)][q]["iou"] for q in blocks[u] if q in by[(c, k)]]))
            for u in units} for (c, k) in by}

    def paired(x, y):
        u = [q for q in units if q in per_unit[x] and q in per_unit[y]]
        d = [per_unit[y][q] - per_unit[x][q] for q in u]
        ci = bootstrap_ci(d, int(cfg.eval.bootstrap_n), int(cfg.eval.bootstrap_seed))
        ci.update({"units_improved": int(np.sum([z > 0 for z in d])), "n_units": len(d),
                   "unit": unit})
        return ci
    PAIRS = []
    for R in ("R", "D"):
        PAIRS += [(f"G51-A-{R}", f"G51-B-{R}"), (f"G51-A-{R}", f"G51-C-{R}"),
                  (f"G51-A-{R}", f"G51-D-{R}"), (f"M0-{R}", f"G51-D-{R}"),
                  (f"M1-{R}", f"G51-D-{R}"), (f"G51-D-{R}", f"OR-{R}")]
    contrasts = {f"{x}->{y}": {**paired(x, y)} for x, y in PAIRS
                 if x in per_unit and y in per_unit}

    margin = float(cfg.eval.non_inferiority_margin_iou)
    ni_ref = "G51-A-D" if a.dataset == "occ3d" else "M1-D"
    ni = paired(ni_ref, "G51-D-D")
    ni.update({"reference": ni_ref, "margin": margin,
               "non_inferior": bool(ni["lo"] > margin),
               "note": "non-inferiority requires the lower bound to exceed the margin; "
                       "a CI crossing zero does not prove equivalence"})

    gain_ret = {}
    for R in ("R", "D"):
        den = agg[f"OR-{R}"]["iou"] - agg[f"M0-{R}"]["iou"]
        for v in VARIANTS:
            num = agg[f"{v}-{R}"]["iou"] - agg[f"M0-{R}"]["iou"]
            gain_ret[f"{v}-{R}"] = float(num / den) if abs(den) > 1e-12 else float("nan")

    bb = {}
    for x in band_rows:
        bb.setdefault(name(x["condition"], x["corrector"]), {}).setdefault(
            x["band"], []).append(x)
    banded = {n: {b: {f: float(np.mean([q[f] for q in v])) for f in
                      ("iou", "precision", "recall", "n_pred_occupied")}
                  for b, v in dd.items()} for n, dd in bb.items()}

    with open(os.path.join(art, f"per_clip_{a.dataset}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(art, f"per_band_{a.dataset}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(band_rows[0])); w.writeheader()
        w.writerows(band_rows)
    write_json(os.path.join(art, f"per_unit_iou_{a.dataset}.json"), per_unit)
    peak = (torch.cuda.max_memory_allocated(dev) / 2**30) if dev.type == "cuda" else 0.0
    write_json(os.path.join(art, f"summary_{a.dataset}.json"), {
        "dataset": a.dataset, "n_clips": len(by[("M0", "raw")]), "n_units": len(units),
        "bootstrap_unit": unit, "skipped_by_condition": skipped,
        "gate5_reproduction": repro, "aggregate": agg, "by_distance": banded,
        "contrasts": contrasts, "non_inferiority": ni,
        "oracle_gain_retention": gain_ret,
        "dilate_r2_expansion_m": v1_kw["radius"] * float(fv.canonical_voxel_size),
        "peak_gpu_gib": peak, "elapsed_s": time.time() - t0})

    print(f"\n{a.dataset}: {len(by[('M0','raw')])} clips / {len(units)} {unit}s")
    print(f"{'cond':10s} {'IoU':>8} {'P':>7} {'R':>7} {'native':>8} {'scale':>8} "
          f"{'in-grid':>8}")
    for n in sorted(agg):
        s = agg[n]
        print(f"{n:9s}{'*' if s['is_oracle'] else ' '} {s['iou']:8.4f} "
              f"{s['precision']:7.3f} {s['recall']:7.3f} {s['native_occupied']:8.0f} "
              f"{s['scale_used']:8.3f} {s['frac_points_in_grid']:8.3f}")
    print(f"\n{'contrast':26s} {'dIoU':>9}  95% CI               {unit}s")
    for k, v in contrasts.items():
        print(f"{k:26s} {v['mean']:+9.4f}  [{v['lo']:+.4f},{v['hi']:+.4f}]"
              f"{'*' if v['excludes_zero'] else ' '} {v['units_improved']:4d}/{v['n_units']}")
    print(f"\nnon-inferiority {ni_ref} -> G51-D-D: d {ni['mean']:+.4f} "
          f"[{ni['lo']:+.4f},{ni['hi']:+.4f}]  margin {margin}  "
          f"-> {'NON-INFERIOR' if ni['non_inferior'] else 'NOT non-inferior'}")
    print("oracle-gain retention: " +
          "  ".join(f"{k} {v:+.1%}" for k, v in gain_ret.items() if k.endswith("-D")))
    print(f"skipped {skipped}  peak {peak:.2f} GiB  {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
