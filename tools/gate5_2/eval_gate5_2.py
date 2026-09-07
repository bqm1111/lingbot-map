#!/usr/bin/env python
"""Gate 5.2 step 6 — the eight deployable conditions plus the diagnostic oracle.

    K360-C0  frozen constant s0        K360-C3  frozen Gate-2 clip head
    K360-A   MoGe, inferred FOV        K360-B   MoGe, calibrated FOV
    K360-OR  LiDAR-oracle scale (diagnostic only)

each with raw occupancy and the frozen deterministic ``dilate_r2`` (0.4 m). V3, the
occupancy CNN and every learned morphology are not run.

Only the *scalar* differs between conditions: depth shape, intrinsics and relative poses
are LingBot's throughout, and MoGe depth, points, mask and predicted intrinsics never
enter the fusion. Targets are opened only after each clip's predictions are built.

    python tools/gate5_2/eval_gate5_2.py
"""
from __future__ import annotations

import argparse, csv, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from prompted_lingbot.occ_datasets import apply_transform
from gates.scale_gate.config import REPO_ROOT, load_config, write_json
from gates.scale_gate.kitti import read_manifest
from gates.scale_gate.scale import bootstrap_ci
from sscbench_kitti360.adapter import (SSCBENCH_KITTI360_GRID as G, binary_target,
                                       load_target)
from gates.voxel_gate.c3 import as4x4, c3_points
from gates.voxel_gate.controls import control
from gates.voxel_gate.voxels import dilate, distance_bins, sparse_voxel_features

CONDITIONS = ["C0", "C3", "A", "B", "OR"]
ORACLE_CONDITIONS = ("OR",)
CORRECTORS = ["raw", "dilate_r2"]


def scores(pred, gt, keep):
    p, t = pred & keep, gt & keep
    tp = int((p & t).sum()); fp = int((p & ~t).sum()); fn = int((~p & t).sum())
    den = tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "iou": tp / den if den else 0.0,
            "n_pred_occupied": int(p.sum())}


def height_bins(bins):
    """Voxel masks split by height above the anchor velodyne origin."""
    iz = np.arange(G.dims[2])
    z = (iz + 0.5) * G.voxel_size + G.origin[2]
    out = {}
    for lo, hi in bins:
        m = np.zeros(G.dims, bool)
        m[:, :, (z >= lo) & (z < hi)] = True
        out[f"{lo:g}..{hi:g}m"] = m
    return out


def read_scales(art):
    S = {}
    for r in csv.DictReader(open(os.path.join(art, "scales_C0C3.csv"))):
        S.setdefault(r["clip_id"], {}).update({"C0": float(r["s_c0"]),
                                               "C3": float(r["s_c3"])})
    for v in ("A", "B"):
        for r in csv.DictReader(open(os.path.join(art, f"scales_{v}.csv"))):
            S.setdefault(r["clip_id"], {})[v] = (float(r["s_moge"]) if int(r["ok"])
                                                 else float("nan"))
    p = os.path.join(art, "oracle_scales.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            S.setdefault(r["clip_id"], {})["OR"] = float(r["s_oracle"])
    return S


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/gate5_2/kitti360_transfer.yaml")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    g4 = load_config(cfg.experiment.gate4_config)
    scfg = load_config("configs/scale_gate/semantickitti.yaml")
    fv = g4.frozen_values
    dev = torch.device(cfg.lingbot.device if torch.cuda.is_available() else "cpu")
    art = os.path.join(REPO_ROOT, cfg.experiment.output_dir)

    v1_kw = dict(fv.v1_control)
    assert v1_kw == {"kind": "dilate", "radius": 2}, "dilate_r2 must be the frozen control"
    r_vox = int(round(float(fv.radius_m) / float(fv.canonical_voxel_size)))
    assert r_vox == 3
    assert abs(v1_kw["radius"] * float(fv.canonical_voxel_size) - 0.4) < 1e-12, \
        "dilate_r2 must remain exactly 0.4 m"
    assert abs(G.voxel_size - float(fv.canonical_voxel_size)) < 1e-12

    conf_thr = float(scfg.lingbot.confidence_threshold)
    dmin, dmax = float(scfg.voxel.min_depth_m), float(scfg.voxel.max_depth_m)
    S = read_scales(art)
    recs = read_manifest(os.path.join(REPO_ROOT, "manifests", "gate5_2", "val.jsonl"))
    if a.limit:
        recs = recs[: a.limit]
    dbands = {k: torch.from_numpy(m).to(dev) for k, m in
              distance_bins(tuple(tuple(b) for b in cfg.eval.distance_bands_m)).items()}
    hbands = {k: torch.from_numpy(m).to(dev) for k, m in
              height_bins([tuple(b) for b in cfg.eval.height_bands_m]).items()}

    rows, dband_rows, hband_rows = [], [], []
    skipped = {c: 0 for c in CONDITIONS}
    t0 = time.time()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    for i, rec in enumerate(recs):
        cid = rec["clip_id"]
        lp = os.path.join(cfg.lingbot.cache_root, cid + ".npz")
        if not os.path.exists(lp) or cid not in S:
            continue
        d = np.load(lp, allow_pickle=False)
        dep = d["pred_depth"].astype(np.float32)
        conf = d["pred_depth_conf"].astype(np.float32)
        K = d["pred_K"].astype(np.float64)
        pose = as4x4(d["pred_pose_c2w"])
        cam_to_velo = d["rect_cam_to_velo"].astype(np.float64)

        pred, meta = {}, {}
        for c in CONDITIONS:
            s = S[cid].get(c)
            if s is None or not np.isfinite(s) or s <= 0:
                skipped[c] += 1
                continue
            pts, fr, cf, dp, _ = c3_points(dep, conf, K, pose, float(s), conf_thr,
                                           dmin, dmax)
            pg = apply_transform(cam_to_velo, pts) if len(pts) else pts
            sp = sparse_voxel_features(pg, fr, cf, dp)
            occ = torch.zeros(int(np.prod(G.dims)), dtype=torch.bool, device=dev)
            if sp["flat"].size:
                occ[torch.from_numpy(sp["flat"]).to(dev)] = True
            occ = occ.view(G.dims)
            idx = (np.floor((pg - np.asarray(G.origin)) / G.voxel_size).astype(np.int64)
                   if len(pg) else np.zeros((0, 3), np.int64))
            kin = np.ones(len(idx), bool)
            for ax in range(3):
                kin &= (idx[:, ax] >= 0) & (idx[:, ax] < G.dims[ax])
            pred[(c, "raw")] = occ
            pred[(c, "dilate_r2")] = control(occ, dilate(occ, r_vox), **v1_kw)
            meta[c] = {"scale": float(s), "n_points": int(len(pts)),
                       "n_in_grid": int(kin.sum()), "n_out_grid": int(len(idx) - kin.sum())}

        target, _ = load_target(cfg.dataset.root, int(rec["anchor"]), G)   # target opened now
        gt_np, keep_np = binary_target(target, G)
        gt = torch.from_numpy(gt_np).to(dev)
        keep = torch.from_numpy(keep_np).to(dev)
        for (c, k), pv in pred.items():
            sc = scores(pv, gt, keep)
            rows.append({"clip_id": cid, "block": int(rec["block"]),
                         "anchor": int(rec["anchor"]), "condition": c, "corrector": k,
                         "is_oracle": int(c in ORACLE_CONDITIONS),
                         "scale_used": meta[c]["scale"], **sc,
                         "n_occupied_pred": int(pv.sum()),
                         "n_valid_eval_voxels": int(keep.sum()),
                         "n_gt_occupied": int((gt & keep).sum()),
                         "frac_pred_in_eval_mask": float((pv & keep).sum()) /
                         max(float(pv.sum()), 1.0),
                         "n_fused_points": meta[c]["n_points"],
                         "n_points_in_grid": meta[c]["n_in_grid"],
                         "n_points_out_grid": meta[c]["n_out_grid"]})
            for bn, bm in dbands.items():
                dband_rows.append({"clip_id": cid, "condition": c, "corrector": k,
                                   "band": bn, **scores(pv & bm, gt & bm, keep)})
            for bn, bm in hbands.items():
                hband_rows.append({"clip_id": cid, "condition": c, "corrector": k,
                                   "band": bn, **scores(pv & bm, gt & bm, keep)})
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(recs)}  {time.time()-t0:.0f}s", flush=True)

    # ---- aggregation ---------------------------------------------------------- #
    by = {}
    for x in rows:
        by.setdefault((x["condition"], x["corrector"]), {})[x["clip_id"]] = x
    name = lambda c, k: f"K360-{c}-{'R' if k == 'raw' else 'D'}"
    mean = lambda key, f: float(np.mean([v[f] for v in by[key].values()]))
    agg = {name(c, k): {
        "condition": c, "corrector": k, "n_clips": len(by[(c, k)]),
        "is_oracle": int(c in ORACLE_CONDITIONS),
        **{f: mean((c, k), f) for f in
           ("iou", "precision", "recall", "tp", "fp", "fn", "n_pred_occupied",
            "n_occupied_pred", "n_valid_eval_voxels", "n_gt_occupied",
            "frac_pred_in_eval_mask", "n_fused_points", "n_points_in_grid",
            "n_points_out_grid", "scale_used")},
        **{f"iou_{q}": float(np.quantile([v["iou"] for v in by[(c, k)].values()], p))
           for q, p in (("p25", .25), ("median", .5), ("p75", .75))}}
        for (c, k) in by}

    block_of = {x["clip_id"]: x["block"] for x in rows}
    units = sorted(set(block_of.values()))
    per_unit = {name(c, k): {u: float(np.mean([v["iou"] for cid, v in by[(c, k)].items()
                                               if block_of[cid] == u])) for u in units
                             if any(block_of[cid] == u for cid in by[(c, k)])}
                for (c, k) in by}

    def paired(x, y):
        u = [v for v in units if v in per_unit[x] and v in per_unit[y]]
        diffs = [per_unit[y][v] - per_unit[x][v] for v in u]
        ci = bootstrap_ci(diffs, int(cfg.eval.bootstrap_n), int(cfg.eval.bootstrap_seed))
        ci.update({"blocks_improved": int(np.sum([q > 0 for q in diffs])),
                   "n_blocks": len(diffs), "unit": "contiguous_block_of_20_clips"})
        return ci

    PAIRS = [("C0", "A"), ("C0", "B"), ("C3", "B"), ("A", "B"), ("B", "OR"),
             ("C0", "C3"), ("C0", "OR"), ("C3", "A")]
    contrasts = {}
    for x, y in PAIRS:
        for R in ("R", "D"):
            nx, ny = f"K360-{x}-{R}", f"K360-{y}-{R}"
            if nx in per_unit and ny in per_unit:
                contrasts[f"{nx}->{ny}"] = paired(nx, ny)

    margin = float(cfg.eval.non_inferiority_margin_iou)
    ni = contrasts.get("K360-A-D->K360-B-D")
    non_inferiority = {
        "contrast": "K360-B-D  -  K360-A-D", "margin": margin,
        "delta": ni["mean"] if ni else None, "ci_lo": ni["lo"] if ni else None,
        "ci_hi": ni["hi"] if ni else None,
        "non_inferior": bool(ni and ni["lo"] > margin)}

    rec_gain = {}
    for tag, R in (("raw", "R"), ("dilate", "D")):
        for cond in ("A", "B"):
            num = agg[f"K360-{cond}-{R}"]["iou"] - agg[f"K360-C0-{R}"]["iou"]
            den = agg[f"K360-OR-{R}"]["iou"] - agg[f"K360-C0-{R}"]["iou"]
            rec_gain[f"retention_{tag}_{cond}"] = float(num / den) if abs(den) > 1e-12 \
                else float("nan")
            rec_gain[f"gain_{tag}_{cond}_minus_C0"] = float(num)
        rec_gain[f"gain_{tag}_OR_minus_C0"] = float(
            agg[f"K360-OR-{R}"]["iou"] - agg[f"K360-C0-{R}"]["iou"])

    def band_agg(brows):
        bb = {}
        for x in brows:
            bb.setdefault(name(x["condition"], x["corrector"]), {}) \
              .setdefault(x["band"], []).append(x)
        return {n: {b: {f: float(np.mean([q[f] for q in v]))
                        for f in ("iou", "precision", "recall", "n_pred_occupied")}
                    for b, v in dd.items()} for n, dd in bb.items()}

    for fn, data in (("gate5_2_per_clip.csv", rows),
                     ("gate5_2_per_distance_band.csv", dband_rows),
                     ("gate5_2_per_height_band.csv", hband_rows)):
        with open(os.path.join(art, fn), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0])); w.writeheader()
            w.writerows(data)
    with open(os.path.join(art, "gate5_2_per_block.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["block"] + sorted(per_unit))
        for u in units:
            w.writerow([u] + [per_unit[n].get(u, "") for n in sorted(per_unit)])

    peak = (torch.cuda.max_memory_allocated(dev) / 2**30) if dev.type == "cuda" else 0.0
    write_json(os.path.join(art, "gate5_2_summary.json"), {
        "dataset": "sscbench_kitti360", "sequence": cfg.dataset.sequence,
        "grid": {"dims": list(G.dims), "voxel_size": G.voxel_size,
                 "origin": list(G.origin), "frame": G.frame},
        "frozen": {"s0": float(fv.s0), "dilate_r2_m": v1_kw["radius"] * G.voxel_size,
                   "conf_threshold": conf_thr, "depth_range_m": [dmin, dmax]},
        "n_clips": len(by[("C0", "raw")]), "scale_failures": skipped,
        "aggregate": agg, "contrasts": contrasts,
        "non_inferiority": non_inferiority, "retention": rec_gain,
        "per_distance_band": band_agg(dband_rows),
        "per_height_band": band_agg(hband_rows),
        "bootstrap": {"n": int(cfg.eval.bootstrap_n), "seed": int(cfg.eval.bootstrap_seed),
                      "unit": "contiguous block of 20 consecutive anchor clips "
                              "(NOT independent scenes)", "n_blocks": len(units)},
        "elapsed_s": time.time() - t0, "peak_gpu_gib": peak})

    print(f"\n{len(by[('C0','raw')])} clips, {time.time()-t0:.0f}s, peak {peak:.2f} GiB")
    for n in sorted(agg):
        print(f"  {n:12s} IoU {agg[n]['iou']:.4f}  P {agg[n]['precision']:.4f}  "
              f"R {agg[n]['recall']:.4f}  scale {agg[n]['scale_used']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
