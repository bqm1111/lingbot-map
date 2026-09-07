#!/usr/bin/env python
"""Gate 4.1 step 2 — the 2x2 geometry factorization on Occ3D-nuScenes val.

Five geometry conditions x four correctors, plus a matched in-band oracle per condition.
The frozen Gate-4 stack is reproduced as a special case: G00/raw must equal T0 and the C3
rows must equal T1/T2/T4/T6, which is asserted before any new number is interpreted.

    G00  const s0    + LingBot poses      DEPLOYABLE
    G01  LiDAR scale + LingBot poses      ORACLE (privileged scale)
    G10  const s0    + GT poses           ORACLE (privileged poses)
    G11  LiDAR scale + GT poses           ORACLE (both)
    C3   learned s   + LingBot poses      reference row, not a factorial cell

Occ3D labels are opened only after every prediction for a clip is finalised.

    python tools/occ3d_zeroshot/eval_factorization.py
"""
from __future__ import annotations

import argparse, csv, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, load_config, write_json
from gates.scale_gate.scale import bootstrap_ci
from gates.voxel_gate.controls import control
from occ3d_zeroshot.factorization import (
    DEPLOYABLE_CELLS, ORACLE_CELLS, fuse, ground_height, gt_camera_to_world,
    relative_transforms,
)
from occ3d_zeroshot.grid import (
    CANONICAL, NATIVE, canonical_to_native, native_binary_target, native_distance_bands,
    points_to_canonical,
)
from occ3d_zeroshot.nuscenes_adapter import load_annotations, scene_frames
from occ3d_zeroshot.pipeline import (
    canonical_features, clip_scale, load_corrector, load_depth_head, region_from,
    run_corrector,
)

CELLS = {"G00": ("const", "pred"), "G01": ("oracle", "pred"),
         "G10": ("const", "gt"), "G11": ("oracle", "gt"),
         "C3": ("c3", "pred")}
CORRECTORS = ["raw", "dilate_r2", "occ_only", "v3"]
# Gate-4 rows this run must reproduce before anything else is read.
GATE4_MAP = {("G00", "raw"): "T0", ("C3", "raw"): "T1", ("C3", "dilate_r2"): "T2",
             ("C3", "occ_only"): "T4", ("C3", "v3"): "T6"}
TOL = 5e-4


def scores(pred, gt, keep):
    p, t = pred & keep, gt & keep
    tp = int((p & t).sum()); fp = int((p & ~t).sum()); fn = int((~p & t).sum())
    den = tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "iou": tp / den if den else 0.0,
            "n_pred_occupied": int(p.sum())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/occ3d_zeroshot/gate41_factorization.yaml")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    g4 = load_config(cfg.experiment.gate4_config)
    fv = g4.frozen_values
    dev = torch.device(g4.lingbot.device if torch.cuda.is_available() else "cpu")
    art = os.path.join(REPO_ROOT, cfg.experiment.output_dir)
    os.makedirs(art, exist_ok=True)

    r_vox = int(round(float(fv.radius_m) / float(fv.canonical_voxel_size)))
    assert r_vox == 3 and abs(r_vox * float(fv.canonical_voxel_size) - 0.6) < 1e-12
    v1_kw = dict(fv.v1_control)
    assert v1_kw == {"kind": "dilate", "radius": 2}
    # dilate_r2 expands by 2 voxels = 0.4 m (a 5x5x5 max-pool); the V3 band is 3 voxels
    # = 0.6 m. These are different physical quantities and are never conflated.
    DILATE_M = v1_kw["radius"] * float(fv.canonical_voxel_size)
    assert abs(DILATE_M - 0.4) < 1e-12 and abs(float(fv.radius_m) - 0.6) < 1e-12

    head, hck = load_depth_head(os.path.join(REPO_ROOT, g4.frozen.depth_head), dev)
    corr = {"v3": load_corrector(os.path.join(REPO_ROOT, g4.frozen.full_s0), dev),
            "occ_only": load_corrector(os.path.join(REPO_ROOT, g4.frozen.occ_only_s0), dev)}
    for m, _ in list(corr.values()) + [(head, None)]:
        assert not any(p.requires_grad for p in m.parameters())
    for k, (_, ck) in corr.items():
        assert int(ck["radius"]) == 3 and float(ck["threshold"]) == float(fv.tau)

    geo = {r["clip_id"]: r for r in csv.DictReader(
        open(os.path.join(art, "geometry_per_clip.csv")))}
    ann = load_annotations(g4.data.occ3d_root)
    recs = [json.loads(l) for l in
            open(os.path.join(REPO_ROOT, "manifests", "occ3d_zeroshot", "val.jsonl"))]
    if a.limit:
        recs = recs[: a.limit]
    bands = {k: torch.from_numpy(m).to(dev) for k, m in
             native_distance_bands([tuple(b) for b in cfg.eval.distance_bands_m]).items()}
    gp = cfg.eval.ground_plane
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    fmap_cache, rows, band_rows = {}, [], []
    t0, t_inf, n_excluded = time.time(), 0.0, 0
    for i, r in enumerate(recs):
        cp = os.path.join(g4.data.cache_root, r["clip_id"] + ".npz")
        lp = os.path.join(g4.data.occ3d_root, r["anchor_gt_path"])
        g = geo.get(r["clip_id"])
        if not (os.path.exists(cp) and os.path.exists(lp) and g):
            continue
        if not int(g["oracle_scale_ok"]):
            n_excluded += 1                     # policy fixed in advance: exclude+report
            continue
        d = np.load(cp)
        dep = d["pred_depth"].astype(np.float32)
        conf = d["pred_depth_conf"].astype(np.float32)
        K = d["pred_K"].astype(np.float64)
        pose_pred = d["pred_pose_c2w"].astype(np.float64)
        Tce = d["T_camera_to_ego"][-1].astype(np.float64)
        anchor = dep.shape[0] - 1
        if r["scene"] not in fmap_cache:
            fmap_cache[r["scene"]] = {f.token: f for f in
                                      scene_frames(ann, r["scene"], g4.data.camera,
                                                   g4.data.nuscenes_root)}
        frames = [fmap_cache[r["scene"]][t] for t in r["sample_tokens"]]
        c2w_gt = gt_camera_to_world(frames)
        _, s_c3, _, _ = clip_scale(head, hck, dep, conf, float(fv.s0),
                                   float(fv.confidence_threshold),
                                   float(fv.min_depth_m), float(fv.max_depth_m), dev)
        S = {"const": float(fv.s0), "oracle": float(g["s_oracle"]), "c3": s_c3}

        canon, meta = {}, {}
        for cell, (stag, pmode) in CELLS.items():
            s = S[stag]
            # GT translations are metric: the scale must not reach that branch
            rel = relative_transforms(pose_pred, c2w_gt, anchor, pmode,
                                      s if pmode == "pred" else float("nan"))
            pe, fr, cf, dp = fuse(dep, conf, K, rel, Tce, s,
                                  float(fv.confidence_threshold),
                                  float(fv.min_depth_m), float(fv.max_depth_m))
            feat5, _ = canonical_features(pe, fr, cf, dp, dev)
            occ = feat5[0] > 0
            R = region_from(occ, r_vox)
            _, keep_in = points_to_canonical(pe) if len(pe) else (None, np.zeros(0, bool))
            meta[cell] = {"n_points": len(pe),
                          "in_grid": float(keep_in.mean()) if len(pe) else 0.0,
                          "ground_z": ground_height(pe, tuple(gp.x_range_m),
                                                    float(gp.abs_y_max_m),
                                                    float(gp.percentile)),
                          "R": R, "scale": s}
            canon[(cell, "raw")] = occ
            canon[(cell, "dilate_r2")] = control(occ, R, **v1_kw)
            for name in ("occ_only", "v3"):
                m, ck = corr[name]
                ts = time.time()
                canon[(cell, name)] = run_corrector(
                    m, feat5, R, ck["norm"], float(ck["threshold"]),
                    name == "occ_only")
                t_inf += time.time() - ts

        # ---- labels opened only now; every prediction above is already fixed ---- #
        lab = np.load(lp)
        gt_np, keep_np = native_binary_target(
            lab, bool(g4.eval.apply_camera_mask), bool(g4.eval.apply_lidar_mask),
            int(g4.eval.single_camera_x_cut))
        gt = torch.from_numpy(gt_np).to(dev)
        keep = torch.from_numpy(keep_np).to(dev)
        gtc = gt.repeat_interleave(2, 0).repeat_interleave(2, 1).repeat_interleave(2, 2)
        for cell in CELLS:
            canon[(cell, "REF")] = gtc & meta[cell]["R"]

        base = {c: canonical_to_native(canon[(c, "raw")]) for c in CELLS}
        for (cell, cor), pc in canon.items():
            pn = canonical_to_native(pc)
            sc = scores(pn, gt, keep)
            rows.append({
                "clip_id": r["clip_id"], "scene": r["scene"], "cell": cell,
                "corrector": cor, "scale_kind": CELLS[cell][0],
                "pose_kind": CELLS[cell][1],
                "is_oracle": int(cell in ORACLE_CELLS), **sc,
                "canonical_occupied": int(pc.sum()), "native_occupied": int(pn.sum()),
                "added_vs_raw_native": int((pn & ~base[cell] & keep).sum()),
                "removed_vs_raw_native": int((~pn & base[cell] & keep).sum()),
                "frac_pred_in_eval_mask": (float((pn & keep).sum()) /
                                           max(float(pn.sum()), 1.0)),
                "n_fused_points": meta[cell]["n_points"],
                "frac_points_in_grid": meta[cell]["in_grid"],
                "ground_z_p5_m": meta[cell]["ground_z"],
                "scale_used": meta[cell]["scale"],
                "n_eval_voxels": int(keep.sum()),
                "n_gt_occupied": int((gt & keep).sum())})
            for bn, bm in bands.items():
                band_rows.append({"clip_id": r["clip_id"], "cell": cell,
                                  "corrector": cor, "band": bn,
                                  **scores(pn & bm, gt & bm, keep)})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(recs)}  {time.time()-t0:.0f}s", flush=True)

    # ---- aggregate ---------------------------------------------------------- #
    by = {}
    for x in rows:
        by.setdefault((x["cell"], x["corrector"]), {})[x["clip_id"]] = x
    n_clips = len(by[("G00", "raw")])
    mean = lambda k, f: float(np.mean([v[f] for v in by[k].values()]))
    agg = {f"{c}|{k}": {"cell": c, "corrector": k, "n_clips": len(by[(c, k)]),
                        "is_oracle": int(c in ORACLE_CELLS),
                        **{f: mean((c, k), f) for f in
                           ("iou", "precision", "recall", "tp", "fp", "fn",
                            "n_pred_occupied", "canonical_occupied", "native_occupied",
                            "added_vs_raw_native", "removed_vs_raw_native",
                            "frac_pred_in_eval_mask", "n_fused_points",
                            "frac_points_in_grid", "ground_z_p5_m", "scale_used")},
                        **{f"iou_{q}": float(np.quantile(
                            [v["iou"] for v in by[(c, k)].values()], p))
                           for q, p in (("p25", .25), ("median", .5), ("p75", .75))}}
           for c in CELLS for k in CORRECTORS + ["REF"]}

    # ---- Gate-4 reproduction gate ------------------------------------------- #
    g4sum = json.load(open(os.path.join(REPO_ROOT, g4.experiment.output_dir,
                                        "eval", "summary.json")))
    repro, ok = {}, True
    for (cell, cor), tid in GATE4_MAP.items():
        got = agg[f"{cell}|{cor}"]["iou"]
        want = g4sum["aggregate"][tid]["iou"]
        d = abs(got - want); ok &= d <= TOL
        repro[f"{cell}|{cor} == {tid}"] = {"gate4": want, "gate41": got,
                                           "abs_diff": d, "within_tolerance": d <= TOL}
    # The Gate-4 figures are means over the full 1 182-clip protocol; a subset run
    # cannot match them, so the gate is enforced only on a complete evaluation.
    full_run = (n_clips == int(g4sum["n_clips"]))
    for k, v in repro.items():
        mark = ("OK " if v["within_tolerance"] else "FAIL") if full_run else "subset"
        print(f"  {mark:6s} {k:22s} {v['gate41']:.5f} vs {v['gate4']:.5f}  "
              f"d={v['abs_diff']:.2e}")
    if not full_run:
        print(f"  (subset run: {n_clips}/{g4sum['n_clips']} clips, reproduction gate "
              f"not enforced)")
    if full_run and not ok:
        write_json(os.path.join(art, "reproduction_failure.json"), repro)
        print("\nGATE4_REPRODUCTION_FAILED"); return 2

    scene_of = {x["clip_id"]: x["scene"] for x in rows}
    scenes = sorted(set(scene_of.values()))
    per_scene = {f"{c}|{k}": {s: float(np.mean(
        [v["iou"] for cid, v in by[(c, k)].items() if scene_of[cid] == s]))
        for s in scenes} for c in CELLS for k in CORRECTORS + ["REF"]}

    def paired(x, y):
        d = [per_scene[y][s] - per_scene[x][s] for s in scenes]
        ci = bootstrap_ci(d, int(cfg.eval.bootstrap_n), int(cfg.eval.bootstrap_seed))
        ci.update({"scenes_improved": int(np.sum([v > 0 for v in d])),
                   "n_scenes": len(d)})
        return ci

    contrasts = {}
    for k in ("raw", "dilate_r2", "occ_only", "v3"):
        for tag, (x, y) in {"pose@const": ("G00", "G10"), "pose@oracle": ("G01", "G11"),
                            "scale@pred": ("G00", "G01"), "scale@gt": ("G10", "G11")}.items():
            contrasts[f"{tag}|{k}"] = {"from": f"{x}|{k}", "to": f"{y}|{k}",
                                       **paired(f"{x}|{k}", f"{y}|{k}")}
    for c in CELLS:
        for x, y in (("raw", "dilate_r2"), ("raw", "occ_only"), ("raw", "v3"),
                     ("dilate_r2", "v3"), ("occ_only", "v3")):
            contrasts[f"{c}:{x}->{y}"] = {"from": f"{c}|{x}", "to": f"{c}|{y}",
                                          **paired(f"{c}|{x}", f"{c}|{y}")}

    # difference in differences: (G11 - G10) - (G01 - G00)
    did = {}
    for k in CORRECTORS:
        d = [(per_scene[f"G11|{k}"][s] - per_scene[f"G10|{k}"][s])
             - (per_scene[f"G01|{k}"][s] - per_scene[f"G00|{k}"][s]) for s in scenes]
        did[k] = {**bootstrap_ci(d, int(cfg.eval.bootstrap_n), int(cfg.eval.bootstrap_seed)),
                  "n_scenes": len(d)}

    bb = {}
    for x in band_rows:
        bb.setdefault(f"{x['cell']}|{x['corrector']}", {}).setdefault(x["band"], []).append(x)
    banded = {k: {b: {f: float(np.mean([q[f] for q in v])) for f in
                      ("iou", "precision", "recall", "n_pred_occupied")}
                  for b, v in dd.items()} for k, dd in bb.items()}

    with open(os.path.join(art, "factorization_per_clip.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(art, "factorization_per_band.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(band_rows[0])); w.writeheader()
        w.writerows(band_rows)
    write_json(os.path.join(art, "factorization_per_scene_iou.json"), per_scene)

    peak = (torch.cuda.max_memory_allocated(dev) / 2**30) if dev.type == "cuda" else 0.0
    write_json(os.path.join(art, "factorization_summary.json"), {
        "n_clips": n_clips, "n_scenes": len(scenes), "n_excluded_oracle_failure": n_excluded,
        "cells": {c: {"scale": CELLS[c][0], "poses": CELLS[c][1],
                      "is_oracle": c in ORACLE_CELLS,
                      "deployable": c in DEPLOYABLE_CELLS} for c in CELLS},
        "dilate_r2_expansion_m": DILATE_M, "v3_band_radius_m": float(fv.radius_m),
        "radius_voxels": r_vox,
        "gate4_reproduction": repro,
        "aggregate": agg, "by_distance": banded, "contrasts": contrasts,
        "difference_in_differences": did,
        "inference_seconds_per_clip": t_inf / max(n_clips, 1),
        "peak_gpu_gib": peak, "elapsed_s": time.time() - t0})

    print(f"\n{n_clips} clips / {len(scenes)} scenes  (excluded {n_excluded})")
    print(f"{'cell':5s} {'corrector':10s} {'IoU':>8} {'P':>7} {'R':>7} "
          f"{'native':>8} {'canon':>9} {'pts in-grid':>11} {'ground z':>9}")
    for c in CELLS:
        for k in CORRECTORS + ["REF"]:
            s = agg[f"{c}|{k}"]
            tag = "*" if c in ORACLE_CELLS else " "
            print(f"{c:4s}{tag} {k:10s} {s['iou']:8.4f} {s['precision']:7.3f} "
                  f"{s['recall']:7.3f} {s['native_occupied']:8.0f} "
                  f"{s['canonical_occupied']:9.0f} {s['frac_points_in_grid']:11.3f} "
                  f"{s['ground_z_p5_m']:9.3f}")
    print("\n* = non-deployable diagnostic oracle")
    print(f"\n{'contrast':22s} {'dIoU':>9}  95% CI               scenes")
    for k, v in contrasts.items():
        if "|" in k and k.split("|")[0] in ("pose@const", "pose@oracle",
                                            "scale@pred", "scale@gt"):
            print(f"{k:22s} {v['mean']:+9.4f}  [{v['lo']:+.4f},{v['hi']:+.4f}]"
                  f"{'*' if v['excludes_zero'] else ' '} {v['scenes_improved']:4d}/150")
    print(f"\n{'DiD (G11-G10)-(G01-G00)':22s}")
    for k, v in did.items():
        print(f"  {k:12s} {v['mean']:+9.4f}  [{v['lo']:+.4f},{v['hi']:+.4f}]"
              f"{'*' if v['excludes_zero'] else ' '}")
    print(f"\npeak GPU {peak:.2f} GiB, {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
