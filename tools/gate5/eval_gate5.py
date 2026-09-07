#!/usr/bin/env python
"""Gate 5 step 2 — the eight frozen conditions on Occ3D-nuScenes and SemanticKITTI.

    M0 = frozen constant s0     M1 = frozen learned C3 scale
    M2 = MoGe-2 metric gauge    OR = LiDAR-oracle scale (diagnostic only)
each with raw and the frozen 0.4 m `dilate_r2`. V3 is not run.

Only the *scalar* differs between conditions: depth shape, intrinsics and relative poses
are LingBot's throughout, and MoGe depth/points/intrinsics never enter the fusion.

    python tools/gate5/eval_gate5.py --dataset occ3d
    python tools/gate5/eval_gate5.py --dataset kitti
"""
from __future__ import annotations

import argparse, csv, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, load_config, write_json
from gates.scale_gate.kitti import read_manifest
from gates.scale_gate.scale import bootstrap_ci
from gates.voxel_gate.controls import control
from gates.voxel_gate.voxels import dilate

CONDITIONS = ["M0", "M1", "M2", "OR"]
CORRECTORS = ["raw", "dilate_r2"]
ORACLE_CONDITIONS = ("OR",)


def scores(pred, gt, keep):
    p, t = pred & keep, gt & keep
    tp = int((p & t).sum()); fp = int((p & ~t).sum()); fn = int((~p & t).sum())
    den = tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "iou": tp / den if den else 0.0,
            "n_pred_occupied": int(p.sum())}


# --------------------------------------------------------------------------- #
def run_occ3d(cfg, dev, moge_scale, v1_kw, r_vox):
    from occ3d_zeroshot.factorization import fuse, gt_camera_to_world, relative_transforms
    from occ3d_zeroshot.grid import (CANONICAL, canonical_to_native,
                                     native_binary_target, native_distance_bands,
                                     points_to_canonical)
    from occ3d_zeroshot.pipeline import (canonical_features, clip_scale,
                                         load_depth_head, region_from)
    g4 = load_config(cfg.experiment.gate4_config)
    g41 = load_config(cfg.experiment.gate41_config)
    fv = g4.frozen_values
    art41 = os.path.join(REPO_ROOT, g41.experiment.output_dir)
    geo = {r["clip_id"]: r for r in
           csv.DictReader(open(os.path.join(art41, "geometry_per_clip.csv")))}
    head, hck = load_depth_head(os.path.join(REPO_ROOT, g4.frozen.depth_head), dev)
    recs = [json.loads(l) for l in
            open(os.path.join(REPO_ROOT, cfg.datasets.occ3d.manifest))]
    bands = {k: torch.from_numpy(m).to(dev) for k, m in
             native_distance_bands([tuple(b) for b in cfg.eval.distance_bands_m]).items()}
    to_native = canonical_to_native

    def clip_iter():
        for r in recs:
            cp = os.path.join(cfg.datasets.occ3d.lingbot_cache, r["clip_id"] + ".npz")
            lp = os.path.join(g4.data.occ3d_root, r["anchor_gt_path"])
            g = geo.get(r["clip_id"])
            if not (os.path.exists(cp) and os.path.exists(lp) and g):
                continue
            d = np.load(cp)
            dep = d["pred_depth"].astype(np.float32)
            conf = d["pred_depth_conf"].astype(np.float32)
            _, s_c3, _, _ = clip_scale(head, hck, dep, conf, float(fv.s0),
                                       float(fv.confidence_threshold),
                                       float(fv.min_depth_m), float(fv.max_depth_m), dev)
            S = {"M0": float(fv.s0), "M1": s_c3, "M2": moge_scale.get(r["clip_id"]),
                 "OR": float(g["s_oracle"]) if int(g["oracle_scale_ok"]) else None}
            pose = d["pred_pose_c2w"].astype(np.float64)
            anchor = dep.shape[0] - 1
            rel_fn = lambda s: relative_transforms(pose, pose, anchor, "pred", s)
            def build(s):
                pe, fr, cf, dp = fuse(dep, conf, d["pred_K"].astype(np.float64),
                                      rel_fn(s), d["T_camera_to_ego"][-1].astype(np.float64),
                                      s, float(fv.confidence_threshold),
                                      float(fv.min_depth_m), float(fv.max_depth_m))
                feat5, _ = canonical_features(pe, fr, cf, dp, dev)
                occ = feat5[0] > 0
                _, kin = points_to_canonical(pe) if len(pe) else (None, np.zeros(0, bool))
                return occ, region_from(occ, r_vox), len(pe), (
                    float(kin.mean()) if len(pe) else 0.0)
            def labels():
                lab = np.load(lp)
                gt_np, keep_np = native_binary_target(
                    lab, bool(g4.eval.apply_camera_mask), bool(g4.eval.apply_lidar_mask),
                    int(g4.eval.single_camera_x_cut))
                return (torch.from_numpy(gt_np).to(dev), torch.from_numpy(keep_np).to(dev))
            yield r["clip_id"], r["scene"], S, build, labels, bands, to_native
    return clip_iter, "scene"


def run_kitti(cfg, dev, moge_scale, v1_kw, r_vox):
    from gates.voxel_gate.voxels import (build_input, dense_from_sparse, sparse_voxel_features,
                                   unpack)
    from gates.voxel_gate.c3 import as4x4, c3_points, clip_scale as kclip_scale, load_head
    from prompted_lingbot.occ_datasets import SemanticKittiOccSpec, apply_transform
    from prompted_lingbot.occupancy import SEMANTICKITTI_GRID as G
    from gates.voxel_gate.voxels import distance_bins
    scfg = load_config("configs/scale_gate/semantickitti.yaml")
    dcfg = load_config("configs/depth_gate/refine.yaml")
    g31 = load_config(cfg.experiment.gate31_config)
    s0 = float(dcfg.scale.constant)
    conf_thr = float(scfg.lingbot.confidence_threshold)
    dmin, dmax = float(scfg.voxel.min_depth_m), float(scfg.voxel.max_depth_m)
    root = os.path.join(REPO_ROOT, scfg.dataset.root)
    head, hck = load_head(os.path.join(REPO_ROOT, dcfg.experiment.output_dir,
                                       "runs", "depth_cnn"), dev)
    oracle = {r["clip_id"]: r for r in
              csv.DictReader(open(os.path.join(REPO_ROOT, cfg.datasets.kitti.oracle_scale_csv)))}
    recs = read_manifest(os.path.join(REPO_ROOT, cfg.datasets.kitti.manifest))
    bands = {k: torch.from_numpy(m).to(dev) for k, m in
             distance_bins(tuple(tuple(b) for b in cfg.eval.distance_bands_m)).items()}
    specs = {}
    identity = lambda v: v                       # KITTI is already the native grid

    def clip_iter():
        for rec in recs:
            cid, seq = rec["clip_id"], rec["sequence"]
            lp = os.path.join(REPO_ROOT, scfg.cache.root, "lingbot", f"{cid}.npz")
            if not os.path.exists(lp):
                continue
            if seq not in specs:
                specs[seq] = SemanticKittiOccSpec.build(root, seq)
            spec = specs[seq]
            target, valid_mask = spec.target(int(rec["frame_ids"][-1]))
            if target is None:
                continue
            L = np.load(lp, allow_pickle=False)
            dep = L["pred_depth"].astype(np.float32)
            conf = L["pred_depth_conf"].astype(np.float32)
            K = L["pred_K"].astype(np.float64)
            pose = as4x4(L["pred_pose_c2w"])
            _, s_c3, _ = kclip_scale(head, hck, dep, conf, s0, conf_thr, dmin, dmax,
                                     None, dev)[:3]
            try:
                s_or = float(oracle[cid]["s_joint"])
                s_or = s_or if np.isfinite(s_or) and s_or > 0 else None
            except (KeyError, TypeError, ValueError):
                s_or = None
            S = {"M0": s0, "M1": s_c3, "M2": moge_scale.get(cid), "OR": s_or}

            def build(s):
                pts, fr, cf, dp, _ = c3_points(dep, conf, K, pose, s, conf_thr, dmin, dmax)
                pg = apply_transform(spec.cam_to_velo, pts) if len(pts) else pts
                sp = sparse_voxel_features(pg, fr, cf, dp)
                feat5 = dense_from_sparse(sp, dev)
                occ = feat5[0] > 0
                idx = np.floor((pg - np.asarray(G.origin)) / G.voxel_size).astype(np.int64) \
                    if len(pg) else np.zeros((0, 3), np.int64)
                kin = np.ones(len(idx), bool)
                for ax in range(3):
                    kin &= (idx[:, ax] >= 0) & (idx[:, ax] < G.dims[ax])
                return occ, dilate(occ, r_vox), len(pts), (
                    float(kin.mean()) if len(idx) else 0.0)

            def labels(target=target, valid_mask=valid_mask):
                keep = (target != G.ignore_label) & valid_mask
                gt = (target != G.empty_class) & keep
                return (torch.from_numpy(gt).to(dev), torch.from_numpy(keep).to(dev))
            yield cid, seq, S, build, labels, bands, identity
    return clip_iter, "block"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/gate5/moge_metric_gauge.yaml")
    ap.add_argument("--dataset", required=True, choices=["occ3d", "kitti"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    g4 = load_config(cfg.experiment.gate4_config)
    fv = g4.frozen_values
    dev = torch.device(cfg.moge.device if torch.cuda.is_available() else "cpu")
    art = os.path.join(REPO_ROOT, cfg.experiment.output_dir)
    v1_kw = dict(fv.v1_control)
    assert v1_kw == {"kind": "dilate", "radius": 2}
    r_vox = int(round(float(fv.radius_m) / float(fv.canonical_voxel_size)))
    assert r_vox == 3
    DIL_M = v1_kw["radius"] * float(fv.canonical_voxel_size)
    assert abs(DIL_M - 0.4) < 1e-12

    ms = {r["clip_id"]: (float(r["s_moge"]) if int(r["ok"]) else None)
          for r in csv.DictReader(open(os.path.join(art, f"moge_scale_{a.dataset}.csv")))}

    builder, unit = (run_occ3d if a.dataset == "occ3d" else run_kitti)(
        cfg, dev, ms, v1_kw, r_vox)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    rows, band_rows, t0, skipped = [], [], time.time(), {c: 0 for c in CONDITIONS}
    for i, (cid, group, S, build, labels, bands, to_native) in enumerate(builder()):
        if a.limit and i >= a.limit:
            break
        canon, meta = {}, {}
        for c in CONDITIONS:
            s = S[c]
            if s is None or not np.isfinite(s) or s <= 0:
                skipped[c] += 1
                continue
            occ, R, npts, ingrid = build(float(s))
            canon[(c, "raw")] = occ
            canon[(c, "dilate_r2")] = control(occ, R, **v1_kw)
            meta[c] = {"n_points": npts, "in_grid": ingrid, "scale": float(s)}
        gt, keep = labels()                    # labels opened only now
        for (c, k), pc in canon.items():
            pn = to_native(pc)
            sc = scores(pn, gt, keep)
            rows.append({"clip_id": cid, "group": group, "condition": c, "corrector": k,
                         "is_oracle": int(c in ORACLE_CONDITIONS),
                         "scale_used": meta[c]["scale"], **sc,
                         "canonical_occupied": int(pc.sum()),
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
    keys = [k for k in by]
    mean = lambda k, f: float(np.mean([v[f] for v in by[k].values()]))
    agg = {f"{c}-{'R' if k=='raw' else 'D'}": {
        "condition": c, "corrector": k, "n_clips": len(by[(c, k)]),
        "is_oracle": int(c in ORACLE_CONDITIONS),
        **{f: mean((c, k), f) for f in
           ("iou", "precision", "recall", "tp", "fp", "fn", "n_pred_occupied",
            "canonical_occupied", "native_occupied", "frac_pred_in_eval_mask",
            "n_fused_points", "frac_points_in_grid", "scale_used")},
        **{f"iou_{q}": float(np.quantile([v["iou"] for v in by[(c, k)].values()], p))
           for q, p in (("p25", .25), ("median", .5), ("p75", .75))}}
        for (c, k) in keys}

    group_of = {x["clip_id"]: x["group"] for x in rows}
    if unit == "scene":
        units = sorted(set(group_of.values()))
        per_unit = {n: {u: float(np.mean([v["iou"] for cid, v in by[k].items()
                                          if group_of[cid] == u])) for u in units}
                    for n, k in ((f"{c}-{'R' if kk=='raw' else 'D'}", (c, kk))
                                 for (c, kk) in keys)}
    else:                                    # one sequence: contiguous clip blocks
        ids = sorted({x["clip_id"] for x in rows})
        bs = int(cfg.eval.kitti_block_size)
        blocks = {f"block{j//bs:03d}": [ids[j2] for j2 in range(j, min(j + bs, len(ids)))]
                  for j in range(0, len(ids), bs)}
        units = sorted(blocks)
        per_unit = {n: {u: float(np.mean([by[k][c]["iou"] for c in blocks[u] if c in by[k]]))
                        for u in units}
                    for n, k in ((f"{c}-{'R' if kk=='raw' else 'D'}", (c, kk))
                                 for (c, kk) in keys)}

    def paired(x, y):
        u = [v for v in units if v in per_unit[x] and v in per_unit[y]]
        d = [per_unit[y][v] - per_unit[x][v] for v in u]
        ci = bootstrap_ci(d, int(cfg.eval.bootstrap_n), int(cfg.eval.bootstrap_seed))
        ci.update({"units_improved": int(np.sum([q > 0 for q in d])), "n_units": len(d),
                   "unit": unit})
        return ci
    PAIRS = [("M0-R", "M2-R"), ("M1-R", "M2-R"), ("M0-D", "M2-D"), ("M1-D", "M2-D"),
             ("M2-R", "OR-R"), ("M2-D", "OR-D"),
             ("M0-R", "M1-R"), ("M0-D", "M1-D"), ("M0-R", "OR-R"), ("M0-D", "OR-D")]
    contrasts = {f"{x}->{y}": paired(x, y) for x, y in PAIRS
                 if x in per_unit and y in per_unit}

    rec = {}
    for tag, R in (("raw", "R"), ("dilate", "D")):
        num = agg[f"M2-{R}"]["iou"] - agg[f"M0-{R}"]["iou"]
        den = agg[f"OR-{R}"]["iou"] - agg[f"M0-{R}"]["iou"]
        rec[f"recovery_{tag}"] = float(num / den) if abs(den) > 1e-12 else float("nan")
        rec[f"gain_{tag}_M2_minus_M0"] = float(num)
        rec[f"gain_{tag}_OR_minus_M0"] = float(den)

    bb = {}
    for x in band_rows:
        n = f"{x['condition']}-{'R' if x['corrector']=='raw' else 'D'}"
        bb.setdefault(n, {}).setdefault(x["band"], []).append(x)
    banded = {n: {b: {f: float(np.mean([q[f] for q in v])) for f in
                      ("iou", "precision", "recall", "n_pred_occupied")}
                  for b, v in dd.items()} for n, dd in bb.items()}

    with open(os.path.join(art, f"gate5_per_clip_{a.dataset}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(art, f"gate5_per_band_{a.dataset}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(band_rows[0])); w.writeheader()
        w.writerows(band_rows)
    write_json(os.path.join(art, f"gate5_per_unit_iou_{a.dataset}.json"), per_unit)
    peak = (torch.cuda.max_memory_allocated(dev) / 2**30) if dev.type == "cuda" else 0.0
    write_json(os.path.join(art, f"gate5_summary_{a.dataset}.json"), {
        "dataset": a.dataset, "n_clips": len(by[("M0", "raw")]),
        "n_units": len(units), "bootstrap_unit": unit,
        "skipped_by_condition": skipped,
        "dilate_r2_expansion_m": DIL_M, "radius_voxels": r_vox,
        "aggregate": agg, "by_distance": banded, "contrasts": contrasts,
        "recovery": rec, "peak_gpu_gib": peak, "elapsed_s": time.time() - t0})

    print(f"\n{a.dataset}: {len(by[('M0','raw')])} clips / {len(units)} {unit}s")
    print(f"{'cond':6s} {'IoU':>8} {'P':>7} {'R':>7} {'native':>8} {'scale':>8} "
          f"{'in-grid':>8} {'in-mask':>8}")
    for n in sorted(agg):
        s = agg[n]
        print(f"{n:5s}{'*' if s['is_oracle'] else ' '} {s['iou']:8.4f} "
              f"{s['precision']:7.3f} {s['recall']:7.3f} {s['native_occupied']:8.0f} "
              f"{s['scale_used']:8.3f} {s['frac_points_in_grid']:8.3f} "
              f"{s['frac_pred_in_eval_mask']:8.3f}")
    print(f"\n{'contrast':16s} {'dIoU':>9}  95% CI               {unit}s improved")
    for k, v in contrasts.items():
        print(f"{k:16s} {v['mean']:+9.4f}  [{v['lo']:+.4f},{v['hi']:+.4f}]"
              f"{'*' if v['excludes_zero'] else ' '} {v['units_improved']:4d}/{v['n_units']}")
    print(f"\nrecovery of oracle gain: raw {rec['recovery_raw']:+.1%}  "
          f"dilate {rec['recovery_dilate']:+.1%}")
    print(f"skipped: {skipped}   peak {peak:.2f} GiB   {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
