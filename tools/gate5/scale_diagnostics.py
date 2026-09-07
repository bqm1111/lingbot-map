#!/usr/bin/env python
"""Gate 5 step 3 — scale and depth diagnostics against LiDAR (diagnostic only).

Run strictly AFTER `eval_gate5.py`: every deployable prediction is already finalised and
written to disk, so opening LiDAR here cannot influence any prediction. M2 never used
LiDAR at any point.

    python tools/gate5/scale_diagnostics.py --dataset occ3d
    python tools/gate5/scale_diagnostics.py --dataset kitti
"""
from __future__ import annotations

import argparse, csv, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..")), "occ3d_zeroshot"))

from gates.scale_gate.config import REPO_ROOT, load_config, write_json
from gates.scale_gate.kitti import Preprocess, read_manifest
from occ3d_zeroshot.factorization import depth_metrics
from occ3d_zeroshot.nuscenes_adapter import load_annotations, scene_frames


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x - x.mean(), y - y.mean()
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / d) if d > 0 else float("nan")


def spearman(x, y):
    r = lambda z: np.argsort(np.argsort(np.asarray(z, float))).astype(float)
    return pearson(r(x), r(y))


def dmetrics(pred_canonical, gt_m, valid, scale):
    m = depth_metrics(pred_canonical, gt_m, valid, scale)
    p = float(scale) * pred_canonical.astype(np.float64)
    g = gt_m.astype(np.float64)
    v = valid & np.isfinite(p) & (p > 0) & (g > 0)
    if v.sum():
        ratio = np.maximum(p[v] / g[v], g[v] / p[v])
        m["delta2"] = float((ratio < 1.25 ** 2).mean())
        m["delta3"] = float((ratio < 1.25 ** 3).mean())
    else:
        m["delta2"] = m["delta3"] = float("nan")
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/gate5/moge_metric_gauge.yaml")
    ap.add_argument("--dataset", required=True, choices=["occ3d", "kitti"])
    a = ap.parse_args()
    cfg = load_config(a.config)
    g4 = load_config(cfg.experiment.gate4_config)
    fv = g4.frozen_values
    art = os.path.join(REPO_ROOT, cfg.experiment.output_dir)
    dev = torch.device(cfg.moge.device if torch.cuda.is_available() else "cpu")
    conf_thr = float(fv.confidence_threshold)

    # every deployable prediction must already exist on disk
    need = os.path.join(art, f"gate5_summary_{a.dataset}.json")
    if not os.path.exists(need):
        raise SystemExit(f"run eval_gate5.py --dataset {a.dataset} first ({need} missing)")

    moge = {r["clip_id"]: r for r in
            csv.DictReader(open(os.path.join(art, f"moge_scale_{a.dataset}.csv")))}
    bands = [tuple(b) for b in cfg.eval.depth_bands_m]
    rows, t0 = [], time.time()

    if a.dataset == "occ3d":
        from lidar_scale_diagnostic import build_lidar_index, lidar_depth_on_lattice
        from occ3d_zeroshot.pipeline import clip_scale, load_depth_head
        g41 = load_config(cfg.experiment.gate41_config)
        geo = {r["clip_id"]: r for r in csv.DictReader(open(os.path.join(
            REPO_ROOT, g41.experiment.output_dir, "geometry_per_clip.csv")))}
        idx = build_lidar_index(g4.data.nuscenes_root, os.path.join(
            REPO_ROOT, g4.experiment.output_dir, "nuscenes_lidar_index.json"))
        ann = load_annotations(g4.data.occ3d_root)
        head, hck = load_depth_head(os.path.join(REPO_ROOT, g4.frozen.depth_head), dev)
        pre = Preprocess.build((900, 1600), int(fv.lingbot_image_size),
                               int(fv.lingbot_patch_size))
        recs = [json.loads(l) for l in
                open(os.path.join(REPO_ROOT, cfg.datasets.occ3d.manifest))]
        fmap = {}
        for i, r in enumerate(recs):
            cp = os.path.join(cfg.datasets.occ3d.lingbot_cache, r["clip_id"] + ".npz")
            g = geo.get(r["clip_id"]); mg = moge.get(r["clip_id"])
            if not (os.path.exists(cp) and g and mg and int(g["oracle_scale_ok"])):
                continue
            d = np.load(cp)
            dep = d["pred_depth"].astype(np.float32)
            conf = d["pred_depth_conf"].astype(np.float32)
            _, s_c3, _, _ = clip_scale(head, hck, dep, conf, float(fv.s0), conf_thr,
                                       float(fv.min_depth_m), float(fv.max_depth_m), dev)
            if r["scene"] not in fmap:
                fmap[r["scene"]] = {f.token: f for f in scene_frames(
                    ann, r["scene"], g4.data.camera, g4.data.nuscenes_root)}
            S = {"c0": float(fv.s0), "c3": s_c3, "moge": float(mg["s_moge"]),
                 "oracle": float(g["s_oracle"])}
            for t, tok in enumerate(r["sample_tokens"]):
                if tok not in idx:
                    continue
                gt, val = lidar_depth_on_lattice(g4.data.nuscenes_root, idx[tok],
                                                 fmap[r["scene"]][tok], pre)
                if gt is None:
                    continue
                vm = val & (gt > 0) & (conf[t] >= conf_thr)
                if vm.sum() < 50:
                    continue
                row = {"clip_id": r["clip_id"], "group": r["scene"], "frame": t,
                       "offset": t - (len(r["sample_tokens"]) - 1),
                       "n_valid": int(vm.sum()), **{f"s_{k}": v for k, v in S.items()}}
                for k, s in S.items():
                    for mk, mv in dmetrics(dep[t], gt, vm, s).items():
                        row[f"{k}_{mk}"] = mv
                    for lo, hi in bands:
                        bm = vm & (gt >= lo) & (gt < hi)
                        row[f"{k}_absrel_{int(lo)}_{int(hi)}"] = (
                            dmetrics(dep[t], gt, bm, s)["abs_rel"] if bm.sum() >= 20
                            else float("nan"))
                rows.append(row)
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(recs)}  {time.time()-t0:.0f}s", flush=True)
    else:
        scfg = load_config("configs/scale_gate/semantickitti.yaml")
        dcfg = load_config("configs/depth_gate/refine.yaml")
        from gates.voxel_gate.c3 import clip_scale as kclip, load_head
        head, hck = load_head(os.path.join(REPO_ROOT, dcfg.experiment.output_dir,
                                           "runs", "depth_cnn"), dev)
        oracle = {r["clip_id"]: r for r in csv.DictReader(open(os.path.join(
            REPO_ROOT, cfg.datasets.kitti.oracle_scale_csv)))}
        recs = read_manifest(os.path.join(REPO_ROOT, cfg.datasets.kitti.manifest))
        for i, rec in enumerate(recs):
            cid = rec["clip_id"]
            lp = os.path.join(REPO_ROOT, scfg.cache.root, "lingbot", f"{cid}.npz")
            dp = os.path.join(REPO_ROOT, scfg.cache.root, "lidar_depth", f"{cid}.npz")
            mg = moge.get(cid)
            if not (os.path.exists(lp) and os.path.exists(dp) and mg):
                continue
            L, D = np.load(lp), np.load(dp)
            dep = L["pred_depth"].astype(np.float32)
            conf = L["pred_depth_conf"].astype(np.float32)
            _, s_c3, _ = kclip(head, hck, dep, conf, float(dcfg.scale.constant), conf_thr,
                               float(scfg.voxel.min_depth_m),
                               float(scfg.voxel.max_depth_m), None, dev)[:3]
            try:
                s_or = float(oracle[cid]["s_joint"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (np.isfinite(s_or) and s_or > 0):
                continue
            S = {"c0": float(dcfg.scale.constant), "c3": s_c3,
                 "moge": float(mg["s_moge"]), "oracle": s_or}
            gtd, val = D["depth"].astype(np.float64), D["valid"].astype(bool)
            for t in range(dep.shape[0]):
                vm = val[t] & (gtd[t] > 0) & (conf[t] >= conf_thr)
                if vm.sum() < 50:
                    continue
                row = {"clip_id": cid, "group": rec["sequence"], "frame": t,
                       "offset": t - (dep.shape[0] - 1), "n_valid": int(vm.sum()),
                       **{f"s_{k}": v for k, v in S.items()}}
                for k, s in S.items():
                    for mk, mv in dmetrics(dep[t], gtd[t], vm, s).items():
                        row[f"{k}_{mk}"] = mv
                    for lo, hi in bands:
                        bm = vm & (gtd[t] >= lo) & (gtd[t] < hi)
                        row[f"{k}_absrel_{int(lo)}_{int(hi)}"] = (
                            dmetrics(dep[t], gtd[t], bm, s)["abs_rel"] if bm.sum() >= 20
                            else float("nan"))
                rows.append(row)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(recs)}  {time.time()-t0:.0f}s", flush=True)

    with open(os.path.join(art, f"gate5_depth_frames_{a.dataset}.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # ---- clip-level scale statistics ---------------------------------------- #
    per_clip = {}
    for r in rows:
        per_clip.setdefault(r["clip_id"], r)
    A = lambda k: np.array([v[k] for v in per_clip.values()], float)
    so = A("s_oracle")
    stats = {"dataset": a.dataset, "n_clips": len(per_clip), "n_frames": len(rows)}
    for k in ("c0", "c3", "moge"):
        s = A(f"s_{k}")
        stats[k] = {
            "median": float(np.median(s)),
            **{f"p{int(p*100):02d}": float(np.quantile(s, p))
               for p in (0.05, 0.25, 0.75, 0.95)},
            "median_abs_log_error": float(np.median(np.abs(np.log(s) - np.log(so)))),
            "median_abs_rel_error": float(np.median(np.abs(s / so - 1.0))),
            "median_ratio_to_oracle": float(np.median(s / so)),
            "pearson_log_vs_log_oracle": pearson(np.log(s), np.log(so)),
            "spearman_log_vs_log_oracle": spearman(np.log(s), np.log(so))}
    stats["oracle"] = {"median": float(np.median(so)),
                       **{f"p{int(p*100):02d}": float(np.quantile(so, p))
                          for p in (0.05, 0.25, 0.75, 0.95)}}
    # depth metrics pooled over frames
    stats["depth"] = {}
    for k in ("c0", "c3", "moge", "oracle"):
        stats["depth"][k] = {
            m: float(np.nanmedian(A(f"{k}_{m}"))) if False else
            float(np.nanmedian(np.array([r[f"{k}_{m}"] for r in rows], float)))
            for m in ("abs_rel", "median_abs_rel", "median_abs_log", "rmse_m",
                      "delta1", "delta2", "delta3")}
        stats["depth"][k]["by_band"] = {
            f"{int(lo)}-{int(hi)}m": float(np.nanmedian(np.array(
                [r[f"{k}_absrel_{int(lo)}_{int(hi)}"] for r in rows], float)))
            for lo, hi in bands}
        stats["depth"][k]["by_offset"] = {
            str(o): float(np.nanmedian(np.array(
                [r[f"{k}_abs_rel"] for r in rows if r["offset"] == o], float)))
            for o in sorted({r["offset"] for r in rows})}
    write_json(os.path.join(art, f"gate5_scale_diagnostics_{a.dataset}.json"), stats)

    print(f"\n{a.dataset}: {len(per_clip)} clips, {len(rows)} frames")
    print(f"{'scale':8s} {'median':>8} {'p05':>8} {'p95':>8} {'|log err|':>10} "
          f"{'|rel err|':>10} {'ratio':>7} {'pearson':>8} {'spearman':>9}")
    print(f"{'oracle':8s} {stats['oracle']['median']:8.3f} "
          f"{stats['oracle']['p05']:8.3f} {stats['oracle']['p95']:8.3f}")
    for k in ("c0", "c3", "moge"):
        s = stats[k]
        print(f"{k:8s} {s['median']:8.3f} {s['p05']:8.3f} {s['p95']:8.3f} "
              f"{s['median_abs_log_error']:10.4f} {s['median_abs_rel_error']:10.4f} "
              f"{s['median_ratio_to_oracle']:7.3f} {s['pearson_log_vs_log_oracle']:8.3f} "
              f"{s['spearman_log_vs_log_oracle']:9.3f}")
    print(f"\n{'scale':8s} {'AbsRel':>8} {'medRel':>8} {'medLog':>8} {'RMSE':>8} "
          f"{'d1':>7} {'d2':>7} {'d3':>7}")
    for k in ("c0", "c3", "moge", "oracle"):
        d = stats["depth"][k]
        print(f"{k:8s} {d['abs_rel']:8.4f} {d['median_abs_rel']:8.4f} "
              f"{d['median_abs_log']:8.4f} {d['rmse_m']:8.3f} {d['delta1']:7.4f} "
              f"{d['delta2']:7.4f} {d['delta3']:7.4f}")
    print(f"\n{time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
