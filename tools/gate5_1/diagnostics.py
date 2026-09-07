#!/usr/bin/env python
"""Gate 5.1 step 3 — scale and depth diagnostics for the four variants (diagnostic only).

Runs strictly AFTER `eval_gate5_1.py`, so every deployable prediction is already final and
written; opening LiDAR here cannot influence any prediction. G51-A/B/C/D never used LiDAR.
LiDAR is projected once per frame and all scales are scored against it in the same pass.
"""
from __future__ import annotations

import argparse, csv, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                                "occ3d_zeroshot")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "gate5")))

from gates.scale_gate.config import REPO_ROOT, load_config, write_json
from gates.scale_gate.kitti import Preprocess, read_manifest
from scale_diagnostics import dmetrics, pearson, spearman                    # noqa: E402
from occ3d_zeroshot.nuscenes_adapter import load_annotations, scene_frames

VARIANTS = ["G51-A", "G51-B", "G51-C", "G51-D"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/gate5_1/calibrated_gauge.yaml")
    ap.add_argument("--dataset", required=True, choices=["occ3d", "kitti"])
    a = ap.parse_args()
    cfg = load_config(a.config)
    g5 = load_config(cfg.experiment.gate5_config)
    g4 = load_config(cfg.experiment.gate4_config)
    fv = g4.frozen_values
    art = os.path.join(REPO_ROOT, cfg.experiment.output_dir)
    dev = torch.device(g5.moge.device if torch.cuda.is_available() else "cpu")
    conf_thr = float(fv.confidence_threshold)
    bands = [tuple(b) for b in cfg.eval.depth_bands_m]
    if not os.path.exists(os.path.join(art, f"summary_{a.dataset}.json")):
        raise SystemExit(f"run eval_gate5_1.py --dataset {a.dataset} first")

    S = {v: {r["clip_id"]: (float(r["s_moge"]) if int(r["ok"]) else None)
             for r in csv.DictReader(open(os.path.join(art, f"scales_{v}_{a.dataset}.csv")))}
         for v in VARIANTS}
    g5rows = {r["clip_id"]: r for r in csv.DictReader(open(os.path.join(
        REPO_ROOT, g5.experiment.output_dir, f"gate5_depth_frames_{a.dataset}.csv")))}
    rows, t0 = [], time.time()

    if a.dataset == "occ3d":
        from lidar_scale_diagnostic import build_lidar_index, lidar_depth_on_lattice
        idx = build_lidar_index(g4.data.nuscenes_root, os.path.join(
            REPO_ROOT, g4.experiment.output_dir, "nuscenes_lidar_index.json"))
        ann = load_annotations(g4.data.occ3d_root)
        pre = Preprocess.build((900, 1600), int(fv.lingbot_image_size),
                               int(fv.lingbot_patch_size))
        recs = [json.loads(l) for l in
                open(os.path.join(REPO_ROOT, g5.datasets.occ3d.manifest))]
        fmap = {}
        for i, r in enumerate(recs):
            cid = r["clip_id"]
            cp = os.path.join(g5.datasets.occ3d.lingbot_cache, cid + ".npz")
            if not os.path.exists(cp) or cid not in g5rows:
                continue
            d = np.load(cp)
            dep = d["pred_depth"].astype(np.float32)
            conf = d["pred_depth_conf"].astype(np.float32)
            s_or = float(g5rows[cid]["s_oracle"])
            if r["scene"] not in fmap:
                fmap[r["scene"]] = {f.token: f for f in scene_frames(
                    ann, r["scene"], g4.data.camera, g4.data.nuscenes_root)}
            sc = {v: S[v].get(cid) for v in VARIANTS}
            sc["oracle"] = s_or
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
                row = {"clip_id": cid, "group": r["scene"], "frame": t,
                       "offset": t - (len(r["sample_tokens"]) - 1),
                       "n_valid": int(vm.sum()),
                       **{f"s_{k}": (v if v else float("nan")) for k, v in sc.items()}}
                for k, s in sc.items():
                    if s is None or not np.isfinite(s):
                        continue
                    for mk, mv in dmetrics(dep[t], gt, vm, s).items():
                        row[f"{k}_{mk}"] = mv
                    for lo, hi in bands:
                        bm = vm & (gt >= lo) & (gt < hi)
                        row[f"{k}_absrel_{int(lo)}_{int(hi)}"] = (
                            dmetrics(dep[t], gt, bm, s)["abs_rel"] if bm.sum() >= 20
                            else float("nan"))
                rows.append(row)
            if (i + 1) % 300 == 0:
                print(f"  {i+1}/{len(recs)}  {time.time()-t0:.0f}s", flush=True)
    else:
        scfg = load_config("configs/scale_gate/semantickitti.yaml")
        recs = read_manifest(os.path.join(REPO_ROOT, g5.datasets.kitti.manifest))
        for i, rec in enumerate(recs):
            cid = rec["clip_id"]
            lp = os.path.join(REPO_ROOT, scfg.cache.root, "lingbot", f"{cid}.npz")
            dp = os.path.join(REPO_ROOT, scfg.cache.root, "lidar_depth", f"{cid}.npz")
            if not (os.path.exists(lp) and os.path.exists(dp) and cid in g5rows):
                continue
            L, D = np.load(lp), np.load(dp)
            dep = L["pred_depth"].astype(np.float32)
            conf = L["pred_depth_conf"].astype(np.float32)
            sc = {v: S[v].get(cid) for v in VARIANTS}
            sc["oracle"] = float(g5rows[cid]["s_oracle"])
            gtd, val = D["depth"].astype(np.float64), D["valid"].astype(bool)
            for t in range(dep.shape[0]):
                vm = val[t] & (gtd[t] > 0) & (conf[t] >= conf_thr)
                if vm.sum() < 50:
                    continue
                row = {"clip_id": cid, "group": rec["sequence"], "frame": t,
                       "offset": t - (dep.shape[0] - 1), "n_valid": int(vm.sum()),
                       **{f"s_{k}": (v if v else float("nan")) for k, v in sc.items()}}
                for k, s in sc.items():
                    if s is None or not np.isfinite(s):
                        continue
                    for mk, mv in dmetrics(dep[t], gtd[t], vm, s).items():
                        row[f"{k}_{mk}"] = mv
                    for lo, hi in bands:
                        bm = vm & (gtd[t] >= lo) & (gtd[t] < hi)
                        row[f"{k}_absrel_{int(lo)}_{int(hi)}"] = (
                            dmetrics(dep[t], gtd[t], bm, s)["abs_rel"] if bm.sum() >= 20
                            else float("nan"))
                rows.append(row)
            if (i + 1) % 60 == 0:
                print(f"  {i+1}/{len(recs)}  {time.time()-t0:.0f}s", flush=True)

    with open(os.path.join(art, f"depth_frames_{a.dataset}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    per_clip = {}
    for r in rows:
        per_clip.setdefault(r["clip_id"], r)
    A = lambda k: np.array([v[k] for v in per_clip.values()], float)
    so = A("s_oracle")
    sf = {v: {"per_frame_log_scale_std": float(np.nanmedian(np.array(
        [float(x["per_frame_log_scale_std"]) for x in csv.DictReader(open(
            os.path.join(art, f"scales_{v}_{a.dataset}.csv")))], float))),
        "mad_log_clip": float(np.nanmedian(np.array(
            [float(x["mad_log_clip"]) for x in csv.DictReader(open(
                os.path.join(art, f"scales_{v}_{a.dataset}.csv")))], float)))}
        for v in VARIANTS}
    stats = {"dataset": a.dataset, "n_clips": len(per_clip), "n_frames": len(rows),
             "oracle": {"median": float(np.median(so)),
                        **{f"p{int(p*100):02d}": float(np.quantile(so, p))
                           for p in (0.05, 0.25, 0.75, 0.95)}}}
    for v in VARIANTS:
        s = A(f"s_{v}")
        stats[v] = {
            "median": float(np.median(s)),
            **{f"p{int(p*100):02d}": float(np.quantile(s, p))
               for p in (0.05, 0.25, 0.75, 0.95)},
            "median_abs_log_error": float(np.median(np.abs(np.log(s) - np.log(so)))),
            "median_abs_rel_error": float(np.median(np.abs(s / so - 1.0))),
            "median_ratio_to_oracle": float(np.median(s / so)),
            "median_abs_bias": float(abs(np.median(s / so) - 1.0)),
            "pearson_log_vs_log_oracle": pearson(np.log(s), np.log(so)),
            "spearman_log_vs_log_oracle": spearman(np.log(s), np.log(so)),
            **sf[v]}
    stats["depth"] = {}
    for k in VARIANTS + ["oracle"]:
        stats["depth"][k] = {m: float(np.nanmedian(np.array(
            [r.get(f"{k}_{m}", np.nan) for r in rows], float)))
            for m in ("abs_rel", "median_abs_rel", "median_abs_log", "rmse_m",
                      "delta1", "delta2", "delta3")}
        stats["depth"][k]["by_band"] = {
            f"{int(lo)}-{int(hi)}m": float(np.nanmedian(np.array(
                [r.get(f"{k}_absrel_{int(lo)}_{int(hi)}", np.nan) for r in rows], float)))
            for lo, hi in bands}
        stats["depth"][k]["by_offset"] = {
            str(o): float(np.nanmedian(np.array(
                [r.get(f"{k}_abs_rel", np.nan) for r in rows if r["offset"] == o], float)))
            for o in sorted({r["offset"] for r in rows})}
    write_json(os.path.join(art, f"diagnostics_{a.dataset}.json"), stats)

    print(f"\n{a.dataset}: {len(per_clip)} clips, {len(rows)} frames")
    print(f"{'variant':9s} {'median':>8} {'p05':>8} {'p95':>8} {'|logerr|':>9} "
          f"{'|relerr|':>9} {'ratio':>7} {'|bias|':>7} {'pears':>7} {'spear':>7} {'disp':>7}")
    o = stats["oracle"]
    print(f"{'oracle':9s} {o['median']:8.3f} {o['p05']:8.3f} {o['p95']:8.3f}")
    for v in VARIANTS:
        s = stats[v]
        print(f"{v:9s} {s['median']:8.3f} {s['p05']:8.3f} {s['p95']:8.3f} "
              f"{s['median_abs_log_error']:9.4f} {s['median_abs_rel_error']:9.4f} "
              f"{s['median_ratio_to_oracle']:7.3f} {s['median_abs_bias']:7.4f} "
              f"{s['pearson_log_vs_log_oracle']:7.3f} "
              f"{s['spearman_log_vs_log_oracle']:7.3f} "
              f"{s['per_frame_log_scale_std']:7.4f}")
    print(f"\n{'variant':9s} {'AbsRel':>8} {'medRel':>8} {'medLog':>8} {'RMSE':>8} "
          f"{'d1':>7} {'d2':>7} {'d3':>7}")
    for k in VARIANTS + ["oracle"]:
        d = stats["depth"][k]
        print(f"{k:9s} {d['abs_rel']:8.4f} {d['median_abs_rel']:8.4f} "
              f"{d['median_abs_log']:8.4f} {d['rmse_m']:8.3f} {d['delta1']:7.4f} "
              f"{d['delta2']:7.4f} {d['delta3']:7.4f}")
    print(f"\n{time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
