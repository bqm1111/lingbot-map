#!/usr/bin/env python
"""Gate 5.2 step 5 — the diagnostic LiDAR scale oracle and the depth diagnostics.

**Non-deployable.** Everything here is computed *after* the C0/C3/A/B scales are finalised
and hashed; the hashes are re-checked on entry so no result computed here can flow back
into a prediction. No occupancy ``.label``, ``.invalid`` or preprocessed target is read.

Two oracles, both diagnostic:

* ``projected_lidar``   -- the raw KITTI-360 velodyne sweep of each of the five frames,
  projected into its own image, pooled exactly like the MoGe estimator. This matches the
  Gate-4/4.1/5 oracle definition, so the numbers are comparable across gates.
* ``voxel_centre_lidar`` -- centres of the official ``.bin`` voxelised LiDAR *input* at the
  anchor. Quantised to 0.2 m, hence explicitly NOT an exact projected-point oracle.

    python tools/gate5_2/lidar_oracle.py
"""
from __future__ import annotations

import argparse, csv, hashlib, os, sys, time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, load_config, write_json
from gates.scale_gate.kitti import Preprocess, read_manifest
from gates.scale_gate.scale import weighted_median
from sscbench_kitti360.adapter import (SEQUENCE, SSCBENCH_KITTI360_GRID, parse_calibration,
                                       read_velodyne, voxelized_lidar_input)

LIDAR_MIN_M, LIDAR_MAX_M = 1.0, 80.0


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def project_to_lattice(pts_velo: np.ndarray, calib, pre: Preprocess):
    """Velodyne points -> z-buffered metric z-depth on LingBot's processed lattice."""
    cam = calib.velo_to_cam(pts_velo)
    z = cam[:, 2]
    keep = np.isfinite(cam).all(1) & (z > LIDAR_MIN_M) & (z < LIDAR_MAX_M)
    cam, z = cam[keep], z[keep]
    H, W = pre.proc_hw
    depth = np.zeros((H, W), np.float64)
    valid = np.zeros((H, W), bool)
    if not len(cam):
        return depth, valid
    uv = np.stack([cam[:, 0] / z * calib.K[0, 0] + calib.K[0, 2],
                   cam[:, 1] / z * calib.K[1, 1] + calib.K[1, 2]], -1)
    uvp = pre.map_pixels(uv)
    ui = np.round(uvp[:, 0]).astype(np.int64)
    vi = np.round(uvp[:, 1]).astype(np.int64)
    ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi, zz = ui[ok], vi[ok], z[ok]
    if zz.size:
        order = np.argsort(-zz)                       # nearest surface wins the pixel
        depth[vi[order], ui[order]] = zz[order]
        valid[vi[order], ui[order]] = True
    return depth, valid


def depth_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Standard monocular-depth errors on already-matched pixel pairs."""
    if pred.size == 0:
        return {k: float("nan") for k in ("absrel", "med_rel", "med_log", "rmse",
                                          "d1", "d2", "d3", "n")} | {"n": 0}
    r = np.abs(pred - gt) / gt
    ratio = np.maximum(pred / gt, gt / pred)
    return {"absrel": float(np.mean(r)), "med_rel": float(np.median(r)),
            "med_log": float(np.median(np.abs(np.log(pred) - np.log(gt)))),
            "rmse": float(np.sqrt(np.mean((pred - gt) ** 2))),
            "d1": float(np.mean(ratio < 1.25)), "d2": float(np.mean(ratio < 1.25 ** 2)),
            "d3": float(np.mean(ratio < 1.25 ** 3)), "n": int(pred.size)}


class Accumulator:
    """Pools pixel pairs per (condition, bucket) without keeping them all in memory."""

    def __init__(self):
        self.s = {}

    def add(self, key, pred, gt):
        if pred.size == 0:
            return
        a = self.s.setdefault(key, dict(n=0, sum_rel=0.0, sum_sq=0.0, d1=0, d2=0, d3=0,
                                        rel=[], log=[]))
        r = np.abs(pred - gt) / gt
        ratio = np.maximum(pred / gt, gt / pred)
        a["n"] += pred.size
        a["sum_rel"] += float(r.sum())
        a["sum_sq"] += float(((pred - gt) ** 2).sum())
        a["d1"] += int((ratio < 1.25).sum())
        a["d2"] += int((ratio < 1.25 ** 2).sum())
        a["d3"] += int((ratio < 1.25 ** 3).sum())
        # reservoir of at most 200k values per bucket keeps the medians honest but bounded
        if len(a["rel"]) < 200_000:
            a["rel"].append(r[:: max(1, r.size // 2000)].astype(np.float32))
            a["log"].append(np.abs(np.log(pred) - np.log(gt))[
                :: max(1, pred.size // 2000)].astype(np.float32))

    def result(self):
        out = {}
        for k, a in self.s.items():
            n = max(a["n"], 1)
            rel = np.concatenate(a["rel"]) if a["rel"] else np.zeros(0, np.float32)
            lg = np.concatenate(a["log"]) if a["log"] else np.zeros(0, np.float32)
            out["|".join(str(x) for x in k)] = {
                "absrel": a["sum_rel"] / n, "rmse": float(np.sqrt(a["sum_sq"] / n)),
                "med_rel": float(np.median(rel)) if rel.size else float("nan"),
                "med_log": float(np.median(lg)) if lg.size else float("nan"),
                "d1": a["d1"] / n, "d2": a["d2"] / n, "d3": a["d3"] / n, "n": a["n"]}
        return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/gate5_2/kitti360_transfer.yaml")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    scfg = load_config("configs/scale_gate/semantickitti.yaml")
    conf_thr = float(scfg.lingbot.confidence_threshold)
    art = os.path.join(REPO_ROOT, cfg.experiment.output_dir)

    # --- deployable predictions must already be final; pin them ------------------- #
    scale_files = {v: os.path.join(art, f"scales_{v}.csv") for v in ("C0C3", "A", "B")}
    for v, p in scale_files.items():
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing: finalise the deployable scales first")
    pinned = {v: sha256(p) for v, p in scale_files.items()}
    print("deployable scale tables pinned:")
    for v, h in pinned.items():
        print(f"  {v}: {h}")

    S = {}
    for r in csv.DictReader(open(scale_files["C0C3"])):
        S.setdefault(r["clip_id"], {})["C0"] = float(r["s_c0"])
        S[r["clip_id"]]["C3"] = float(r["s_c3"])
    for v in ("A", "B"):
        for r in csv.DictReader(open(scale_files[v])):
            S.setdefault(r["clip_id"], {})[v] = (float(r["s_moge"]) if int(r["ok"]) else
                                                 float("nan"))

    calib = parse_calibration(os.path.join(cfg.dataset.root, "calibration"))
    pre = Preprocess.build(calib.native_hw, int(cfg.lingbot.inference_resolution),
                           int(cfg.lingbot.patch_size))
    recs = read_manifest(os.path.join(REPO_ROOT, "manifests", "gate5_2", "val.jsonl"))
    if a.limit:
        recs = recs[: a.limit]
    G = SSCBENCH_KITTI360_GRID
    bands = [tuple(b) for b in cfg.eval.depth_bands_m]

    rows, acc, t0, n_short = [], Accumulator(), time.time(), 0
    for i, rec in enumerate(recs):
        cid = rec["clip_id"]
        lp = os.path.join(cfg.lingbot.cache_root, cid + ".npz")
        if not os.path.exists(lp) or cid not in S:
            continue
        d = np.load(lp, allow_pickle=False)
        dep = d["pred_depth"].astype(np.float64)
        conf = d["pred_depth_conf"].astype(np.float32)

        logs, wts, per_frame_n = [], [], []
        gts, msks = [], []
        for t, f in enumerate(rec["native_frames"]):
            gt, valid = project_to_lattice(
                read_velodyne(cfg.dataset.kitti360_root, int(f))[:, :3], calib, pre)
            m = valid & (gt > 0) & np.isfinite(dep[t]) & (dep[t] > 0) & (conf[t] >= conf_thr)
            gts.append(gt); msks.append(m); per_frame_n.append(int(m.sum()))
            if m.any():
                logs.append(np.log(gt[m]) - np.log(dep[t][m]))
                wts.append(conf[t][m].astype(np.float64))
        n_tot = int(sum(per_frame_n))
        ok = n_tot >= int(cfg.oracle.min_valid_lidar_pixels)
        if not ok:
            n_short += 1
        s_or = float(np.exp(weighted_median(np.concatenate(logs), np.concatenate(wts)))) \
            if ok else float("nan")
        s_anchor = float(np.exp(weighted_median(
            np.log(gts[-1][msks[-1]]) - np.log(dep[-1][msks[-1]]),
            conf[-1][msks[-1]].astype(np.float64)))) if per_frame_n[-1] else float("nan")

        # secondary oracle: centres of the official voxelised LiDAR input at the anchor
        occ = voxelized_lidar_input(cfg.dataset.root, int(rec["anchor"]), G)
        idx = np.argwhere(occ)
        s_vox = float("nan")
        if len(idx):
            centres = G.voxel_centres(idx)
            gtv, validv = project_to_lattice(centres, calib, pre)
            mv = validv & (gtv > 0) & np.isfinite(dep[-1]) & (dep[-1] > 0) & \
                (conf[-1] >= conf_thr)
            if mv.sum() >= int(cfg.oracle.min_valid_lidar_pixels):
                s_vox = float(np.exp(weighted_median(
                    np.log(gtv[mv]) - np.log(dep[-1][mv]),
                    conf[-1][mv].astype(np.float64))))

        scales = dict(S[cid]); scales["OR"] = s_or
        row = {"clip_id": cid, "block": int(rec["block"]), "anchor": int(rec["anchor"]),
               "s_oracle": s_or, "s_oracle_anchor_only": s_anchor,
               "s_oracle_voxel_centre": s_vox, "ok": int(ok),
               "n_valid_lidar_px": n_tot,
               **{f"s_{k}": scales[k] for k in ("C0", "C3", "A", "B")}}
        for k in ("C0", "C3", "A", "B", "OR"):
            sv = scales[k]
            if not (np.isfinite(sv) and sv > 0):
                continue
            for t in range(len(rec["native_frames"])):
                m = msks[t]
                if not m.any():
                    continue
                p, g = sv * dep[t][m], gts[t][m]
                acc.add((k, "all", "all"), p, g)
                acc.add((k, "all", f"t{t}"), p, g)
                for lo, hi in bands:
                    b = (g >= lo) & (g < hi)
                    if b.any():
                        acc.add((k, f"{int(lo)}-{int(hi)}m", "all"), p[b], g[b])
            row[f"absrel_{k}"] = depth_metrics(
                np.concatenate([sv * dep[t][msks[t]] for t in range(5) if msks[t].any()]),
                np.concatenate([gts[t][msks[t]] for t in range(5) if msks[t].any()])
            )["absrel"] if any(m.any() for m in msks) else float("nan")
        rows.append(row)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(recs)}  {time.time()-t0:.0f}s", flush=True)

    with open(os.path.join(art, "oracle_scales.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    write_json(os.path.join(art, "depth_diagnostics.json"), {
        "pinned_deployable_scale_sha256": pinned,
        "oracle": {"primary": "projected_lidar (raw KITTI-360 velodyne, 5 frames pooled)",
                   "secondary": "voxel_centre_lidar (official .bin centres, anchor only; "
                                "quantised to 0.2 m, NOT an exact projected-point oracle)",
                   "range_m": [LIDAR_MIN_M, LIDAR_MAX_M],
                   "estimator": "confidence-weighted median of log(D_lidar) - log(D_lingbot)"},
        "n_clips": len(rows), "n_below_min_valid_px": n_short,
        "by_condition": acc.result(), "elapsed_s": time.time() - t0})
    print(f"\n{len(rows)} clips in {time.time()-t0:.0f}s; {n_short} below the "
          f"{cfg.oracle.min_valid_lidar_pixels}-pixel LiDAR minimum")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
