#!/usr/bin/env python
"""Gate 8C-0 Stage 4: where does KITTI-360 degradation enter -- depth, pose, or neither?

Four cells on one fixed audit subset, all through the **real** incremental mapper, so the
map/export path is identical to Gate 8B's:

    GT depth  x GT pose      -- the ceiling of the current map representation
    GT depth  x LingBot pose -- isolates pose error
    LingBot depth x GT pose  -- isolates depth error
    LingBot depth x LingBot pose -- the deployed configuration

"GT depth" is the anchor's own velodyne sweep projected into the network's image lattice
and kept only where a real return lands: a sparse, legitimate measurement, never densified
with target labels. Predicted cells apply the frozen scale policy exactly -- the pinned
G51-B scalar, the same number on depth and on pose translation -- and scale is never
estimated from target occupancy.

    python tools/gate8c0/stage4_factorization.py --device cuda:1
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, cfg, default_device                          # noqa: E402
from gates.scale_gate.config import write_json                                         # noqa: E402
from gates.gate6 import grids as G6G, targets as G6T                                   # noqa: E402
from gates.gate7b import depth as D7                                                   # noqa: E402
from gates.gate8 import sources as S                                                   # noqa: E402
from gates.gate8.mapper import FrameInput, IncrementalMapper                           # noqa: E402
from gates.gate8c0 import oracle as OR, transforms as TF                               # noqa: E402

HISTORIES = (1, 5, 20)


def lidar_depth_on_lattice(g: TF.DriveGeometry, native: int, K: np.ndarray, hw) -> tuple:
    """Sparse metric z-depth on the processed lattice from the sweep. No densification."""
    pts = g.read_velodyne(native)
    pc = TF.apply(TF.inv(g.rect_cam_to_velo), pts)          # velodyne -> rectified cam0
    z = pc[:, 2]
    ok = (z > D7.MIN_DEPTH_M) & (z < D7.MAX_DEPTH_M)
    pc, z = pc[ok], z[ok]
    u = (K[0, 0] * pc[:, 0] / z + K[0, 2]).astype(np.int64)
    v = (K[1, 1] * pc[:, 1] / z + K[1, 2]).astype(np.int64)
    H, W = hw
    m = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[m], v[m], z[m]
    dep = np.full((H, W), np.inf)
    np.minimum.at(dep, (v, u), z)                            # nearest return wins
    hit = np.isfinite(dep)
    dep[~hit] = 0.0
    return dep.astype(np.float32), hit


def build_map(dev, frames, n_teacher=0):
    m = IncrementalMapper(dev, n_teacher=n_teacher)
    m.scale_state.force(1.0)
    for f in frames:
        m.step(f)
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--per-drive", type=int, default=3)
    a = ap.parse_args()
    c = cfg(); dev = torch.device(a.device); torch.cuda.set_device(dev)
    t0 = time.time()
    part_of = {}
    for part, ds in (("train", c["train_drives"]), ("source_validation", [c["val_drive"]]),
                     ("heldout", [c["heldout_drive"]])):
        for d in ds:
            part_of[d] = part
    rng = np.random.default_rng(c["seed"])
    per, store = [], {}
    for d, part in part_of.items():
        src = "kitti360" if d == c["heldout_drive"] else "k360_train"
        g = TF.DriveGeometry(d, c["sscbench_root"], c["kitti360_root"])
        seg = [s for s in S.segments(src, REPO_ROOT) if s.name == d][0]
        with np.load(S.stream_path(src, d)) as z:
            dep_p = z["pred_depth"].astype(np.float32); conf_p = z["pred_depth_conf"].astype(np.float32)
            K_p = z["pred_K"].astype(np.float64); pose_p = z["pred_pose_c2w"].astype(np.float64)
            keys = [str(k) for k in z["keys"]]
        with np.load(S.scale_path(src, d)) as z:
            log_s = z["log_s"].astype(np.float64)
        scale = float(np.exp(np.median(log_s[:5])))          # the frozen five-frame anchor
        pos = {k: i for i, k in enumerate(keys)}
        anchors = [f for f in seg.frames if f.gt_ref is not None and f.index >= 20]
        pick = [anchors[int(q * (len(anchors) - 1))] for q in np.linspace(0.25, 0.75, a.per_drive)]
        hw = dep_p.shape[1:]
        for fr in pick:
            i = fr.index
            anc = int(fr.gt_ref["anchor"]); native = g.native(anc)
            target, keep = G6T.semantic_target("kitti360", fr.gt_ref, REPO_ROOT)
            occ = OR.occupied_from_target(target, keep)
            rec = {"partition": part, "drive": d, "anchor": anc, "stream_index": i,
                   "scale": scale, "cells": {}}
            for cell in ("gt_depth_gt_pose", "gt_depth_lb_pose", "lb_depth_gt_pose",
                         "lb_depth_lb_pose"):
                gtd, gtp = cell.startswith("gt_depth"), cell.endswith("gt_pose")
                for H in HISTORIES:
                    idxs = [j for j in range(max(0, i - H + 1), i + 1)]
                    frames = []
                    for j in idxs:
                        nj = g.native(int(seg.frames[j].gt_ref["anchor"])) \
                            if seg.frames[j].gt_ref else None
                        if nj is None:
                            nj = int(g.pose_frames[np.searchsorted(
                                g.pose_frames, g.native(anc) - (i - j) * 5)])
                        K = K_p[j]
                        if gtd:
                            dm, hit = lidar_depth_on_lattice(g, nj, K, hw)
                            dt = torch.from_numpy(dm)
                            cf = torch.from_numpy((hit * (D7.CONF_THRESHOLD + 1.0)).astype(np.float32))
                        else:
                            dt = torch.from_numpy(dep_p[j] * scale)
                            cf = torch.from_numpy(conf_p[j])
                        if gtp:
                            P = TF.inv(g.rect_cam_to_world(native)) @ g.rect_cam_to_world(nj)
                        else:
                            Pw = pose_p[j].copy(); Pw[:3, 3] *= scale
                            Pa = pose_p[i].copy(); Pa[:3, 3] *= scale
                            P = TF.inv(Pa) @ Pw               # relative to the anchor camera
                        frames.append(FrameInput(index=j, depth_canonical=dt.to(dev),
                                                 conf=cf.to(dev), K=K,
                                                 pose_c2w_canonical=P,
                                                 log_scale_candidate=0.0))
                    m = build_map(dev, frames)
                    q = m.query(G6G.PREDICTION_GRID["kitti360"],
                                TF.inv(np.asarray(fr.T_cam_to_grid, np.float64)))
                    pred = (q["logodds"] > 0).cpu().numpy().reshape(TF.DIMS)
                    cc = OR.counts(pred, occ, keep)
                    dd = OR.distance_stats(pred & keep, occ)
                    rec["cells"][f"{cell}_h{H}"] = {
                        "counts": cc, "distance": dd,
                        "predicted_occupied_fraction": cc["pred_fraction"],
                        "target_coverage": cc["recall"]}
                    k = (part, cell, H); store.setdefault(k, [0, 0, 0])
                    for z_, f_ in enumerate(("tp", "fp", "fn")):
                        store[k][z_] += cc[f_]
            per.append(rec)
            print(f"  {part:17s} {d[-9:-5]} anc {anc:6d} " + " ".join(
                f"{cl[:11]}h{H}:IoU{rec['cells'][f'{cl}_h{H}']['counts']['iou']:.3f}"
                for cl in ("gt_depth_gt_pose", "lb_depth_lb_pose") for H in HISTORIES), flush=True)
    agg = {}
    for (part, cell, H), (tp, fp, fn) in store.items():
        agg.setdefault(part, {}).setdefault(cell, {})[str(H)] = {
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "iou": tp / max(tp + fp + fn, 1)}
    out = {"histories": list(HISTORIES), "n_samples": len(per),
           "gt_depth": "anchor-frame velodyne projected to the processed lattice; sparse, "
                       "never densified with target labels",
           "scale_policy": "frozen five-frame G51-B median, applied identically to depth "
                           "and to pose translation; never estimated from target occupancy",
           "aggregate": agg, "per_sample": per, "seconds": time.time() - t0,
           "peak_gpu_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30}
    write_json(os.path.join(ART, "stage4_factorization.json"), out)
    print(f"\nstage 4 aggregate ({len(per)} samples)")
    for part, v in agg.items():
        print(f"  {part}")
        for cell, hv in v.items():
            print(f"     {cell:20s} " + "  ".join(
                f"h{H}: P {hv[str(H)]['precision']:.3f} R {hv[str(H)]['recall']:.3f} "
                f"IoU {hv[str(H)]['iou']:.3f}" for H in HISTORIES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
