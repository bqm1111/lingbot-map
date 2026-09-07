#!/usr/bin/env python
"""Gate 8C-1: head-to-head against the released OccAny checkpoint on SemanticKITTI 08.

The counterpart of ``compare_occany.py`` / ``figure_3d_compare.py``, which do the same job
on Occ3D-nuScenes. OccAny's own model was run here (``extract_output_occany.py``, EXP_ID 0
settings: kitti, 5frames, ``--gen``, conf 2.5), so both sides are measured, not quoted.

**Target frames differ between the two protocols and are matched explicitly.** OccAny builds
a 10-frame video at ``frame_interval=5`` and takes ``recon_view_idx = [0, 2, 4, 6, 8]``, so
its target is the *first* frame and the other four are in its future. Ours is the mirror
image: five frames *ending* at the target. A sample directory ``08_000765`` therefore pairs
with our clip ``08_000745_000765_s5`` -- same target frame 765, opposite temporal direction.
Ours is the strictly harder input and the figure says so.

Ground truth is OccAny's stored ``voxel_label``, after checking it against the target our own
evaluator builds. SemanticKITTI's empty class is 0 (not 17 as on nuScenes) and 255 is ignore.

    python tools/gate8c1/compare_occany_kitti.py --n 8 --device cuda:2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import time

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                  # noqa: E402
from matplotlib.patches import Patch                                             # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, default_device                               # noqa: E402
from gates.scale_gate.config import write_json                                         # noqa: E402
from gates.gate6 import grids as G6G, targets as G6T, vocab                            # noqa: E402
from gates.gate8 import sources as S, vocab as V8                                      # noqa: E402
from gates.gate8.net import load_checkpoint                                            # noqa: E402
from gates.gate8b import pooling as PL                                                 # noqa: E402
from gates.gate8c1 import occany_eval as OE, render3d as R3                            # noqa: E402
from tools.gate8c1.eval_target import MANIFEST, PAST5_OFFSETS                    # noqa: E402
from tools.gate8c1.figure_3d_compare import (C_BOTH, C_FP, C_GT, C_MISS,         # noqa: E402
                                             C_OCC, C_OURS, C_TP, GHOST, SOLID,
                                             compose, write_pairs)
from tools.gate8c1.figure_3d_gallery import predict as our_predict               # noqa: E402

OCCANY_PRED = ("/media/SSD1/MINH_DATASETS/occany_out/ssc_voxel_pred/"
               "OccAny_5frames_kitti512_rot60_vpi10_fwd3_sTrans2")
PRED_KEY = "render_recon_gen_recon2.5_gen2.5"
FREE, IGNORE = 0, 255                       # SemanticKITTI: class 0 is empty
DS = "semantickitti"
#: KITTI cam2, from the published intrinsics (fx 718.8, 1226x370)
FOV_X_DEG, FOV_Y_DEG = 81.8, 29.1


def occany_samples():
    """``target frame id -> path``, from the ``08_000765`` directory names."""
    out = {}
    for f in sorted(glob.glob(os.path.join(OCCANY_PRED, "*", "voxel_predictions.pkl"))):
        name = os.path.basename(os.path.dirname(f))
        try:
            out[int(name.split("_")[1])] = f
        except (IndexError, ValueError):
            continue
    return out


def load_occany(path, dims):
    d = pickle.load(open(path, "rb"))
    lab = np.asarray(d["voxel_label"]).reshape(dims)
    return {"pred": np.asarray(d[PRED_KEY]).reshape(dims).astype(bool),
            "label": lab,
            "images": np.asarray(d.get("estimated_input_images"))}


def counts(pred, gt, valid):
    p, g = pred & valid, gt & valid
    tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "iou": tp / max(tp + fp + fn, 1),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "pred_over_gt": (tp + fp) / max(tp + fn, 1)}


def official(pred, label, n_classes):
    """Score through OccAny's own SSCMetrics, with SemanticKITTI's empty class."""
    lab = np.where(pred, 1, FREE).astype(np.uint8)
    return OE.official_sc_counts(lab, label.astype(np.uint8), n_classes=n_classes,
                                 empty_class=FREE)


def mesh(mask, colour, grid, shrink=SOLID):
    idx = np.argwhere(mask)
    if not len(idx):
        return None
    return R3.cube_mesh(idx, np.tile(colour, (len(idx), 1)),
                        float(grid.voxel_size), grid.origin, shrink=shrink)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=8, help="scenes in the figure")
    ap.add_argument("--per-figure", type=int, default=2,
                    help="scenes per extra split figure")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--max-metric", type=int, default=100000)
    ap.add_argument("--metrics-only", action="store_true")
    a = ap.parse_args()

    fz = json.load(open(MANIFEST))["seeds"][str(a.seed)]
    tau = float(fz["occupancy_threshold"])
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    v = vocab.load(DS)
    C = len(v)
    MAP, EVAL = G6G.PREDICTION_GRID[DS], G6G.EVAL_GRID[DS]
    edims = tuple(int(d) for d in EVAL.dims)
    into = V8.into_matrix(DS)
    net = load_checkpoint(os.path.join(REPO_ROOT, fz["checkpoint"]), dev)
    seg = S.segments(DS, REPO_ROOT)[0]

    oc = occany_samples()
    # our anchor's target frame is the LAST frame of its clip; OccAny's is its FIRST
    ours_by_frame = {}
    for i in sorted(seg.anchors):
        if i + min(PAST5_OFFSETS) < 0:
            continue
        ours_by_frame[int(seg.frames[i].gt_ref["frame_ids"][-1])] = i
    shared = sorted(set(oc) & set(ours_by_frame))
    print(f"OccAny samples {len(oc)}   our anchors {len(ours_by_frame)}   "
          f"shared target frames {len(shared)}", flush=True)
    if not shared:
        print("no shared target frames -- is the OccAny KITTI run finished?")
        return 1

    n_cls = 20            # SemanticKITTI SSC: class 0 empty + 19 semantic classes
    VARIANTS = ("occany", "occany_pooled", "ours", "ours_pooled")
    agg = {k: [0, 0, 0] for k in VARIANTS}
    rows, label_identical, t0 = [], 0, time.time()
    for f in shared[:a.max_metric]:
        d = load_occany(oc[f], edims)
        label = d["label"]
        gt = (label != FREE) & (label != IGNORE)
        valid = label != IGNORE
        i = ours_by_frame[f]
        r = our_predict(DS, seg, i, a.seed, dev, net, tau, into, C, MAP, EVAL, edims)
        if r is None:
            continue
        # is their stored ground truth the same volume our own evaluator builds?
        both_valid = valid & r["valid"]
        label_identical += int(bool((gt[both_valid] == r["gt"][both_valid]).all()
                                    and bool((valid == r["valid"]).all())))
        preds = {"occany": d["pred"], "ours": r["pred"]}
        for nm in list(preds):
            preds[nm + "_pooled"] = PL.geometry_dilation(
                torch.from_numpy(preds[nm])).numpy()
        row = {"frame": f, "stream_index": i}
        for nm, p in preds.items():
            tp, fp, fn = official(p, label, n_cls)
            for j, x in enumerate((tp, fp, fn)):
                agg[nm][j] += x
            row[nm] = counts(p, gt, valid)
            row[nm]["official"] = {"tp": tp, "fp": fp, "fn": fn,
                                   "iou": tp / max(tp + fp + fn, 1)}
        rows.append(row)
        if len(rows) % 25 == 0:
            print(f"  {len(rows)} scored ({(time.time()-t0)/len(rows):.2f}s each)",
                  flush=True)

    out = {"n_scored": len(rows), "dataset": DS, "threshold": tau, "seed": a.seed,
           "occany_pred_dir": OCCANY_PRED, "occany_pred_key": PRED_KEY,
           "gt_label_identical_count": label_identical,
           "scored_with": "OccAny SSCMetrics.get_score_completion",
           "protocol_note": ("OccAny's target is the FIRST of its five frames (the other "
                             "four are its future); ours is the LAST (all five are past). "
                             "Same target frame, opposite temporal direction."),
           "pooled": {}, "per_sample": rows[:200], "seconds": time.time() - t0}
    for nm, (tp, fp, fn) in agg.items():
        out["pooled"][nm] = {"tp": tp, "fp": fp, "fn": fn,
                             "iou": tp / max(tp + fp + fn, 1),
                             "precision": tp / max(tp + fp, 1),
                             "recall": tp / max(tp + fn, 1),
                             "pred_over_gt": (tp + fp) / max(tp + fn, 1)}
    write_json(os.path.join(ART, "occany_kitti_headtohead.json"), out)
    print(f"\npooled over {len(rows)} shared target frames (OccAny's own evaluator):")
    for nm, m in out["pooled"].items():
        print(f"   {nm:15s} IoU {100*m['iou']:6.2f}  P {100*m['precision']:6.2f}  "
              f"R {100*m['recall']:6.2f}  pred/GT {m['pred_over_gt']:.2f}")
    print(f"   GT identical to ours on {label_identical}/{len(rows)}")
    if a.metrics_only:
        return 0

    # ------------------------------------------------------------------ figure
    pick = [rows[int(round(q * (len(rows) - 1)))]
            for q in np.linspace(0.04, 0.96, min(a.n, len(rows)))]
    handles = [Patch(color=C_GT, label="ground truth — the shape to be reproduced"),
               Patch(color=C_TP, label="recovered correctly"),
               Patch(color=C_MISS, label="left out by this method"),
               Patch(color=C_FP, label="predicted where the ground truth says free"),
               Patch(color=C_OURS, label="only ours recovers it"),
               Patch(color=C_OCC, label="only OccAny recovers it"),
               Patch(color=C_BOTH, label="both recover it"),
               Patch(color=(0.05, 0.05, 0.05), label="camera frustum at the target frame")]
    TITLE = "Visualization on SemanticKITTI 08"
    VIEW = dict(eye_offset=(-34.0, -26.0, 30.0), fov=46.0)

    rendered = []
    for row in pick:
        f = row["frame"]
        d = load_occany(oc[f], edims)
        label = d["label"]
        gt = (label != FREE) & (label != IGNORE)
        valid = label != IGNORE
        rr = our_predict(DS, seg, ours_by_frame[f], a.seed, dev, net, tau, into, C,
                         MAP, EVAL, edims)
        p_oc, p_us = d["pred"], rr["pred"]
        g = gt & valid
        ao, au = p_oc & valid, p_us & valid
        ghost_o = mesh(~p_oc & g, C_MISS, EVAL, GHOST)
        ghost_u = mesh(~p_us & g, C_MISS, EVAL, GHOST)
        scenes = [
            [mesh(g, C_GT, EVAL)],
            [ghost_o, mesh(p_oc & g, C_TP, EVAL)],
            [ghost_u, mesh(p_us & g, C_TP, EVAL)],
            [ghost_o, mesh(p_oc & g, C_TP, EVAL), mesh(ao & ~g, C_FP, EVAL)],
            [ghost_u, mesh(p_us & g, C_TP, EVAL), mesh(au & ~g, C_FP, EVAL)],
            [mesh(g & ao & au, C_BOTH, EVAL, 0.55), mesh(g & au & ~ao, C_OURS, EVAL),
             mesh(g & ao & ~au, C_OCC, EVAL)],
        ]
        # the grid frame is the velodyne of this frame; the camera is the pose we already
        # use to place the map, so the frustum is exact rather than assumed
        fr = R3.frustum_mesh(np.asarray(seg.frames[ours_by_frame[f]].T_cam_to_grid, float),
                             FOV_X_DEG, FOV_Y_DEG, depth=6.0, radius=0.16)
        idx = np.argwhere(g)
        ctr = (((idx + 0.5) * float(EVAL.voxel_size) + np.asarray(EVAL.origin)).mean(0)
               if len(idx) else None)
        rendered.append({"images": [R3.render(ms + [fr], size=(980, 700),
                                              look_at=ctr, **VIEW)
                                    for ms in scenes],
                         "label": f"frame {f:06d}"})
        print(f"  rendered frame {f:06d}", flush=True)

    base = os.path.join(ART, "fig_occany_kitti_3d.png")
    compose(rendered, base, TITLE, handles, dpi=100)
    write_pairs(rendered, base, TITLE, handles, per_fig=a.per_figure, dpi=100)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
