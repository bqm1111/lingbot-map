#!/usr/bin/env python
"""Qualitative panel: the same anchors drawn as ground truth, map only, ours and both OccAny arms.

Anchor choice is a recorded rule, not a curation. All 801 canonical anchors are ranked by the
per-anchor SC IoU difference (ours KITTI-360 minus OccAny's causal one-frame arm, raw, over the
whole valid volume) and one anchor is taken at each fixed quantile. The low quantiles are
anchors where OccAny wins, so failures are in the figure by construction.

Columns
    1  ground truth, coloured by the strict visibility region it falls in
    2  map only -- the same pipeline with the completion head removed
    3  ours, KITTI-360 head
    4  OccAny, one causal frame
    5  OccAny, five forward frames (t..t+40) -- it sees 4.0 s of future
    6  who recovers which GT-occupied voxel, ours against OccAny's causal arm

Columns 2-5 use one colour scheme: recovered / missed / predicted where the GT says free. Every
panel is the same camera, the same anchor, the same evaluation grid and the same valid mask.

    PYTHONPATH=$PWD python tools/method_vs_data/qualitative_panel.py --device cuda:3
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch                                                                           # noqa: E402
from matplotlib.patches import Patch                                                   # noqa: E402
from gates.gate8c1 import render3d as R3                                               # noqa: E402
from tools.gate8c1.figure_3d_compare import (C_BOTH, C_FP, C_GT, C_MISS, C_OCC,         # noqa: E402
                                             C_OURS, C_TP, SOLID, compose)
from tools.method_vs_data.canonical_rescore import OCCANY_ARMS, OURS_ROOTS             # noqa: E402
from tools.method_vs_data.map_only_predictions import load_map_only                     # noqa: E402
from tools.method_vs_data.observed_hidden import IMAGE_SIZE, strict_four                # noqa: E402
from tools.method_vs_data.observed_strict import frame_transforms, sk_pose_chain        # noqa: E402
from tools.observability_split import runs as R                                         # noqa: E402
from tools.zero_shot_eval import config as ZC, samples as ZS                            # noqa: E402
from tools.zero_shot_eval.canonical.prediction import CanonicalPrediction               # noqa: E402
from tools.zero_shot_eval.practical.protocol import DATASET_SPECS                       # noqa: E402

OUT_DIR = os.path.join(ZC.REPO_ROOT, "artifacts", "method_vs_data", "figures",
                       "phase_runtime_hidden")
QUANTILES = (0.04, 0.27, 0.50, 0.73, 0.96)
SK_FOV_DEG = (81.8, 29.1)
VIEW = dict(eye_offset=(-34.0, -26.0, 30.0), fov=46.0)
FRUSTUM = dict(depth=6.0, radius=0.16)
C_REG = {"anchor_observed": (0.20, 0.60, 0.25),
         "earlier_only": (0.98, 0.76, 0.18),
         "never_observed_in_frustum": (0.85, 0.33, 0.20),
         "never_observed_out_frustum": (0.42, 0.26, 0.62)}
TITLES = ("1 · ground truth, by strict visibility region", "2 · map only — no learned completion",
          "3 · ours (KITTI-360)", "4 · OccAny — 1 causal frame",
          "5 · OccAny — 5 forward frames [non-causal]", "6 · ours vs OccAny causal")
HANDLES = ([Patch(color=c, label="GT occupied, " + k.replace("_", " ")) for k, c in C_REG.items()]
           + [Patch(color=C_TP, label="recovered correctly"),
              Patch(color=C_MISS, label="left out by this method"),
              Patch(color=C_FP, label="predicted where the ground truth says free"),
              Patch(color=C_OURS, label="only ours recovers it"),
              Patch(color=C_OCC, label="only OccAny (causal) recovers it"),
              Patch(color=C_BOTH, label="both recover it"),
              Patch(color=(0.05, 0.05, 0.05), label="anchor camera frustum")])


class Grid:
    def __init__(self, dims, vs, origin):
        self.dims, self.voxel_size, self.origin = dims, vs, origin


def _mesh(mask, colour, g, shrink=SOLID):
    idx = np.argwhere(np.asarray(mask).reshape(g.dims))
    if not len(idx):
        return None
    return R3.cube_mesh(idx, np.tile(colour, (len(idx), 1)), float(g.voxel_size), g.origin,
                        shrink=shrink)


def iou(pred, gt_occ, valid):
    tp = int((pred & gt_occ & valid).sum())
    fp = int((pred & ~gt_occ & valid).sum())
    fn = int((~pred & gt_occ & valid).sum())
    return 100.0 * tp / max(tp + fp + fn, 1), tp, fp, fn


def rank(dataset: str, runs, spec, dims):
    """Per-anchor (ours - OccAny causal) SC IoU, raw, over the whole valid volume."""
    sids = [s for s in R.samples(dataset)]
    out = []
    for i, sid in enumerate(sids):
        label, vol = R.load_occany(runs["occany_1f_geom"], sid, dims)
        valid = label != 255
        gt = valid & (label != spec.empty_class)
        ours = CanonicalPrediction.load(os.path.join(OURS_ROOTS["k360"], dataset,
                                                     f"{sid}.npz")).occupied
        occ = (vol != spec.empty_class) & (vol != 255)
        a, *_ = iou(ours, gt, valid)
        b, *_ = iou(occ, gt, valid)
        out.append({"sample_id": sid, "ours": a, "occany_1f": b, "delta": a - b})
        if (i + 1) % 200 == 0:
            print(f"  ranked {i + 1}/{len(sids)}", flush=True)
    out.sort(key=lambda r: r["delta"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--post", default="raw", choices=["raw", "pooled"])
    ap.add_argument("--ranking", default=os.path.join(OUT_DIR, "panel_ranking.json"))
    a = ap.parse_args()
    ds = "semantickitti"
    spec = DATASET_SPECS[ds]
    dev = torch.device(a.device)
    runs = {r.key: r for r in R.RUNS[ds]}
    by_sid = {s.sample_id: s for s in ZS.evaluated_samples(ds)}
    ref = CanonicalPrediction.load(os.path.join(OURS_ROOTS["k360"], ds,
                                                f"{R.samples(ds)[0]}.npz"))
    dims, vs, origin = ref.dims, ref.voxel_size, ref.origin
    g = Grid(dims, vs, origin)
    os.makedirs(OUT_DIR, exist_ok=True)

    if os.path.exists(a.ranking):
        ranked = json.load(open(a.ranking))["ranked"]
    else:
        ranked = rank(ds, runs, spec, dims)
        json.dump({"rule": "per-anchor SC IoU, ours(KITTI-360) minus OccAny 1-frame causal, "
                           "raw, whole valid volume; ascending", "ranked": ranked},
                  open(a.ranking, "w"), indent=1, default=float)
    picks = [(q, ranked[min(int(round(q * (len(ranked) - 1))), len(ranked) - 1)])
             for q in QUANTILES]

    poses, cam_to_velo = sk_pose_chain(ZC.REPO_ROOT, "08")
    rows, record = [], []
    for q, pk in picks:
        sid = pk["sample_id"]
        s = by_sid[sid]
        label, _ = R.load_occany(runs["occany_1f_geom"], sid, dims)
        valid = label != 255
        gt = valid & (label != spec.empty_class)
        p0 = CanonicalPrediction.load(os.path.join(OURS_ROOTS["k360"], ds, f"{sid}.npz"))
        natives = [int(k.split("_")[1]) for k in (f["key"] for f in ZS.history_frames(s))]
        Ts = frame_transforms(poses, cam_to_velo, natives, natives[-1])
        masks, _seen, _fov = strict_four(gt, valid, Ts, [p0.K] * len(Ts), IMAGE_SIZE[ds],
                                         dims, vs, origin, dev, 2, 0.1)
        mo, _T = load_map_only(ds, sid)
        preds = {"map_only": mo, "ours": p0.occupied}
        for k, run_key in OCCANY_ARMS.items():
            _l, vol = R.load_occany(runs[run_key], sid, dims)
            preds[k] = (vol != spec.empty_class) & (vol != 255)
        if a.post == "pooled":
            from gates.gate8b import pooling as PL
            preds = {k: PL.geometry_dilation(
                torch.from_numpy(v).reshape(dims)).reshape(-1).numpy() for k, v in preds.items()}

        fr = R3.frustum_mesh(p0.T_cam_to_grid, *SK_FOV_DEG, **FRUSTUM)
        idx = np.argwhere(gt.reshape(dims))
        ctr = ((idx + 0.5) * float(vs) + np.asarray(origin)).mean(0) if len(idx) else None
        scenes = [[_mesh(masks[k] & gt, c, g) for k, c in C_REG.items()]]
        for k in ("map_only", "ours", "occany_1f", "occany_5f_fwd"):
            p = preds[k]
            scenes.append([_mesh(~p & gt, C_MISS, g), _mesh(p & gt, C_TP, g),
                           _mesh(p & ~gt & valid, C_FP, g)])
        o, u = preds["occany_1f"], preds["ours"]
        scenes.append([_mesh(gt & o & u, C_BOTH, g, 0.55), _mesh(gt & u & ~o, C_OURS, g),
                       _mesh(gt & o & ~u, C_OCC, g)])
        imgs = [R3.render([m for m in ms if m is not None] + [fr], size=(980, 700),
                          look_at=ctr, **VIEW) for ms in scenes]
        stats = {k: iou(v, gt, valid)[0] for k, v in preds.items()}
        rows.append({"images": imgs,
                     "label": f"{sid}   q={q:.2f}   ours {stats['ours']:.1f} vs "
                              f"OccAny-1f {stats['occany_1f']:.1f}  "
                              f"(Δ {stats['ours'] - stats['occany_1f']:+.1f})"})
        record.append({"quantile": q, "sample_id": sid, "post": a.post, "sc_iou": stats,
                       "region_gt_occupied": {k: int((masks[k] & gt).sum()) for k in C_REG}})
        print(f"  rendered {sid} (q={q:.2f})", flush=True)

    out = os.path.join(OUT_DIR, f"qualitative_panel_{a.post}.png")
    compose(rows, out, "Observed, remembered and never-observed geometry on identical anchors "
                       f"(SemanticKITTI 08, {a.post})", HANDLES, titles=TITLES, dpi=100)
    json.dump({"post": a.post, "quantiles": list(QUANTILES), "rows": record,
               "selection_rule": "fixed quantiles of the recorded per-anchor ranking in "
                                 "panel_ranking.json; low quantiles are OccAny wins"},
              open(os.path.join(OUT_DIR, f"qualitative_panel_{a.post}.json"), "w"),
              indent=1, default=float)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
