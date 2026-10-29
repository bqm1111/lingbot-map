#!/usr/bin/env python
"""Qualitative panel: the same anchors under both inference modes, against the visibility history.

Anchor choice is a recorded rule, not a curation. Every canonical anchor is ranked by the
per-anchor SC IoU difference (**streaming minus clip-reset**, KITTI-360 seed 0, raw, whole
valid volume) and one anchor is taken at each fixed quantile. The low quantiles are anchors
where the streaming mode LOSES, so failures are in the figure by construction.

Columns
    1  ground truth, coloured by the history-aware visibility region it falls in
    2  map only, streaming mode -- the same pipeline with the completion head removed
    3  ours, clip-reset reference
    4  ours, persistent state + startup-only scale
    5  OccAny, one causal frame
    6  which GT-occupied voxel each mode recovers -- the A/B difference itself

The point of column 1 beside columns 3-4 is the distinction the protocol asks for: geometry
RETRIEVED from older evidence (``earlier_in_stream``) versus geometry PREDICTED where nothing
in the processed stream ever saw it (``never_in_frustum`` / ``never_out_frustum``).

    PYTHONPATH=$PWD python tools/method_vs_data/streaming/panel.py --device cuda:2
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import torch                                                                            # noqa: E402
from matplotlib.patches import Patch                                                    # noqa: E402
from gates.gate8b import pooling as PL                                                  # noqa: E402
from gates.gate8c1 import render3d as R3                                                # noqa: E402
from tools.gate8c1.figure_3d_compare import (C_BOTH, C_FP, C_MISS, C_OCC,               # noqa: E402
                                             C_OURS, C_TP, SOLID, compose)
from tools.method_vs_data.observed_strict import frame_transforms, sk_pose_chain         # noqa: E402
from tools.method_vs_data.streaming import (build_maps as BM, config as SV, heads as HD,  # noqa: E402
                                            streams as ST, visibility as VS)
from tools.observability_split import runs as R                                          # noqa: E402
from tools.zero_shot_eval import config as ZC, samples as ZS                             # noqa: E402
from tools.zero_shot_eval.canonical.prediction import CanonicalPrediction                # noqa: E402
from tools.zero_shot_eval.practical.protocol import DATASET_SPECS                        # noqa: E402

OUT_DIR = os.path.join(SV.FIGURE_DIR)
QUANTILES = (0.03, 0.27, 0.50, 0.73, 0.97)
SK_FOV_DEG = (81.8, 29.1)
VIEW = dict(eye_offset=(-34.0, -26.0, 30.0), fov=46.0)
FRUSTUM = dict(depth=6.0, radius=0.16)
C_REG = {"current_frame": (0.20, 0.60, 0.25),
         "recent_other_four": (0.98, 0.76, 0.18),
         "earlier_in_stream": (0.25, 0.55, 0.85),
         "never_in_frustum": (0.85, 0.33, 0.20),
         "never_out_frustum": (0.42, 0.26, 0.62)}
TITLES = ("1 · ground truth, by history-aware visibility region",
          "2 · map only (streaming) — no learned completion",
          "3 · ours — clip-reset reference",
          "4 · ours — persistent state, startup-only scale",
          "5 · OccAny — 1 causal frame",
          "6 · streaming vs clip-reset")
HANDLES = ([Patch(color=c, label="GT occupied, " + k.replace("_", " ")) for k, c in C_REG.items()]
           + [Patch(color=C_TP, label="recovered correctly"),
              Patch(color=C_MISS, label="left out by this mode"),
              Patch(color=C_FP, label="predicted where the ground truth says free"),
              Patch(color=C_OURS, label="only the streaming mode recovers it"),
              Patch(color=C_OCC, label="only the clip-reset mode recovers it"),
              Patch(color=C_BOTH, label="both modes recover it"),
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
    return 100.0 * tp / max(tp + fp + fn, 1)


def rank(dataset: str, dev, rd, runs, spec) -> list:
    """Per-anchor streaming-minus-clip-reset SC IoU, raw, over the whole valid volume."""
    net = HD.load_head("k360", 0, dev)
    out = []
    sids = sorted(s.sample_id for s in ZS.evaluated_samples(dataset))
    for i, sid in enumerate(sids):
        if not all(os.path.exists(SV.map_cache_path(dataset, m, sid)) for m in ("A", "B")):
            continue
        label, _ = R.load_occany(runs["occany_1f_geom"], sid, rd.native.dims)
        valid = label != 255
        gt = valid & (label != spec.empty_class)
        a = rd.completion(net, *BM.load_map(dataset, "A", sid)[:2])
        b = rd.completion(net, *BM.load_map(dataset, "B", sid)[:2])
        ia, ib = iou(a, gt, valid), iou(b, gt, valid)
        out.append({"sample_id": sid, "clip_reset": ia, "streaming": ib, "delta": ib - ia})
        if (i + 1) % 200 == 0:
            print(f"  ranked {i+1}/{len(sids)}", flush=True)
    out.sort(key=lambda r: r["delta"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--post", default="raw", choices=["raw", "pooled"])
    ap.add_argument("--ranking", default=None)
    a = ap.parse_args()
    ds = "semantickitti"
    spec = DATASET_SPECS[ds]
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    rd = HD.Readout(ds, dev)
    dims, vs, origin = rd.native.dims, rd.native.voxel_size, rd.native.origin
    g = Grid(dims, vs, origin)
    runs = {r.key: r for r in R.RUNS[ds]}
    by_sid = {s.sample_id: s for s in ZS.evaluated_samples(ds)}
    seg = ST.build(ds)[0]
    os.makedirs(OUT_DIR, exist_ok=True)
    rk = a.ranking or os.path.join(OUT_DIR, "panel_ranking.json")

    if os.path.exists(rk):
        ranked = json.load(open(rk))["ranked"]
    else:
        ranked = rank(ds, dev, rd, runs, spec)
        json.dump({"rule": "per-anchor SC IoU, streaming minus clip-reset (KITTI-360 seed 0), "
                           "raw, whole valid volume; ascending so low quantiles are "
                           "streaming LOSSES", "ranked": ranked},
                  open(rk, "w"), indent=1, default=float)
    picks = [(q, ranked[min(int(round(q * (len(ranked) - 1))), len(ranked) - 1)])
             for q in QUANTILES]

    poses, cam_to_velo = sk_pose_chain(SV.REPO_ROOT, "08")
    net = HD.load_head("k360", 0, dev)
    rows, record = [], []
    for q, pk in picks:
        sid = pk["sample_id"]
        s = by_sid[sid]
        p = s.target_pos
        label, _ = R.load_occany(runs["occany_1f_geom"], sid, dims)
        valid = label != 255
        gt = valid & (label != spec.empty_class)
        K0 = CanonicalPrediction.load(ZC.prediction_path(ds, sid)).K
        prefix = list(range(0, p + 1))
        natives = [int(seg.frame_keys[j].split("_")[1]) for j in prefix]
        Ts = frame_transforms(poses, cam_to_velo, natives, natives[p])
        masks, cast = VS.regions_for_anchor(
            gt, valid, Ts, [K0] * len(prefix), prefix, [p + o for o in ST.CAUSAL.offsets],
            p, SV.IMAGE_SIZE[ds], dims, vs, origin, dev)
        SB, FB, _ = BM.load_map(ds, "B", sid)
        SA_, FA_, _ = BM.load_map(ds, "A", sid)
        preds = {"map_only": rd.map_only(SB),
                 "clip_reset": rd.completion(net, SA_, FA_),
                 "streaming": rd.completion(net, SB, FB)}
        _l, vol = R.load_occany(runs["occany_1f_geom"], sid, dims)
        preds["occany_1f"] = (vol != spec.empty_class) & (vol != 255)
        if a.post == "pooled":
            preds = {k: PL.geometry_dilation(
                torch.from_numpy(v).reshape(dims)).reshape(-1).numpy() for k, v in preds.items()}

        fr = R3.frustum_mesh(np.asarray(s.T_cam_to_grid_ours, np.float64), *SK_FOV_DEG, **FRUSTUM)
        idx = np.argwhere(gt.reshape(dims))
        ctr = ((idx + 0.5) * float(vs) + np.asarray(origin)).mean(0) if len(idx) else None
        scenes = [[_mesh(masks[k] & gt, c, g) for k, c in C_REG.items()]]
        for k in ("map_only", "clip_reset", "streaming", "occany_1f"):
            pr = preds[k]
            scenes.append([_mesh(~pr & gt, C_MISS, g), _mesh(pr & gt, C_TP, g),
                           _mesh(pr & ~gt & valid, C_FP, g)])
        A_, B_ = preds["clip_reset"], preds["streaming"]
        scenes.append([_mesh(gt & A_ & B_, C_BOTH, g, 0.55), _mesh(gt & B_ & ~A_, C_OURS, g),
                       _mesh(gt & A_ & ~B_, C_OCC, g)])
        imgs = [R3.render([m for m in ms if m is not None] + [fr], size=(980, 700),
                          look_at=ctr, **VIEW) for ms in scenes]
        st = {k: iou(v, gt, valid) for k, v in preds.items()}
        rows.append({"images": imgs,
                     "label": f"{sid}  q={q:.2f}  t={seg.times_s[p]:.0f}s  "
                              f"clip-reset {st['clip_reset']:.1f} vs streaming "
                              f"{st['streaming']:.1f} (Δ {st['streaming']-st['clip_reset']:+.1f})"})
        record.append({"quantile": q, "sample_id": sid, "post": a.post,
                       "stream_position": int(p), "time_s": float(seg.times_s[p]),
                       "sc_iou": st, "frames_cast": cast["n_frames_cast"],
                       "frames_in_prefix": cast["n_frames_in_prefix"],
                       "region_gt_occupied": {k: int((masks[k] & gt).sum()) for k in C_REG}})
        print(f"  rendered {sid} (q={q:.2f})", flush=True)

    out = os.path.join(OUT_DIR, f"qualitative_panel_{a.post}.png")
    compose(rows, out, "Clip-reset and persistent-state streaming on identical anchors, against "
                       f"the visibility history of the processed stream (SemanticKITTI 08, {a.post})",
            HANDLES, titles=TITLES, dpi=100)
    json.dump({"post": a.post, "quantiles": list(QUANTILES), "rows": record,
               "selection_rule": ("fixed quantiles of the recorded per-anchor ranking in "
                                  "panel_ranking.json; low quantiles are streaming losses")},
              open(os.path.join(OUT_DIR, f"qualitative_panel_{a.post}.json"), "w"),
              indent=1, default=float)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
