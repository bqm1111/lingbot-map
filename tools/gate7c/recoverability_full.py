#!/usr/bin/env python
"""Gate 7B Phase 6 — does a later viewpoint reconstruct what the five-frame map missed?

Gate 7A could not separate genuine occlusion from depth failure. This can: for every
B-D coverage-miss voxel at an official timestamp ``t`` it asks whether the *same frozen
model*, streamed, ever places geometry there -- from the causal past, from a later
viewpoint, or never.

**This is diagnostic only.** Future frames are fused into a separate volume that is
written to a separate artifact and can never reach a causal prediction; the causal map of
Gate 7B's own evaluation is produced by ``run_stream_eval.py`` and never sees this code.

    python tools/gate7b/recoverability.py --dataset semantickitti --device cuda:0
"""
from __future__ import annotations

import argparse, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT, write_json                              # noqa: E402
from gates.gate6 import frames as G6F, grids as G6G, metrics as G6M, targets as G6T, \
    vocab                                                                        # noqa: E402
from gates.gate7a import frustum as FR7A                                               # noqa: E402
from gates.gate7b import config as C, depth as D7, fuse as FZ, scale as SC, \
    streams as ST                                                                # noqa: E402
from gates.gate7b.voxmap import EvidenceVolume                                         # noqa: E402
from tools.gate7b.stream_lingbot import seg_path                                 # noqa: E402
from tools.gate7b.scale_candidates import OUT_ROOT as SCALE_ROOT                 # noqa: E402
from tools.gate7b.run_stream_eval import anchor_transform                        # noqa: E402

ART = os.path.join(REPO_ROOT, "artifacts", "gate7c")
HORIZONS = (1, 5, 10, 20, 50)
#: Classes of a B-D coverage miss, in the order they are resolved.
CLASSES = ("recovered_from_causal_past", "recovered_from_a_later_viewpoint",
           "in_frustum_but_never_reconstructed", "outside_all_frusta", "unresolved")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=list(C.DATASETS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--stride", type=int, default=1, help="evaluate every Nth anchor")
    ap.add_argument("--limit-anchors", type=int, default=None)
    a = ap.parse_args()
    ds = a.dataset
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    torch.cuda.init()

    v = vocab.load(ds)
    Cn = len(v)
    MAP = G6G.PREDICTION_GRID[ds]
    EVAL = G6G.EVAL_GRID[ds]
    n_eval = int(np.prod(EVAL.dims))
    dist_bands, _h = G6M.band_masks(EVAL, dist_bands=C.RANGE_BANDS, height_bands=())
    man = json.load(open(os.path.join(REPO_ROOT, "artifacts", "gate6",
                                      f"prediction_manifest_{ds}.json")))
    proot = man["prediction_root"]
    recs = {r.clip_id: r.raw for r in G6F.read_manifest(ds, REPO_ROOT)}
    segs = ST.build(ds, REPO_ROOT)
    calib = {}
    nb = len(C.RANGE_BANDS)
    # [class, 1 + range bands]
    counts = np.zeros((len(CLASSES), 1 + nb), np.int64)
    latency = np.zeros(len(HORIZONS) + 1, np.int64)          # per horizon, + "any later"
    per_class_sem = np.zeros((Cn, len(CLASSES)), np.int64)
    n_miss_total = 0
    vol_c = vol_f = None
    t0, n_anchor = time.time(), 0

    for seg in segs:
        with np.load(seg_path(ds, seg.name)) as z:
            stream = {k: z[k] for k in ("pred_depth", "pred_depth_conf",
                                        "pred_pose_c2w", "pred_K")}
            keys = [str(k) for k in z["keys"]]
        cand = np.load(os.path.join(SCALE_ROOT, ds, f"{seg.name}.npz"))["log_s"]
        st = SC.ScaleState(policy="G-A")
        s_of = np.array([st.observe(i, {"log_s": float(cand[i])})
                         for i in range(len(keys))])
        poses = stream["pred_pose_c2w"].astype(np.float64)
        anchors = sorted(seg.anchors.items(), key=lambda kv: kv[1])[::a.stride]
        for cid, t in anchors:
            rec = recs.get(cid)
            name = cid + ".npz"
            if rec is None or name not in man["per_file_sha256"]:
                continue
            target, keep = G6T.semantic_target(ds, rec, REPO_ROOT)
            if target is None:
                continue
            with np.load(os.path.join(proot, name)) as z:
                bd = z["dil_flat"].astype(np.int64)
            occ_s0 = np.zeros(n_eval, bool)
            occ_s0[bd] = True
            k = keep.reshape(-1)
            tgt = target.reshape(-1)
            miss = k & (tgt != v.empty_label) & ~occ_s0
            if not miss.any():
                continue

            T_ag = anchor_transform(ds, rec, REPO_ROOT, calib)
            reach = FZ.reach_mask(poses, t, float(s_of[t]), T_ag, MAP, D7.MAX_DEPTH_M)
            past = [f for f in range(0, t + 1) if reach[f]]
            future = [f for f in range(t + 1, len(keys)) if reach[f]]

            if vol_c is None:
                vol_c = EvidenceVolume(tuple(MAP.dims), dev, Cn)
                vol_f = EvidenceVolume(tuple(MAP.dims), dev, Cn)
            else:
                vol_c.reset(); vol_f.reset()
            FZ.fuse_window(vol_c, stream, past, t, lambda _f: s_of[t], T_ag, MAP,
                           "S1", 1.5, dev)
            FZ.fuse_window(vol_f, stream, future, t, lambda _f: s_of[t], T_ag, MAP,
                           "S1", 1.5, dev)

            def to_eval(vol):
                o = vol.occupied()
                ft = vol.first_time
                if not G6G.NEEDS_REDUCTION[ds]:
                    return (o.cpu().numpy(), ft.cpu().numpy())
                fl = o.nonzero(as_tuple=True)[0].cpu().numpy().astype(np.int64)
                oe = np.zeros(n_eval, bool)
                fe = np.full(n_eval, -1, np.int32)
                if len(fl):
                    idx = G6G.unflat(fl, MAP) // G6G.RATIO
                    nat = np.ravel_multi_index((idx[:, 0], idx[:, 1], idx[:, 2]),
                                               EVAL.dims)
                    oe[nat] = True
                    ftv = ft[o].cpu().numpy()
                    order = np.argsort(ftv, kind="stable")
                    fe[nat[order]] = ftv[order]
                return oe, fe

            occ_past, _ = to_eval(vol_c)
            occ_fut, ft_fut = to_eval(vol_f)

            centres = FR7A.voxel_centres(np.flatnonzero(miss), EVAL)
            fa = FR7A.analyse(centres, T_ag, poses, stream["pred_K"].astype(np.float64),
                              stream["pred_depth"].astype(np.float32),
                              stream["pred_depth_conf"].astype(np.float32),
                              float(s_of[t]), D7.CONF_THRESHOLD, D7.MIN_DEPTH_M,
                              D7.MAX_DEPTH_M, float(np.sqrt(3) * EVAL.voxel_size))
            mi = np.flatnonzero(miss)
            cls = np.full(len(mi), CLASSES.index("unresolved"), np.int64)
            in_any = fa["in_any"]
            cls[in_any] = CLASSES.index("in_frustum_but_never_reconstructed")
            cls[~in_any] = CLASSES.index("outside_all_frusta")
            rec_f = occ_fut[mi]
            cls[rec_f] = CLASSES.index("recovered_from_a_later_viewpoint")
            rec_p = occ_past[mi]
            cls[rec_p] = CLASSES.index("recovered_from_causal_past")

            band_of = np.zeros(len(mi), np.int64)
            for bi, (_n, m) in enumerate(dist_bands):
                band_of[m.reshape(-1)[mi]] = bi + 1
            np.add.at(counts, (cls, 0), 1)
            np.add.at(counts, (cls, band_of), 1)
            n_miss_total += len(mi)

            lat = ft_fut[mi] - t
            for hi, h in enumerate(HORIZONS):
                latency[hi] += int(((cls == 1) & (lat > 0) & (lat <= h)).sum())
            latency[-1] += int((cls == 1).sum())

            ti = tgt[mi]
            chan = np.full(len(mi), -1, np.int64)
            lab = np.asarray(v.labels, np.int32)
            lut = np.full(int(max(lab.max(), v.empty_label)) + 2, -1, np.int64)
            lut[lab] = np.arange(Cn)
            chan = lut[np.clip(ti, 0, len(lut) - 1)]
            ok = chan >= 0
            np.add.at(per_class_sem, (chan[ok], cls[ok]), 1)

            n_anchor += 1
            if a.limit_anchors and n_anchor >= a.limit_anchors:
                break
            if n_anchor % 100 == 0:
                print(f"  [{ds}] {n_anchor} anchors {(time.time()-t0)/n_anchor:.2f}s",
                      flush=True)
        if a.limit_anchors and n_anchor >= a.limit_anchors:
            break

    band_names = ["all"] + [f"{int(lo)}-{int(hi)}m" for lo, hi in C.RANGE_BANDS]
    out = {"dataset": ds, "n_anchors": n_anchor, "anchor_stride": a.stride,
           "n_b_d_coverage_misses": n_miss_total,
           "classes": list(CLASSES),
           "counts": {cn: {bn: int(counts[ci, bi]) for bi, bn in enumerate(band_names)}
                      for ci, cn in enumerate(CLASSES)},
           "fractions": {cn: (float(counts[ci, 0]) / max(n_miss_total, 1))
                         for ci, cn in enumerate(CLASSES)},
           "recovery_latency": {**{f"within_{h}_frames": int(latency[i])
                                   for i, h in enumerate(HORIZONS)},
                                "any_later_frame": int(latency[-1])},
           "recovery_latency_fraction": {
               **{f"within_{h}_frames": float(latency[i]) / max(n_miss_total, 1)
                  for i, h in enumerate(HORIZONS)},
               "any_later_frame": float(latency[-1]) / max(n_miss_total, 1)},
           "per_class": {v.names[i]: {cn: int(per_class_sem[i, ci])
                                      for ci, cn in enumerate(CLASSES)}
                         for i in range(Cn)},
           "seconds": time.time() - t0,
           "diagnostic_only": ("future frames are fused in a separate volume and never "
                               "reach a causal prediction"),
           "semantic_class_use": "analysis only; no class drives any decision"}
    write_json(os.path.join(ART, f"recoverability_full_{ds}.json"), out)
    print(f"[{ds}] {n_anchor} anchors, {n_miss_total:,} B-D misses  "
          + "  ".join(f"{cn.split('_')[0]}={out['fractions'][cn]:.3f}"
                      for cn in CLASSES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
