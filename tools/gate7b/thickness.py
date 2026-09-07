#!/usr/bin/env python
"""Map thickness and duplicate-surface rate, per metric-gauge policy.

A gauge that drifts leaves the same wall in two places. Two fixed, gauge-blind statistics
detect that on the evaluation grid:

* **thickness** -- occupied voxels per non-empty vertical column. A single clean surface
  gives a thin column; a duplicated one gives a thick column;
* **duplicate-surface rate** -- the fraction of non-empty columns containing two or more
  occupied runs separated by at least two empty voxels. One surface seen twice at
  different scales is exactly that.

Both are computed on the same anchors for all three policies, so the comparison is paired.

    python tools/gate7b/thickness.py --dataset semantickitti --device cuda:0
"""
from __future__ import annotations

import argparse, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT, write_json                              # noqa: E402
from gates.gate6 import frames as G6F, grids as G6G                                    # noqa: E402
from gates.gate6 import vocab                                                          # noqa: E402
from gates.gate7b import config as C, depth as D7, fuse as FZ, scale as SC, streams as ST  # noqa: E402
from gates.gate7b.voxmap import EvidenceVolume                                         # noqa: E402
from tools.gate7b.stream_lingbot import seg_path                                 # noqa: E402
from tools.gate7b.scale_candidates import OUT_ROOT as SCALE_ROOT                 # noqa: E402
from tools.gate7b.run_stream_eval import anchor_transform                        # noqa: E402

ART = os.path.join(REPO_ROOT, "artifacts", "gate7b")


def column_stats(occ: torch.Tensor, dims) -> dict:
    """Thickness and duplicate-surface rate over vertical columns."""
    o = occ.reshape(int(dims[0]), int(dims[1]), int(dims[2]))
    cols = o.any(dim=2)
    n_cols = int(cols.sum())
    if n_cols == 0:
        return {"n_columns": 0, "thickness": 0.0, "duplicate_rate": 0.0,
                "n_occupied": 0}
    n_occ = int(o.sum())
    oi = o.to(torch.int8)
    # a run starts where an occupied voxel follows a non-occupied one
    prev = torch.cat([torch.zeros_like(oi[:, :, :1]), oi[:, :, :-1]], dim=2)
    starts = ((oi == 1) & (prev == 0)).sum(dim=2)
    # gaps of at least two empty voxels between runs
    gap2 = torch.zeros_like(oi, dtype=torch.bool)
    gap2[:, :, 2:] = (oi[:, :, 2:] == 1) & (oi[:, :, 1:-1] == 0) & (oi[:, :, :-2] == 0)
    sep = (gap2.sum(dim=2) >= 1) & (starts >= 2)
    return {"n_columns": n_cols, "n_occupied": n_occ,
            "thickness": n_occ / n_cols,
            "duplicate_rate": float(sep.sum()) / n_cols}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=list(C.DATASETS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--stride", type=int, default=10)
    a = ap.parse_args()
    ds = a.dataset
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    v = vocab.load(ds)
    MAP = G6G.PREDICTION_GRID[ds]
    recs = {r.clip_id: r.raw for r in G6F.read_manifest(ds, REPO_ROOT)}
    segs = ST.build(ds, REPO_ROOT)
    calib = {}
    out = {p: {"thickness": [], "duplicate_rate": [], "n_occupied": []}
           for p in C.SCALE_POLICIES}
    vol = None
    t0, n = time.time(), 0
    for seg in segs:
        with np.load(seg_path(ds, seg.name)) as z:
            stream = {k: z[k] for k in ("pred_depth", "pred_depth_conf",
                                        "pred_pose_c2w", "pred_K")}
            keys = [str(k) for k in z["keys"]]
        cand = np.load(os.path.join(SCALE_ROOT, ds, f"{seg.name}.npz"))["log_s"]
        series = {}
        for pol in C.SCALE_POLICIES:
            st = SC.ScaleState(policy=pol)
            series[pol] = np.array([st.observe(i, {"log_s": float(cand[i])})
                                    for i in range(len(keys))])
        poses = stream["pred_pose_c2w"].astype(np.float64)
        anchors = sorted(seg.anchors.items(), key=lambda kv: kv[1])[::a.stride]
        for cid, t in anchors:
            rec = recs.get(cid)
            if rec is None:
                continue
            T_ag = anchor_transform(ds, rec, REPO_ROOT, calib)
            for pol in C.SCALE_POLICIES:
                s_of = series[pol]
                reach = FZ.reach_mask(poses, t, float(s_of[t]), T_ag, MAP, D7.MAX_DEPTH_M)
                frames = [f for f in range(0, t + 1) if reach[f]]
                if vol is None:
                    vol = EvidenceVolume(tuple(MAP.dims), dev, len(v))
                else:
                    vol.reset()
                # G-C is the only policy whose gauge differs per frame; it is applied
                # per frame precisely so its duplication is visible rather than hidden
                sfun = ((lambda f: s_of[f]) if pol == "G-C" else (lambda _f: s_of[t]))
                FZ.fuse_window(vol, stream, frames, t, sfun, T_ag, MAP, "S1", 1.5, dev)
                cs = column_stats(vol.occupied(), MAP.dims)
                out[pol]["thickness"].append(cs["thickness"])
                out[pol]["duplicate_rate"].append(cs["duplicate_rate"])
                out[pol]["n_occupied"].append(cs["n_occupied"])
            n += 1
            if n % 25 == 0:
                print(f"  [{ds}] {n} anchors {(time.time()-t0)/n:.2f}s", flush=True)
    res = {"dataset": ds, "n_anchors": n, "anchor_stride": a.stride,
           "seconds": time.time() - t0,
           "definition": {"thickness": "occupied voxels per non-empty vertical column",
                          "duplicate_rate": ("fraction of non-empty columns with two or "
                                             "more occupied runs separated by >= 2 empty "
                                             "voxels")},
           "policies": {p: {"thickness_mean": float(np.mean(out[p]["thickness"])),
                            "thickness_median": float(np.median(out[p]["thickness"])),
                            "duplicate_rate_mean": float(np.mean(out[p]["duplicate_rate"])),
                            "occupied_mean": float(np.mean(out[p]["n_occupied"]))}
                        for p in C.SCALE_POLICIES}}
    write_json(os.path.join(ART, f"thickness_{ds}.json"), res)
    for p in C.SCALE_POLICIES:
        d = res["policies"][p]
        print(f"[{ds} {p}] thickness {d['thickness_mean']:.3f}  "
              f"duplicate rate {d['duplicate_rate_mean']:.4f}  "
              f"occupied {d['occupied_mean']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
