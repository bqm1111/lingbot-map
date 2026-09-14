#!/usr/bin/env python
"""Why is per-voxel multi-view evidence so rare? For every (voxel, frame) evidence pair,
find the nearest voxel carrying evidence from a *different* future frame and record the
distance and whether the two single-frame top-1 classes agree. If other views of the same
surface land a voxel or two away rather than in the same voxel, the single-view structure
is endpoint scatter of the frozen depth at 0.2 m, not absence of views. Diagnostic only;
no target is defined from it.

    python tools/future_semantic_completion/target_consistency/neighbourhood.py
"""
from __future__ import annotations
import argparse, glob, os, sys, time
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.future_semantic_completion.target_consistency import common as TC             # noqa: E402
from tools.future_semantic_completion import study as ST                                  # noqa: E402
from core.datasets import kitti360_splits as SRC, union_vocab as V8                       # noqa: E402
from core.datasets.audit import FileAudit                                                 # noqa: E402

EDGES = np.asarray([0, 0.5, 1.5, 2.5, 3.5, 5.5, 10.5, 1e9])
NAMES = ["same voxel", "1", "2", "3", "4-5", "6-10", "none within 10 (of the 32 nearest)"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stride", type=int, default=2)
    a = ap.parse_args()
    ST.require_data_root()
    out = {"edges_vox": EDGES.tolist(), "names": NAMES, "drives": {}}
    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        for d in list(SRC.TRAIN_DRIVES) + [SRC.VAL_DRIVE]:
            files = sorted(glob.glob(os.path.join(TC.EVIDENCE_DIR, d, "*.npz")))[::a.stride]
            hist = np.zeros((len(NAMES), 2)); n_pairs = 0; n_single = 0; t0 = time.time()
            for f in files:
                with np.load(f) as z:
                    dims = tuple(int(x) for x in z["dims"]); elig = z["elig"].astype(np.int64)
                    row = z["pair_row"].astype(np.int64); fr = z["pair_frame"].astype(np.int64)
                    w = z["pair_w"].astype(np.float32); s = z["pair_s"].astype(np.float32)
                if len(row) < 2:
                    continue
                xyz = np.stack(np.unravel_index(elig[row], dims), 1).astype(np.float32)
                top = (s / w[:, None]).argmax(1)
                nf = np.bincount(row, minlength=len(elig))
                n_pairs += len(row); n_single += int((nf[row] == 1).sum())
                tree = cKDTree(xyz)
                # k nearest, then take the first with a different frame
                k = min(32, len(row))
                dist, idx = tree.query(xyz, k=k)
                other = fr[idx] != fr[:, None]
                first = np.argmax(other, axis=1)
                has = other[np.arange(len(row)), first]
                dd = np.where(has, dist[np.arange(len(row)), first], np.inf)
                same_voxel = (dd == 0)
                ok = top[idx[np.arange(len(row)), first]] == top
                b = np.clip(np.searchsorted(EDGES, dd, side="right") - 1, 0, len(NAMES) - 1)
                b = np.where(has, b, len(NAMES) - 1)
                np.add.at(hist, b, np.stack([np.ones(len(row)), ok & has & (b < len(NAMES) - 1)], 1))
            out["drives"][d] = {"n_evidence_pairs": int(n_pairs), "single_frame_fraction": n_single / max(n_pairs, 1),
                                "nearest_other_frame": {NAMES[k]: {"fraction": float(hist[k, 0] / max(hist[:, 0].sum(), 1)),
                                                                   "top1_agreement": (None if k == len(NAMES) - 1 else float(hist[k, 1] / max(hist[k, 0], 1)))}
                                                        for k in range(len(NAMES))},
                                "seconds": time.time() - t0}
            r = out["drives"][d]["nearest_other_frame"]
            print(f"  {d[17:21]}: pairs {n_pairs:,} single-frame {out['drives'][d]['single_frame_fraction']:.3f} | "
                  + " ".join(f"{k}:{v['fraction']:.3f}/{v['top1_agreement'] if v['top1_agreement'] is None else round(v['top1_agreement'], 2)}" for k, v in r.items()), flush=True)
    out["stride"] = a.stride; out["firewall"] = audit.summary()
    ST.write_json(os.path.join(TC.ART, "neighbourhood_support.json"), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
