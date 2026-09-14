#!/usr/bin/env python
"""Evaluation regions for 0006 (590 anchors), from raw LiDAR + GT poses only (evaluation side):

  past5            voxels measured (return or traversal) by the five causal sweeps t-4..t
  unobs5_later     target valid & not past5 & resolved by the future sweeps t+1..t+20 (return-only xor traversal-only)
  observed20 / unobserved_later20   the frozen data-scaling evaluation masks (20-sweep past window), copied

No prediction, no input and no LingBot state is read.

    python tools/gate12_suf/build_regions.py --device cuda:0 [--shard i --shards n]
"""
from __future__ import annotations
import argparse, glob, os, sys, time
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g12                                                                                  # noqa: E402
sys.path.insert(0, g12.REPO_ROOT)
from core.datasets import kitti360_transforms as TF, kitti360_splits as SRC, sources as S   # noqa: E402
from core.supervision import rawtarget as RT                                                # noqa: E402
from core.run.build_targets import load as load_target                                      # noqa: E402
from core.run._common import REPO_ROOT                                                      # noqa: E402

OUT = os.path.join(g12.DATA_ROOT, "regions", g12.VAL_DRIVE)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1); a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev); d = g12.VAL_DRIVE
    geo = TF.DriveGeometry(d, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
    seg = [s for s in S.segments(SRC.source_of(d), REPO_ROOT) if s.name == d][0]; nat = [int(f.order) for f in seg.frames]
    anc = {x.stream_index: x for x in SRC.anchors(d, REPO_ROOT)}
    ids = sorted(int(os.path.basename(f)[:-4]) for f in glob.glob(os.path.join(g12.EVAL_MASK_ROOT, d, "*.npz")))[a.shard::a.shards]
    os.makedirs(OUT, exist_ok=True); t0 = time.time(); nw = 0
    for t in ids:
        op = os.path.join(OUT, f"{t:05d}.npz")
        if os.path.exists(op):
            continue
        tgt = load_target(g12.target_path(d, t)); valid, occ = tgt["valid"], tgt["occupied"]
        em = np.load(os.path.join(g12.EVAL_MASK_ROOT, d, f"{t:05d}.npz")); n = valid.size
        rp = RT.build(geo, anc[t].native_frame, [nat[j] for j in range(t - 4, t + 1)], dev)
        rf = RT.build(geo, anc[t].native_frame, list(anc[t].future_natives), dev)
        past5 = np.asarray(rp.n_obs > 0) | np.asarray(rp.n_free > 0)
        mo, mf = np.asarray(rf.n_obs > 0), np.asarray(rf.n_free > 0)
        later5 = valid & ~past5 & (mo ^ mf)
        tmp = op + ".tmp.npz"
        np.savez_compressed(tmp, valid=np.packbits(valid), occupied=np.packbits(occ), past5=np.packbits(valid & past5), unobs5_later=np.packbits(later5),
                            observed20=em["observed"], unobserved_later20=em["unobserved_later"], t=np.int64(t))
        os.replace(tmp, op); nw += 1
        if nw % 50 == 0:
            print(f"  [regions s{a.shard}] {nw}/{len(ids)} {(time.time()-t0)/nw:.1f}s each", flush=True)
    print(f"[regions s{a.shard}] {nw} written, {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
