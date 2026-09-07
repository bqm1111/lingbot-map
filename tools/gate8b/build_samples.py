#!/usr/bin/env python
"""Cached (causal input, privileged target) samples for Gate 8B: the new training source
``k360_train`` and the KITTI-360 source-validation drive (``kitti360``, drive 0006).

Identical construction to ``tools/gate8/build_samples.py`` -- one incremental mapper per
segment, stepped frame by frame; at each labelled anchor the causal state through ``t`` is
exported and its targets built from frames ``t+1 .. t+20`` in a separate volume with the
same frozen scale. Written under the Gate 8B sample root so nothing of Gate 8 is touched.

    python tools/gate8b/build_samples.py --source k360_train --shard 0 --shards 3
    python tools/gate8b/build_samples.py --source kitti360 --anchor-stride 10
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, default_device                               # noqa: E402
from gates.scale_gate.config import write_json                                         # noqa: E402
from gates.gate6 import grids as G6G                                                   # noqa: E402
from gates.gate8 import sources as S, vocab as V8, targets as TG                       # noqa: E402
from gates.gate8.feed import CachedFeed                                                # noqa: E402
from gates.gate8.mapper import IncrementalMapper                                       # noqa: E402
from gates.gate8b.sources import G8B_ROOT                                              # noqa: E402

SAMPLE_ROOT = f"{G8B_ROOT}/samples"


def sample_path(source, seg, t):
    return f"{SAMPLE_ROOT}/{source}/{seg}_{t:05d}.npz"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, choices=["k360_train", "kitti360"])
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--anchor-stride", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    ds = S.DATASET_OF[a.source]
    MAP = G6G.PREDICTION_GRID[ds]
    into = V8.into_matrix(ds)
    os.makedirs(f"{SAMPLE_ROOT}/{a.source}", exist_ok=True)
    segs = S.segments(a.source, REPO_ROOT)[a.shard::a.shards]
    audit, n, t0 = [], 0, time.time()
    for seg in segs:
        if not os.path.exists(S.stream_path(a.source, seg.name)):
            print(f"  skip {seg.name}: no stream yet"); continue
        assert seg.anchors, f"{seg.name}: no labelled anchors (targets not fetched?)"
        feed = CachedFeed(seg, dev)
        m = IncrementalMapper(dev, sem_into=into)
        anchors = set(seg.anchors[::a.anchor_stride])
        for i in range(len(seg)):
            m.step(feed.frame(i))
            if i not in anchors or not m.scale_state.frozen:
                continue
            f = seg.frames[i]
            out_p = sample_path(a.source, seg.name, i)
            if os.path.exists(out_p):
                n += 1; continue
            P = feed.pose[i].copy(); P[:3, 3] *= m.scale_state.scale
            Tgw = P @ np.linalg.inv(np.asarray(f.T_cam_to_grid, np.float64))
            smp = TG.build_sample(feed, m, i, MAP, Tgw, ds, f.gt_ref, REPO_ROOT, into, dev)
            if smp is None:
                continue
            tmp = out_p + ".tmp.npz"
            np.savez_compressed(tmp, **smp)
            os.replace(tmp, out_p)
            audit.append({"segment": seg.name, "t": i, "clip": f.gt_ref["clip_id"],
                          "input_frames": [0, i], "target_frames":
                          [int(smp["target_frames"][0]), int(smp["target_frames"][-1])],
                          "n_rows": int(len(smp["rows"])), "n_fut_rows": int(len(smp["fut_rows"])),
                          "bytes": os.path.getsize(out_p)})
            n += 1
            if a.limit and n >= a.limit:
                break
            if n % 25 == 0:
                torch.cuda.empty_cache()
            if n % 100 == 0:
                print(f"  [{a.source} s{a.shard}] {n} samples {(time.time()-t0)/n:.2f}s each", flush=True)
        if feed.n_missing_sem:
            print(f"  WARNING {seg.name}: {feed.n_missing_sem} frames had no Trident cache")
        if a.limit and n >= a.limit:
            break
    os.makedirs(ART, exist_ok=True)
    write_json(os.path.join(ART, f"samples_{a.source}_s{a.shard}.json"),
               {"source": a.source, "n_samples": n, "seconds": time.time() - t0,
                "anchor_stride": a.anchor_stride, "future_frames": TG.FUTURE_FRAMES,
                "audit": audit[:50], "total_bytes": int(sum(x["bytes"] for x in audit))})
    print(f"[{a.source} s{a.shard}] {n} samples in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
