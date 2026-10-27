#!/usr/bin/env python
"""Raw-LiDAR occupancy supervision for ONCE anchors, under the frozen Gate 8C-1 rule.

ONCE has a **single** roof LiDAR with a published sensor origin, so
``core/supervision/rawtarget.py`` applies directly and the multisensor path Waymo needed is
not used. The rule is untouched: endpoints occupied, ray interiors free from ``CARVE_NEAR_M``
to ``BAND_HALF_M``, nothing beyond an endpoint, same-sweep returns never carved, cross-sweep
conflict -> unknown, sky rule unchanged.

The window is the authorized one: real native sweeps within ~2.1 s of the anchor, five of
them at ONCE's 0.500 s cadence. Nothing is interpolated or duplicated.

The npz contract is the KITTI-360 one, so
``baseline_reconstruction/current/dataset.py::HInputDataset`` reads this arm unchanged.

    PYTHONPATH=$PWD python tools/method_vs_data/once_arm/cli/build_targets.py --device cuda:3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
from core.supervision import rawtarget as RT                                          # noqa: E402
from tools.method_vs_data.once_arm import anchors as AN, config as A, geometry as G    # noqa: E402
from tools.method_vs_data.once_reader import OnceSequence                              # noqa: E402


def save(path: str, t, seq: str, anchor: int, frames: list, sq: OnceSequence) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    occ = t.occupied
    rows = np.flatnonzero(occ).astype(np.int32)
    tmp = path + ".tmp.npz"
    np.savez_compressed(
        tmp, valid_packed=np.packbits(t.valid), occ_packed=np.packbits(occ),
        occ_rows=rows, occ_n_obs=t.n_obs[rows].astype(np.uint16),
        occ_n_frames=t.n_frames_occ[rows].astype(np.uint8),
        dims=np.asarray(A.GRID_DIMS, np.int32),
        stream_index=np.int32(anchor), native_frame=np.int32(anchor),
        future_stream_indices=np.asarray(frames[1:], np.int32),
        future_natives=np.asarray(frames[1:], np.int32),
        frames_used=np.asarray(t.frames_used, np.int32), segment=seq,
        supervision_frame_ids=np.asarray([sq.frame_ids[f] for f in frames]),
        supervision_timestamps_s=np.asarray([sq.t_s[f] for f in frames], np.float64),
        supervision_span_s=np.float64(sq.t_s[frames[-1]] - sq.t_s[frames[0]]),
        n_sweeps=np.int32(len(frames)),
        stats_json=json.dumps(t.stats))
    os.replace(tmp, path)
    return os.path.getsize(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only-with-clips", action="store_true",
                    help="restrict to anchors whose causal clip already exists")
    ap.add_argument("--tag", default="build_targets")
    a = ap.parse_args()

    A.ensure_dirs()
    p = A.Paths()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    man = AN.load_or_build(p)
    rows_all = man["anchors"][a.shard::a.shards]
    if a.only_with_clips:
        rows_all = [r for r in rows_all
                    if os.path.exists(p.clip_path(r["sequence"], int(r["anchor"])))]
    if a.limit:
        rows_all = rows_all[:a.limit]
    print(f"[targets] manifest {man['digest'][:12]}, {len(rows_all)} anchors, "
          f"grid {A.GRID.dims} @ {A.GRID.voxel_size} m origin "
          f"{list(A.GRID.origin)} frame {A.GRID.frame}", flush=True)

    seqs, rows, t0, n = {}, [], time.time(), 0
    for i, r in enumerate(rows_all):
        s, anc = r["sequence"], int(r["anchor"])
        out = p.target_path(s, anc)
        if os.path.exists(out):
            rows.append({"sequence": s, "anchor": anc, "skipped": True})
            continue
        if s not in seqs:
            seqs[s] = OnceSequence(A.ONCE_DATA_ROOT, s)
        sq = seqs[s]
        win = G.supervision_window(sq, anc)
        assert len(win) == G.EXPECTED_SWEEPS, f"{s}/{anc}: window is {len(win)} sweeps"
        try:
            t = RT.build(sq, anc, win, dev, grid=A.GRID)
        except Exception as exc:                                              # noqa: BLE001
            rows.append({"sequence": s, "anchor": anc,
                         "error": f"{type(exc).__name__}: {exc}"})
            continue
        nb = save(out, t, s, anc, win, sq)
        n += 1
        st = dict(t.stats)
        rows.append({"sequence": s, "anchor": anc, "bytes": nb, "n_sweeps": len(win),
                     "span_s": float(sq.t_s[win[-1]] - sq.t_s[win[0]]), **st})
        if n % 100 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(rows_all)} built {n}, {el / n:.2f}s each, "
                  f"eta {(len(rows_all) - i - 1) * el / n / 60:.0f} min", flush=True)

    ok = [r for r in rows if "prevalence_in_valid" in r]
    f_ = lambda k: np.array([r[k] for r in ok], np.float64)
    agg = None
    if ok:
        agg = {k: {"min": float(f_(k).min()), "p50": float(np.median(f_(k))),
                   "max": float(f_(k).max())}
               for k in ("prevalence_in_valid", "valid_fraction",
                         "conflict_fraction_of_touched", "n_occupied", "n_free", "n_unknown")}
    rec = {"provenance": A.provenance(), "manifest_digest": man["digest"],
           "shard": a.shard, "shards": a.shards, "n_built": len(ok),
           "n_skipped": sum(1 for r in rows if r.get("skipped")),
           "n_failed": sum(1 for r in rows if "error" in r),
           "minutes": (time.time() - t0) / 60, "aggregate": agg,
           "band_reminder": A.PREVELANCE_BAND_STATUS,
           "errors": [r for r in rows if "error" in r][:10]}
    os.makedirs(p.logs, exist_ok=True)
    fp = os.path.join(p.logs, f"{a.tag}_shard{a.shard}.json")
    json.dump(rec, open(fp, "w"), indent=2, default=float)
    if agg:
        print(f"\n[targets] {rec['n_built']} built; prevalence p50 "
              f"{agg['prevalence_in_valid']['p50']:.4f} valid p50 "
              f"{agg['valid_fraction']['p50']:.4f} conflict p50 "
              f"{agg['conflict_fraction_of_touched']['p50']:.4f} -> {fp}")
    return 0 if rec["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
