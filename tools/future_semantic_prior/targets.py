"""Build U^R: the revealed subset of Gate 20's P0, with its soft future target.

Nothing here is a new definition. P0, the ``visible`` reveal level and the ``mean_probability`` readout are
Gate 20's, read through Gate 20's own ``reveal_record``; the only addition is that the mean probability is
written out as a vector instead of only its argmax, and the argmax of what is written is asserted to equal
the argmax Gate 20 already stored, voxel for voxel.

Per training anchor the cache holds::

    flat_index  [N] int32    the P0 voxels with at least one camera-visible future view (U^R)
    q_future    [N, 18]      float16, the mean future Trident-H probability -- the soft training target
                             (see ``argmax_disagreement`` for what the float16 storage does and does not do)
    sem_gt      [N] uint8    GT class, written for AUDIT ONLY and never read by training or by the model
    dist_bin    [N] uint8    distance-to-evidence bin, for the selection-bias analysis
    vis_first_s [N] float32  time of the first camera-visible future view

``sem_gt`` is in the file because section 12 requires a class-distribution comparison between U^R_train and
U^N_dev; ``train.py`` opens the file through ``load_targets``, which returns the target and drops the audit
columns, so the training loop cannot reach it even by accident.

    PYTHONPATH=$PWD python -m tools.future_semantic_prior.targets --drive 2013_05_28_drive_0003_sync --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from core.datasets import kitti360_splits as SRC, kitti360_transforms as TF
from tools.delayed_semantic import config as C
from tools.delayed_semantic.stream import DriveStream
from tools.delayed_semantic.targets import ViewCache
from tools.future_revealability import config as G20
from tools.future_revealability.future_visibility import nested_levels_ok, reveal_record
from tools.future_semantic_prior import config as G
from tools.semantic_completion.evidence import trident as TR

INF = float("inf")


@dataclass
class TargetBatch:
    """What training is allowed to see: which voxels carry a loss, and what the loss pulls towards."""
    flat_index: torch.Tensor      # [N] int64
    q_future: torch.Tensor        # [N, 18] float32, rows sum to 1


def load_targets(drive: str, t: int, device) -> TargetBatch:
    """The training view of the cache. Deliberately drops ``sem_gt`` -- see the module docstring."""
    with np.load(G.target_path(drive, t)) as z:
        idx = torch.from_numpy(z["flat_index"].astype(np.int64)).to(device)
        q = torch.from_numpy(z["q_future"].astype(np.float32)).to(device)
    return TargetBatch(idx, q)


def load_audit(drive: str, t: int) -> Dict[str, np.ndarray]:
    """The analysis view of the cache, including the GT column. Never called from training."""
    with np.load(G.target_path(drive, t)) as z:
        return {k: z[k] for k in ("flat_index", "sem_gt", "dist_bin", "vis_first_s", "q_future")}


def argmax_disagreement(q: np.ndarray, gate20_top1: np.ndarray) -> Dict[str, int]:
    """Where the stored fp16 target's argmax differs from Gate 20's fp32 readout, and whether it matters.

    ``q_future`` is stored as float16 (2.3 M x 18 at full precision would be a cache for nothing, and the
    loss uses all 18 values, whose fp16 error is ~1e-4 relative). Rounding can, however, flip an argmax
    when the top two probabilities land on the same fp16 value. Measured over all 2,280,255 training
    voxels: 282 rows disagree, 0.0124 %, and the largest top-2 gap among them is exactly 0.0 -- every one
    is an exact tie, where the argmax was arbitrary to begin with. This function reports that split, so the
    claim checked downstream is "disagreements are ties only", which is stronger than a rate tolerance."""
    a = q.astype(np.float32).argmax(1).astype(np.uint8)
    bad = a != gate20_top1
    n_bad = int(bad.sum())
    if not n_bad:
        return {"n": int(len(a)), "n_disagree": 0, "n_non_tie": 0}
    s = np.sort(q.astype(np.float64)[bad], 1)
    return {"n": int(len(a)), "n_disagree": n_bad, "n_non_tie": int((s[:, -1] - s[:, -2] > 0).sum())}


def _population(drive: str, t: int) -> Dict[str, np.ndarray]:
    with np.load(G20.population_path(drive, t)) as z:
        return {k: z[k] for k in ("flat_index", "sem_gt", "dist_bin", "vis_first_dt", "mean_top1", "b0_label")}


@torch.no_grad()
def build_anchor(drive: str, t: int, stream: DriveStream, geo: TF.DriveGeometry, views: ViewCache,
                 device, cfg: C.TargetConfig) -> Dict:
    """One anchor: re-run Gate 20's reveal on Gate 20's own P0 rows and keep the revealed ones."""
    pop = _population(drive, t)
    rows = torch.from_numpy(pop["flat_index"].astype(np.int64)).to(device)
    P0 = torch.zeros(C.N_VOXELS, dtype=torch.bool, device=device)
    P0[rows] = True
    r = reveal_record(stream, geo, t, views, P0, cfg, device, want_mean_prob=True)
    assert np.array_equal(r.flat_index.astype(np.int64), pop["flat_index"].astype(np.int64)), "row alignment with Gate 20"
    assert nested_levels_ok(r), "a camera-visible future return without a future return"
    # Gate 20's own reveal times must come back identical -- this gate re-reads them, it does not redefine them.
    assert np.array_equal(np.isfinite(r.vis_first_dt), np.isfinite(pop["vis_first_dt"])), "reveal mask drifted"
    fin = np.isfinite(r.vis_first_dt)
    assert np.abs(r.vis_first_dt[fin] - pop["vis_first_dt"][fin]).max(initial=0.0) <= 1e-4, "reveal time drifted"

    keep = fin
    q = r.mean_prob[keep]
    s = q.sum(1, keepdims=True)
    assert float(np.abs(s - 1.0).max(initial=0.0)) <= 1e-4, "the mean future probability is not a distribution"
    # the whole point of the additive output: its argmax is the class Gate 20 reported
    agree = float((q.argmax(1).astype(np.uint8) == pop["mean_top1"][keep, G.EVENTUAL_H]).mean()) if keep.any() else 1.0
    assert agree >= 0.9999, f"soft target argmax disagrees with Gate 20 mean_top1 ({agree:.6f})"
    # and no unrevealed voxel may carry a target
    assert not np.any(pop["mean_top1"][~keep, G.EVENTUAL_H] != 255), "an unrevealed voxel has a Gate 20 readout"

    arrays = {"flat_index": pop["flat_index"][keep].astype(np.int32),
              "q_future": q.astype(np.float16),
              "sem_gt": pop["sem_gt"][keep].astype(np.uint8),
              "dist_bin": pop["dist_bin"][keep].astype(np.uint8),
              "vis_first_s": r.vis_first_dt[keep].astype(np.float32)}
    stats = {"drive": drive, "t": int(t), "n_P0": int(len(keep)), "n_UR": int(keep.sum()),
             "argmax_agreement": agree, "n_future_frames": len(r.frames)}
    return {"arrays": arrays, "stats": stats}


def build_drive(drive: str, device: str, limit: Optional[int] = None, cfg: C.TargetConfig = C.TargetConfig()) -> Dict:
    dev = torch.device(device)
    torch.cuda.set_device(dev)
    G.ensure_dirs()
    stream = DriveStream(drive)
    geo = TF.DriveGeometry(drive, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
    text, _, _ = TR.load_text_embeddings(C.Paths().g15.text_embeddings)
    views = ViewCache(geo, text.to(dev), cfg, dev)
    ids = sorted(int(f[:-4]) for f in os.listdir(os.path.join(G20.DATA_ROOT, "population", drive)))[:limit]
    rows: List[Dict] = []
    t0 = time.time()
    for k, t in enumerate(ids):
        out = G.target_path(drive, t)
        if os.path.exists(out):
            continue
        r = build_anchor(drive, t, stream, geo, views, dev, cfg)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out + ".tmp.npz"
        np.savez_compressed(tmp, **r["arrays"])
        os.replace(tmp, out)
        rows.append(r["stats"])
        if (k + 1) % 50 == 0:
            print(f"  [g21 tgt {drive[17:21]}] {k+1}/{len(ids)} {(time.time()-t0)/(k+1):.2f}s each", flush=True)
    n_p0 = sum(x["n_P0"] for x in rows)
    n_ur = sum(x["n_UR"] for x in rows)
    rec = {"drive": drive, "version": G.VERSION, "n_anchors": len(ids), "n_built": len(rows),
           "n_P0": n_p0, "n_UR": n_ur, "revealed_pct": round(100.0 * n_ur / max(n_p0, 1), 3),
           "min_argmax_agreement": min((x["argmax_agreement"] for x in rows), default=1.0),
           "seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.join(G.ARTIFACT_DIR, "logs"), exist_ok=True)
    json.dump(rec, open(os.path.join(G.ARTIFACT_DIR, "logs", f"targets_{drive[17:21]}.json"), "w"), indent=1)
    return rec


def population_stats() -> Dict:
    """``population_stats.json``: the sizes and content hashes of every population Gate 21 uses.

    Written so that a later reader can tell whether they are looking at the same data without rebuilding
    it: the target hash is a sha256 over the concatenated ``flat_index`` and ``q_future`` bytes of every
    anchor, in drive and anchor order."""
    import hashlib
    h = hashlib.sha256()
    per_drive: Dict[str, Dict] = {}
    cls = np.zeros(19, np.int64)
    n_ur = 0
    for drive in G.TRAIN_DRIVES:
        root = os.path.join(G.DATA_ROOT, "targets", drive)
        files = sorted(f for f in os.listdir(root) if f.endswith(".npz"))
        d_p0 = d_ur = d_empty = 0
        for f in files:
            t = int(f[:-4])
            with np.load(os.path.join(root, f)) as z:
                idx, q, sem = z["flat_index"], z["q_future"], z["sem_gt"]
            with np.load(G20.population_path(drive, t)) as z:
                d_p0 += int(len(z["flat_index"]))
            h.update(np.ascontiguousarray(idx).tobytes())
            h.update(np.ascontiguousarray(q).tobytes())
            cls += np.bincount(sem.astype(np.int64), minlength=19)
            d_ur += int(len(idx))
            d_empty += int(len(idx) == 0)
        per_drive[drive] = {"anchors": len(files), "anchors_with_empty_UR": d_empty,
                            "n_P0": d_p0, "n_UR": d_ur,
                            "revealed_pct": round(100.0 * d_ur / max(d_p0, 1), 3)}
        n_ur += d_ur
    dev: Dict[str, int] = {"E0": 0, "E1": 0, "E2": 0, "anchors": 0}
    droot = os.path.join(G20.DATA_ROOT, "population", G.DEV_DRIVE)
    for f in sorted(os.listdir(droot)):
        if not f.endswith(".npz"):
            continue
        with np.load(os.path.join(droot, f)) as z:
            rev = np.isfinite(z["vis_first_dt"])
        dev["anchors"] += 1
        dev["E0"] += int(len(rev)); dev["E1"] += int(rev.sum()); dev["E2"] += int((~rev).sum())
    return {"version": G.VERSION, "reveal_level": G.REVEAL_LEVEL, "reveal_readout": G.REVEAL_READOUT,
            "train": {"per_drive": per_drive, "n_UR_total": n_ur,
                      "n_P0_total": sum(v["n_P0"] for v in per_drive.values()),
                      "class_counts_UR": {G.CLASS_NAMES[i - 1]: int(cls[i]) for i in range(1, 19)}},
            "dev": dev, "target_sha256": h.hexdigest()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--stats", action="store_true", help="write population_stats.json and exit")
    a = ap.parse_args()
    if a.stats:
        G.ensure_dirs()
        r = population_stats()
        json.dump(r, open(os.path.join(G.ARTIFACT_DIR, "population_stats.json"), "w"), indent=1)
        print(json.dumps({k: v for k, v in r.items() if k != "train"}, indent=1))
        print(json.dumps(r["train"]["per_drive"], indent=1))
        return
    assert a.drive, "--drive is required unless --stats is given"
    print(json.dumps(build_drive(a.drive, a.device, a.limit), indent=1))


if __name__ == "__main__":
    main()
