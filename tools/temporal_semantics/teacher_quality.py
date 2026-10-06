"""Gate 18 section 2: how good is the frozen teacher itself, before any propagation?

Every completion result so far (B1, B4, B5, the Gate 17 instance oracle) ends at the same wall, so this measures the
wall directly: at voxels where the teacher HAS evidence and ground truth HAS a label, how often is the teacher right?

Nothing is accumulated and nothing is registered here -- this is the anchor's own frozen 5-frame evidence, scored
against the 0006 labels with the repository's frozen readout and metric. Fully deployable, no oracle.

    PYTHONPATH=$PWD python -m tools.temporal_semantics.teacher_quality --device cuda:0 --shard 0 --shards 4
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Optional

import numpy as np
import torch

from core.datasets.audit import FileAudit
from tools.delayed_semantic import config as C
from tools.delayed_semantic.memory import load_observed
from tools.semantic_completion.evaluation import semantic_metrics as SM
from tools.semantic_completion.evidence import build as G15B, trident as TR
from tools.temporal_semantics import config as G18


def in_bin(count: torch.Tensor, i: int) -> torch.Tensor:
    """Closed integer range ``OBS_RANGES[i]``. Explicit so the bin labels cannot drift from the bin contents."""
    lo, hi = G18.OBS_RANGES[i]
    return (count >= lo) & (count <= hi)


def evaluate(device: str, shard: int = 0, shards: int = 1, limit: Optional[int] = None,
             paths: C.Paths = C.Paths()) -> Dict:
    dev = torch.device(device); torch.cuda.set_device(dev)
    text, _, _ = TR.load_text_embeddings(paths.g15.text_embeddings); text = text.to(dev)
    ids = G15B.anchor_ids(G18.DEV_DRIVE, paths.g15)[shard::shards][:limit]
    nO = len(G18.OBS_NAMES)
    conf_all, conf_obs, conf_ray, n_ev, n_lab = [], [], [], [], []
    t0 = time.time()
    with FileAudit(patterns=C.EVAL_FORBIDDEN, strict=True) as audit:
        for k, t in enumerate(ids):
            ev = load_observed(G18.DEV_DRIVE, t, dev, paths)
            with np.load(paths.g15.eval_mask_path(t)) as z:
                sem_gt = torch.from_numpy(z["sem_gt"].astype(np.int64)).to(dev)
            pred = torch.zeros(C.N_VOXELS, dtype=torch.long, device=dev)
            pred[ev.flat_index] = TR.classify(ev.feature, text)
            has_ev = torch.zeros(C.N_VOXELS, dtype=torch.bool, device=dev); has_ev[ev.flat_index] = True
            scored = has_ev & (sem_gt >= 1) & (sem_gt <= 18)          # teacher evidence AND a GT label
            conf_all.append(SM.confusion(sem_gt, pred, scored))
            # Two different quantities, and the brief asks to tell them apart:
            #   frames  = how many DISTINCT history frames saw the voxel (independent observations), from frame_bits
            #   rays    = how many accepted ray endpoints landed in it, which one frame can supply many of
            bits = ev.frame_bits.long()
            frames = sum((bits >> q) & 1 for q in range(len(C.HISTORY_OFFSETS)))
            nfr = torch.zeros(C.N_VOXELS, dtype=torch.long, device=dev); nfr[ev.flat_index] = frames
            nray = torch.zeros(C.N_VOXELS, dtype=torch.long, device=dev); nray[ev.flat_index] = ev.count.long()
            conf_obs.append(np.stack([SM.confusion(sem_gt, pred, scored & in_bin(nfr, i)) for i in range(nO)]))
            conf_ray.append(np.stack([SM.confusion(sem_gt, pred, scored & in_bin(nray, i)) for i in range(nO)]))
            n_ev.append(int(has_ev.sum())); n_lab.append(int(scored.sum()))
            if (k + 1) % 100 == 0:
                print(f"  [teacher s{shard}] {k+1}/{len(ids)} {(time.time()-t0)/(k+1):.2f}s each", flush=True)
    out = {"ids": np.asarray(ids), "conf": np.stack(conf_all), "conf_obs": np.stack(conf_obs),
           "conf_ray": np.stack(conf_ray),
           "n_evidence": np.asarray(n_ev), "n_scored": np.asarray(n_lab)}
    p = G18.shard_path("teacher", shard, shards); os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez_compressed(p, **out)
    meta = {"version": G18.VERSION, "shard": shard, "shards": shards, "n_anchors": len(ids),
            "seconds": time.time() - t0, "firewall": {"opened": len(audit.opened), "violations": audit.violations}}
    json.dump(meta, open(p[:-4] + ".json", "w"), indent=1)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    print(json.dumps(evaluate(a.device, a.shard, a.shards, a.limit), indent=1))


if __name__ == "__main__":
    main()
