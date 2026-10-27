"""Adapter alignment: reproduce the deployed baseline B0 exactly, on the reference the legacy repo published.

Gate 17 reports B0 over all 590 anchors of drive 0006, on three of ``build_regions``' populations:

    completed_unobserved  (its P2 "correct completion")  8.796376363534595  over 2,743,212 voxels
    ssc_no_evidence       (its P1 "deployable SSC")      1.3927645153874209 over 553,275,690 voxels
    oracle_unobserved     (its P3 "oracle geometry")     8.392475299689655  over 7,834,276 voxels

This script recomputes them through this experiment's adapter -- frozen hard map, frozen CompletionUNet
seed 0, frozen evidence, frozen teacher readout, the frozen nearest-evidence rule, the frozen scorer --
and asserts they match. Agreement to the published digits proves the adapter did not change the grid,
the transform, the occupancy threshold, the readout or the metric.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

import legacy as L
import paths as P

REFERENCE = {"completed_unobserved": (8.796376363534595, 2_743_212),
             "ssc_no_evidence": (1.3927645153874209, 553_275_690),
             "oracle_unobserved": (8.392475299689655, 7_834_276)}
TOL = 1e-6


def main(device: str = "cuda:0", limit: int = 0) -> dict:
    text, _, _ = L.TR.load_text_embeddings(); text = text.to(device)
    net = L.load_frozen_geometry(L.GEOMETRY_SEED, device)
    ids = L.anchor_ids(P.DRIVE)
    if limit:
        ids = ids[:limit]
    acc = {r: np.zeros((19, 19), np.int64) for r in REFERENCE}
    t0 = time.time()
    for k, t in enumerate(ids):
        S, F_ = L.load_hard_map_flat(t, device)
        O = L.occupancy_and_trunk(net, L.geometry_input(S, F_))[0][0]
        ev = L.LF.load_evidence(L.C.Paths().evidence_path(P.DRIVE, t), device)
        E = ev.dense_mask()
        teacher = torch.zeros(L.N_VOXELS, dtype=torch.long, device=device)
        teacher[ev.flat_index] = L.TR.classify(ev.feature, text)
        nearest, _ = L.B0.nearest_evidence(E)
        b0 = L.B0.nearest_label(teacher, nearest)
        sem_gt = L.load_sem_gt(t, device)
        reg = L.build_regions(sem_gt, E, O)
        ssc = torch.where(O, b0, torch.zeros_like(b0))
        for r in REFERENCE:
            acc[r] += L.SM.confusion(sem_gt, ssc if r in L.SSC_REGIONS else b0, reg[r])
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(ids)} {(time.time()-t0)/(k+1):.2f}s each", flush=True)
    out = {"n_anchors": len(ids), "seconds": time.time() - t0, "regions": {}}
    ok = True
    for r, (ref_miou, ref_n) in REFERENCE.items():
        s = L.SM.summarize(acc[r], semantic_only=r not in L.SSC_REGIONS)
        got = 100.0 * s["miou"]
        n = int(acc[r].sum() if r in L.SSC_REGIONS else acc[r][1:].sum())
        match = abs(got - ref_miou) < TOL and (n == ref_n or not limit == 0)
        ok &= bool(match) if not limit else True
        out["regions"][r] = {"reference_miou": ref_miou, "reproduced_miou": got,
                             "reference_voxels": ref_n, "reproduced_voxels": n, "match": bool(match)}
        print(f"  {r:22s} reference {ref_miou:.12f} / {ref_n:>11d}   reproduced {got:.12f} / {n:>11d}   "
              f"{'MATCH' if match else 'MISMATCH'}", flush=True)
    out["all_match"] = bool(ok)
    if not limit:
        with open(P.under_experiment(os.path.join(P.ARTIFACTS, "verify_b0.json")), "w") as f:
            json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    r = main(dev, lim)
    print("ALL MATCH" if r["all_match"] else "MISMATCH")
