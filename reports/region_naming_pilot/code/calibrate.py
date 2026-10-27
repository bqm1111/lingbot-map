"""Stage 3: choose the one free constant of the extension rule -- ``max_hops`` -- on calibration anchors.

Only the growth limit is selected here, with pooled-Trident naming (P), on the four calibration anchors.
Those anchors share no frame and no temporal block with any evaluation anchor, so their human labels may
be read for this choice. The association thresholds are *not* searched: they are the a-priori values in
``proposals.SETTINGS``, recorded as given.

The criterion is the frozen primary population's mIoU, ``completed_unobserved = GT_occupied & O & ~E``,
with accuracy reported beside it. Writes ``artifacts/calibration.json``; nothing else reads GT before
the evaluation predictions are written.
"""
from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np
import torch

import extension as EX
import legacy as L
import manifest as MF
import paths as P
import proposals as PR
import recognize as RC


def main(device: str = "cuda:0") -> Dict:
    a = MF.select()
    ids = [t for t in a["calibration"] if os.path.exists(os.path.join(P.PROPOSALS, f"{t:05d}.npz"))]
    assert ids, "no calibration anchor has proposals yet"
    text, _, _ = L.TR.load_text_embeddings(); text = text.to(device)
    net = L.load_frozen_geometry(L.GEOMETRY_SEED, device)
    cand = EX.SETTINGS["max_hops_candidates"]
    acc = {h: [] for h in cand}; conf = {h: [] for h in cand}; cover = {h: [] for h in cand}
    base_conf, base_acc = [], []
    for t in ids:
        S, F_ = L.load_hard_map_flat(t, device)
        O = L.occupancy_and_trunk(net, L.geometry_input(S, F_))[0][0]
        prop = PR.load(t)
        vr = torch.from_numpy(prop["voxel_region"].astype(np.int64)).to(device)
        n_regions = max(int(vr.max().item()) + 1, 1)
        ev = L.LF.load_evidence(L.C.Paths().evidence_path(P.DRIVE, t), device)
        E = ev.dense_mask()
        pooled, _ = RC.pooled_trident_labels(vr, ev, text, n_regions)
        teacher = torch.zeros(L.N_VOXELS, dtype=torch.long, device=device)
        teacher[ev.flat_index] = L.TR.classify(ev.feature, text)
        ev_P = torch.where(E, torch.where(vr >= 0, pooled[vr.clamp(min=0)], teacher), torch.zeros_like(teacher))
        nearest, _ = L.B0.nearest_evidence(E)
        p_n = L.B0.nearest_label(ev_P, nearest)
        sem_gt = L.load_sem_gt(t, device)
        prim = L.build_regions(sem_gt, E, O)[L.PRIMARY]
        base_conf.append(L.SM.confusion(sem_gt, p_n, prim))
        base_acc.append([int((prim & (p_n == sem_gt)).sum()), int(prim.sum())])
        for h in cand:
            mem = EX.grow(vr, O, F_, h)
            assigned = (mem["member"] >= 0) & ~mem["contested"]
            rid = mem["member"].clamp(min=0).long()
            ok = assigned & (pooled[rid] > 0)
            lab = torch.where(ok, pooled[rid], p_n)
            conf[h].append(L.SM.confusion(sem_gt, lab, prim))
            acc[h].append([int((prim & (lab == sem_gt)).sum()), int(prim.sum())])
            cover[h].append(mem["stats"]["occupied_coverage"])
        print(f"  calibrated on t={t}", flush=True)

    def mi(cs):
        return L.SM.summarize(np.sum(cs, axis=0), semantic_only=True)["miou"]

    rows = [{"max_hops": "P_N (no region extension)", "miou": mi(base_conf),
             "accuracy": sum(x[0] for x in base_acc) / sum(x[1] for x in base_acc), "occupied_coverage": 0.0}]
    for h in cand:
        rows.append({"max_hops": h, "miou": mi(conf[h]),
                     "accuracy": sum(x[0] for x in acc[h]) / sum(x[1] for x in acc[h]),
                     "occupied_coverage": float(np.mean(cover[h]))})
    best = max((r for r in rows if isinstance(r["max_hops"], int)), key=lambda r: r["miou"])
    out = {"anchors": ids, "criterion": f"mIoU on {L.PRIMARY}, pooled-Trident naming",
           "candidates": EX.SETTINGS["max_hops_candidates"], "rows": rows,
           "chosen_max_hops": best["max_hops"],
           "association_thresholds_not_searched": {k: PR.SETTINGS[k] for k in ("assoc_iou_min", "assoc_rgb_max")}}
    with open(P.under_experiment(os.path.join(P.ARTIFACTS, "calibration.json")), "w") as f:
        json.dump(out, f, indent=2)
    for r in rows:
        print(f"  hops {str(r['max_hops']):28s} mIoU {100*r['miou']:.2f}  acc {100*r['accuracy']:.2f}  cover {r['occupied_coverage']:.3f}")
    print("chosen max_hops =", out["chosen_max_hops"])
    return out


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
