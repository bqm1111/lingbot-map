"""The five label sets of one anchor. Occupancy is computed once and shared by all of them.

    B0    the deployed baseline, reproduced: nearest evidence, each evidence voxel labelled by its own
          teacher feature
    P_N   pooled Trident region label at the region's own evidence voxels, then the SAME nearest-evidence rule
    P_R   the same pooled labels, then bounded region extension
    Q_N   Qwen region label (pooled-Trident fallback) at the same evidence voxels, same nearest rule
    Q_R   the same Qwen labels, same bounded extension

``P_N`` is not expected to equal ``B0``: pooling replaces per-voxel teacher argmax by one argmax per
region, which changes the seeds the nearest-evidence rule copies from. That difference is reported.

Nothing here reads a human label: the run happens inside a strict file audit that refuses eval_masks,
the target benchmarks and any LiDAR path.
"""
from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np
import torch

import extension as EX
import legacy as L
import paths as P
import proposals as PR
import recognize as RC

METHODS = ("B0", "P_N", "P_R", "Q_N", "Q_R")


def _pack(x: torch.Tensor) -> np.ndarray:
    return np.packbits(x.cpu().numpy().astype(bool))


def qwen_labels(t: int, regions, pooled: torch.Tensor) -> Dict:
    """Region label table under Q, plus the outcome of every query. Falls back to ``pooled``."""
    path = os.path.join(P.QWEN_CACHE, f"{t:05d}.json")
    rec = json.load(open(path))
    lab = pooled.clone()
    outcomes, changed = {}, 0
    for k, v in rec["regions"].items():
        r = int(k)
        outcomes[r] = v["parsed"]["outcome"]
        if v["parsed"]["outcome"] == "ok":
            lab[r] = int(v["parsed"]["label"])
            changed += int(lab[r] != pooled[r])
    return {"label": lab, "outcomes": outcomes, "queried": sorted(outcomes),
            "n_changed_vs_pooled": changed, "raw": rec}


def predict_anchor(t: int, net, text: torch.Tensor, device, max_hops: int = None) -> Dict:
    clip_scale = None
    S, F_ = L.load_hard_map_flat(t, device)
    O = L.occupancy_and_trunk(net, L.geometry_input(S, F_))[0][0]
    prop = PR.load(t)
    voxel_region = torch.from_numpy(prop["voxel_region"].astype(np.int64)).to(device)
    n_regions = max(int(voxel_region.max().item()) + 1, 1)
    ev = L.LF.load_evidence(L.C.Paths().evidence_path(P.DRIVE, t), device)
    E = ev.dense_mask()

    # --- seeds -------------------------------------------------------------------------
    teacher = torch.zeros(L.N_VOXELS, dtype=torch.long, device=device)
    teacher[ev.flat_index] = L.TR.classify(ev.feature, text)                 # B0's own evidence labels
    pooled, support = RC.pooled_trident_labels(voxel_region, ev, text, n_regions)
    q = qwen_labels(t, prop["regions"], pooled)

    owned = voxel_region >= 0
    ev_P = torch.where(owned, pooled[voxel_region.clamp(min=0)], teacher)
    ev_Q = torch.where(owned, q["label"][voxel_region.clamp(min=0)], teacher)
    ev_P = torch.where(E, ev_P, torch.zeros_like(ev_P))
    ev_Q = torch.where(E, ev_Q, torch.zeros_like(ev_Q))

    # --- nearest-evidence extension (the original rule) --------------------------------
    nearest, dist_m = L.B0.nearest_evidence(E)
    lab = {"B0": L.B0.nearest_label(teacher, nearest),
           "P_N": L.B0.nearest_label(ev_P, nearest),
           "Q_N": L.B0.nearest_label(ev_Q, nearest)}

    # --- bounded region extension ------------------------------------------------------
    mem = EX.grow(voxel_region, O, F_, max_hops)
    assigned = (mem["member"] >= 0) & ~mem["contested"]
    rid = mem["member"].clamp(min=0).long()
    for src, dst in (("P_N", "P_R"), ("Q_N", "Q_R")):
        tab = pooled if src == "P_N" else q["label"]
        ok = assigned & (tab[rid] > 0)
        lab[dst] = torch.where(ok, tab[rid], lab[src])

    stats = {"t": t, "n_regions": n_regions, "n_queried": len(q["queried"]),
             "qwen_outcomes": {k: sum(1 for v in q["outcomes"].values() if v == k)
                               for k in ("ok", "abstain", "out_of_vocabulary", "malformed")},
             "qwen_changed_vs_pooled": q["n_changed_vs_pooled"],
             "n_S": int(S.sum()), "n_F": int(F_.sum()), "n_O": int(O.sum()), "n_E": int(E.sum()),
             "seeds_changed_P_vs_B0": int(((ev_P != teacher) & E).sum()),
             "seeds_changed_Q_vs_P": int(((ev_Q != ev_P) & E).sum()),
             "extension": mem["stats"],
             "n_occupied_relabelled": {m: int(((lab[m] != lab["B0"]) & O).sum()) for m in METHODS},
             "membership_fallback_fraction_on_O": float((~assigned)[O].sum() / max(int(O.sum()), 1))}
    return {"labels": lab, "O": O, "E": E, "S": S, "F": F_, "dist_m": dist_m,
            "member": mem["member"], "contested": mem["contested"], "hops": mem["hops"],
            "pooled": pooled, "qwen": q["label"], "support": support, "stats": stats}


def save(t: int, out: Dict) -> str:
    path = P.under_experiment(os.path.join(P.PREDICTIONS, f"{t:05d}.npz"))
    arr = {f"label__{m}": out["labels"][m].cpu().numpy().astype(np.uint8) for m in METHODS}
    np.savez_compressed(path + ".tmp.npz", O=_pack(out["O"]), E=_pack(out["E"]), S=_pack(out["S"]),
                        F=_pack(out["F"]), dist_m=out["dist_m"].astype(np.float16),
                        member=out["member"].cpu().numpy().astype(np.int32),
                        contested=_pack(out["contested"]), hops=out["hops"].cpu().numpy().astype(np.int8),
                        pooled=out["pooled"].cpu().numpy().astype(np.uint8),
                        qwen=out["qwen"].cpu().numpy().astype(np.uint8),
                        support=out["support"].cpu().numpy().astype(np.int32),
                        stats=json.dumps(out["stats"]), **arr)
    os.replace(path + ".tmp.npz", path)
    return path


def load(t: int, device) -> Dict:
    def unp(a):
        return torch.from_numpy(np.unpackbits(a)[:L.N_VOXELS].astype(bool)).to(device)
    with np.load(os.path.join(P.PREDICTIONS, f"{t:05d}.npz")) as z:
        out = {"labels": {m: torch.from_numpy(z[f"label__{m}"].astype(np.int64)).to(device) for m in METHODS},
               "O": unp(z["O"]), "E": unp(z["E"]), "S": unp(z["S"]), "F": unp(z["F"]),
               "contested": unp(z["contested"]),
               "member": torch.from_numpy(z["member"].astype(np.int64)).to(device),
               "hops": torch.from_numpy(z["hops"].astype(np.int64)).to(device),
               "dist_m": z["dist_m"].astype(np.float32),
               "pooled": torch.from_numpy(z["pooled"].astype(np.int64)).to(device),
               "qwen": torch.from_numpy(z["qwen"].astype(np.int64)).to(device),
               "support": torch.from_numpy(z["support"].astype(np.int64)).to(device),
               "stats": json.loads(str(z["stats"]))}
    return out
