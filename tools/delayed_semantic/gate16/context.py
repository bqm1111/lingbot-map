"""Per-anchor inputs for Gate 16 evaluation, split by what may be seen when.

``student_context``   the prediction side (PREDICT_FORBIDDEN file audit): hard map, frozen occupancy and trunk, H_t evidence
                      with provenance, the pilot's B0 labels and distances. Identical calls to the pilot's ``predict_anchor``.
``truth``             EVALUATION ONLY, opened after predictions: semantic GT, Gate 12 occupancy regions, 0006 delayed targets.
``query_candidates``  candidate sets for a query list, chunked exactly like the pilot's ``head_predict`` (16,384 queries).
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import torch

from core.datasets.audit import FileAudit
from tools.delayed_semantic import config as PC
from tools.delayed_semantic.gate16 import candidates as CS
from tools.delayed_semantic.memory import load_observed
from tools.delayed_semantic.stream import DriveStream
from tools.semantic_completion.evaluation import baseline_nearest as B0
from tools.semantic_completion.evaluation.evaluate import REGIONS_ROOT
from tools.semantic_completion.evidence import trident as TR
from tools.semantic_completion.frozen_geometry import geometry_input, load_hard_map, occupancy_and_trunk

CHUNK = 16384


@torch.no_grad()
def student_context(stream: DriveStream, t: int, net, text: torch.Tensor, device, paths: PC.Paths = PC.Paths()) -> Dict:
    hist = stream.history(t)
    with FileAudit(patterns=PC.PREDICT_FORBIDDEN, strict=True):
        S, F_ = load_hard_map(stream.drive, t, device, paths.g15)
        occ, trunk = occupancy_and_trunk(net, geometry_input(S, F_)); O = occ[0]; trunk = trunk[0]
        ev = load_observed(stream.drive, t, device, paths)
    assert ev.t == hist.t
    E = ev.dense_mask(); Ec = torch.zeros_like(E); Ec[ev.flat_index[ev.current()]] = True
    ev_label = torch.zeros(PC.N_VOXELS, dtype=torch.long, device=device); ev_label[ev.flat_index] = TR.classify(ev.feature, text)
    nearest, dist_m = B0.nearest_evidence(E)
    return {"t": t, "S": S, "F": F_, "O": O, "E": E, "Ec": Ec, "trunk": trunk, "ev": ev, "evd": CS.evidence_tensors(ev, trunk),
            "ev_label": ev_label, "b0": B0.nearest_label(ev_label, nearest), "dist_m": dist_m,
            "masks": torch.stack([F_, S, O]).reshape(3, *PC.GRID_DIMS)}


def truth(t: int, device, paths: PC.Paths = PC.Paths()) -> Dict:
    """EVALUATION ONLY (0006)."""
    with np.load(paths.g15.eval_mask_path(t)) as z:
        sem_gt = torch.from_numpy(z["sem_gt"].astype(np.int64)).to(device)
    with np.load(os.path.join(REGIONS_ROOT, PC.DEV_DRIVE, f"{t:05d}.npz")) as z:
        valid = torch.from_numpy(np.unpackbits(z["valid"])[:PC.N_VOXELS].astype(bool)).to(device)
        gt_occ = torch.from_numpy(np.unpackbits(z["occupied"])[:PC.N_VOXELS].astype(bool)).to(device) & valid
    with np.load(paths.target_path(PC.DEV_DRIVE, t)) as z:
        keep = ~z["in_evidence"]
        rflat = torch.from_numpy(z["flat_index"][keep].astype(np.int64)).to(device)
        rprob = torch.from_numpy(z["prob"][keep].astype(np.float32)).to(device)
        rdt = torch.from_numpy(z["first_dt_s"][keep].astype(np.float32)).to(device)
        rh = torch.from_numpy(z["hist_seen"][keep]).to(device)
    R = torch.zeros(PC.N_VOXELS, dtype=torch.bool, device=device); R[rflat] = True
    Rh = torch.zeros_like(R); Rh[rflat[rh]] = True
    nR = len(PC.REVEAL_NAMES)
    rbin = torch.full((PC.N_VOXELS,), nR - 1, dtype=torch.long, device=device)
    rbin[rflat] = torch.bucketize(rdt, torch.tensor(PC.REVEAL_EDGES_S, device=device), right=True)
    return {"sem_gt": sem_gt, "valid": valid, "gt_occ": gt_occ, "R": R, "Rh": Rh, "rbin": rbin, "rflat": rflat, "rprob": rprob}


@torch.no_grad()
def query_candidates(ctx: Dict, query_flat: torch.Tensor, text: torch.Tensor) -> List[Tuple[torch.Tensor, CS.CandidateSet]]:
    return [(query_flat[s:s + CHUNK], CS.build(query_flat[s:s + CHUNK], ctx["evd"], text)) for s in range(0, len(query_flat), CHUNK)]


def cat_sets(parts: List[CS.CandidateSet]) -> CS.CandidateSet:
    return CS.CandidateSet(*[torch.cat([getattr(p, f) for p in parts]) for f in ("query_flat", "rows", "voxel_ids", "distance_m", "semantic_probs", "classes", "valid")])


__all__ = ["CHUNK", "student_context", "truth", "query_candidates", "cat_sets"]
