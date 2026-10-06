"""Oracle instance identity, and the five label maps compared on each population.

**The oracle is a grouping, not a label.** For every method except `C2_gt_class`, the semantic class of a completed
voxel still comes from the frozen Trident-H teacher, read out through the frozen text embeddings. The only thing
ground truth supplies is *which voxels belong together*.

Instance identity is 6-connected components of same-class GT-occupied voxels. KITTI-360 instance annotations are not
available on this machine (see `artifacts/gate17_instance_oracle/plan.md`), so this is the strongest legitimate
substitute. It is an upper bound in two ways that matter and are reported throughout: it knows GT class boundaries
exactly, and adjacent same-class objects merge into one component.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from scipy import ndimage

from tools.delayed_semantic import config as C
from tools.instance_oracle import config as G

DIMS = tuple(int(d) for d in C.GRID_DIMS)
#: 6-connectivity structuring element (faces only).
STRUCT = ndimage.generate_binary_structure(3, 1)


def instance_map(sem_gt: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(instance_id [N] int32, instance_label [n_inst+1] uint8)``; id 0 means "no instance".

    Components are computed per class, so a component never spans two classes."""
    g = sem_gt.reshape(DIMS); v = valid.reshape(DIMS)
    out = np.zeros(DIMS, np.int32)
    labels = [0]
    nxt = 1
    for c in range(1, 19):
        m = (g == c) & v
        if not m.any():
            continue
        lab, n = ndimage.label(m, structure=STRUCT)
        if n:
            out[m] = lab[m] + (nxt - 1)
            labels.extend([c] * n)
            nxt += n
    return out.reshape(-1), np.asarray(labels, np.uint8)


def instance_stats(inst: np.ndarray, evidence: np.ndarray, n_inst: int) -> Dict[str, np.ndarray]:
    """Per instance: total voxels, voxels carrying visible evidence, visible fraction."""
    total = np.bincount(inst, minlength=n_inst + 1).astype(np.int64)
    seen = np.bincount(inst[evidence], minlength=n_inst + 1).astype(np.int64)
    frac = np.divide(seen, np.maximum(total, 1), dtype=np.float64)
    return {"total": total, "seen": seen, "visible_fraction": frac}


@torch.no_grad()
def instance_semantics(inst: torch.Tensor, ev_flat: torch.Tensor, ev_feature: torch.Tensor, ev_label_dense: torch.Tensor,
                       n_inst: int, text: torch.Tensor, device) -> Dict[str, torch.Tensor]:
    """Per instance, from its OWN visible evidence only.

    ``inst`` and ``ev_label_dense`` are dense ``[N_VOXELS]``; ``ev_flat`` and ``ev_feature`` are per evidence row.

    ``mean_feature_label``: readout of the mean unit teacher feature (method A).
    ``majority_label``:     most frequent per-voxel teacher label (control C3).
    Both are 0 where an instance has no visible evidence."""
    ev_inst = inst[ev_flat]                                        # instance of each evidence voxel, 0 if none
    keep = ev_inst > 0
    ei, ef, el = ev_inst[keep], ev_feature[keep], ev_label_dense[ev_flat][keep]
    fsum = torch.zeros(n_inst + 1, ef.shape[1], device=device, dtype=torch.float32).index_add_(0, ei, ef.float())
    cnt = torch.zeros(n_inst + 1, device=device, dtype=torch.float32).index_add_(0, ei, torch.ones_like(ei, dtype=torch.float32))
    has = cnt > 0
    mean_label = torch.zeros(n_inst + 1, dtype=torch.long, device=device)
    if has.any():
        z = torch.nn.functional.normalize(fsum[has], dim=-1)
        mean_label[has] = (z @ text.T).argmax(1) + 1               # +1: text index 0 == sem label 1
    votes = torch.zeros(n_inst + 1, 19, device=device, dtype=torch.float32)
    votes.index_put_((ei, el), torch.ones_like(ei, dtype=torch.float32), accumulate=True)
    majority = votes.argmax(1)
    majority[~has] = 0
    return {"mean_feature_label": mean_label, "majority_label": majority, "has_evidence": has, "n_evidence": cnt}


def shuffled(labels: torch.Tensor, has: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """C1: permute the per-instance labels among instances that HAVE evidence, keeping coverage identical.

    If instance identity is what matters, destroying which identity goes with which voxel group must hurt."""
    out = labels.clone()
    idx = has.nonzero(as_tuple=True)[0]
    if len(idx) > 1:
        perm = idx[torch.randperm(len(idx), generator=gen, device=labels.device)]
        out[idx] = labels[perm]
    return out


@torch.no_grad()
def method_labels(b0_label: torch.Tensor, inst: torch.Tensor, sem: Dict[str, torch.Tensor], sem_gt: torch.Tensor,
                  gen: torch.Generator) -> Dict[str, torch.Tensor]:
    """Dense [N] label per method. Voxels whose instance has no visible evidence fall back to B0, and that
    fallback is exactly what ``coverage`` measures -- the oracle cannot invent evidence it never saw."""
    has_i = sem["has_evidence"][inst]                              # [N] bool: this voxel's instance has evidence
    take = lambda per_inst: torch.where(has_i, per_inst[inst], b0_label)
    sh = shuffled(sem["mean_feature_label"], sem["has_evidence"], gen)
    return {"C0_b0": b0_label,
            "A_oracle_instance": take(sem["mean_feature_label"]),
            "C1_shuffled_instance": take(sh),
            "C3_visible_majority": take(sem["majority_label"]),
            "C2_gt_class": sem_gt.clone(),
            "_covered": has_i}


__all__ = ["instance_map", "instance_stats", "instance_semantics", "shuffled", "method_labels", "DIMS"]
