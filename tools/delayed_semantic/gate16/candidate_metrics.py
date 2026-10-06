"""Selection metrics. Label-free ones compare scorer weights with the candidate target; GT ones are evaluation-only.

``SelectionTally`` accumulates, per method, over delayed-target voxels of 0006:
    n, top1 (argmax weight == argmax compatibility), top3, sum of weight on the best candidate, sum of KL(target || weights),
    and (where the voxel has a GT class) GT correctness of the top-weight candidate's class, of the delayed-best candidate's
    class and of the nearest candidate's class.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from tools.delayed_semantic.gate16.candidate_targets import CandidateTarget
from tools.delayed_semantic.gate16.candidates import CandidateSet

FIELDS = ("n", "top1", "top3", "p_best", "kl", "n_gt", "top_weight_gt", "delayed_best_gt", "nearest_gt")


class SelectionTally:
    def __init__(self):
        self.v: Dict[str, np.ndarray] = {}

    @torch.no_grad()
    def add(self, method: str, cs: CandidateSet, ct: CandidateTarget, weights: Optional[torch.Tensor], gt: torch.Tensor) -> None:
        """weights [Q, K] (None = nearest candidate one-hot, i.e. B0); gt [Q] int64, values outside 1..18 are unlabelled."""
        if weights is None:
            weights = torch.zeros_like(ct.distribution); weights[:, 0] = 1.0
        best = ct.best_index
        top = weights.argmax(-1); top3 = weights.topk(3, -1).indices
        kl = (ct.distribution * (ct.distribution.clamp_min(1e-12).log() - weights.clamp_min(1e-12).log())).sum(-1)
        lab = (gt >= 1) & (gt <= 18)
        cls_top = cs.classes.gather(1, top[:, None]).squeeze(1); cls_best = cs.classes.gather(1, best[:, None]).squeeze(1)
        row = np.array([len(best), int((top == best).sum()), int((top3 == best[:, None]).any(-1).sum()), float(weights.gather(1, best[:, None]).sum()),
                        float(kl.sum()), int(lab.sum()), int((cls_top == gt)[lab].sum()), int((cls_best == gt)[lab].sum()), int((cs.classes[:, 0] == gt)[lab].sum())], np.float64)
        self.v[method] = self.v.get(method, np.zeros(len(FIELDS))) + row

    def array(self, methods) -> np.ndarray:
        return np.stack([self.v.get(m, np.zeros(len(FIELDS))) for m in methods])


def summarize_selection(row: np.ndarray) -> Dict[str, float]:
    d = dict(zip(FIELDS, row))
    return {"voxels": int(d["n"]), "top1": d["top1"] / max(d["n"], 1), "top3": d["top3"] / max(d["n"], 1), "mean_weight_on_best": d["p_best"] / max(d["n"], 1),
            "kl_to_candidate_target": d["kl"] / max(d["n"], 1), "gt_voxels": int(d["n_gt"]), "top_weight_gt_acc": d["top_weight_gt"] / max(d["n_gt"], 1),
            "delayed_best_gt_acc": d["delayed_best_gt"] / max(d["n_gt"], 1), "nearest_gt_acc": d["nearest_gt"] / max(d["n_gt"], 1)}


def accuracy(conf: np.ndarray) -> float:
    """Overall GT-class accuracy of a 19x19 semantic-region confusion (rows = GT 1..18)."""
    return float(np.trace(conf[1:, 1:]) / max(conf[1:].sum(), 1))


def recovered_fraction(acc: float, nearest: float, oracle: float) -> float:
    return (acc - nearest) / max(oracle - nearest, 1e-12)


__all__ = ["FIELDS", "SelectionTally", "summarize_selection", "accuracy", "recovered_fraction"]
