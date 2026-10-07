"""ANNOTATION-ASSISTED DIAGNOSTIC. Human labels are read here and nowhere else.

Teacher agreement measures whether two views of a frozen teacher say the same thing. It does
not measure whether they are right. This module answers the second question on a source
subset that has human labels, so the report can state correctness instead of implying it from
consistency.

Hard rules, enforced by construction:

* nothing here feeds a training target, a filtering threshold or a model selection. It is
  called once, by the report, after every measurement is already written;
* the label mapping is the explicit SSCBench-KITTI-360 one (``1..18``, ``0`` empty,
  ``255`` ignore) and is used only to score;
* every number this module returns is prefixed ``annotated_`` in the report so it can never
  be mistaken for a label-free measurement.

If the labels are absent the functions return ``available = False`` and the report says
correctness could not be measured, rather than substituting agreement for it.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from tools.future_revealability import config as G20
from tools.waymo_pilot import config as C, k360 as K, populations as POP

#: the evaluation-only human labels resampled onto the anchor grid (Gate 20's cache)
LABEL_ROOT = os.path.join(G20.DATA_ROOT, "eval_masks")
IGNORE = 255


def label_path(drive: str, t: int) -> str:
    return os.path.join(LABEL_ROOT, drive, f"{t:05d}.npz")


def available(drive: str, t: int) -> bool:
    return os.path.isfile(label_path(drive, t))


def load_labels(drive: str, t: int, device) -> Optional[torch.Tensor]:
    p = label_path(drive, t)
    if not os.path.isfile(p):
        return None
    with np.load(p) as z:
        if not bool(z["has_label"]):
            return None
        return torch.from_numpy(z["sem_gt"].astype(np.int64)).to(device)


@torch.no_grad()
def teacher_correctness(drive: str, t: int, device, text: torch.Tensor) -> Dict:
    """Top-1 correctness of the privileged semantic target on ``U``, against human labels."""
    gt = load_labels(drive, t, device)
    if gt is None:
        return {"available": False, "scene": drive, "anchor": int(t)}
    from tools.semantic_completion.evidence import build as G15B
    cs = K.causal_state(drive, t, device, text)
    centres = POP.grid_centres(device)
    obs = POP.causal_observability(centres, np.linalg.inv(G15B.camera_to_grid()),
                                   K.causal_views(cs["clip"], device), cs["S"], cs["F"],
                                   cs["evidence_mask"])
    O_priv, _v, _m = K.privileged_occupancy(drive, t, device)
    sem = K.privileged_semantics(drive, t, device)
    pop = POP.build(O_priv, sem["valid"], obs)

    readout = torch.zeros(C.N_VOXELS, dtype=torch.long, device=device)
    readout[sem["flat"]] = (1 + sem["prob"].argmax(1)).long()
    labelled = (gt >= 1) & (gt <= C.N_CLASSES)

    scored = pop.U & labelled
    n = int(scored.sum())
    correct = int(((readout == gt) & scored).sum())
    # per-class confusion over the scored set, for a label-space mIoU of the TARGET itself
    conf = torch.zeros(C.N_CLASSES, C.N_CLASSES, dtype=torch.long, device=device)
    if n:
        ti = (gt[scored] - 1).clamp(0, C.N_CLASSES - 1)
        pi = (readout[scored] - 1).clamp(0, C.N_CLASSES - 1)
        conf.view(-1).index_add_(0, ti * C.N_CLASSES + pi,
                                 torch.ones_like(ti, dtype=torch.long))
    # the same question restricted to the two-view-consistent subset
    two = torch.zeros(C.N_VOXELS, dtype=torch.bool, device=device)
    agree = torch.zeros(C.N_VOXELS, dtype=torch.bool, device=device)
    two[sem["flat"]] = sem["n_views"] >= 2
    agree[sem["flat"]] = (sem["view0_top1"] == sem["view1_top1"]) & (sem["n_views"] >= 2)
    s2 = scored & two
    sa = scored & agree
    # correctness split by the recipe's own reliability tier: A = occupancy at t confirmed by
    # a causal-window sweep, B = occupancy at t uncertain, carried by >= 2 future sweeps.
    tier = torch.zeros(C.N_VOXELS, dtype=torch.uint8, device=device)
    tier[sem["flat"]] = sem["tier"].to(torch.uint8)
    per_tier = {}
    for name, val in (("A", 1), ("B", 2)):
        m = scored & (tier == val)
        per_tier[f"n_tier{name}"] = int(m.sum())
        per_tier[f"n_tier{name}_correct"] = int(((readout == gt) & m).sum())
    return {"available": True, "scene": drive, "anchor": int(t), **per_tier,
            "n_U": int(pop.U.sum()), "n_U_with_label": n,
            "n_correct": correct,
            "confusion": conf.cpu().numpy(),
            "n_two_view": int(s2.sum()),
            "n_two_view_correct": int(((readout == gt) & s2).sum()),
            "n_two_view_agreeing": int(sa.sum()),
            "n_two_view_agreeing_correct": int(((readout == gt) & sa).sum())}


def summarize(rows: Sequence[Dict]) -> Dict:
    """Pool the per-anchor diagnostics. Counts are summed; ratios come from the sums."""
    got = [r for r in rows if r.get("available")]
    if not got:
        return {"available": False,
                "note": "no human-labelled subset was available; teacher CORRECTNESS could "
                        "not be measured, and agreement must not be read as a substitute"}
    conf = np.sum([r["confusion"] for r in got], axis=0)
    tp = np.diag(conf).astype(np.int64)
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    den = tp + fp + fn
    iou = np.where(den > 0, tp / np.maximum(den, 1), np.nan)
    present = den > 0
    n = int(sum(r["n_U_with_label"] for r in got))
    c = int(sum(r["n_correct"] for r in got))
    n2 = int(sum(r["n_two_view"] for r in got))
    c2 = int(sum(r["n_two_view_correct"] for r in got))
    na = int(sum(r["n_two_view_agreeing"] for r in got))
    ca = int(sum(r["n_two_view_agreeing_correct"] for r in got))
    tiers = {}
    for name in ("A", "B"):
        nt = int(sum(r.get(f"n_tier{name}", 0) for r in got))
        ct = int(sum(r.get(f"n_tier{name}_correct", 0) for r in got))
        tiers[f"n_tier{name}"] = nt
        tiers[f"annotated_top1_tier{name}"] = (ct / nt) if nt else None
    return {
        "available": True, "n_anchors": len(got),
        "n_U_with_label": n, "n_correct": c,
        "annotated_top1_accuracy": (c / n) if n else None,
        "annotated_target_miou": float(np.nanmean(iou[present])) if present.any() else None,
        "n_classes_scored": int(present.sum()),
        "annotated_per_class_iou": {C.CLASS_NAMES[i]: (None if not present[i] else float(iou[i]))
                                    for i in range(C.N_CLASSES)},
        **tiers,
        "annotated_top1_two_view": (c2 / n2) if n2 else None,
        "annotated_top1_two_view_agreeing": (ca / na) if na else None,
        "n_two_view": n2, "n_two_view_agreeing": na,
        "note": ("annotation-assisted; human labels are read ONLY here and never enter a "
                 "target, a threshold or a selection. 'target mIoU' is the quality of the "
                 "privileged SUPERVISION, not of any model."),
    }


__all__ = ["LABEL_ROOT", "available", "load_labels", "teacher_correctness", "summarize"]
