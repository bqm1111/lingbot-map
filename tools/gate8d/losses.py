"""Gate 8D: the training objective.

Gate 8C-1's occupancy and semantic losses unchanged, plus the metric surface-distance term
from Phase 4. The distance term is auxiliary: it shapes the representation, and the
occupancy logit remains the only thing inference reads.
"""
from __future__ import annotations

import torch

from gates.gate8 import losses as L
from gates.gate8a import losses as L8A
from gates.gate8a.regions import editable_native
from gates.gate8d import protocol as P
from gates.gate8d.net import apply_residual, surface_distance_loss


def step_loss(net, b, cfg):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        occ_res, sem_log, dist = net(b["input"])
    occ_res = occ_res[:, 0].float()
    sem_log = sem_log.float()
    dist = dist[:, 0].float()
    final = apply_residual(b["base_logodds"], occ_res)

    #: the semantic loss ignores unreliable teacher evidence: ``fut_valid`` is already
    #: masked to voxels where the future window is geometrically supported AND its two
    #: halves agree on the top-1 class
    l_sem = L.semantic_kl(sem_log, b["fut_p"], b["fut_valid"].float())
    l_dist = surface_distance_loss(dist, b["tudf_m"], b["tudf_valid"])
    parts = {"sem_kl": l_sem.item(), "surface_l1_m": l_dist.item()}

    if cfg["occ_loss"] == "bce":
        edit = editable_native(b["base_logodds"], cfg["lock_logodds"])
        l_occ = L8A.bce_editable(final, b["gt_occ"], b["gt_valid"], edit)
        loss = cfg["w_bce"] * l_occ + cfg["w_sem"] * l_sem
        parts.update({"bce": l_occ.item(), "focal": float("nan"), "dice": float("nan")})
    else:
        w = L.balanced_weights(b["gt_occ"], b["gt_valid"], b["observed"], cfg["w_unknown"])
        n_pos = (b["gt_occ"] * b["gt_valid"].float()).sum()
        n_neg = b["gt_valid"].float().sum() - n_pos
        pw = float(min(cfg["pos_weight_cap"], max(1.0, (n_neg / n_pos.clamp_min(1)).item())))
        l_focal = L.focal_bce(final, b["gt_occ"], w, cfg["focal_gamma"], pw)
        l_dice = L.soft_dice(final, b["gt_occ"], b["gt_valid"].float())
        loss = cfg["w_focal"] * l_focal + cfg["w_dice"] * l_dice + cfg["w_sem"] * l_sem
        parts.update({"focal": l_focal.item(), "dice": l_dice.item(), "bce": float("nan")})
    loss = loss + float(cfg.get("w_surface", P.TUDF_LOSS_WEIGHT)) * l_dist

    with torch.no_grad():
        gt = (b["gt_occ"] > 0) & b["gt_valid"]
        pred = (final > 0) & b["gt_valid"]
        base = (b["base_logodds"] > 0) & b["gt_valid"]

        def iou(p):
            tp = (p & gt).sum().item(); fp = (p & ~gt).sum().item()
            fn = (~p & gt).sum().item()
            return tp / max(tp + fp + fn, 1)

        parts.update({"iou": iou(pred), "base_iou": iou(base),
                      "prevalence": float(gt.sum().item()
                                          / max(b["gt_valid"].sum().item(), 1)),
                      "density": float(pred.sum().item()
                                       / max(b["gt_valid"].sum().item(), 1))})
    return loss, parts


__all__ = ["step_loss"]
