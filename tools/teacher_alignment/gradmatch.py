"""Losses and the matched-gradient combination rule.

Shared parameters are the completion encoder/decoder -- every trainable tensor except the
three output heads. At each step two task gradients are taken over *all* parameters:

    g_occ  = d L_occ        / d theta
    g_aux  = d (0.5 * L_KL) / d theta

and the update gradient is assembled per parameter group:

    shared         : g_occ + c * g_aux,   c = rho * ||g_occ_shared|| / ||g_aux_shared||
    occupancy head : g_occ                (unchanged)
    semantic head  : g_aux                (unchanged -- never multiplied by c)
    feature head   : none                 (no loss term)

``c`` is a plain float computed from detached float32 norms; nothing differentiates through
it. If either shared norm is at or below ``NORM_FLOOR``, or non-finite, ``c = 0`` for that
step and the event is counted. Clipping and the optimizer run afterwards on the assembled
gradients exactly as the adopted trainer does.

With ``c = 1`` this is ordinary joint backpropagation of ``L_occ + 0.5 * L_KL``; ``verify.py``
asserts that equality numerically.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from core.model import losses as L
from core.model.net import apply_residual
from tools.teacher_alignment.study import NORM_FLOOR

HEAD_PREFIXES = ("head_occ", "head_sem", "head_feat")


def param_groups(net) -> Dict[str, List[torch.nn.Parameter]]:
    g = {"shared": [], "head_occ": [], "head_sem": [], "head_feat": []}
    for n, p in net.named_parameters():
        for h in HEAD_PREFIXES:
            if n.startswith(h):
                g[h].append(p); break
        else:
            g["shared"].append(p)
    return g


def losses(net, b, cfg, fut_p: torch.Tensor, aux_mask: torch.Tensor, autocast: bool = True):
    """``(l_occ, l_kl, metrics)`` for one batch, with the auxiliary target supplied by the
    caller (aligned or shuffled) and the common auxiliary mask already applied.

    ``autocast=False`` runs the forward in fp32; training never does, the exactness check
    in ``verify.py`` does."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        occ_res, sem_log = net(b["input"])
    occ_res = occ_res[:, 0].float(); sem_log = sem_log.float()
    final = apply_residual(b["base_logodds"], occ_res)
    w = L.balanced_weights(b["gt_occ"], b["gt_valid"], b["observed"], cfg["w_unknown"])
    n_pos = (b["gt_occ"] * b["gt_valid"].float()).sum()
    n_neg = b["gt_valid"].float().sum() - n_pos
    pw = float(min(cfg["pos_weight_cap"], max(1.0, (n_neg / n_pos.clamp_min(1)).item())))
    l_focal = L.focal_bce(final, b["gt_occ"], w, cfg["focal_gamma"], pw)
    l_dice = L.soft_dice(final, b["gt_occ"], b["gt_valid"].float())
    l_occ = cfg["w_focal"] * l_focal + cfg["w_dice"] * l_dice
    l_kl = L.semantic_kl(sem_log, fut_p, aux_mask.float())
    with torch.no_grad():
        gt = (b["gt_occ"] > 0) & b["gt_valid"]
        pred = (final > 0) & b["gt_valid"]
        base = (b["base_logodds"] > 0) & b["gt_valid"]

        def iou(p):
            tp = (p & gt).sum().item(); fp = (p & ~gt).sum().item(); fn = (~p & gt).sum().item()
            return tp / max(tp + fp + fn, 1)
        # top-1 agreement of the semantic head with the target it is being trained on
        agree = float(((sem_log.argmax(1) == fut_p.argmax(1)) & aux_mask).sum()
                      / aux_mask.sum().clamp_min(1))
    m = {"focal": l_focal.item(), "dice": l_dice.item(), "occ": l_occ.item(),
         "kl": l_kl.item(), "n_aux_voxels": int(aux_mask.sum()), "aux_top1_agree": agree,
         "iou": iou(pred), "base_iou": iou(base),
         "density": float(pred.sum().item() / max(b["gt_valid"].sum().item(), 1))}
    return l_occ, l_kl, m


def _norm(ts: List[torch.Tensor]) -> float:
    return float(torch.sqrt(sum((t.float() ** 2).sum() for t in ts)))


def _dot(a: List[torch.Tensor], b: List[torch.Tensor]) -> float:
    return float(sum((x.float() * y.float()).sum() for x, y in zip(a, b)))


def task_gradients(l_occ, l_kl, groups, w_aux: float = 0.5):
    """Detached float32 task gradients over every parameter group.

    Both calls read the same graph; the first keeps it alive for the second.
    """
    allp = [p for g in ("shared", "head_occ", "head_sem") for p in groups[g]]
    g_occ = torch.autograd.grad(l_occ, allp, retain_graph=True, allow_unused=True)
    g_aux = torch.autograd.grad(w_aux * l_kl, allp, retain_graph=False, allow_unused=True)
    z = lambda gs: [torch.zeros_like(p) if g is None else g.detach().float()
                    for g, p in zip(gs, allp)]
    g_occ, g_aux = z(g_occ), z(g_aux)
    n_sh = len(groups["shared"]); n_ho = len(groups["head_occ"])
    return {"shared": (g_occ[:n_sh], g_aux[:n_sh]),
            "head_occ": (g_occ[n_sh:n_sh + n_ho], g_aux[n_sh:n_sh + n_ho]),
            "head_sem": (g_occ[n_sh + n_ho:], g_aux[n_sh + n_ho:])}


def assemble(groups, tg, rho: float) -> Dict[str, float]:
    """Write the matched gradients into ``.grad`` and return the step's diagnostics."""
    go, ga = tg["shared"]
    n_occ, n_aux = _norm(go), _norm(ga)
    finite = all(torch.isfinite(t).all() for t in go) and all(torch.isfinite(t).all() for t in ga)
    degenerate = (not finite) or n_occ <= NORM_FLOOR or n_aux <= NORM_FLOOR
    c = 0.0 if degenerate else rho * n_occ / n_aux
    cos = 0.0 if (n_occ <= NORM_FLOOR or n_aux <= NORM_FLOOR) else _dot(go, ga) / (n_occ * n_aux)
    for p, a, b in zip(groups["shared"], go, ga):
        p.grad = a + c * b
    for p, a, _ in zip(groups["head_occ"], *tg["head_occ"]):
        p.grad = a.clone()
    for p, _, b in zip(groups["head_sem"], *tg["head_sem"]):
        p.grad = b.clone()
    for p in groups["head_feat"]:
        p.grad = None
    achieved = 0.0 if n_occ <= NORM_FLOOR else c * n_aux / n_occ
    return {"occ_grad_norm": n_occ, "aux_grad_norm": n_aux,
            "raw_ratio": (n_aux / n_occ if n_occ > NORM_FLOOR else float("nan")),
            "achieved_ratio": achieved, "norm_factor": c, "grad_cosine": cos,
            "degenerate": int(degenerate), "finite": int(finite),
            "head_sem_grad_norm": _norm(tg["head_sem"][1]),
            "head_occ_grad_norm": _norm(tg["head_occ"][0])}


__all__ = ["param_groups", "losses", "task_gradients", "assemble", "HEAD_PREFIXES"]
