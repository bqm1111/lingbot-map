"""Metrics: teacher fidelity, cross-view consistency, boundary preservation.

**Anti-degeneracy note.** Mean cross-view cosine is not interpretable on its own: a
probe that emits one constant vector scores 1.0 while carrying no information.  The
earlier sidecar study in this repository hit exactly that failure
(``docs/semantic_sidecar_feasibility_report.md`` §2).  Every cross-view number here is
therefore reported together with

* ``neg``  -- cosine between *non*-corresponding patches of the same frame pair, and
* ``margin`` = ``pos - neg``, and
* ``recall@1`` -- can the true match be retrieved among the destination frame's
  patches by cosine alone.

``margin`` and ``recall@1`` are invariant to the collapse that inflates ``pos``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from research.lingbot_semantic_memory.reprojection import Correspondences


def _norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)


# --------------------------------------------------------------------------- #
# Semantic fidelity
# --------------------------------------------------------------------------- #
def teacher_fidelity(pred: torch.Tensor, teacher: torch.Tensor) -> Dict[str, float]:
    """Cosine agreement between predicted and teacher features.

    Args:
        pred: ``[N, D]`` probe output (already L2-normalised).
        teacher: ``[N, D]`` frozen teacher features.

    Returns:
        Mean/median cosine, the ``1 - cos`` reconstruction loss, and the
        **centred** cosine, which removes the corpus mean direction.  DINO features
        are anisotropic, so raw cosine is optimistic; centred cosine is the part that
        actually discriminates between regions.
    """
    p, t = _norm(pred.float()), _norm(teacher.float())
    cos = (p * t).sum(-1)
    mu = _norm(t.mean(0, keepdim=True))
    pc, tc = _norm(p - (p * mu).sum(-1, keepdim=True) * mu), _norm(t - (t * mu).sum(-1, keepdim=True) * mu)
    return {
        "cos_mean": float(cos.mean()),
        "cos_median": float(cos.median()),
        "recon_loss": float((1.0 - cos).mean()),
        "cos_centered": float((pc * tc).sum(-1).mean()),
    }


def feature_diversity(feat: torch.Tensor, sample: int = 50000,
                      generator: Optional[torch.Generator] = None) -> float:
    """Mean pairwise cosine *distance* between random feature pairs.

    For unit vectors the expected cosine between two independent draws is
    ``||mean||^2``, so this is ``1 - ||mean||^2``: it is 0.0 for a fully collapsed
    representation and grows as the features spread out.  Read every cross-view
    number against this one.
    """
    f = _norm(feat.float())
    n = f.shape[0]
    if n > sample:
        idx = torch.randperm(n, generator=generator)[:sample]
        f = f[idx]
    return float(1.0 - f.mean(0).pow(2).sum())


# --------------------------------------------------------------------------- #
# Cross-view consistency
# --------------------------------------------------------------------------- #
def cross_view_stats(
    feat_src: torch.Tensor, feat_dst: torch.Tensor, corr: Correspondences,
    num_negatives: int = 4096, generator: Optional[torch.Generator] = None,
) -> Optional[Dict[str, float]]:
    """Consistency of one representation over one frame pair's correspondences.

    Args:
        feat_src, feat_dst: ``[P, D]`` features on the patch lattice of each frame.
        corr: correspondences produced by :mod:`reprojection`.
        num_negatives: random non-corresponding pairs for the dispersion control.

    Returns:
        ``None`` when the pair has no valid correspondence, otherwise the positive
        cosine, the negative-control cosine, their margin, the feature variance over
        matched pairs, and retrieval recall@1.
    """
    if len(corr) == 0:
        return None
    a = _norm(feat_src.float()[corr.src_idx])
    b = _norm(feat_dst.float()[corr.dst_idx])
    pos = (a * b).sum(-1)

    n = a.shape[0]
    k = min(num_negatives, n)
    perm = torch.randperm(n, generator=generator, device=a.device)[:k]
    shift = torch.randperm(n, generator=generator, device=a.device)[:k]
    bad = perm == shift
    shift[bad] = (shift[bad] + 1) % n
    neg = (a[perm] * b[shift]).sum(-1)

    # Retrieval: is the geometrically correct patch the nearest neighbour in frame dst?
    sim = a @ _norm(feat_dst.float()).T                      # [N, P]
    recall1 = float((sim.argmax(-1) == corr.dst_idx).float().mean())

    # Variance of the matched pair's feature (mean over dims of the 2-sample variance).
    var = float(((a - b) ** 2).sum(-1).mean() / 2.0)

    return {
        "pos": float(pos.mean()),
        "neg": float(neg.mean()),
        "margin": float(pos.mean() - neg.mean()),
        "variance": var,
        "recall@1": recall1,
        "n": n,
        "valid_fraction": corr.valid_fraction,
    }


def aggregate(rows: Sequence[Dict[str, float]], weight_key: str = "n") -> Dict[str, float]:
    """Correspondence-count-weighted mean of per-pair stats."""
    rows = [r for r in rows if r]
    if not rows:
        return {}
    w = np.array([r.get(weight_key, 1) for r in rows], dtype=np.float64)
    out: Dict[str, float] = {}
    for k in rows[0]:
        if k == weight_key:
            continue
        v = np.array([r[k] for r in rows], dtype=np.float64)
        out[k] = float((v * w).sum() / w.sum()) if w.sum() > 0 else float(v.mean())
    out["n_total"] = float(w.sum())
    out["n_pairs"] = float(len(rows))
    return out


# --------------------------------------------------------------------------- #
# Boundary preservation
# --------------------------------------------------------------------------- #
def boundary_stats_from_depth(
    feat: torch.Tensor, depth_patch: torch.Tensor, grid_hw,
    same_rel: float = 0.02, edge_rel: float = 0.10,
) -> Optional[Dict[str, float]]:
    """Within-surface vs across-depth-discontinuity similarity over adjacent patches.

    A label-free boundary definition usable on any dataset with ground-truth depth: an
    adjacent patch pair is *within-region* when their relative depth differs by less
    than ``same_rel`` and *across-boundary* when it differs by more than ``edge_rel``.
    The gap between the two thresholds drops ambiguous pairs (gently slanted surfaces)
    instead of forcing them into one class.

    This is the metric that makes KITTI and Replica comparable, since the Replica
    release carries no semantic annotation.
    """
    gh, gw = grid_hw
    f = _norm(feat.float()).view(gh, gw, -1)
    d = depth_patch.float().view(gh, gw)

    cos_l, same_l, edge_l = [], [], []
    for da, db in ((0, 1), (1, 0)):
        a = f[: gh - da, : gw - db].reshape(-1, f.shape[-1])
        b = f[da:, db:].reshape(-1, f.shape[-1])
        za = d[: gh - da, : gw - db].reshape(-1)
        zb = d[da:, db:].reshape(-1)
        ok = (za > 0) & (zb > 0) & torch.isfinite(za) & torch.isfinite(zb)
        if not ok.any():
            continue
        rel = (za - zb).abs() / torch.minimum(za, zb).clamp_min(1e-6)
        cos_l.append((a[ok] * b[ok]).sum(-1))
        same_l.append(rel[ok] < same_rel)
        edge_l.append(rel[ok] > edge_rel)
    if not cos_l:
        return None
    cos = torch.cat(cos_l); same = torch.cat(same_l); edge = torch.cat(edge_l)
    if same.sum() == 0 or edge.sum() == 0:
        return None
    within, across = float(cos[same].mean()), float(cos[edge].mean())
    return {"within": within, "across": across, "margin": within - across,
            "n_within": int(same.sum()), "n_across": int(edge.sum())}


def boundary_stats(feat: torch.Tensor, labels: torch.Tensor, grid_hw) -> Optional[Dict[str, float]]:
    """Within-region vs across-boundary similarity over 4-neighbour patch pairs.

    Regions come from the projected SemanticKITTI labels (evaluation only).  A pair of
    adjacent patches is *within-region* when both carry the same class and
    *across-boundary* when the classes differ; 255 (unlabelled) pairs are dropped.

    Returns:
        ``within``, ``across`` and their ``margin``.  A representation that has blurred
        object boundaries shows a small margin.
    """
    gh, gw = grid_hw
    f = _norm(feat.float()).view(gh, gw, -1)
    lab = labels.view(gh, gw)

    pairs = []
    for da, db in ((0, 1), (1, 0)):
        a = f[: gh - da, : gw - db].reshape(-1, f.shape[-1])
        b = f[da:, db:].reshape(-1, f.shape[-1])
        la = lab[: gh - da, : gw - db].reshape(-1)
        lb = lab[da:, db:].reshape(-1)
        ok = (la != 255) & (lb != 255)
        if ok.any():
            pairs.append(((a[ok] * b[ok]).sum(-1), la[ok] == lb[ok]))
    if not pairs:
        return None
    cos = torch.cat([p[0] for p in pairs])
    same = torch.cat([p[1] for p in pairs])
    if same.sum() == 0 or (~same).sum() == 0:
        return None
    within, across = float(cos[same].mean()), float(cos[~same].mean())
    return {"within": within, "across": across, "margin": within - across,
            "n_within": int(same.sum()), "n_across": int((~same).sum())}
