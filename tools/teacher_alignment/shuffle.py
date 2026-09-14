"""The controlled shuffle: permute class-probability targets within (z, observed) strata.

Given one training crop with its auxiliary support mask, the future-teacher vectors and the
frozen input map's observed flag, every supervised voxel is assigned to a stratum
``(z layer, observed)``. Within each stratum the *complete* probability vectors are permuted
uniformly at random among the stratum's voxels. Nothing else moves: the support mask, the
occupancy targets, the inputs and the multiset of vectors per stratum are all preserved, so
the coarse priors "what classes live at this height in map-observed vs unobserved space"
survive and only the *local* correspondence between a voxel and its own teacher opinion is
destroyed.

Strata with fewer than two eligible voxels cannot be permuted and are dropped from the
auxiliary loss -- in **both** conditions, so R and S are supervised on identical voxels.

The permutation's randomness comes from its own generator, keyed by ``(training seed,
optimizer step, crop index)``, so it neither consumes nor perturbs the trainer's RNG.

No scalar confidence weight rides with the vector: the adopted KL uses a binary mask
(``core.model.losses.semantic_kl``), so there is nothing to move besides the vector.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


def _gen(seed: int, step: int, crop: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(np.uint64(seed) * np.uint64(1_000_003) + np.uint64(step) * np.uint64(1_009)
                      + np.uint64(crop) + np.uint64(0x5EED)) & 0x7FFF_FFFF_FFFF_FFFF)
    return g


def strata(valid: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    """``[cx, cy, cz]`` int64 stratum id ``2*z + observed`` (only meaningful where valid)."""
    cz = valid.shape[-1]
    z = torch.arange(cz, device=valid.device).view(1, 1, cz).expand_as(valid)
    return 2 * z + observed.long()


def eligible_mask(valid: torch.Tensor, observed: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
    """The common auxiliary mask: supervised voxels in strata of size >= 2.

    Returned for R and S alike. The dict reports how much supervision the size rule dropped.
    """
    sid = strata(valid, observed)
    ids = sid[valid]
    n_strata = 2 * valid.shape[-1]
    cnt = torch.bincount(ids, minlength=n_strata)
    ok = cnt[sid] >= 2
    keep = valid & ok
    n_valid = int(valid.sum())
    return keep, {"n_valid": n_valid, "n_kept": int(keep.sum()),
                  "n_dropped_tiny_strata": int((valid & ~ok).sum())}


@torch.no_grad()
def shuffle_targets(fut_p: torch.Tensor, keep: torch.Tensor, observed: torch.Tensor,
                    seed: int, step: int, crop: int) -> Tuple[torch.Tensor, Dict]:
    """Return a copy of ``fut_p`` ``[U, cx, cy, cz]`` with vectors permuted within strata.

    Only positions in ``keep`` are touched. Statistics describe what the permutation did.
    """
    U = fut_p.shape[0]
    idx = keep.reshape(-1).nonzero(as_tuple=True)[0]                 # eligible flat positions
    n = int(idx.numel())
    out = fut_p.clone()
    stats = {"n_eligible": n, "n_changed": 0, "n_argmax_changed": 0,
             "mean_l1": 0.0, "n_in_homogeneous_strata": 0}
    if n < 2:
        return out, stats
    sid = strata(keep, observed).reshape(-1)[idx]                     # stratum per eligible
    g = _gen(seed, step, crop)
    r = torch.rand(n, generator=g).to(idx.device)
    # both orders group by stratum; the first is deterministic, the second random inside it
    order_det = torch.argsort(sid, stable=True)
    order_rnd = torch.argsort(sid.double() + 0.5 * r, stable=True)
    src = idx[order_rnd]; dst = idx[order_det]
    flat = fut_p.reshape(U, -1)
    new = flat.clone()
    new[:, dst] = flat[:, src]
    out = new.reshape_as(fut_p)
    # ---- measure the manipulation -------------------------------------------------
    before = flat[:, idx].T; after = new[:, idx].T                   # [n, U]
    diff = (before - after).abs().sum(1)
    changed = diff > 1e-6
    stats["n_changed"] = int(changed.sum())
    stats["n_argmax_changed"] = int((before.argmax(1) != after.argmax(1)).sum())
    stats["mean_l1"] = float(diff.mean())
    # a stratum is homogeneous when every vector in it is (numerically) the same
    # -> a permutation there changes nothing; count the supervision that sits in such strata
    n_strata = 2 * keep.shape[-1]
    per = torch.zeros(n_strata, device=idx.device, dtype=torch.float32)
    per.index_add_(0, sid, diff)
    cnt = torch.bincount(sid, minlength=n_strata).float()
    homog = (per <= 1e-6) & (cnt >= 2)
    stats["n_in_homogeneous_strata"] = int(cnt[homog].sum())
    return out, stats


def multiset_preserved(fut_p: torch.Tensor, shuffled: torch.Tensor, keep: torch.Tensor,
                       observed: torch.Tensor) -> bool:
    """Per stratum, the sorted multiset of vectors must be identical before and after."""
    U = fut_p.shape[0]
    idx = keep.reshape(-1).nonzero(as_tuple=True)[0]
    sid = strata(keep, observed).reshape(-1)[idx].cpu().numpy()
    a = fut_p.reshape(U, -1)[:, idx].T.cpu().numpy()
    b = shuffled.reshape(U, -1)[:, idx].T.cpu().numpy()
    for s_ in np.unique(sid):
        m = sid == s_
        x, y = a[m], b[m]
        x = x[np.lexsort(x.T[::-1])]; y = y[np.lexsort(y.T[::-1])]
        if not np.allclose(x, y, atol=1e-6):
            return False
    return True


__all__ = ["strata", "eligible_mask", "shuffle_targets", "multiset_preserved"]
