"""The §4 population decomposition, stated without human annotations.

Every mask here is a flat ``[N_VOXELS]`` bool on the anchor grid. The definitions are the
brief's, and the three that are easy to get wrong are written out:

* ``E_causal`` is *semantic* evidence -- a voxel the causal clip's accepted rays reached AND
  that the frozen hard map marks occupied, i.e. exactly the Gate 15 evidence set. A voxel
  whose geometry the causal clip observed but which carries no teacher token is **not** in
  ``E_causal``; it is a distinct sub-case of ``U``, not an occlusion.
* ``occluded_confirmed`` requires a *trustworthy* causal depth in front of the voxel. A ray
  the frozen gate rejected (low confidence, out of range) proves nothing, so such a voxel is
  ``unresolved``, never occluded. This is the brief's "do not infer physical occlusion from a
  missing teacher output or a ray with no trustworthy depth".
* a voxel observed in *any* allowed causal frame is causally observed, even if occluded at
  the anchor -- so the surface test is taken over the whole clip, not over frame ``t``.

``O_priv`` and ``Z_priv_valid`` come from privileged sensors and are built elsewhere
(``k360.py`` / ``waymo_arm.py``); this module only combines them and partitions the result.
No function here opens a human label.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from core.mapping import depth as D7
from tools.waymo_pilot import config as C

#: the four sub-cases of "no causal semantic evidence", mutually exclusive and exhaustive
SUBCASES: Tuple[str, ...] = ("surface_observed_no_teacher", "outside_every_frustum",
                             "occluded_throughout", "unresolved_visibility")


@dataclass
class CausalObservability:
    """What the causal clip alone establishes about every voxel of the anchor grid."""
    in_frustum: torch.Tensor        # [N] bool, centre inside >= 1 causal view's image at z >= near
    surface: torch.Tensor           # [N] bool, hard-map S: an accepted causal ray ended here
    free: torch.Tensor              # [N] bool, hard-map F: carved free by a causal ray
    evidence: torch.Tensor          # [N] bool, Gate 15 semantic evidence voxel
    occluded_all: torch.Tensor      # [N] bool, in frustum and behind a trustworthy surface in
                                    #          EVERY view where it is in frustum
    n_frustum_views: torch.Tensor   # [N] int16, causal views whose image contains the centre
    n_trusted_views: torch.Tensor   # [N] int16, of those, views with a trustworthy depth on the ray

    def stats(self) -> Dict[str, int]:
        return {k: int(getattr(self, k).sum()) for k in
                ("in_frustum", "surface", "free", "evidence", "occluded_all")}


def grid_centres(device) -> torch.Tensor:
    """``[N, 3]`` float64 voxel centres in the anchor grid frame, flat C order."""
    X, Y, Z = C.GRID_DIMS
    ix, iy, iz = torch.meshgrid(torch.arange(X, device=device), torch.arange(Y, device=device),
                                torch.arange(Z, device=device), indexing="ij")
    c = torch.stack([ix, iy, iz], -1).reshape(-1, 3).double()
    return (c + 0.5) * C.VOXEL_SIZE_M + torch.as_tensor(C.GRID_ORIGIN_M, dtype=torch.float64,
                                                        device=device)


@torch.no_grad()
def causal_observability(centres_grid: torch.Tensor, T_grid_to_cam: np.ndarray,
                         views, S: torch.Tensor, F_: torch.Tensor, evidence: torch.Tensor,
                         band_half_m: float = 0.2) -> CausalObservability:
    """Frustum / occlusion / trust state of every voxel from the causal clip alone.

    ``views`` is a sequence of ``(T_cam_to_anchor [4,4] metric, K [3,3], depth_m [H,W],
    accepted [H,W] bool)`` -- one entry per allowed causal frame, in the anchor camera's
    metric frame. ``T_grid_to_cam`` maps the grid frame to the anchor camera.
    """
    device = centres_grid.device
    n = centres_grid.shape[0]
    T = torch.as_tensor(np.asarray(T_grid_to_cam, np.float64), device=device)
    p_anchor = centres_grid @ T[:3, :3].T + T[:3, 3]             # anchor camera, metres

    n_frustum = torch.zeros(n, dtype=torch.int16, device=device)
    n_trusted = torch.zeros(n, dtype=torch.int16, device=device)
    n_occluded = torch.zeros(n, dtype=torch.int16, device=device)

    for T_c2a, K, depth_m, accepted in views:
        Ta = torch.as_tensor(np.asarray(T_c2a, np.float64), device=device)
        Ri = Ta[:3, :3].T                                        # anchor -> this camera
        p = (p_anchor - Ta[:3, 3]) @ Ri.T
        z = p[:, 2]
        Kt = torch.as_tensor(np.asarray(K, np.float64), device=device)
        H, W = int(depth_m.shape[0]), int(depth_m.shape[1])
        u = Kt[0, 0] * p[:, 0] / z + Kt[0, 2]
        v = Kt[1, 1] * p[:, 1] / z + Kt[1, 2]
        inimg = ((z >= D7.MIN_DEPTH_M) & (z < D7.MAX_DEPTH_M) & (u >= -0.5) & (u < W - 0.5)
                 & (v >= -0.5) & (v < H - 0.5))
        n_frustum += inimg.to(torch.int16)
        idx = inimg.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        uu = torch.round(u[idx]).long().clamp(0, W - 1)
        vv = torch.round(v[idx]).long().clamp(0, H - 1)
        ok = accepted[vv, uu]                                    # trustworthy depth on this ray
        ds = depth_m[vv, uu].to(torch.float64)
        trusted = torch.zeros(n, dtype=torch.bool, device=device)
        trusted[idx] = ok
        n_trusted += trusted.to(torch.int16)
        # behind a trustworthy surface by more than the occupied band -> occluded in this view
        occ = torch.zeros(n, dtype=torch.bool, device=device)
        occ[idx] = ok & (z[idx] > ds + band_half_m)
        n_occluded += occ.to(torch.int16)

    in_frustum = n_frustum > 0
    occluded_all = in_frustum & (n_occluded == n_frustum)
    return CausalObservability(in_frustum=in_frustum, surface=S.bool(), free=F_.bool(),
                               evidence=evidence.bool(), occluded_all=occluded_all,
                               n_frustum_views=n_frustum, n_trusted_views=n_trusted)


def unknown_observability(S: torch.Tensor, F_: torch.Tensor,
                          evidence: torch.Tensor) -> CausalObservability:
    """Observability when the causal clip is unavailable: frusta and occlusion are UNKNOWN.

    ``in_frustum`` and ``occluded_all`` are set false everywhere, which routes every
    non-evidence, non-surface voxel into ``outside_every_frustum``. That label would be a lie,
    so ``measure_anchor`` flags these anchors ``subcases_available = False`` and the
    aggregator drops their sub-case counts instead of pooling them.
    """
    n = S.numel()
    z = torch.zeros(n, dtype=torch.bool, device=S.device)
    return CausalObservability(in_frustum=z, surface=S.bool(), free=F_.bool(),
                               evidence=evidence.bool(), occluded_all=z.clone(),
                               n_frustum_views=torch.zeros(n, dtype=torch.int16, device=S.device),
                               n_trusted_views=torch.zeros(n, dtype=torch.int16, device=S.device))


def diagnostic_masks(obs: CausalObservability) -> Dict[str, torch.Tensor]:
    """Finer cuts that do NOT change the four-way partition, reported alongside it.

    ``causal_free_disagreement`` is the one that matters for reading the result: the causal
    map carved the voxel free, and privileged LiDAR says it is occupied. That is a geometry
    disagreement, not a missing-semantics problem, and it sits inside
    ``unresolved_visibility`` only because no causal surface was measured in front of it.
    Counting it separately keeps a geometry failure from being read as semantic headroom.
    """
    sub = subcase_masks(obs)
    unres = sub["unresolved_visibility"]
    free_dis = unres & obs.free
    out = {"causal_free_disagreement": free_dis,
           "unresolved_not_carved": unres & ~obs.free}
    assert bool(((out["causal_free_disagreement"] | out["unresolved_not_carved"])
                 == unres).all()), "the diagnostic split must partition unresolved_visibility"
    return out


def subcase_masks(obs: CausalObservability) -> Dict[str, torch.Tensor]:
    """Partition "no causal semantic evidence" into the brief's four cases.

    Precedence is deliberate and stated: geometry first. A voxel whose surface an allowed
    causal frame measured is *causally observed* whatever happens at the anchor, so it can
    only be the "surface observed, teacher silent" case -- never an occlusion.
    """
    surface = obs.surface & ~obs.evidence
    outside = ~obs.surface & ~obs.evidence & ~obs.in_frustum
    occl = ~obs.surface & ~obs.evidence & obs.in_frustum & obs.occluded_all
    unres = ~obs.surface & ~obs.evidence & obs.in_frustum & ~obs.occluded_all
    out = {"surface_observed_no_teacher": surface, "outside_every_frustum": outside,
           "occluded_throughout": occl, "unresolved_visibility": unres}
    assert set(out) == set(SUBCASES)
    return out


@dataclass
class Populations:
    """The brief's populations for one anchor, plus the partition of each."""
    O_priv: torch.Tensor            # privileged geometry says occupied
    E_causal: torch.Tensor          # causal semantic evidence
    Z_valid: torch.Tensor           # a reliable privileged semantic target exists
    hidden: torch.Tensor            # O_priv & ~E_causal     (BEFORE requiring a target)
    U: torch.Tensor                 # hidden & Z_valid       (the supervised set)
    subcases: Dict[str, torch.Tensor]
    diagnostics: Dict[str, torch.Tensor] = field(default_factory=dict)

    def counts(self) -> Dict[str, int]:
        out = {"O_priv": int(self.O_priv.sum()), "E_causal": int(self.E_causal.sum()),
               "Z_valid": int(self.Z_valid.sum()), "hidden": int(self.hidden.sum()),
               "U": int(self.U.sum())}
        for k, m in list(self.subcases.items()) + list(self.diagnostics.items()):
            out[f"hidden_{k}"] = int((self.hidden & m).sum())
            out[f"U_{k}"] = int((self.U & m).sum())
        # the four-way partition must still account for every no-evidence voxel
        tot = sum(out[f"hidden_{k}"] for k in SUBCASES)
        assert tot == out["hidden"], f"subcases sum to {tot}, hidden is {out['hidden']}"
        return out


def build(O_priv: torch.Tensor, Z_valid: torch.Tensor, obs: CausalObservability,
          roi: Optional[torch.Tensor] = None) -> Populations:
    """Combine the privileged and causal sides. ``roi`` is the existing task ROI, if any."""
    keep = torch.ones_like(O_priv, dtype=torch.bool) if roi is None else roi.bool()
    O = O_priv.bool() & keep
    hidden = O & ~obs.evidence
    U = hidden & Z_valid.bool()
    return Populations(O_priv=O, E_causal=obs.evidence.bool(), Z_valid=Z_valid.bool(),
                       hidden=hidden, U=U, subcases=subcase_masks(obs),
                       diagnostics=diagnostic_masks(obs))


__all__ = ["SUBCASES", "CausalObservability", "Populations", "grid_centres",
           "causal_observability", "unknown_observability", "subcase_masks",
           "diagnostic_masks", "build"]
