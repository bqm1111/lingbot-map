"""The five depth-evidence variants, as fixed acceptance rules over one frame.

``S0`` is the frozen five-frame comparator and is not produced here -- it is Gate 6's own
pinned prediction, read back unchanged. The other four all consume the same streamed
LingBot output, the same gauge and the same map update, and differ **only** in which rays
they accept and with what reliability:

* ``S1`` LingBot's frozen gate: ``conf >= 1.5`` and ``1 m < s*d < 60 m``;
* ``S2`` the same, with a globally fixed relaxed confidence threshold. The depth range is
  *not* relaxed: the evaluation volume is 51.2 m across, so nothing beyond 60 m can enter
  it and pretending otherwise would be dishonest;
* ``S3`` ``S1`` plus MoGe on the rays ``S1`` rejects, at a lower reliability weight;
* ``S4`` ``S3`` with a continuous, analytic reliability weight and a free-space veto: a
  MoGe candidate is refused where the causal map has already carved that voxel free, and
  is down-weighted smoothly by its disagreement with whatever LingBot depth exists.

Nothing here is learned. Every constant is in the precommit and is identical on all three
benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .depth import CONF_THRESHOLD, MAX_DEPTH_M, MIN_DEPTH_M
from .voxmap import SRC_LINGBOT, SRC_MOGE, W_LINGBOT, W_MOGE

VARIANTS = ("S0", "S1", "S2", "S3", "S4")
#: The globally fixed relaxed-confidence sweep of S2. One set for all three benchmarks.
S2_CONF_SWEEP = (1.5, 1.0, 0.5, 0.25, 0.0)
S2_PRIMARY_CONF = 0.5
#: S4 disagreement scale, in log-depth units: a factor-e disagreement halves the weight
#: about twice over. Fixed, not fitted.
S4_TAU_LOG = 0.5
#: S4 refuses a MoGe candidate where the causal map's log-odds are already below this.
S4_FREE_VETO_L = -0.8
#: Floor on the S4 weight; below it the candidate is dropped rather than fused faintly.
S4_MIN_WEIGHT = 0.05


@dataclass
class Acceptance:
    """One frame's decision, split by evidence source."""
    lingbot: torch.Tensor          # [H, W] bool
    moge: torch.Tensor             # [H, W] bool
    moge_weight: torch.Tensor      # [H, W] float32
    depth_lingbot_m: torch.Tensor  # [H, W] float64, metric
    depth_moge_m: torch.Tensor     # [H, W] float64, metric


def lingbot_gate(depth_canonical: torch.Tensor, conf: torch.Tensor, scale: float,
                 conf_threshold: float = CONF_THRESHOLD,
                 min_depth_m: float = MIN_DEPTH_M,
                 max_depth_m: float = MAX_DEPTH_M) -> Tuple[torch.Tensor, torch.Tensor]:
    d = float(scale) * depth_canonical.to(torch.float64)
    ok = (conf.to(torch.float64) >= conf_threshold) & torch.isfinite(d) \
        & (d > min_depth_m) & (d < max_depth_m)
    return ok, d


def moge_eligible(moge_depth: torch.Tensor, moge_mask: torch.Tensor,
                  min_depth_m: float = MIN_DEPTH_M,
                  max_depth_m: float = MAX_DEPTH_M) -> torch.Tensor:
    d = moge_depth.to(torch.float64)
    return moge_mask & torch.isfinite(d) & (d > min_depth_m) & (d < max_depth_m)


def accept(variant: str, depth_canonical: torch.Tensor, conf: torch.Tensor, scale: float,
           moge_depth: Optional[torch.Tensor], moge_mask: Optional[torch.Tensor],
           conf_threshold: float = CONF_THRESHOLD) -> Acceptance:
    """The variant's acceptance decision for one frame. No map state is consulted here;
    the S4 free-space veto is applied at fusion time, where the causal map exists."""
    zeros = torch.zeros_like(conf, dtype=torch.bool)
    thr = conf_threshold if variant == "S2" else CONF_THRESHOLD
    lb, d_lb = lingbot_gate(depth_canonical, conf, scale, conf_threshold=thr)
    if variant in ("S1", "S2") or moge_depth is None:
        return Acceptance(lingbot=lb, moge=zeros,
                          moge_weight=torch.zeros_like(conf, dtype=torch.float32),
                          depth_lingbot_m=d_lb,
                          depth_moge_m=torch.zeros_like(d_lb))
    d_mg = moge_depth.to(torch.float64)
    elig = moge_eligible(d_mg, moge_mask) & ~lb
    if variant == "S3":
        w = torch.where(elig, torch.full_like(conf, W_MOGE, dtype=torch.float32),
                        torch.zeros_like(conf, dtype=torch.float32))
        return Acceptance(lingbot=lb, moge=elig, moge_weight=w, depth_lingbot_m=d_lb,
                          depth_moge_m=d_mg)
    if variant == "S4":
        # continuous reliability: agreement with whatever LingBot depth exists, and the
        # frame's own confidence, both smooth -- no target-tuned binary thresholds
        have_lb = torch.isfinite(d_lb) & (d_lb > 0)
        dis = torch.where(have_lb, (torch.log(d_mg.clamp_min(1e-6))
                                    - torch.log(d_lb.clamp_min(1e-6))).abs(),
                          torch.zeros_like(d_lb))
        f_agree = torch.where(have_lb, torch.exp(-dis / S4_TAU_LOG),
                              torch.ones_like(dis))
        f_conf = (conf.to(torch.float64) / CONF_THRESHOLD).clamp(0.0, 1.0)
        w = (W_MOGE * (0.5 * f_agree + 0.5 * f_conf)).to(torch.float32)
        w = torch.where(elig, w, torch.zeros_like(w))
        keep = elig & (w >= S4_MIN_WEIGHT)
        return Acceptance(lingbot=lb, moge=keep, moge_weight=w * keep.to(w.dtype),
                          depth_lingbot_m=d_lb, depth_moge_m=d_mg)
    raise KeyError(variant)


def semantic_weight(variant: str, acc: Acceptance, conf: torch.Tensor) -> torch.Tensor:
    """Geometry-reliability weight for the semantic accumulator.

    Gate 7A showed the teacher's own maximum probability does not track propagation
    accuracy, so semantic evidence is weighted by **geometry** reliability and observation
    quality -- LingBot confidence where LingBot supplied the ray, the MoGe reliability
    weight where MoGe did -- never by the teacher's confidence.
    """
    c = (conf.to(torch.float32) / CONF_THRESHOLD).clamp(0.0, 2.0)
    w = torch.where(acc.lingbot, W_LINGBOT * c.clamp(min=0.25), torch.zeros_like(c))
    return torch.where(acc.moge, acc.moge_weight, w)


__all__ = ["VARIANTS", "S2_CONF_SWEEP", "S2_PRIMARY_CONF", "S4_TAU_LOG",
           "S4_FREE_VETO_L", "S4_MIN_WEIGHT", "Acceptance", "accept", "lingbot_gate",
           "moge_eligible", "semantic_weight"]
