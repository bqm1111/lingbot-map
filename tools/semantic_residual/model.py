"""B5: an additive correction to B0's nearest-evidence feature.

The whole model is one equation::

    corrected = b0_feature + residual_head(context)
    prediction = normalize(corrected) @ text.T        # the frozen readout, unchanged

``b0_feature`` is a unit vector in the RAW Trident-H space (the space the frozen text embeddings live in), so with a
zero residual ``normalize(b0 + 0) == b0`` and the prediction is B0's, voxel for voxel. That identity is the point of
the design and is asserted in ``tests/semantic_residual``.

**Why this is not B4.** B4 (``tools/semantic_field/model/semantic_field_head.py``) predicts the feature outright --
its own docstring calls it "a free vector, not a combination of input features" -- and its ``semantic_head`` is a plain
``nn.Linear`` with default initialisation. Its ``ResidualBlock``s are internal convolutional residuals, which give the
network depth, not a path from B0 to the output. B5 adds the missing skip and starts from exactly B0.

**Where spatial context enters, and its receptive field.** The head itself is a per-voxel MLP: it mixes nothing across
voxels. All spatial context arrives through ``trunk``, the frozen CompletionUNet decoder feature already computed for
the geometry, whose receptive field is that network's (three resolutions over the 256x256x32 grid). This is deliberate
for a first pilot: no attention, no new geometry backbone, nothing that would make a positive result ambiguous about
where the information came from.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.semantic_residual.config import ResidualConfig


class QueryContext(dict):
    """Per-query inputs, all ``[N, ...]`` on one device, all available causally at time t.

    b0_feature  [N, 1024] unit, raw teacher space -- the feature B0 would use (nearest evidence voxel)
    trunk       [N, 24]   frozen CompletionUNet decoder feature at the query voxel
    geometry    [N, 3]    hard map S, hard map F, frozen occupancy O at the query voxel
    delta       [N, 3]    (evidence centre - query centre) / range_norm_m, in the anchor's velodyne frame
    distance    [N, 1]    |delta|, same normalisation
    valid       [N, 1]    1.0 where a nearest evidence voxel exists at all
    log_count   [N, 1]    log1p(ray count of that evidence voxel) / 5, matching the pilot's convention
    """

    REQUIRED = ("b0_feature", "trunk", "geometry", "delta", "distance", "valid", "log_count")

    def check(self, cfg: ResidualConfig) -> None:
        missing = [k for k in self.REQUIRED if k not in self]
        if missing:
            raise KeyError(f"QueryContext is missing {missing}")
        n = len(self["b0_feature"])
        shapes = {"b0_feature": cfg.teacher_dim, "trunk": cfg.trunk_channels, "geometry": cfg.geometry_channels,
                  "delta": 3, "distance": 1, "valid": 1, "log_count": 1}
        for k, want in shapes.items():
            got = self[k]
            if got.shape != (n, want):
                raise ValueError(f"QueryContext['{k}'] has shape {tuple(got.shape)}, expected {(n, want)}. "
                                 f"A teacher-dimension mismatch here usually means a different teacher cache.")


class ResidualHead(nn.Module):
    """Predicts a correction in the raw teacher space. Zero at initialisation, so the model starts as B0."""

    def __init__(self, cfg: ResidualConfig = ResidualConfig()):
        super().__init__()
        self.cfg = cfg
        self.project_b0 = nn.Linear(cfg.teacher_dim, cfg.b0_projection)
        cin = cfg.b0_projection + cfg.trunk_channels + cfg.geometry_channels + cfg.evidence_context
        layers = [nn.Linear(cin, cfg.width), nn.GELU()]
        for _ in range(cfg.depth - 1):
            layers += [nn.Linear(cfg.width, cfg.width), nn.GELU()]
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(cfg.width, cfg.teacher_dim)
        # Only the OUTPUT projection is zeroed: the body keeps its normal initialisation, so gradients are well scaled
        # from the first step. Zeroing the whole network would make it start dead, not start at B0.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def residual(self, ctx: QueryContext) -> torch.Tensor:
        """``[N, 1024]`` correction, unnormalised."""
        ctx.check(self.cfg)
        x = torch.cat([self.project_b0(ctx["b0_feature"]), ctx["trunk"], ctx["geometry"],
                       ctx["delta"], ctx["distance"], ctx["valid"], ctx["log_count"]], dim=-1)
        return self.out(self.body(x))

    def corrected_feature(self, ctx: QueryContext) -> torch.Tensor:
        """``[N, 1024]`` unit feature for the frozen readout. At initialisation this equals ``b0_feature``.

        The correction is gated by ``valid``: with no evidence anywhere there is no B0 feature to correct, and the
        caller's existing B0 fallback (the empty class) must stand. Writing the gate here keeps that guarantee in the
        model rather than relying on every call site to remember it."""
        r = self.residual(ctx) * ctx["valid"]
        return F.normalize(ctx["b0_feature"] + r, dim=-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def config(self) -> Dict:
        return {"residual": asdict(self.cfg)}


__all__ = ["QueryContext", "ResidualHead"]
