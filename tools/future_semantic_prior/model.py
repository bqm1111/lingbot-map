"""The Gate 21 head: 18 class logits per voxel, from frozen causal context only.

The body is Gate 15's ``PropagationHead`` body, imported unchanged -- four residual blocks of dilated
3x3x3 convolutions at width 32, dilations 1/2/4/8, receptive-field radius 30 voxels (6 m). Gate 15 showed
that body can carry semantics across completed space; Gate 21 asks a different question of it, so changing
it would confound the answer. Two things are different and both follow from the brief:

* the output is 18 logits, not a 1024-d feature. Section 8 rules out regressing foundation features: B4 and
  B5 already tested that and it made the result ambiguous about where the information came from. Gate 21
  distils ``q_future`` directly, so a gain can only mean the model predicts the right distribution.
* the input channel set is selectable, so C2 (geometry only), C3 (full) and C4 (semantics only) are the
  same body on different information rather than three different networks.

``C1_prior`` is not this module: it is a single learned 18-vector with no input path at all
(``ClassPriorHead``), and its optimum under the Gate 21 loss is the mean training target, which the gate
also computes in closed form and asserts against.

Nothing in here can see the future. ``features.causal_channels`` builds every input channel under a file
audit that forbids the Gate 21 target cache, and the forward signature has no place to put a reveal mask.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from tools.future_semantic_prior import config as G
from tools.semantic_completion.model.propagation_head import DilatedResidualBlock


def channel_spec(control: str) -> Tuple[Tuple[str, int], ...]:
    """The named channels this control is allowed to see, in a fixed order (section 9)."""
    if control == "C2_geometry":
        return G.GEOMETRY_CHANNELS
    if control == "C3_full":
        return G.GEOMETRY_CHANNELS + G.SEMANTIC_CHANNELS
    if control == "C4_semantics":
        keep = tuple(c for c in G.GEOMETRY_CHANNELS if c[0] in G.C4_GEOMETRY_KEEP)
        return keep + G.SEMANTIC_CHANNELS
    if control == "C1_prior":
        return ()
    raise KeyError(f"unknown control {control!r}; expected one of {sorted(G.CONTROLS)}")


def uses_evidence(control: str) -> bool:
    return any(n == "evidence_proj" for n, _ in channel_spec(control))


class SemanticCompletionHead(nn.Module):
    """``[B, cin, X, Y, Z] -> [B, 18, X, Y, Z]`` class logits."""

    def __init__(self, control: str, cfg: G.HeadConfig = G.HeadConfig()):
        super().__init__()
        self.control, self.cfg = control, cfg
        self.spec = channel_spec(control)
        self.cin = sum(w for _, w in self.spec)
        assert self.cin > 0, "SemanticCompletionHead needs at least one input channel"
        #: 1024 -> 32 projection of the sparse unit evidence feature, exactly Gate 15's ``project_evidence``
        self.project_evidence = nn.Linear(cfg.teacher_dim, cfg.evidence_dim) if uses_evidence(control) else None
        self.stem = nn.Conv3d(self.cin, cfg.width, 1)
        self.blocks = nn.Sequential(*[DilatedResidualBlock(cfg.width, d) for d in cfg.dilations])
        self.out = nn.Conv3d(cfg.width, G.N_CLASSES, 1)

    def scatter_evidence(self, index: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        """Sparse unit ``feature [N, 1024]`` at flat ``index [N]`` -> dense ``[evidence_dim, X, Y, Z]``.

        The projection is learned, so the head never sees a class id on the evidence path -- it sees the
        teacher's feature, as every earlier gate's head did."""
        X, Y, Z = (int(d) for d in G.GRID_DIMS)
        dense = torch.zeros(X * Y * Z, self.cfg.evidence_dim, device=feature.device, dtype=feature.dtype)
        if index.numel():
            dense[index] = self.project_evidence(feature)
        return dense.T.reshape(self.cfg.evidence_dim, X, Y, Z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.cin, f"{self.control} expects {self.cin} channels, got {x.shape[1]}"
        return self.out(self.blocks(self.stem(x)))

    def logits_at(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """``[N, 18]`` logits at flat voxel ``index`` of sample 0. One forward, then a gather: the voxels
        that happen to carry a loss are chosen AFTER the prediction, never before it."""
        return self.forward(x)[0].reshape(G.N_CLASSES, -1).T[index]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> Dict:
        return {"control": self.control, "channels": [{"name": n, "width": w} for n, w in self.spec],
                "in_channels": self.cin, "out_channels": G.N_CLASSES, "params": self.n_params(),
                "receptive_field_radius_voxels": 2 * sum(self.cfg.dilations),
                "receptive_field_radius_m": 2 * sum(self.cfg.dilations) * 0.2}


class ClassPriorHead(nn.Module):
    """C1: input-independent class logits. 18 parameters, no spatial path, no evidence path.

    Under the Gate 21 soft cross-entropy this converges to ``softmax(b) = mean_x q_future(x)``, so it
    answers exactly one question: how much of any mIoU gain is available from the training class frequency
    alone, with no knowledge of the voxel at all."""

    def __init__(self, control: str = "C1_prior", cfg: G.HeadConfig = G.HeadConfig()):
        super().__init__()
        self.control, self.cfg, self.spec, self.cin = control, cfg, (), 0
        self.bias = nn.Parameter(torch.zeros(G.N_CLASSES))
        self.project_evidence = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0] if x is not None and x.ndim == 5 else 1
        return self.bias.reshape(1, G.N_CLASSES, 1, 1, 1).expand(b, G.N_CLASSES, *[int(d) for d in G.GRID_DIMS])

    def logits_at(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return self.bias.reshape(1, G.N_CLASSES).expand(len(index), G.N_CLASSES)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> Dict:
        return {"control": self.control, "channels": [], "in_channels": 0, "out_channels": G.N_CLASSES,
                "params": self.n_params(), "receptive_field_radius_voxels": 0, "receptive_field_radius_m": 0.0}


def build_head(control: str, cfg: G.HeadConfig = G.HeadConfig()) -> nn.Module:
    return ClassPriorHead(control, cfg) if control == "C1_prior" else SemanticCompletionHead(control, cfg)


def soft_distillation(logits: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """``- sum_c q_c log p_c`` averaged over voxels, with ``p = softmax(logits)`` (section 8's default loss).

    The targets are never hardened. Equivalent to KL up to the target's own entropy, which is constant in
    the parameters, so the gradient is the same."""
    assert logits.shape == q.shape, f"{tuple(logits.shape)} vs {tuple(q.shape)}"
    if logits.numel() == 0:
        return logits.sum() * 0.0
    return -(q * torch.log_softmax(logits.float(), dim=-1)).sum(-1).mean()


__all__ = ["channel_spec", "uses_evidence", "SemanticCompletionHead", "ClassPriorHead", "build_head",
           "soft_distillation"]
