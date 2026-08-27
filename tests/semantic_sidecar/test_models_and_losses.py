"""Sidecar architectures, the frozen-gradient guarantee, and the loss terms."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_sidecar.config import LossConfig, ModelConfig
from semantic_sidecar.losses import (
    SidecarLoss,
    cosine_loss,
    multiview_consistency_loss,
    relational_loss,
)
from semantic_sidecar.models import RidgeAccumulator, build_sidecar, parameter_report
from semantic_sidecar.models.linear_sidecar import LinearSidecar


@pytest.mark.parametrize("arch,limit", [("linear", 1_000_000), ("mlp", 20_000_000)])
def test_parameter_budget_and_output_shape(arch, limit):
    model = build_sidecar(ModelConfig(arch=arch), num_layers=2)
    report = parameter_report(model, 1_157_943_540)
    assert report["sidecar_trainable"] < limit
    assert report["trainable_fraction_pct"] < 2.0
    out = model(torch.randn(5, 11, 2, 2048))
    assert out.shape == (5, 11, 64)
    assert torch.allclose(out.norm(dim=-1), torch.ones(5, 11), atol=1e-5)


def test_wrong_input_shape_is_rejected():
    model = build_sidecar(ModelConfig(arch="mlp"), num_layers=2)
    with pytest.raises(ValueError):
        model(torch.randn(4, 3, 1024))


class _FakeLingBot(nn.Module):
    """Stands in for the frozen model: parameters that must never receive gradients."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(2048, 2048)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        with torch.no_grad():
            return self.backbone(x)


def test_no_gradient_reaches_the_frozen_model():
    """Fails if any LingBot-side parameter is trainable or accumulates a gradient."""
    frozen = _FakeLingBot()
    sidecar = build_sidecar(ModelConfig(arch="mlp"), num_layers=1)
    tokens = frozen(torch.randn(16, 2048)).unsqueeze(1)  # [N, 1, 2048]
    target = F.normalize(torch.randn(16, 64), dim=-1)
    loss = cosine_loss(sidecar(tokens), target)
    loss.backward()

    for name, p in frozen.named_parameters():
        assert not p.requires_grad, f"{name} is trainable"
        assert p.grad is None, f"{name} received a gradient"
    assert any(p.grad is not None for p in sidecar.parameters())


def test_detached_features_carry_no_graph():
    x = torch.randn(4, 2048, requires_grad=True)
    captured = x.detach()
    assert not captured.requires_grad
    sidecar = build_sidecar(ModelConfig(arch="linear"), num_layers=1)
    sidecar(captured.unsqueeze(1)).sum().backward()
    assert x.grad is None


def test_ridge_recovers_a_known_linear_map():
    torch.manual_seed(0)
    in_dim, out_dim, n = 32, 8, 4000
    true_w = torch.randn(out_dim, in_dim) * 0.1
    x = torch.randn(n, in_dim)
    y = x @ true_w.T
    acc = RidgeAccumulator(in_dim, out_dim, device="cpu", dtype=torch.float64)
    for i in range(0, n, 500):
        acc.add(x[i : i + 500], y[i : i + 500])
    w = acc.solve(ridge=1e-8)
    assert torch.allclose(w, true_w, atol=1e-3)

    model = LinearSidecar(in_dim, out_dim)
    model.load_ridge(w)
    pred = model(x[:16])
    assert torch.allclose(pred, F.normalize(y[:16], dim=-1), atol=1e-3)


def test_ridge_rejects_wrong_shape():
    model = LinearSidecar(16, 4)
    with pytest.raises(ValueError):
        model.load_ridge(torch.zeros(4, 8))


def test_cosine_loss_masking():
    pred = F.normalize(torch.randn(6, 4), dim=-1)
    assert float(cosine_loss(pred, pred)) == pytest.approx(0.0, abs=1e-6)
    assert float(cosine_loss(pred, -pred)) == pytest.approx(2.0, abs=1e-6)
    mask = torch.tensor([True, False, True, False, True, False])
    target = pred.clone()
    target[~mask] = -target[~mask]
    assert float(cosine_loss(pred, target, mask)) == pytest.approx(0.0, abs=1e-6)


def test_multiview_loss_is_zero_for_consistent_predictions():
    base = F.normalize(torch.randn(3, 8), dim=-1)
    ids = torch.tensor([0, 0, 1, 1, 2, 2])
    pred = base[ids]
    assert float(multiview_consistency_loss(pred, ids)) == pytest.approx(0.0, abs=1e-6)

    scattered = F.normalize(torch.randn(6, 8), dim=-1)
    assert float(multiview_consistency_loss(scattered, ids)) > 0.1
    # Singleton tracks carry no pairwise signal and must not contribute.
    assert float(multiview_consistency_loss(scattered, torch.arange(6))) == pytest.approx(0.0, abs=1e-6)
    assert float(multiview_consistency_loss(scattered, torch.full((6,), -1))) == pytest.approx(0.0)


def test_multiview_loss_scales_linearly_with_track_size():
    """A large track must not trigger quadratic pair enumeration."""
    import time

    pred = F.normalize(torch.randn(20000, 64), dim=-1)
    ids = torch.zeros(20000, dtype=torch.long)
    t0 = time.time()
    multiview_consistency_loss(pred, ids)
    assert time.time() - t0 < 1.0


def test_relational_loss_zero_on_identical_structure():
    torch.manual_seed(0)
    y = F.normalize(torch.randn(64, 16), dim=-1)
    assert float(relational_loss(y, y, group_size=32, num_groups=2)) == pytest.approx(0.0, abs=1e-6)
    assert float(relational_loss(F.normalize(torch.randn(64, 16), dim=-1), y, 32, 2)) > 0.0


def test_sidecar_loss_reports_every_term():
    cfg = LossConfig(lambda_pixel=1.0, lambda_consensus=1.0, lambda_mv=0.5, lambda_rel=0.1)
    loss = SidecarLoss(cfg)
    pred = F.normalize(torch.randn(32, 16), dim=-1).requires_grad_(True)
    target = F.normalize(torch.randn(32, 16), dim=-1)
    mask = torch.ones(32, dtype=torch.bool)
    terms = loss(pred, target, mask, target, mask, torch.randint(0, 4, (32,)))
    assert set(terms) == {"pixel", "consensus", "mv", "rel", "total"}
    terms["total"].backward()
    assert pred.grad is not None


def test_sidecar_loss_requires_the_targets_it_weights():
    loss = SidecarLoss(LossConfig(lambda_consensus=1.0))
    with pytest.raises(ValueError):
        loss(torch.randn(4, 8))


def test_centered_cosine_separates_a_collapsed_model():
    """The plain cosine cannot; the centred one must."""
    from semantic_sidecar.losses import centered_cosine_loss

    torch.manual_seed(0)
    mu = F.normalize(torch.randn(32), dim=-1)
    # Targets concentrated around mu with mean cosine ~0.87, matching the measured
    # anisotropy of the real consensus targets.
    target = F.normalize(11.0 * mu + torch.randn(256, 32), dim=-1)
    assert float((target @ mu).mean()) > 0.85
    collapsed = mu.expand(256, 32)

    plain_collapsed = float(cosine_loss(collapsed, target))
    assert plain_collapsed < 0.15, "the plain cosine should look deceptively good"
    assert float(centered_cosine_loss(collapsed, target, mu)) > 0.8
    assert float(centered_cosine_loss(target, target, mu)) == pytest.approx(0.0, abs=1e-5)


def test_center_term_requires_a_mean_direction():
    from semantic_sidecar.losses import SidecarLoss

    with pytest.raises(ValueError, match="mean_direction"):
        SidecarLoss(LossConfig(lambda_center=1.0))

    loss = SidecarLoss(LossConfig(lambda_consensus=1.0, lambda_center=0.5),
                       mean_direction=F.normalize(torch.randn(16), dim=-1))
    pred = F.normalize(torch.randn(8, 16), dim=-1).requires_grad_(True)
    target = F.normalize(torch.randn(8, 16), dim=-1)
    terms = loss(pred, consensus_target=target, consensus_mask=torch.ones(8, dtype=torch.bool))
    assert "center" in terms
    terms["total"].backward()
    assert pred.grad is not None
