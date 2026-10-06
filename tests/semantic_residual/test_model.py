"""Checks for the properties B5's conclusion depends on.

These are cheap and synthetic on purpose: they pin the contract of the residual head (starts at B0, corrects only where
B0 has something to correct, gradients reach the right parameters). Data-dependent checks -- causal independence of the
targets, correspondence transforms -- live with the target builder and reuse the existing pilot checks.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from tools.semantic_residual.config import ResidualConfig
from tools.semantic_residual.model import QueryContext, ResidualHead


def ctx(n: int = 16, dim: int = 1024, valid: float = 1.0) -> QueryContext:
    g = torch.Generator().manual_seed(0)
    return QueryContext(b0_feature=F.normalize(torch.randn(n, dim, generator=g), dim=-1),
                        trunk=torch.randn(n, 24, generator=g), geometry=torch.rand(n, 3, generator=g),
                        delta=torch.randn(n, 3, generator=g), distance=torch.rand(n, 1, generator=g),
                        valid=torch.full((n, 1), valid), log_count=torch.rand(n, 1, generator=g))


def test_starts_exactly_at_b0():
    """The zero-residual path must reproduce B0, or a 'gain' could just be a different starting point."""
    m = ResidualHead(); c = ctx()
    assert torch.allclose(m.corrected_feature(c), c["b0_feature"], atol=1e-6)


def test_only_the_output_projection_is_zeroed():
    """Zeroing the whole network would start it dead; zeroing only the output starts it at B0 with live gradients."""
    m = ResidualHead()
    assert m.out.weight.abs().max() == 0 and m.out.bias.abs().max() == 0
    assert m.project_b0.weight.abs().max() > 0
    assert any(p.abs().max() > 0 for p in m.body.parameters())


def test_no_evidence_leaves_b0_untouched_even_when_trained():
    """With no evidence anywhere there is no B0 feature to correct: the existing fallback must survive."""
    m = ResidualHead()
    with torch.no_grad():                                   # simulate a trained, non-zero head
        m.out.weight.normal_(0, 0.1); m.out.bias.normal_(0, 0.1)
    c = ctx(valid=0.0)
    assert torch.allclose(m.corrected_feature(c), c["b0_feature"], atol=1e-6)
    assert not torch.allclose(m.corrected_feature(ctx(valid=1.0)), ctx(valid=1.0)["b0_feature"], atol=1e-6)


def test_trained_head_moves_the_prediction():
    """A non-zero residual must actually change the feature, otherwise the skip would be a no-op by construction."""
    m = ResidualHead()
    with torch.no_grad():
        m.out.weight.normal_(0, 0.1)
    c = ctx()
    assert (m.corrected_feature(c) - c["b0_feature"]).abs().max() > 1e-3


def test_output_is_unit_norm_for_the_frozen_readout():
    m = ResidualHead()
    with torch.no_grad():
        m.out.weight.normal_(0, 0.5)
    n = m.corrected_feature(ctx()).norm(dim=-1)
    assert torch.allclose(n, torch.ones_like(n), atol=1e-5)


def test_gradient_starts_at_the_output_then_reaches_everything():
    """Zero-init has a known consequence worth pinning down rather than discovering during training.

    On step 0 the output projection is zero, so d(loss)/d(body output) is zero and only ``out`` receives gradient.
    That is correct and intended -- it is what makes the model start exactly at B0 -- but it means "all parameters have
    gradient" is false at initialisation. Once ``out`` has moved, every parameter trains."""
    torch.manual_seed(0)
    m = ResidualHead(); c = ctx()
    for v in c.values():
        v.requires_grad_(False)
    target = F.normalize(torch.randn(16, 1024, generator=torch.Generator().manual_seed(1)), dim=-1)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)

    loss0 = (1 - (m.corrected_feature(c) * target).sum(-1)).mean()
    loss0.backward()
    assert m.out.weight.grad.abs().max() > 0, "the output projection must train from the first step"
    assert m.project_b0.weight.grad.abs().max() == 0, "with a zero output, earlier layers see no gradient on step 0"
    assert all(v.grad is None for v in c.values()), "frozen inputs must not accumulate gradient"

    opt.step(); opt.zero_grad(set_to_none=True)
    loss1 = (1 - (m.corrected_feature(c) * target).sum(-1)).mean()
    loss1.backward()
    for name, p_ in m.named_parameters():
        assert p_.grad is not None and torch.isfinite(p_.grad).all(), name
        assert p_.grad.abs().max() > 0, f"{name} still receives no gradient after one step"
    assert float(loss1) < float(loss0), "one step on a fixed batch must reduce the loss"


def test_teacher_dimension_mismatch_is_a_clear_error():
    """A silently wrong teacher cache is the failure mode most likely to produce a meaningless number."""
    m = ResidualHead()
    c = ctx(dim=512)
    with pytest.raises(ValueError, match="teacher-dimension|expected"):
        m.corrected_feature(c)


def test_missing_context_field_names_the_field():
    m = ResidualHead(); c = ctx(); del c["trunk"]
    with pytest.raises(KeyError, match="trunk"):
        m.corrected_feature(c)


def test_config_records_the_dimension_actually_used():
    assert ResidualConfig().teacher_dim == 1024
