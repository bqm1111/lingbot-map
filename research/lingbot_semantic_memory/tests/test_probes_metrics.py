"""Probe capacity/normalisation and metric behaviour on known tensors."""

import math
import pytest
import torch

from research.lingbot_semantic_memory.metrics import (
    aggregate, boundary_stats, cross_view_stats, feature_diversity, teacher_fidelity,
)
from research.lingbot_semantic_memory.probes import (
    SemanticProbe, build_matched_probe, cosine_distillation_loss, probe_param_count,
    solve_hidden,
)
from research.lingbot_semantic_memory.reprojection import Correspondences

D_OUT = 768


def test_probe_output_is_l2_normalised():
    p = SemanticProbe(1024, D_OUT, 64)
    y = p(torch.randn(7, 5, 1024))
    assert y.shape == (7, 5, D_OUT)
    assert torch.allclose(y.norm(dim=-1), torch.ones(7, 5), atol=1e-5)


def test_param_count_formula_matches_the_module():
    for d_in, hidden in ((1024, 97), (2048, 33)):
        p = SemanticProbe(d_in, D_OUT, hidden)
        assert p.num_params == probe_param_count(d_in, hidden, D_OUT)


def test_capacity_is_matched_across_input_widths():
    """The whole comparison rests on this: 1024-d and 2048-d probes get equal budgets."""
    target, cap = 2_000_000, 10_000_000
    counts = []
    for d_in in (1024, 2048):
        probe, hidden = build_matched_probe(d_in, D_OUT, target, cap)
        assert probe.num_params <= target
        counts.append(probe.num_params)
    lo, hi = min(counts), max(counts)
    assert (hi - lo) / hi < 0.002, f"probe budgets differ by {(hi-lo)/hi:.4%}: {counts}"


def test_capacity_stays_under_the_ten_million_cap():
    for d_in in (1024, 2048):
        probe, _ = build_matched_probe(d_in, D_OUT, 2_000_000, 10_000_000)
        assert probe.num_params < 10_000_000


def test_oversized_target_is_rejected():
    with pytest.raises(ValueError):
        build_matched_probe(2048, D_OUT, 50_000_000, 10_000_000)


def test_solve_hidden_is_the_largest_admissible_width():
    d_in, target = 2048, 2_000_000
    h = solve_hidden(d_in, D_OUT, target)
    assert probe_param_count(d_in, h, D_OUT) <= target
    assert probe_param_count(d_in, h + 1, D_OUT) > target


def test_distillation_loss_is_zero_for_a_perfect_prediction():
    t = torch.randn(64, D_OUT)
    p = t / t.norm(dim=-1, keepdim=True)
    assert cosine_distillation_loss(p, t).item() == pytest.approx(0.0, abs=1e-6)
    assert cosine_distillation_loss(-p, t).item() == pytest.approx(2.0, abs=1e-6)


def test_teacher_fidelity_on_identical_features():
    t = torch.randn(256, D_OUT)
    m = teacher_fidelity(t / t.norm(dim=-1, keepdim=True), t)
    assert m["cos_mean"] == pytest.approx(1.0, abs=1e-5)
    assert m["cos_median"] == pytest.approx(1.0, abs=1e-5)
    assert m["recon_loss"] == pytest.approx(0.0, abs=1e-5)
    assert m["cos_centered"] == pytest.approx(1.0, abs=1e-4)


def test_diversity_is_zero_for_a_collapsed_representation():
    collapsed = torch.ones(500, 16)
    assert feature_diversity(collapsed) == pytest.approx(0.0, abs=1e-5)
    torch.manual_seed(0)
    spread = torch.randn(5000, 16)
    assert feature_diversity(spread) > 0.9


def test_cross_view_margin_exposes_a_collapsed_representation():
    """A constant feature scores 1.0 on raw cosine but 0.0 on margin and chance recall."""
    n_patch, d = 64, 32
    corr = Correspondences(torch.arange(n_patch), torch.arange(n_patch),
                           torch.ones(n_patch), n_patch)
    collapsed = torch.ones(n_patch, d)
    s = cross_view_stats(collapsed, collapsed, corr, num_negatives=64,
                         generator=torch.Generator().manual_seed(0))
    assert s["pos"] == pytest.approx(1.0, abs=1e-5)
    assert s["margin"] == pytest.approx(0.0, abs=1e-5)

    torch.manual_seed(0)
    informative = torch.randn(n_patch, d)
    s2 = cross_view_stats(informative, informative, corr, num_negatives=64,
                          generator=torch.Generator().manual_seed(0))
    assert s2["pos"] == pytest.approx(1.0, abs=1e-5)
    assert s2["margin"] > 0.8
    assert s2["recall@1"] == pytest.approx(1.0)
    assert s2["variance"] == pytest.approx(0.0, abs=1e-5)


def test_cross_view_returns_none_without_correspondences():
    empty = Correspondences(torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long),
                            torch.zeros(0), 64)
    assert cross_view_stats(torch.randn(64, 8), torch.randn(64, 8), empty) is None


def test_aggregate_weights_by_correspondence_count():
    rows = [{"pos": 1.0, "n": 90}, {"pos": 0.0, "n": 10}]
    out = aggregate(rows)
    assert out["pos"] == pytest.approx(0.9)
    assert out["n_total"] == pytest.approx(100.0)
    assert out["n_pairs"] == pytest.approx(2.0)
    assert aggregate([]) == {}


def test_boundary_margin_is_positive_for_label_aligned_features():
    gh, gw, d = 4, 8, 16
    torch.manual_seed(0)
    labels = torch.zeros(gh, gw, dtype=torch.uint8)
    labels[:, gw // 2:] = 1
    protos = torch.randn(2, d)
    feat = protos[labels.reshape(-1).long()] + 0.01 * torch.randn(gh * gw, d)
    s = boundary_stats(feat, labels.reshape(-1), (gh, gw))
    assert s["within"] > 0.99 and s["across"] < 0.5 and s["margin"] > 0.5

    flat = torch.ones(gh * gw, d)
    s2 = boundary_stats(flat, labels.reshape(-1), (gh, gw))
    assert s2["margin"] == pytest.approx(0.0, abs=1e-5)


def test_boundary_ignores_unlabelled_patches():
    gh, gw, d = 4, 8, 16
    labels = torch.full((gh * gw,), 255, dtype=torch.uint8)
    assert boundary_stats(torch.randn(gh * gw, d), labels, (gh, gw)) is None
