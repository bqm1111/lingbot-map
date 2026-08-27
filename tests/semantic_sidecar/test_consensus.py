"""Consensus estimator behaviour on controlled toy data."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from semantic_sidecar.config import ConsensusConfig
from semantic_sidecar.consensus import compute_consensus, observation_weights, segment_ids


def _toy(num_tracks=3, obs_per_track=5, dim=8, noise=0.15, seed=0):
    """Each track has a clean direction; observations are noisy copies of it."""
    g = torch.Generator().manual_seed(seed)
    clean = F.normalize(torch.randn(num_tracks, dim, generator=g), dim=-1)
    ptr = torch.arange(num_tracks + 1, dtype=torch.int64) * obs_per_track
    seg = segment_ids(ptr)
    obs = F.normalize(clean[seg] + noise * torch.randn(len(seg), dim, generator=g), dim=-1)
    conf = torch.full((len(seg),), 2.0)
    residual = torch.zeros(len(seg))
    cos = torch.ones(len(seg))
    has = torch.ones(len(seg), dtype=torch.bool)
    return clean, obs, ptr, conf, residual, cos, has


@pytest.mark.parametrize("estimator", ["mean", "geometric", "robust"])
def test_output_is_normalised(estimator):
    clean, obs, ptr, conf, res, cos, has = _toy()
    out = compute_consensus(obs, ptr, conf, res, cos, has, ConsensusConfig(estimator=estimator))
    norms = out.features[out.valid].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert bool(out.valid.all())


@pytest.mark.parametrize("estimator", ["mean", "geometric", "robust"])
def test_consensus_beats_any_single_observation(estimator):
    """Fusing many noisy views must land closer to the clean signal than one view."""
    clean, obs, ptr, conf, res, cos, has = _toy(obs_per_track=9, noise=0.35, seed=3)
    out = compute_consensus(obs, ptr, conf, res, cos, has, ConsensusConfig(estimator=estimator))
    seg = segment_ids(ptr)
    single = (obs * clean[seg]).sum(-1).mean()
    fused = (out.features * clean).sum(-1).mean()
    assert float(fused) > float(single) + 0.05


def test_corrupted_observations_are_downweighted():
    """A view with bad geometry must move the result less than under a plain mean."""
    clean, obs, ptr, conf, res, cos, has = _toy(num_tracks=1, obs_per_track=6, noise=0.05, seed=5)
    corrupt = F.normalize(-clean[0] + 0.01 * torch.randn(8), dim=-1)
    obs = obs.clone()
    obs[-1] = corrupt
    # The corrupted view is flagged by geometry alone: high reprojection residual,
    # low confidence, grazing angle.  No semantic information is used.
    res = res.clone(); res[-1] = 1.0
    conf = conf.clone(); conf[-1] = 1.01
    cos = cos.clone(); cos[-1] = 0.05

    unweighted = compute_consensus(obs, ptr, conf, res, cos, has, ConsensusConfig(estimator="mean"))
    weighted = compute_consensus(obs, ptr, conf, res, cos, has, ConsensusConfig(estimator="geometric"))
    robust = compute_consensus(obs, ptr, conf, res, cos, has, ConsensusConfig(estimator="robust"))

    sim_u = float((unweighted.features[0] * clean[0]).sum())
    sim_w = float((weighted.features[0] * clean[0]).sum())
    sim_r = float((robust.features[0] * clean[0]).sum())
    assert sim_w > sim_u
    assert sim_r >= sim_w - 1e-6


def test_robust_estimator_records_rejections():
    clean, obs, ptr, conf, res, cos, has = _toy(num_tracks=1, obs_per_track=8, noise=0.05, seed=11)
    obs = obs.clone()
    obs[-1] = F.normalize(-clean[0], dim=-1)
    out = compute_consensus(
        obs, ptr, conf, res, cos, has, ConsensusConfig(estimator="robust", agreement_floor=0.1)
    )
    assert int(out.n_rejected[0]) >= 1
    assert float(out.dispersion[0]) > 0


def test_determinism():
    args = _toy(seed=7)
    cfg = ConsensusConfig(estimator="robust")
    a = compute_consensus(args[1], args[2], args[3], args[4], args[5], args[6], cfg)
    b = compute_consensus(args[1], args[2], args[3], args[4], args[5], args[6], cfg)
    assert torch.equal(a.features, b.features)
    assert torch.equal(a.valid, b.valid)


def test_sparse_teacher_marks_tracks_invalid():
    """A track with no teacher evidence is invalid, never silently filled."""
    clean, obs, ptr, conf, res, cos, has = _toy(num_tracks=3, obs_per_track=4)
    has = has.clone()
    has[: 4] = False  # track 0 loses all of its teacher observations
    out = compute_consensus(obs, ptr, conf, res, cos, has, ConsensusConfig(), min_teacher_obs=1)
    assert not bool(out.valid[0])
    assert bool(out.valid[1]) and bool(out.valid[2])
    assert float(out.features[0].abs().sum()) == 0.0
    assert int(out.n_teacher_obs[0]) == 0


def test_pixelwise_fallback_only_when_configured():
    clean, obs, ptr, conf, res, cos, has = _toy(num_tracks=2, obs_per_track=4)
    has = has.clone()
    has[:4] = False
    fallback = F.normalize(torch.randn(2, 8), dim=-1)
    out = compute_consensus(
        obs, ptr, conf, res, cos, has,
        ConsensusConfig(fallback="pixelwise"), min_teacher_obs=1, pixel_fallback=fallback,
    )
    assert bool(out.valid[0])
    assert torch.allclose(out.features[0], fallback[0], atol=1e-5)


def test_min_teacher_obs_threshold():
    clean, obs, ptr, conf, res, cos, has = _toy(num_tracks=2, obs_per_track=4)
    has = has.clone()
    has[:4] = torch.tensor([True, False, False, False])
    out = compute_consensus(obs, ptr, conf, res, cos, has, ConsensusConfig(), min_teacher_obs=2)
    assert not bool(out.valid[0])
    assert bool(out.valid[1])


def test_weights_are_class_agnostic_and_monotone():
    cfg = ConsensusConfig(conf_power=1.0, residual_tau=0.05, angle_power=1.0)
    conf = torch.tensor([1.5, 3.0, 1.5, 1.5])
    res = torch.tensor([0.0, 0.0, 0.5, 0.0])
    cos = torch.tensor([1.0, 1.0, 1.0, 0.1])
    w = observation_weights(conf, res, cos, cfg)
    assert w[1] > w[0]      # higher confidence
    assert w[0] > w[2]      # lower reprojection residual
    assert w[0] > w[3]      # more head-on view
    assert bool((w > 0).all())


def test_roundtrip(tmp_path):
    clean, obs, ptr, conf, res, cos, has = _toy()
    out = compute_consensus(obs, ptr, conf, res, cos, has, ConsensusConfig())
    out.save(str(tmp_path), "scene")

    from semantic_sidecar.consensus import ConsensusTargets

    loaded = ConsensusTargets.load(str(tmp_path), "scene")
    assert torch.allclose(loaded.features, out.features, atol=1e-3)
    assert torch.equal(loaded.valid, out.valid)
    assert loaded.estimator == out.estimator
