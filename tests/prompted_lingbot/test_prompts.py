"""Prompt simulation: determinism, causality, and that every corruption knob bites."""

import numpy as np
import pytest

from prompted_lingbot.prompts import (
    PromptConfig, make_prompt, make_prompt_stream, standard_configs, training_config_sampler,
)


def _gt(n=40, h=16, w=16, seed=0):
    rng = np.random.default_rng(seed)
    depth = rng.uniform(2.0, 30.0, size=(n, h, w))
    valid = rng.random((n, h, w)) > 0.1
    poses = np.tile(np.concatenate([np.eye(3), np.zeros((3, 1))], 1), (n, 1, 1))
    poses[:, :3, 3] = np.cumsum(rng.normal(scale=0.3, size=(n, 3)), axis=0)
    return depth, valid, poses


def test_stream_is_deterministic():
    d, v, p = _gt()
    cfg = PromptConfig(name="t", depth_interval=5, pose_interval=5, depth_noise_sigma=0.1,
                       translation_noise_m=0.2, dropout=0.2, depth_outlier_rate=0.1)
    a = make_prompt_stream(cfg, "seq", len(p), d, v, p)
    b = make_prompt_stream(cfg, "seq", len(p), d, v, p)
    for x, y in zip(a, b):
        assert x.has_depth == y.has_depth and x.has_pose == y.has_pose
        if x.has_depth:
            assert np.array_equal(x.depth_values, y.depth_values)
            assert np.array_equal(x.depth_rows, y.depth_rows)
        if x.has_pose:
            assert np.array_equal(x.pose_c2w, y.pose_c2w)


def test_stream_depends_on_the_sequence_name_and_seed():
    d, v, p = _gt()
    cfg = PromptConfig(name="t", depth_interval=1, depth_noise_sigma=0.1)
    a = make_prompt_stream(cfg, "seq_a", len(p), d, v, p)
    b = make_prompt_stream(cfg, "seq_b", len(p), d, v, p)
    c = make_prompt_stream(cfg.with_seed(7), "seq_a", len(p), d, v, p)
    assert not np.array_equal(a[3].depth_values, b[3].depth_values)
    assert not np.array_equal(a[3].depth_values, c[3].depth_values)


def test_prompt_at_t_is_independent_of_every_other_frame():
    """Causality by construction: rebuild frame t with all other GT scrambled."""
    d, v, p = _gt()
    cfg = PromptConfig(name="t", depth_interval=3, pose_interval=3, depth_noise_sigma=0.1,
                       translation_noise_m=0.3, dropout=0.3, depth_missing=0.2)
    ref = make_prompt_stream(cfg, "seq", len(p), d, v, p)

    rng = np.random.default_rng(99)
    d2, v2, p2 = d.copy(), v.copy(), p.copy()
    t = 12
    keep = (d[t].copy(), v[t].copy(), p[t].copy())
    d2[:] = rng.uniform(1.0, 50.0, size=d.shape)
    v2[:] = rng.random(v.shape) > 0.5
    p2[:] = rng.normal(size=p.shape)
    d2[t], v2[t], p2[t] = keep

    alt = make_prompt(cfg, "seq", t, len(p), d2[t], v2[t], p2[t])
    assert alt.has_depth == ref[t].has_depth and alt.has_pose == ref[t].has_pose
    if ref[t].has_depth:
        assert np.array_equal(alt.depth_values, ref[t].depth_values)
    if ref[t].has_pose:
        assert np.array_equal(alt.pose_c2w, ref[t].pose_c2w)


def test_intervals_select_the_right_frames():
    d, v, p = _gt(n=100)
    for k in (1, 5, 10, 30, 100):
        cfg = PromptConfig(name=f"k{k}", depth_interval=k, pose_enabled=False)
        st = make_prompt_stream(cfg, "s", 100, d, v, p)
        idx = [i for i, x in enumerate(st) if x.has_depth]
        assert idx == list(range(0, 100, k))
        assert all(not x.has_pose for x in st)


def test_first_frame_only_prompts_once():
    d, v, p = _gt(n=50)
    cfg = PromptConfig(name="f", first_frame_only=True)
    st = make_prompt_stream(cfg, "s", 50, d, v, p)
    assert st[0].has_depth and st[0].has_pose
    assert all(x.is_empty for x in st[1:])


def test_bernoulli_schedule_hits_roughly_the_right_rate():
    d, v, p = _gt(n=2000)
    cfg = PromptConfig(name="b", depth_bernoulli=0.1, pose_enabled=False)
    st = make_prompt_stream(cfg, "s", 2000, d, v, p)
    rate = np.mean([x.has_depth for x in st])
    assert 0.07 < rate < 0.13


def test_rgb_only_yields_no_prompts():
    d, v, p = _gt()
    st = make_prompt_stream(standard_configs()["rgb_only"], "s", len(p), d, v, p)
    assert all(x.is_empty for x in st)


def test_dropout_removes_prompts():
    d, v, p = _gt(n=400)
    base = PromptConfig(name="d0", depth_interval=1, pose_enabled=False)
    drop = PromptConfig(name="d5", depth_interval=1, pose_enabled=False, dropout=0.5)
    n0 = sum(x.has_depth for x in make_prompt_stream(base, "s", 400, d, v, p))
    n1 = sum(x.has_depth for x in make_prompt_stream(drop, "s", 400, d, v, p))
    assert n0 == 400
    assert 150 < n1 < 250


def test_sparsification_and_missing_pixels():
    d, v, p = _gt(h=64, w=64)
    a = make_prompt(PromptConfig(name="a", depth_interval=1, num_depth_samples=100),
                    "s", 0, len(p), d[0], v[0], p[0])
    b = make_prompt(PromptConfig(name="b", depth_interval=1, num_depth_samples=100,
                                 depth_missing=0.5), "s", 0, len(p), d[0], v[0], p[0])
    assert a.num_depth_samples == 100
    assert 25 < b.num_depth_samples < 75


def test_depth_samples_only_come_from_valid_pixels():
    d, v, p = _gt(h=32, w=32)
    v[0, :16] = False
    pr = make_prompt(PromptConfig(name="a", depth_interval=1, num_depth_samples=200),
                     "s", 0, len(p), d[0], v[0], p[0])
    assert pr.depth_rows.min() >= 16


def test_multiplicative_noise_and_outliers_perturb_the_values():
    d, v, p = _gt(h=64, w=64)
    clean = make_prompt(PromptConfig(name="c", depth_interval=1, num_depth_samples=2000),
                        "s", 0, len(p), d[0], v[0], p[0])
    noisy = make_prompt(PromptConfig(name="n", depth_interval=1, num_depth_samples=2000,
                                     depth_noise_sigma=0.1), "s", 0, len(p), d[0], v[0], p[0])
    ratio = noisy.depth_values / clean.depth_values
    assert 0.05 < np.std(np.log(ratio)) < 0.2

    out = make_prompt(PromptConfig(name="o", depth_interval=1, num_depth_samples=2000,
                                   depth_outlier_rate=0.2, depth_outlier_scale=4.0),
                      "s", 0, len(p), d[0], v[0], p[0])
    wild = np.mean(np.abs(np.log(out.depth_values / clean.depth_values)) > 1.0)
    assert 0.1 < wild < 0.3


def test_pose_noise_perturbs_translation_and_rotation():
    d, v, p = _gt()
    clean = make_prompt(PromptConfig(name="c", pose_interval=1, depth_enabled=False),
                        "s", 5, len(p), None, None, p[5])
    noisy = make_prompt(PromptConfig(name="n", pose_interval=1, depth_enabled=False,
                                     translation_noise_m=0.5, rotation_noise_deg=2.0),
                        "s", 5, len(p), None, None, p[5])
    assert np.allclose(clean.pose_c2w, p[5])
    assert 0.1 < np.linalg.norm(noisy.pose_c2w[:3, 3] - p[5][:3, 3]) < 3.0
    assert not np.allclose(noisy.pose_c2w[:3, :3], p[5][:3, :3])


def test_standard_grid_covers_the_required_modalities_and_intervals():
    cfgs = standard_configs()
    assert "rgb_only" in cfgs
    for k in (1, 5, 10, 30, 100):
        for mode in ("depth", "pose", "both"):
            assert f"{mode}_k{k}_clean" in cfgs
            assert f"{mode}_k{k}_moderate" in cfgs
    assert any("bern" in k for k in cfgs)
    assert any("first_only" in k for k in cfgs)


def test_training_sampler_is_deterministic_per_seed():
    a, b = training_config_sampler(3), training_config_sampler(3)
    assert a == b
    assert training_config_sampler(4) != a
