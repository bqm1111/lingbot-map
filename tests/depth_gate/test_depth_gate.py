"""Gate-2 unit tests: synthetic residual recovery, contracts, leakage, metrics."""

import os

import numpy as np
import pytest
import torch

from gates.depth_gate.losses import (
    compute_loss, log_smooth_l1, relative_loss, residual_regularizer, sparse_gradient_loss,
)
from gates.depth_gate.metrics import binned_metrics, depth_metrics
from gates.depth_gate.models import ARCHS, build_inputs, build_model, refine
from gates.scale_gate.config import REPO_ROOT, load_config

STATS = {"log_depth_mean": 2.5, "log_depth_std": 0.8, "conf_mean": 3.0, "conf_std": 2.0}


def fake_batch(B=2, H=32, W=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(B, 1, H, W, generator=g) * 30 + 2
    return {
        "rgb": torch.rand(B, 3, H, W, generator=g),
        "lingbot_depth": base / 27.0,
        "lingbot_confidence": torch.rand(B, 1, H, W, generator=g) * 10 + 1,
        "valid_lingbot_mask": torch.ones(B, 1, H, W, dtype=torch.bool),
        "base_depth": base,
        "projected_lidar_depth": base.clone(),
        "projected_lidar_valid_mask": torch.rand(B, 1, H, W, generator=g) > 0.5,
    }


# ---------------- model contract ---------------- #
def test_refined_depth_is_finite_and_positive_for_any_residual():
    base = torch.rand(4, 1, 16, 16) * 50 + 0.1
    for r in (torch.full_like(base, -5.0), torch.full_like(base, 5.0), torch.randn_like(base) * 3):
        out = refine(base, torch.tanh(r) * 0.7)
        assert torch.isfinite(out).all() and (out > 0).all()


def test_heads_start_as_identity():
    """Zero-initialised head means training begins from unrefined LingBot depth."""
    for arch, ch in (("depth_cnn", 5), ("rgbd_unet", 8)):
        m = build_model(arch, ch)
        assert torch.allclose(m(torch.randn(2, ch, 32, 48)), torch.zeros(1), atol=1e-7)


def test_identity_baseline_returns_the_base_depth():
    b = fake_batch()
    m = build_model("identity", 5)
    r = m(build_inputs(b, STATS, use_rgb=False))
    assert torch.allclose(refine(b["base_depth"], r), b["base_depth"])


def test_models_stay_under_the_five_million_parameter_cap():
    for arch, ch in (("depth_cnn", 5), ("rgbd_unet", 8)):
        n = sum(p.numel() for p in build_model(arch, ch).parameters())
        assert n < 5_000_000, f"{arch} has {n:,} parameters"


def test_unknown_arch_is_refused():
    with pytest.raises(ValueError):
        build_model("transformer", 5)


def test_input_channel_count_matches_use_rgb():
    b = fake_batch()
    assert build_inputs(b, STATS, use_rgb=False).shape[1] == 5
    assert build_inputs(b, STATS, use_rgb=True).shape[1] == 8


def test_inputs_never_contain_lidar_or_metadata():
    """The network input must be assembled only from LingBot-side tensors."""
    b = fake_batch()
    b["projected_lidar_depth"] = torch.full_like(b["projected_lidar_depth"], 999.0)
    x1 = build_inputs(b, STATS, use_rgb=True)
    b["projected_lidar_depth"] = torch.zeros_like(b["projected_lidar_depth"])
    x2 = build_inputs(b, STATS, use_rgb=True)
    assert torch.equal(x1, x2), "changing the LiDAR target changed the model input"


# ---------------- synthetic residual recovery ---------------- #
def test_head_recovers_a_known_constant_log_residual():
    """The core learnability check: base depth is 20 % low everywhere; can the head fix it?"""
    torch.manual_seed(0)
    b = fake_batch(B=4, H=32, W=48)
    true_r = 0.1823                                   # log(1.2)
    b["base_depth"] = b["projected_lidar_depth"] * np.exp(-true_r)
    b["lingbot_depth"] = b["base_depth"] / 27.0
    m = build_model("depth_cnn", 5, base=16)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    x = build_inputs(b, STATS, use_rgb=False)
    for _ in range(400):
        r = m(x)
        loss = log_smooth_l1(refine(b["base_depth"], r), b["projected_lidar_depth"],
                             b["projected_lidar_valid_mask"], beta=0.1)
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(r.mean()) == pytest.approx(true_r, abs=0.03)
    assert float(loss) < 1e-3


def test_head_recovers_a_spatially_varying_residual():
    torch.manual_seed(0)
    b = fake_batch(B=4, H=32, W=48)
    H, W = 32, 48
    ramp = torch.linspace(-0.3, 0.3, W).view(1, 1, 1, W).expand(4, 1, H, W)
    b["base_depth"] = b["projected_lidar_depth"] * torch.exp(-ramp)
    b["lingbot_depth"] = b["base_depth"] / 27.0
    m = build_model("depth_cnn", 5, base=32)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    x = build_inputs(b, STATS, use_rgb=False)
    before = None
    for i in range(600):
        r = m(x)
        loss = log_smooth_l1(refine(b["base_depth"], r), b["projected_lidar_depth"],
                             b["projected_lidar_valid_mask"], beta=0.05)
        if i == 0:
            before = float(loss)
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(loss) < 0.1 * before, f"loss only fell {before:.4f} -> {float(loss):.4f}"


# ---------------- losses ---------------- #
def test_losses_are_zero_for_a_perfect_prediction():
    b = fake_batch()
    v = b["projected_lidar_valid_mask"]
    g = b["projected_lidar_depth"]
    assert log_smooth_l1(g, g, v).item() == pytest.approx(0.0, abs=1e-9)
    assert relative_loss(g, g, v).item() == pytest.approx(0.0, abs=1e-9)
    assert sparse_gradient_loss(g, g, v).item() == pytest.approx(0.0, abs=1e-6)


def test_losses_handle_an_empty_valid_mask():
    b = fake_batch()
    v = torch.zeros_like(b["projected_lidar_valid_mask"])
    for fn in (log_smooth_l1, relative_loss):
        out = fn(b["base_depth"], b["projected_lidar_depth"], v)
        assert torch.isfinite(out) and float(out) == 0.0


def test_confidence_weighting_changes_the_loss():
    """Weighting only matters when the error varies spatially -- a uniform offset gives a
    constant per-pixel loss whose weighted mean equals its plain mean for any weights."""
    torch.manual_seed(0)
    b = fake_batch()
    v, g = b["projected_lidar_valid_mask"], b["projected_lidar_depth"]
    err = torch.linspace(0.9, 1.3, g.shape[-1]).view(1, 1, 1, -1).expand_as(g)
    pred = g * err
    plain = log_smooth_l1(pred, g, v)
    w = (err > 1.1).float()                       # up-weight exactly the worst pixels
    assert not torch.isclose(plain, log_smooth_l1(pred, g, v, w), atol=1e-4)
    # and a constant weight must reproduce the unweighted loss
    assert torch.isclose(plain, log_smooth_l1(pred, g, v, torch.ones_like(g)), atol=1e-6)


def test_residual_regularizer_penalises_large_corrections():
    assert residual_regularizer(torch.zeros(4, 1, 8, 8)).item() == 0.0
    assert residual_regularizer(torch.full((4, 1, 8, 8), 0.5)).item() == pytest.approx(0.5)


def test_compute_loss_respects_the_primary_selector():
    cfg = load_config("configs/depth_gate/refine.yaml")
    b = fake_batch()
    r = torch.zeros_like(b["base_depth"])
    out = compute_loss(cfg, refine(b["base_depth"], r), r, b)
    assert set(out) == {"loss", "depth", "residual"} and torch.isfinite(out["loss"])


# ---------------- metrics ---------------- #
def test_metrics_are_exact_on_a_known_offset():
    g = torch.rand(1, 1, 16, 16) * 30 + 5
    v = torch.ones_like(g, dtype=torch.bool)
    m = depth_metrics(g * 1.1, g, v)
    assert m["abs_rel"] == pytest.approx(0.1, abs=1e-6)
    assert m["delta1"] == pytest.approx(1.0)
    assert m["n_valid"] == 256


def test_distance_bins_partition_the_valid_pixels():
    g = torch.rand(1, 1, 40, 40) * 79 + 0.5
    v = torch.ones_like(g, dtype=torch.bool)
    b = binned_metrics(g, g, v, [(0, 10), (10, 20), (20, 40), (40, 80)])
    assert sum(x["n_valid"] for x in b.values()) == int(((g >= 0) & (g < 80)).sum())


# ---------------- split hygiene ---------------- #
def test_configured_splits_are_sequence_disjoint():
    cfg = load_config("configs/depth_gate/refine.yaml")
    tr = set(cfg.data.train_sequences); se = set(cfg.data.select_sequences)
    va = set(cfg.data.val_sequences)
    assert tr.isdisjoint(se) and tr.isdisjoint(va) and se.isdisjoint(va)
    assert va == {"08"} and "08" not in tr | se


@pytest.mark.skipif(not os.path.isdir(os.path.join(REPO_ROOT, "artifacts/depth_gate/cache_rgb")),
                    reason="RGB cache not built")
def test_dataset_splits_do_not_share_sequences():
    from gates.depth_gate.data import DepthRefineDataset
    d = load_config("configs/depth_gate/refine.yaml")
    s = load_config(d.data.scale_gate_config)
    tr = DepthRefineDataset(s, d, d.data.train_sequences, "train", d.scale.constant)
    se = DepthRefineDataset(s, d, d.data.select_sequences, "train", d.scale.constant)
    va = DepthRefineDataset(s, d, d.data.val_sequences, "val", d.scale.constant)
    assert set(tr.sequences).isdisjoint(se.sequences)
    assert set(tr.sequences).isdisjoint(va.sequences)
    assert va.sequences == ["08"]
    item = tr[0]
    assert item["base_depth"].shape == item["projected_lidar_depth"].shape
    assert (item["base_depth"][item["valid_lingbot_mask"]] > 0).all()
    assert item["projected_lidar_valid_mask"].sum() > 0, "no LiDAR supervision in this frame"
