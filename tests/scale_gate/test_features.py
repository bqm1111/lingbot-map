"""Feature-contract and model tests (plan tests 14-15)."""

import numpy as np
import pytest
import torch

from gates.scale_gate.features import FORBIDDEN_KEYS, clip_feature, depth_features, image_features, pose_features


class FakeCache(dict):
    """A cache-like object that raises if a target-only field is ever read."""

    def __getitem__(self, k):
        if k in FORBIDDEN_KEYS:
            raise AssertionError(f"feature builder touched forbidden key {k!r}")
        return super().__getitem__(k)


def make(T=5, H=8, W=12, C=16, seed=0):
    rng = np.random.default_rng(seed)
    c2w = np.tile(np.eye(4), (T, 1, 1))
    c2w[:, 0, 3] = np.arange(T) * 0.4
    return FakeCache({
        "pred_depth": rng.uniform(0.1, 5.0, (T, H, W)).astype(np.float16),
        "pred_depth_conf": rng.uniform(1.0, 20.0, (T, H, W)).astype(np.float16),
        "pred_pose_c2w": c2w.astype(np.float32),
        "pred_K": np.tile(np.array([[300.0, 0, 250.0], [0, 300.0, 80.0], [0, 0, 1.0]]), (T, 1, 1)),
        "pooled_pre_gct": rng.normal(size=(T, C)).astype(np.float16),
        # target-only fields present but forbidden
        "gt_pose_c2w": np.tile(np.eye(4), (T, 1, 1)),
        "gt_K_native": np.eye(3),
    })


# ---------------- 14. inference contract ---------------- #
def test_feature_blocks_never_read_target_only_fields():
    z = make()
    for fn in (depth_features, pose_features, image_features):
        fn(z)                                   # FakeCache raises if a target leaks in
    clip_feature(z, ["depth", "pose", "image"])


def test_clip_feature_is_deterministic_and_finite():
    z = make()
    a = clip_feature(z, ["depth", "pose", "image"])
    b = clip_feature(z, ["depth", "pose", "image"])
    assert np.array_equal(a, b) and np.isfinite(a).all()


def test_block_selection_changes_dimensionality_predictably():
    z = make(C=16)
    d = clip_feature(z, ["depth"]).size
    i = clip_feature(z, ["image"]).size
    dp = clip_feature(z, ["depth", "pose"]).size
    assert i == 2 * 16                              # mean + std over the clip
    assert dp > d and clip_feature(z, ["depth", "pose", "image"]).size == dp + i


def test_features_survive_degenerate_depth():
    z = make()
    z["pred_depth"] = np.zeros_like(z["pred_depth"])       # nothing positive
    v = clip_feature(z, ["depth", "pose"])
    assert np.isfinite(v).all()


def test_static_clip_gives_finite_pose_features():
    z = make()
    z["pred_pose_c2w"] = np.tile(np.eye(4), (5, 1, 1)).astype(np.float32)
    assert np.isfinite(pose_features(z)).all()


def test_missing_pooled_feature_degrades_gracefully():
    z = make()
    z["pooled_pre_gct"] = np.zeros((5, 1), np.float16)     # cache without image features
    assert image_features(z).shape[1] == 0
    assert np.isfinite(clip_feature(z, ["depth", "image"])).all()


# ---------------- 15. MLP overfit ---------------- #
def test_mlp_overfits_a_tiny_synthetic_dataset():
    from tools.scale_gate.train_scale import ScaleMLP     # noqa: PLC0415
    torch.manual_seed(0)
    X = torch.randn(16, 24)
    w = torch.randn(24)
    y = X @ w * 0.1 + 3.3
    m = ScaleMLP(24, hidden=128, dropout=0.0)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(1500):
        loss = torch.nn.functional.smooth_l1_loss(m(X), y)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        err = float(torch.median(torch.abs(m(X) - y)))
    assert err < 0.02, f"failed to overfit 16 clips: median |err| {err:.4f}"


def test_scale_mlp_predicts_one_value_per_clip():
    from tools.scale_gate.train_scale import ScaleMLP     # noqa: PLC0415
    assert ScaleMLP(10)(torch.randn(7, 10)).shape == (7,)
