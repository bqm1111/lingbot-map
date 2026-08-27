"""SemBypass invariants: freezing, shapes, normalisation, geometry, determinism."""

import json
import os

import numpy as np
import pytest
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CACHE_NEW = os.path.join(REPO, "research/sem_bypass/outputs/cache_encoder")
CACHE_OLD = os.path.join(REPO, "output/semantic_sidecar/cache")

from research.sem_bypass.encoder_features import ENCODER_CHANNELS
from research.sem_bypass.model import build_sidecar
from semantic_sidecar.config import load_config
from semantic_sidecar.teacher_features import SemanticProjection, sample_teacher_frames

CFG = os.path.join(REPO, "research/sem_bypass/configs/gate1.yaml")


# --------------------------------------------------------------------------- #
# Architecture / parameters
# --------------------------------------------------------------------------- #
def test_sidecar_parameter_count_is_the_audited_architecture():
    """Same module as the 9.1M study; only the input width differs (1024 vs 2x2048)."""
    s = build_sidecar()
    n = sum(p.numel() for p in s.parameters())
    assert n == 6_156_864, n
    assert n < 0.01 * 1_157_943_540, "must stay under 1% of LingBot for the paper claim"


def test_sidecar_output_dim_and_normalisation():
    s = build_sidecar(out_dim=64)
    y = s(torch.randn(3, 407, 1, ENCODER_CHANNELS))
    assert y.shape == (3, 407, 64)
    assert torch.allclose(y.norm(dim=-1), torch.ones(3, 407), atol=1e-5)


def test_sidecar_rejects_wrong_input_width():
    s = build_sidecar()
    with pytest.raises(ValueError):
        s(torch.randn(2, 407, 1, 2048))
    with pytest.raises(ValueError):
        s(torch.randn(2, 407, 2, ENCODER_CHANNELS))


def test_only_sidecar_parameters_receive_gradients():
    s = build_sidecar()
    x = torch.randn(2, 16, 1, ENCODER_CHANNELS)
    s(x).square().sum().backward()
    assert all(p.grad is not None for p in s.parameters())
    assert x.grad is None


# --------------------------------------------------------------------------- #
# Teacher space
# --------------------------------------------------------------------------- #
def test_pca_basis_is_orthonormal_so_cosine_is_preserved():
    proj = SemanticProjection.load(os.path.join(CACHE_OLD, "pca.safetensors"))
    B = proj.components if hasattr(proj, "components") else None
    assert B is not None and B.shape == (64, 512)
    assert torch.allclose(B @ B.T, torch.eye(64), atol=1e-4)


def test_projection_preserves_cosine_within_the_subspace():
    """Text and image features must be comparable after projection."""
    proj = SemanticProjection.load(os.path.join(CACHE_OLD, "pca.safetensors"))
    B = proj.components
    torch.manual_seed(0)
    # vectors already inside the retained subspace project isometrically
    c = torch.randn(8, 64)
    x = c @ B                                     # [8, 512] in-subspace
    y = proj.project(x, normalize=False)          # back to [8, 64]
    cx = torch.nn.functional.normalize(c, dim=-1)
    cy = torch.nn.functional.normalize(y, dim=-1)
    assert torch.allclose(cx @ cx.T, cy @ cy.T, atol=1e-3)


def test_teacher_features_are_cached_and_l2_normalised():
    from safetensors.torch import load_file
    p = os.path.join(CACHE_OLD, "teacher", "kitti08_c000", "teacher_00000.safetensors")
    if not os.path.exists(p):
        pytest.skip("teacher cache unavailable")
    f = load_file(p)["features"].float()
    assert f.shape[1:] == (407, 512)
    n = f.reshape(-1, 512).norm(dim=-1)
    assert torch.allclose(n, torch.ones_like(n), atol=2e-2)


# --------------------------------------------------------------------------- #
# Deterministic teacher-frame selection
# --------------------------------------------------------------------------- #
def test_teacher_frame_selection_is_deterministic_per_seed():
    a = sample_teacher_frames(250, 0.25, 1234)
    assert np.array_equal(a, sample_teacher_frames(250, 0.25, 1234))
    assert a.sum() == round(250 * 0.25)
    for seed in (0, 1, 2):
        m = sample_teacher_frames(250, 0.25, seed)
        assert m.sum() == round(250 * 0.25)
        assert np.array_equal(m, sample_teacher_frames(250, 0.25, seed))
    # independent seeds must give genuinely different draws
    masks = [sample_teacher_frames(250, 0.25, s) for s in (0, 1, 2)]
    assert not np.array_equal(masks[0], masks[1])
    assert not np.array_equal(masks[1], masks[2])


def test_full_density_keeps_every_frame():
    assert sample_teacher_frames(37, 1.0, 0).all()


# --------------------------------------------------------------------------- #
# Cache integrity and geometry invariance
# --------------------------------------------------------------------------- #
def _cached_scenes():
    if not os.path.isdir(CACHE_NEW):
        return []
    return [d for d in sorted(os.listdir(CACHE_NEW))
            if os.path.isdir(os.path.join(CACHE_NEW, d))
            and os.path.exists(os.path.join(CACHE_NEW, d, "manifest.json"))]


def test_encoder_cache_manifest_records_the_pre_gct_representation():
    scenes = _cached_scenes()
    if not scenes:
        pytest.skip("encoder cache not built yet")
    m = json.load(open(os.path.join(CACHE_NEW, scenes[0], "manifest.json")))
    assert m["layers"] == ["patch_embed"]
    assert m["token_channels"] == ENCODER_CHANNELS
    assert m["representation"].startswith("pre_gct_encoder:")
    assert m["patch_grid"] == [11, 37] and m["num_tokens"] == 407


def test_geometry_is_bit_exact_against_the_existing_gct_cache():
    """The decisive invariance test: swapping the hook cannot move geometry."""
    from safetensors.torch import load_file
    scenes = [s for s in _cached_scenes() if os.path.isdir(os.path.join(CACHE_OLD, s))]
    if not scenes:
        pytest.skip("no overlapping scene cached in both layouts")
    scene = scenes[0]
    mn = json.load(open(os.path.join(CACHE_NEW, scene, "manifest.json")))
    for i in range(len(mn["shards"])):
        a = load_file(os.path.join(CACHE_NEW, scene, f"shard_{i:05d}.safetensors"))
        b = load_file(os.path.join(CACHE_OLD, scene, f"shard_{i:05d}.safetensors"))
        for k in ("depth", "depth_conf", "extrinsic", "intrinsic", "pose_enc", "frame_index"):
            assert torch.equal(a[k], b[k]), f"{scene} shard {i}: {k} differs"
        assert a["tokens"].shape[1:] == (1, 407, ENCODER_CHANNELS)
        assert b["tokens"].shape[1:] == (2, 407, 2048)


def test_cached_tokens_reload_exactly():
    from safetensors.torch import load_file
    scenes = _cached_scenes()
    if not scenes:
        pytest.skip("encoder cache not built yet")
    p = os.path.join(CACHE_NEW, scenes[0], "shard_00000.safetensors")
    a, b = load_file(p), load_file(p)
    for k in a:
        assert torch.equal(a[k], b[k])
    assert torch.isfinite(a["tokens"].float()).all()
    assert not a["tokens"].requires_grad


def test_depth_confidence_is_a_precision():
    """expp1 activation: conf = 1 + exp(raw), higher is better."""
    from safetensors.torch import load_file
    scenes = _cached_scenes()
    if not scenes:
        pytest.skip("encoder cache not built yet")
    c = load_file(os.path.join(CACHE_NEW, scenes[0], "shard_00000.safetensors"))["depth_conf"].float()
    assert float(c.min()) >= 1.0 - 1e-3
    assert float(c.max()) > 1.0


# --------------------------------------------------------------------------- #
# Config / protocol invariants
# --------------------------------------------------------------------------- #
def test_gate1_config_uses_pre_gct_tokens_and_no_consensus():
    cfg = load_config(CFG, [])
    assert cfg.lingbot.layers == ["patch_embed"]
    assert cfg.loss.lambda_pixel == 1.0
    assert cfg.loss.lambda_consensus == 0.0
    assert cfg.loss.lambda_mv == 0.0 and cfg.loss.lambda_rel == 0.0
    assert cfg.train.teacher_density == 0.25
    assert cfg.teacher.backend == "maskclip"
    assert cfg.teacher.model_name == "ViT-B-16-quickgelu" and cfg.teacher.pretrained == "openai"


def test_gate1_train_and_eval_scenes_are_disjoint_and_unlabelled_for_training():
    cfg = load_config(CFG, [])
    train = [s for s in cfg.scenes if s.role == "train"]
    ev = [s for s in cfg.scenes if s.role == "eval"]
    assert train and ev
    train_seq = {s.image_folder.split("sequences/")[1].split("/")[0] for s in train}
    eval_seq = {s.image_folder.split("sequences/")[1].split("/")[0] for s in ev}
    assert train_seq.isdisjoint(eval_seq)
    # KITTI odometry 13-16 have no released semantic labels at all
    assert train_seq <= {"13", "14", "15", "16"}
    for s in train:
        assert not os.path.isdir(os.path.join(REPO, "data/kitti/dataset/sequences",
                                              s.image_folder.split("sequences/")[1].split("/")[0],
                                              "labels"))


def test_no_loss_module_references_semantic_labels():
    import semantic_sidecar.losses as L
    src = open(L.__file__).read().lower()
    for bad in ("learning_map", "semantic_label", "gt_label"):
        assert bad not in src
