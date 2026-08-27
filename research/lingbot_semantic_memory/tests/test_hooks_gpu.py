"""GPU tests: frozen parameters, hook shapes, and that hooks change nothing.

Skipped when no CUDA device or no LingBot checkpoint is available.
"""

import os

import pytest
import torch

from research.lingbot_semantic_memory.config import (
    LingBotConfig, REPO_ROOT, TeacherConfig, default_representations,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

CKPT = os.path.join(REPO_ROOT, LingBotConfig().model_path)
IMAGES = os.path.join(REPO_ROOT, "data/kitti/dataset/sequences/08/image_2")


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda:0")


@pytest.fixture(scope="module")
def images(device):
    from lingbot_map.utils.load_fn import load_and_preprocess_images
    if not os.path.isdir(IMAGES):
        pytest.skip("KITTI sequence 08 unavailable")
    paths = [os.path.join(IMAGES, f"{i:06d}.png") for i in range(6)]
    return load_and_preprocess_images(paths, mode="crop", image_size=518, patch_size=14)


@pytest.fixture(scope="module")
def lingbot(device):
    from research.lingbot_semantic_memory.hooks import FrozenLingBot
    if not os.path.exists(CKPT):
        pytest.skip("LingBot checkpoint unavailable")
    return FrozenLingBot(LingBotConfig(), default_representations(), device)


def test_lingbot_parameters_are_all_frozen(lingbot):
    trainable = [n for n, p in lingbot.model.named_parameters() if p.requires_grad]
    assert trainable == []
    assert all(p.grad is None for p in lingbot.model.parameters())
    assert lingbot.num_params > 1e9


def test_teacher_parameters_are_all_frozen(device):
    from research.lingbot_semantic_memory.hooks import FrozenDinoTeacher
    t = FrozenDinoTeacher(TeacherConfig(), device)
    assert [n for n, p in t.model.named_parameters() if p.requires_grad] == []
    assert all(p.grad is None for p in t.model.parameters())


def test_hook_shapes_match_the_patch_lattice(lingbot, images):
    preds, reps = lingbot.run_chunk(images, capture=True)
    S, _, H, W = images.shape
    P = (H // 14) * (W // 14)
    assert set(reps) == {r.name for r in default_representations()}
    for r in default_representations():
        t = reps[r.name]
        assert t.shape == (S, P, r.dim), f"{r.name}: {tuple(t.shape)}"
        assert t.dtype == torch.float16
        assert torch.isfinite(t.float()).all()
        assert not t.requires_grad and t.grad_fn is None
    assert preds["depth"].shape == (S, H, W, 1)
    assert torch.isfinite(preds["depth"]).all()
    assert torch.isfinite(preds["pose_enc"]).all()


def test_depth_confidence_is_a_precision_not_an_uncertainty(lingbot, images):
    """`conf_activation="expp1"` means conf = 1 + exp(raw), so conf >= 1, higher = better."""
    preds, _ = lingbot.run_chunk(images, capture=False)
    c = preds["depth_conf"]
    assert float(c.min()) >= 1.0 - 1e-6
    assert float(c.max()) > 1.0


def test_hooks_do_not_alter_the_model_output(lingbot, images):
    """Forward hooks return None, so geometry must be bit-identical either way."""
    with_hooks, _ = lingbot.run_chunk(images, capture=True)
    without, reps = lingbot.run_chunk(images, capture=False)
    assert reps == {}
    assert torch.equal(with_hooks["depth"], without["depth"])
    assert torch.equal(with_hooks["depth_conf"], without["depth_conf"])
    assert torch.equal(with_hooks["pose_enc"], without["pose_enc"])


def test_inference_is_deterministic(lingbot, images):
    a, ra = lingbot.run_chunk(images, capture=True)
    b, rb = lingbot.run_chunk(images, capture=True)
    assert torch.equal(a["depth"], b["depth"])
    assert torch.equal(a["pose_enc"], b["pose_enc"])
    for k in ra:
        assert torch.equal(ra[k], rb[k]), k


def test_teacher_grid_matches_lingbot_grid(device, images, lingbot):
    """Both use patch size 14, so distillation needs no resize."""
    from research.lingbot_semantic_memory.hooks import FrozenDinoTeacher
    t = FrozenDinoTeacher(TeacherConfig(), device)
    f = t.encode(images)
    _, reps = lingbot.run_chunk(images, capture=True)
    assert f.shape[:2] == reps["lingbot_encoder"].shape[:2]
    assert f.shape[2] == TeacherConfig().dim
    assert torch.isfinite(f.float()).all()


def test_probe_training_never_touches_frozen_parameters(lingbot, device):
    """A backward pass through probe outputs must leave LingBot gradient-free."""
    from research.lingbot_semantic_memory.probes import build_matched_probe
    probe, _ = build_matched_probe(2048, 768, 2_000_000, 10_000_000)
    probe = probe.to(device)
    x = torch.randn(4, 2048, device=device)
    probe(x).sum().backward()
    assert all(p.grad is not None for p in probe.parameters())
    assert all(p.grad is None for p in lingbot.model.parameters())
