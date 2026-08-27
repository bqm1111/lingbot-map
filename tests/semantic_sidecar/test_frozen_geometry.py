"""The hook must not perturb LingBot, and LingBot must stay frozen.

These tests load the real 1.16 B-parameter checkpoint, so they are skipped when no
GPU or checkpoint is available.  Run them explicitly with::

    CUDA_VISIBLE_DEVICES=0 PATH=<conda-env>/bin:/usr/local/cuda-12.8/bin:$PATH \\
      FLASHINFER_CUDA_ARCH_LIST=12.0a python -m pytest \\
      tests/semantic_sidecar/test_frozen_geometry.py -q
"""

from __future__ import annotations

import glob
import os

import pytest
import torch

from semantic_sidecar.config import LingBotConfig

CHECKPOINT = "checkpoints/lingbot-map/204754b/lingbot-map.pt"
IMAGES = "example/courthouse"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.exists(CHECKPOINT),
    reason="needs a CUDA device and the LingBot checkpoint",
)


@pytest.fixture(scope="module")
def images():
    from lingbot_map.utils.load_fn import load_and_preprocess_images

    paths = sorted(glob.glob(os.path.join(IMAGES, "*.png")))[:12]
    assert paths, f"no images under {IMAGES}"
    return load_and_preprocess_images(paths, mode="crop", image_size=518, patch_size=14)


@pytest.fixture(scope="module")
def frozen():
    from semantic_sidecar.lingbot_features import FrozenLingBot

    return FrozenLingBot(LingBotConfig(model_path=CHECKPOINT, layers=[11, 23], num_scale_frames=8))


def test_all_parameters_are_frozen(frozen):
    frozen.assert_frozen()
    report = frozen.parameter_report()
    assert report["lingbot_trainable"] == 0
    assert report["lingbot_total"] > 1_000_000_000


def test_capture_does_not_change_geometry(frozen, images):
    """Depth and pose must be bit-identical with the feature hook on and off."""
    frozen.capture_tokens = True
    with_hook, tokens = frozen.run_sequence(images)
    frozen.capture_tokens = False
    without_hook, none_tokens = frozen.run_sequence(images)
    frozen.capture_tokens = True

    assert none_tokens is None
    assert tokens is not None
    for key in ("depth", "depth_conf", "pose_enc"):
        a, b = with_hook[key], without_hook[key]
        assert a.shape == b.shape
        assert torch.equal(a, b), f"{key} changed: max |d| = {(a - b).abs().max()}"


def test_repeated_inference_is_deterministic(frozen, images):
    a, _ = frozen.run_sequence(images)
    b, _ = frozen.run_sequence(images)
    for key in ("depth", "pose_enc"):
        assert torch.equal(a[key], b[key])


def test_captured_token_shapes_match_the_image_grid(frozen, images):
    from semantic_sidecar.config import TOKEN_CHANNELS
    from semantic_sidecar.lingbot_features import patch_grid

    _, tokens = frozen.run_sequence(images)
    S, _, H, W = images.shape
    h, w = patch_grid(H, W, 14)
    assert tokens.shape == (S, len(frozen.layers), h * w, TOKEN_CHANNELS)
    assert tokens.dtype == torch.float16
    assert not tokens.requires_grad


def test_cache_scene_writes_a_reusable_manifest(frozen, tmp_path):
    from semantic_sidecar.config import SceneSpec
    from semantic_sidecar.lingbot_features import FeatureCacheReader, cache_scene, scene_is_cached

    spec = SceneSpec(name="unit_courthouse", image_folder=IMAGES, role="train", max_frames=12)
    manifest = cache_scene(frozen, spec, str(tmp_path))
    assert manifest["num_frames"] == 12
    assert manifest["coordinate_system"] == "first_camera_w2c_opencv"
    assert manifest["checkpoint_sha256"] != "missing"
    assert scene_is_cached(str(tmp_path), spec.name)

    reader = FeatureCacheReader(str(tmp_path), spec.name)
    assert reader.num_frames == 12
    g = reader.geometry(3)
    assert g["depth"].shape == tuple(manifest["processed_hw"])
    assert g["extrinsic"].shape == (3, 4)
    assert reader.tokens(3).shape == (2, manifest["num_tokens"], 2048)


def test_unprojected_frames_agree_across_views(frozen, images, tmp_path):
    """Two nearby frames must unproject into overlapping world points.

    This is the empirical check that the stored ``extrinsic`` is consumed as
    world-to-camera (see docs/semantic_sidecar_repo_audit.md §3).
    """
    from semantic_sidecar.lingbot_features import FrozenLingBot as _F
    from semantic_sidecar.lingbot_features import unproject_depth

    preds, _ = frozen.run_sequence(images)
    H, W = preds["depth"].shape[1:3]
    extr, intr = _F.to_extrinsics(preds["pose_enc"], (H, W))

    clouds = []
    for i in (0, 6):
        depth = preds["depth"][i, ..., 0]
        conf = preds["depth_conf"][i]
        keep = (depth > 1e-3) & (conf > 2.0)
        clouds.append(unproject_depth(depth, intr[i], extr[i])[keep][::53])
    dist = torch.cdist(clouds[1], clouds[0]).min(dim=1).values
    scale = clouds[0].norm(dim=-1).median()
    assert float((dist < 0.05 * scale).float().mean()) > 0.7
