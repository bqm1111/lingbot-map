"""Config round-tripping and the training-batch assembly."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from semantic_sidecar.config import (
    SceneSpec,
    SidecarConfig,
    collect_provenance,
    config_to_dict,
    load_config,
    save_config,
    set_seed,
)


def test_config_roundtrip(tmp_path):
    cfg = SidecarConfig(
        name="unit",
        scenes=[SceneSpec(name="a", image_folder="/tmp/a", role="train", chunk_size=64)],
    )
    path = str(tmp_path / "cfg.yaml")
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.name == "unit"
    assert loaded.scenes[0].chunk_size == 64
    assert config_to_dict(loaded) == config_to_dict(cfg)


def test_dotted_overrides(tmp_path):
    cfg = SidecarConfig(name="unit")
    path = str(tmp_path / "cfg.yaml")
    save_config(cfg, path)
    loaded = load_config(path, ["model.arch=linear", "train.lr=5e-4", "lingbot.layers=[23]"])
    assert loaded.model.arch == "linear"
    assert loaded.train.lr == pytest.approx(5e-4)
    assert loaded.lingbot.layers == [23]


def test_unknown_keys_are_rejected(tmp_path):
    path = str(tmp_path / "cfg.yaml")
    with open(path, "w") as fh:
        yaml.safe_dump({"model": {"arch": "mlp", "nonsense": 1}}, fh)
    with pytest.raises(ValueError, match="nonsense"):
        load_config(path)


def test_bad_override_shape(tmp_path):
    cfg = SidecarConfig()
    path = str(tmp_path / "cfg.yaml")
    save_config(cfg, path)
    with pytest.raises(ValueError):
        load_config(path, ["model.arch"])


def test_shipped_configs_parse():
    for name in ("feasibility", "linear", "consensus_mlp", "smoke"):
        cfg = load_config(f"configs/semantic_sidecar/{name}.yaml")
        assert cfg.scenes, name
        assert cfg.teacher.pca_dim in (64, 128)
        roles = {s.role for s in cfg.scenes}
        assert "train" in roles
        # No evaluation dataset may ever appear with role "train".
        for spec in cfg.scenes:
            if spec.role == "train":
                assert spec.dataset not in {"semantickitti", "scannet", "replica", "speira"}


def test_provenance_records_identity():
    info = collect_provenance(SidecarConfig(name="x"))
    assert "git_commit" in info and "torch" in info and info["config_name"] == "x"


def test_set_seed_is_reproducible():
    set_seed(3)
    a = (torch.randn(4), np.random.rand(4))
    set_seed(3)
    b = (torch.randn(4), np.random.rand(4))
    assert torch.equal(a[0], b[0])
    assert np.allclose(a[1], b[1])


# --------------------------------------------------------------------------- #
class _StubReader:
    """Minimal stand-in for FeatureCacheReader over synthetic tokens."""

    def __init__(self, num_frames: int, num_tokens: int, num_layers: int = 2, channels: int = 8):
        self._tokens = torch.randn(num_frames, num_layers, num_tokens, channels)
        self.num_frames = num_frames
        self.patch_hw = (1, num_tokens)
        self.manifest = {"layers": list(range(num_layers)), "patch_size": 14}

    def tokens(self, f: int) -> torch.Tensor:
        return self._tokens[f]


def _bundle(num_frames=4, num_tokens=6, dim=5):
    from semantic_sidecar.datasets import SceneBundle
    from semantic_sidecar.tracks import _empty_trackset
    from semantic_sidecar.config import TrackConfig

    # Two tracks, each observed by every frame at a distinct token.
    frames = torch.arange(num_frames).repeat_interleave(2).to(torch.int32)
    tokens = torch.tensor([1, 4] * num_frames, dtype=torch.int32)
    order = torch.argsort(torch.tensor([0, 1] * num_frames) * 1000 + frames)
    tracks = _empty_trackset(0.1, TrackConfig())
    tracks.obs_frame = frames[order]
    tracks.obs_token = tokens[order]
    tracks.obs_ptr = torch.tensor([0, num_frames, 2 * num_frames], dtype=torch.int64)
    tracks.obs_conf = torch.full((2 * num_frames,), 2.0)
    tracks.obs_residual = torch.zeros(2 * num_frames)
    tracks.obs_cos = torch.ones(2 * num_frames)
    tracks.centres = torch.zeros(2, 3)

    consensus = F.normalize(torch.randn(2, dim), dim=-1)
    return SceneBundle(
        name="stub",
        reader=_StubReader(num_frames, num_tokens),
        tracks=tracks,
        obs_teacher=F.normalize(torch.randn(2 * num_frames, dim), dim=-1),
        obs_has_teacher=torch.ones(2 * num_frames, dtype=torch.bool),
        consensus=consensus,
        consensus_valid=torch.ones(2, dtype=torch.bool),
        frame_teacher_mask=np.ones(num_frames, dtype=bool),
    )


def test_frame_targets_maps_tokens_to_tracks():
    bundle = _bundle()
    tgt = bundle.frame_targets(2)
    assert tgt["track_id"].shape == (6,)
    assert int(tgt["track_id"][1]) == 0
    assert int(tgt["track_id"][4]) == 1
    assert int(tgt["track_id"][0]) == -1
    assert bool(tgt["consensus_mask"][1]) and not bool(tgt["consensus_mask"][0])
    assert torch.allclose(tgt["consensus"][1], bundle.consensus[0])


def test_collate_frame_batch_shapes_and_tracks():
    from semantic_sidecar.datasets import collate_frame_batch

    bundle = _bundle()
    rng = np.random.default_rng(0)
    batch = collate_frame_batch(bundle, [0, 1, 2], tokens_per_frame=8, rng=rng, device=torch.device("cpu"))
    assert batch["tokens"].shape == (6, 2, 8)   # 3 frames x 2 tracked tokens
    assert batch["consensus"].shape == (6, 5)
    assert bool(batch["consensus_mask"].all())
    # Both tracks appear more than once, which is what the cross-view term needs.
    counts = torch.bincount(batch["track_id"].clamp_min(0))
    assert int(counts.min()) >= 2


def test_frame_batch_sampler_stays_within_one_scene():
    from semantic_sidecar.datasets import FrameBatchSampler

    sampler = FrameBatchSampler([_bundle(), _bundle()], frames_per_batch=2, seed=0)
    for _ in range(10):
        bundle, frames = sampler.sample()
        assert len(frames) == 2
        assert all(0 <= f < bundle.num_frames for f in frames)


def test_expand_scene_chunks(tmp_path):
    from semantic_sidecar.datasets import expand_scene

    folder = tmp_path / "imgs"
    folder.mkdir()
    for i in range(70):
        (folder / f"{i:04d}.png").write_bytes(b"x")
    spec = SceneSpec(name="s", image_folder=str(folder), chunk_size=24)
    chunks = expand_scene(spec)
    # 70 frames -> 24 + 24 + 22 (the tail is kept because it clears the 8-frame floor).
    assert [c.max_frames for c in chunks] == [24, 24, 22]
    assert [c.start for c in chunks] == [0, 24, 48]
    assert all(c.chunk_size is None for c in chunks)


def test_expand_scene_drops_a_useless_tail(tmp_path):
    from semantic_sidecar.datasets import expand_scene

    folder = tmp_path / "imgs"
    folder.mkdir()
    for i in range(52):
        (folder / f"{i:04d}.png").write_bytes(b"x")
    chunks = expand_scene(SceneSpec(name="s", image_folder=str(folder), chunk_size=24))
    assert [c.max_frames for c in chunks] == [24, 24]  # the 4-frame tail is discarded


def test_sampler_reuses_a_window_for_shard_locality():
    """Consecutive batches must stay in one scene/window, or shard IO thrashes."""
    from semantic_sidecar.datasets import FrameBatchSampler

    a, b = _bundle(num_frames=8), _bundle(num_frames=8)
    a.name, b.name = "a", "b"
    sampler = FrameBatchSampler([a, b], frames_per_batch=2, window=4, seed=0, resample_every=4)
    names = [sampler.sample()[0].name for _ in range(8)]
    assert len(set(names[:4])) == 1 and len(set(names[4:])) == 1

    # resample_every=1 restores per-batch switching.
    switching = FrameBatchSampler([a, b], frames_per_batch=2, window=4, seed=0, resample_every=1)
    starts = {tuple(switching.sample()[1]) for _ in range(20)}
    assert len(starts) > 1


def test_shard_cache_is_globally_bounded(tmp_path):
    from semantic_sidecar.lingbot_features import (
        _SHARD_CACHE,
        FeatureCacheWriter,
        FeatureCacheReader,
        FrameFeatures,
        clear_shard_cache,
        set_shard_cache_limit,
    )

    clear_shard_cache()
    set_shard_cache_limit(2)
    for scene in ("s0", "s1", "s2"):
        writer = FeatureCacheWriter(str(tmp_path), scene, shard_frames=1)
        for i in range(3):
            writer.add(FrameFeatures(
                frame_index=i, image_path=f"{scene}_{i}.png",
                tokens=torch.zeros(1, 4, 8, dtype=torch.float16),
                depth=torch.zeros(4, 4, dtype=torch.float16),
                depth_conf=torch.zeros(4, 4, dtype=torch.float16),
                intrinsic=torch.eye(3), extrinsic=torch.zeros(3, 4),
                pose_enc=torch.zeros(9), image_patch_rgb=torch.zeros(3, 8, dtype=torch.uint8),
            ))
        writer.write_manifest({"scene": scene, "patch_grid": [2, 4], "processed_hw": [4, 4],
                               "patch_size": 14, "layers": [23]})

    readers = [FeatureCacheReader(str(tmp_path), s) for s in ("s0", "s1", "s2")]
    for r in readers:
        for f in range(3):
            r.geometry(f)
    assert len(_SHARD_CACHE) <= 2
    # Data is still correct after eviction.
    assert readers[0].tokens(2).shape == (1, 4, 8)
    assert readers[2].image_path(1).endswith("s2_1.png")
    clear_shard_cache()
    set_shard_cache_limit(4)
