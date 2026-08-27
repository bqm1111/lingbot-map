"""Cache round-trip, deterministic sampling and teacher-budget selection."""

import os

import numpy as np
import pytest
import torch

from research.lingbot_semantic_memory.config import ChunkSpec, DataConfig, Phase1Config
from research.lingbot_semantic_memory.dataset_adapter import build_splits, scale_intrinsics
from research.lingbot_semantic_memory.feature_cache import CachedChunk, load_chunk, save_chunk
from research.lingbot_semantic_memory.run_phase1 import (
    build_pairs, patch_confidence, teacher_frame_subset,
)


def _dummy_chunk() -> CachedChunk:
    rng = np.random.default_rng(0)
    S, P, gh, gw, H, W = 4, 12, 3, 4, 12, 16
    return CachedChunk(
        name="seq08_000000_4", role="val", sequence="08", start=0,
        grid_hw=(gh, gw), image_hw=(H, W),
        reps={"lingbot_encoder": rng.standard_normal((S, P, 8)).astype(np.float16),
              "lingbot_gct_final": rng.standard_normal((S, P, 16)).astype(np.float16)},
        teacher=rng.standard_normal((S, P, 6)).astype(np.float16),
        pred_depth=rng.random((S, H, W)).astype(np.float32) + 1,
        pred_conf=rng.random((S, H, W)).astype(np.float32) + 1,
        pred_extrinsic=rng.standard_normal((S, 3, 4)).astype(np.float32),
        pred_intrinsic=np.eye(3, dtype=np.float32),
        oracle_depth=rng.random((S, H, W)).astype(np.float32) + 1,
        oracle_extrinsic=rng.standard_normal((S, 3, 4)).astype(np.float32),
        oracle_intrinsic=np.eye(3, dtype=np.float32),
        labels=rng.integers(0, 19, (S, gh, gw)).astype(np.uint8),
    )


def test_cache_save_load_is_lossless(tmp_path):
    c = _dummy_chunk()
    p = str(tmp_path / "c.npz")
    save_chunk(p, c)
    d = load_chunk(p)
    assert d.name == c.name and d.role == c.role and d.sequence == c.sequence
    assert d.grid_hw == c.grid_hw and d.image_hw == c.image_hw
    assert set(d.reps) == set(c.reps)
    for k in c.reps:
        assert np.array_equal(d.reps[k], c.reps[k])
    for k in ("teacher", "pred_depth", "pred_conf", "pred_extrinsic", "pred_intrinsic",
              "oracle_depth", "oracle_extrinsic", "oracle_intrinsic", "labels"):
        assert np.array_equal(getattr(d, k), getattr(c, k)), k


def test_cache_handles_missing_optional_fields(tmp_path):
    c = _dummy_chunk()
    c.oracle_depth = c.oracle_extrinsic = c.oracle_intrinsic = c.labels = None
    p = str(tmp_path / "t.npz")
    save_chunk(p, c)
    d = load_chunk(p)
    assert d.oracle_depth is None and d.labels is None
    assert np.array_equal(d.teacher, c.teacher)


def test_cached_arrays_carry_no_gradient(tmp_path):
    c = _dummy_chunk()
    p = str(tmp_path / "c.npz")
    save_chunk(p, c)
    d = load_chunk(p)
    t = d.rep("lingbot_encoder")
    assert not t.requires_grad and t.grad_fn is None


def test_split_construction_is_deterministic():
    cfg = DataConfig()
    a1, b1 = build_splits(cfg, seed=0)
    a2, b2 = build_splits(cfg, seed=0)
    assert [c.name for c in a1] == [c.name for c in a2]
    assert [c.name for c in b1] == [c.name for c in b2]
    a3, _ = build_splits(cfg, seed=1)
    assert [c.name for c in a1] != [c.name for c in a3]


def test_splits_are_sequence_disjoint_and_sized_as_configured():
    cfg = DataConfig()
    tr, va = build_splits(cfg, seed=0)
    assert set(c.sequence for c in tr).isdisjoint(set(c.sequence for c in va))
    assert len(tr) == len(cfg.train_sequences) * cfg.train_chunks_per_sequence
    assert len(va) == cfg.val_chunks
    assert all(c.length == cfg.chunk_length for c in tr + va)
    assert all(c.start >= 0 for c in tr + va)


def test_teacher_budget_subsets_are_deterministic_and_nested_in_size():
    n = 384
    s10 = teacher_frame_subset(n, 0.10, 0)
    assert np.array_equal(s10, teacher_frame_subset(n, 0.10, 0))
    s25 = teacher_frame_subset(n, 0.25, 0)
    s100 = teacher_frame_subset(n, 1.0, 0)
    assert len(s10) < len(s25) < len(s100) == n
    assert len(s10) == pytest.approx(38, abs=1)
    assert len(s25) == pytest.approx(96, abs=1)
    for s in (s10, s25, s100):
        assert s.min() >= 0 and s.max() < n and len(np.unique(s)) == len(s)


def test_pair_construction_is_deterministic_and_respects_the_cap():
    cfg = Phase1Config()
    p1 = build_pairs(cfg, n_chunks=2, chunk_len=64)
    p2 = build_pairs(cfg, n_chunks=2, chunk_len=64)
    assert [(x.chunk, x.i, x.j) for x in p1] == [(x.chunk, x.i, x.j) for x in p2]
    for c in (0, 1):
        assert len([x for x in p1 if x.chunk == c]) <= cfg.eval.max_pairs_per_chunk + len(cfg.eval.short_gaps + cfg.eval.long_gaps)
    for x in p1:
        assert x.j == x.i + x.gap and 0 <= x.i < x.j < 64
        assert x.band == ("short" if x.gap in cfg.eval.short_gaps else "long")


def test_patch_confidence_reads_the_grid_correctly():
    S, gh, gw, H, W = 2, 3, 4, 12, 16
    conf = np.zeros((S, H, W), dtype=np.float32)
    conf[0, 6, 10] = 5.0            # centre of patch (1, 2) -> flat index 6
    pc = patch_confidence(conf, (gh, gw))
    assert pc.shape == (S, gh * gw)
    assert pc[0, 1 * gw + 2] == pytest.approx(5.0)
    assert pc[0].sum() == pytest.approx(5.0)


def test_intrinsics_scale_under_a_pure_resize():
    K = np.array([[707.0912, 0, 601.8873], [0, 707.0912, 183.1104], [0, 0, 1.0]])
    Ks = scale_intrinsics(K, (370, 1226), (154, 518))
    assert Ks[0, 0] == pytest.approx(707.0912 * 518 / 1226)
    assert Ks[1, 1] == pytest.approx(707.0912 * 154 / 370)
    assert Ks[0, 2] == pytest.approx(601.8873 * 518 / 1226)
    assert Ks[1, 2] == pytest.approx(183.1104 * 154 / 370)
    assert Ks[2, 2] == pytest.approx(1.0)
