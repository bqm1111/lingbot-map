"""PCA basis, text/image calibration, sparse-teacher sampling and the semantic map."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from semantic_sidecar.metrics import Confusion, CoverageCounter, cross_view_consistency, umeyama_sim3
from semantic_sidecar.semantic_map import (
    SemanticMapAccumulator,
    SparseSemanticMap,
    embeddings_to_pixels,
)
from semantic_sidecar.teacher_features import (
    SemanticProjection,
    fit_semantic_projection,
    sample_teacher_frames,
)


def test_projection_preserves_cosine_similarity():
    """The basis is orthonormal and uncentered, so similarity survives compression."""
    torch.manual_seed(0)
    # Features living in a 12-d subspace of a 64-d space: the top-12 basis is exact.
    basis = torch.linalg.qr(torch.randn(64, 12))[0]
    feats = F.normalize(torch.randn(500, 12) @ basis.T, dim=-1)
    proj = fit_semantic_projection(feats, 12, meta={"fit_scenes": ["synthetic"]})

    full = feats[:50] @ feats[50:100].T
    compressed = proj.project(feats[:50], normalize=True) @ proj.project(feats[50:100], normalize=True).T
    assert torch.allclose(full, compressed, atol=1e-4)
    assert proj.explained_ratio > 0.999


def test_text_and_image_take_the_identical_transform():
    torch.manual_seed(1)
    basis = torch.linalg.qr(torch.randn(64, 16))[0]
    feats = F.normalize(torch.randn(400, 16) @ basis.T, dim=-1)
    proj = fit_semantic_projection(feats, 16, meta={})
    text = F.normalize(torch.randn(5, 16) @ basis.T, dim=-1)
    before = feats[:20] @ text.T
    after = proj.project(feats[:20]) @ proj.project(text).T
    assert torch.allclose(before, after, atol=1e-4)


def test_projection_roundtrip_and_metadata(tmp_path):
    feats = F.normalize(torch.randn(200, 32), dim=-1)
    proj = fit_semantic_projection(feats, 8, meta={"fit_scenes": ["a", "b"], "uses_target_labels": False})
    path = str(tmp_path / "pca.safetensors")
    proj.save(path)
    loaded = SemanticProjection.load(path)
    assert torch.allclose(loaded.components, proj.components)
    assert loaded.meta["fit_scenes"] == ["a", "b"]
    assert loaded.meta["uses_target_labels"] is False
    assert loaded.dim == 8 and loaded.source_dim == 32


def test_projection_rejects_mismatched_width():
    proj = fit_semantic_projection(F.normalize(torch.randn(100, 16), dim=-1), 4, meta={})
    with pytest.raises(ValueError):
        proj.project(torch.randn(10, 32))


@pytest.mark.parametrize("density", [0.05, 0.10, 0.25, 0.50, 1.00])
def test_teacher_density_sampling(density):
    n = 400
    mask = sample_teacher_frames(n, density, seed=1234)
    assert mask.sum() == (n if density >= 1.0 else round(n * density))
    # Reproducible from the recorded seed, and different seeds really differ.
    assert np.array_equal(mask, sample_teacher_frames(n, density, seed=1234))
    if density < 1.0:
        assert not np.array_equal(mask, sample_teacher_frames(n, density, seed=99))


def test_teacher_density_keeps_at_least_one_frame():
    assert sample_teacher_frames(3, 0.05, seed=0).sum() == 1


def test_teacher_density_rejects_bad_values():
    with pytest.raises(ValueError):
        sample_teacher_frames(10, 0.0, seed=0)
    with pytest.raises(ValueError):
        sample_teacher_frames(10, 1.5, seed=0)


def test_map_accumulate_query_and_roundtrip(tmp_path):
    torch.manual_seed(0)
    dim = 8
    text = F.normalize(torch.randn(3, dim), dim=-1)
    # Three spatially separated blobs, each carrying one of the text directions.
    # The centres sit mid-voxel so a blob cannot straddle a voxel boundary.
    blobs = [(0.25, 0.25, 0.25), (5.25, 0.25, 0.25), (0.25, 5.25, 0.25)]
    pts, feats, rgb = [], [], []
    for c, centre in enumerate(blobs):
        p = torch.tensor(centre) + 0.01 * torch.randn(200, 3)
        pts.append(p)
        feats.append(F.normalize(text[c].expand(200, dim) + 0.01 * torch.randn(200, dim), dim=-1))
        rgb.append(torch.full((200, 3), 40.0 * (c + 1)))
    accum = SemanticMapAccumulator(0.5, dim, torch.device("cpu"))
    accum.add(torch.cat(pts), torch.cat(feats), torch.cat(rgb))
    smap = accum.finalize(min_count=1, meta={"scene": "toy"})

    assert smap.num_voxels == 3
    labels, scores = smap.classify(text)
    centres = smap.centres()
    for i in range(3):
        nearest = int((centres - torch.tensor(blobs[i])).norm(dim=-1).argmin())
        assert int(labels[nearest]) == i
    assert bool((scores > 0.9).all())
    assert torch.allclose(smap.embeddings.float().norm(dim=-1), torch.ones(3), atol=1e-2)
    assert bool((smap.dispersion >= 0).all()) and bool((smap.dispersion < 0.01).all())

    path = str(tmp_path / "map.npz")
    smap.save(path)
    loaded = SparseSemanticMap.load(path)
    assert torch.equal(loaded.keys, smap.keys)
    assert loaded.meta["scene"] == "toy"

    idx = loaded.lookup(torch.tensor([[5.25, 0.25, 0.25], [100.0, 0.0, 0.0]]))
    assert int(idx[0]) >= 0 and int(idx[1]) == -1

    ply = str(tmp_path / "map.ply")
    loaded.export_ply(ply)
    assert open(ply).readline().strip() == "ply"


def test_dispersion_grows_with_disagreement():
    dim = 8
    accum_a = SemanticMapAccumulator(1.0, dim, torch.device("cpu"))
    v = F.normalize(torch.randn(1, dim), dim=-1)
    accum_a.add(torch.zeros(10, 3), v.expand(10, dim), torch.zeros(10, 3))
    tight = accum_a.finalize(1)

    accum_b = SemanticMapAccumulator(1.0, dim, torch.device("cpu"))
    accum_b.add(torch.zeros(10, 3), F.normalize(torch.randn(10, dim), dim=-1), torch.zeros(10, 3))
    loose = accum_b.finalize(1)
    assert float(loose.dispersion[0]) > float(tight.dispersion[0]) + 0.2


def test_embeddings_to_pixels_upsamples_and_normalises():
    tokens = F.normalize(torch.randn(6, 8), dim=-1)
    up = embeddings_to_pixels(tokens, 2, 3, (28, 42))
    assert up.shape == (28, 42, 8)
    assert torch.allclose(up.norm(dim=-1), torch.ones(28, 42), atol=1e-5)


def test_confusion_and_coverage():
    conf = Confusion(4)
    gt = torch.tensor([1, 1, 2, 2, 3])
    pred = torch.tensor([1, 2, 2, 2, 0])  # one confusion, one "unknown"
    conf.update(gt, pred)
    assert conf.miou() > 0
    assert conf.per_class(["a", "b", "c"])["c"] == 0.0  # class 3 was never predicted
    cov = CoverageCounter()
    cov.update(10, 4)
    cov.update(10, 6)
    assert cov.ratio == pytest.approx(0.5)


def test_cross_view_consistency_bounds():
    v = F.normalize(torch.randn(1, 6), dim=-1)
    ids = torch.zeros(8, dtype=torch.long)
    same, n = cross_view_consistency(v.expand(8, 6), ids)
    assert same == pytest.approx(1.0, abs=1e-5) and n == 1
    random_c, _ = cross_view_consistency(F.normalize(torch.randn(8, 6), dim=-1), ids)
    assert random_c < same


def test_umeyama_is_geometry_only():
    rng = np.random.default_rng(0)
    src = rng.standard_normal((40, 3))
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    dst = 3.0 * (q @ src.T).T + np.array([1.0, -2.0, 0.5])
    R, t, s, rmse = umeyama_sim3(src, dst)
    assert s == pytest.approx(3.0, rel=1e-5)
    assert rmse < 1e-6
    with pytest.raises(ValueError):
        umeyama_sim3(src, dst[:10])
