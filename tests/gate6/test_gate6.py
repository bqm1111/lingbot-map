"""Gate 6 — frozen Trident semantic lifting: the tests the report's claims rest on.

Grouped by the claim each one defends. Tests that need a produced artifact skip with an
explicit reason when it is absent, so the file is runnable at any point in the pipeline
and never silently passes by doing nothing.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gates.gate6 import (audit as g6audit, frames as F, grids, lifting, metrics, permute,  # noqa
                   pipelines, stats, vocab)
from gates.gate6.trident_adapter import keep_ratio_size, resample_to_lattice, OFFICIAL_CITYSCAPES
from gates.scale_gate.config import REPO_ROOT

ART = os.path.join(ROOT, "artifacts", "gate6")


def _art(name):
    p = os.path.join(ART, name)
    if not os.path.exists(p):
        pytest.skip(f"{name} not produced yet")
    return p


def _json(name):
    return json.load(open(_art(name)))


# =========================================================================== #
# Vocabulary, class mapping and prompts
# =========================================================================== #
OFFICIAL_COUNTS = {"semantickitti": 19, "kitti360": 18, "occ3d": 17}


@pytest.mark.parametrize("ds", list(vocab.DATASETS))
def test_every_official_non_empty_class_appears_exactly_once(ds):
    v = vocab.load(ds)
    assert len(v) == OFFICIAL_COUNTS[ds]
    assert len(set(v.labels)) == len(v.labels)
    assert len(set(v.names)) == len(v.names)
    assert v.empty_label not in v.labels
    # contiguous over the benchmark's own occupied label range
    assert list(v.labels) == sorted(v.labels)
    assert v.labels[-1] - v.labels[0] == len(v) - 1


def test_official_class_names_match_the_official_sources():
    """The three lists are the official ones, not a paraphrase."""
    sk = vocab.load("semantickitti")
    assert sk.labels[0] == 1 and sk.names[0] == "car"
    assert sk.names[-1] == "traffic-sign" and sk.labels[-1] == 19
    assert "trunk" in sk.names and "motorcyclist" in sk.names      # rare classes kept
    k3 = vocab.load("kitti360")
    assert k3.names[-2:] == ("other-structure", "other-object")
    assert "bicyclist" not in k3.names          # KITTI-360 has no rider classes
    oc = vocab.load("occ3d")
    assert oc.labels[0] == 0 and oc.names[0] == "others"           # catch-all kept
    assert oc.empty_label == 17 and 17 not in oc.labels


@pytest.mark.parametrize("ds", list(vocab.DATASETS))
def test_channel_to_label_mapping_is_a_bijection(ds):
    v = vocab.load(ds)
    m = v.channel_of_label
    assert sorted(m) == sorted(v.labels)
    assert sorted(m.values()) == list(range(len(v)))
    for c, lab in enumerate(v.labels):
        assert m[lab] == c


def test_prompts_are_deterministic_and_mechanical():
    for _ in range(3):
        assert vocab.load("occ3d").phrases == vocab.load("occ3d").phrases
    # the eight rewrites the brief enumerates
    assert vocab.phrase_for("other-vehicle") == "other vehicle"
    assert vocab.phrase_for("bicyclist") == "person riding a bicycle"
    assert vocab.phrase_for("motorcyclist") == "person riding a motorcycle"
    assert vocab.phrase_for("traffic-sign") == "traffic sign"
    assert vocab.phrase_for("construction_vehicle") == "construction vehicle"
    assert vocab.phrase_for("driveable_surface") == "drivable road surface"
    assert vocab.phrase_for("other_flat") == "other flat ground"
    assert vocab.phrase_for("manmade") == "man-made structure"
    # and nothing else: everything not rewritten is a pure separator substitution
    for ds in vocab.DATASETS:
        v = vocab.load(ds)
        for n, p in zip(v.names, v.phrases):
            if n not in vocab.EXPLICIT_REWRITES:
                assert p == n.replace("-", " ").replace("_", " ")


def test_no_nuisance_class_is_added():
    for ds in vocab.DATASETS:
        low = {p.lower() for p in vocab.load(ds).phrases}
        for bad in ("sky", "background", "unlabeled", "empty", "free", "ignore", "void"):
            assert bad not in low


# =========================================================================== #
# The teacher adapter: dense scores and official parity
# =========================================================================== #
def test_trident_adapter_reproduces_official_predictions():
    d = _json("trident_parity.json")["summary"]
    assert d["n_frames"] >= 100, "the brief requires at least 100 deterministic images"
    assert d["n_datasets"] == 3
    assert d["argmax_parity_exact"], (
        f"{d['total_tie_pixels']} of {d['total_pixels']} pixels disagree")
    assert d["total_tie_pixels"] == 0
    assert d["pristine_all_identical"], "the recording wrapper changed the prediction"
    assert d["n_pristine_checked"] >= 10
    assert d["max_prob_sum_err"] < 1e-5      # the dense tensor is a distribution


def test_dense_scores_are_probabilities_over_the_official_vocabulary():
    d = _json("trident_parity.json")
    for r in d["rows"]:
        assert r["n_classes"] == OFFICIAL_COUNTS[r["dataset"]]
        assert r["prob_sum_max_abs_err"] < 1e-5


def test_official_configuration_is_the_cityscapes_one():
    c = OFFICIAL_CITYSCAPES
    assert (c["model_type"], c["clip_type"]) == ("ViT-H-14", "laion2b_s32b_b79k")
    assert c["sam_model_type"] == "vit_h" and c["sam_refinement"] is True
    assert (c["cos_fac"], c["refine_neg_cos"], c["coarse_thresh"]) == (3.0, False, 0.10)
    assert (c["slide_stride"], c["slide_crop"]) == (224, 336)
    assert c["dataset_type"] == "CityscapesDataset"


def test_teacher_is_frozen_and_its_supervision_is_recorded():
    p = _json("teacher_provenance.json")
    assert p["trained_by_us"] is False
    assert p["trident"]["tracked_files_unmodified"], "the upstream sources were edited"
    assert p["clip"]["patch_size"] == 14 and p["vfm"]["patch_size"] == 16
    for part in ("clip", "sam"):
        assert len(p[part]["weight_sha256"]) == 64
    assert p["total_params"] > 1.5e9


# =========================================================================== #
# Pixel transforms: native <-> Trident <-> LingBot lattice
# =========================================================================== #
def test_keep_ratio_resize_is_uniform_and_full_extent():
    for hw in [(370, 1226), (376, 1408), (900, 1600), (1024, 2048)]:
        th, tw = keep_ratio_size(hw)
        assert abs((th / hw[0]) - (tw / hw[1])) < 2e-3, "resize must be uniform"
        assert max(th, tw) <= 2048 and min(th, tw) <= 688


def test_resample_to_lattice_preserves_a_synthetic_coordinate_field():
    """A field that is linear in normalised image coordinates must survive resampling.

    That is the whole content of the 'full extent to full extent' claim: if the two
    lattices really cover the same rectangle, a ramp in u stays the same ramp in u.
    """
    C, H, W = 3, 64, 200
    u = (torch.arange(W).float() + 0.5) / W
    v = (torch.arange(H).float() + 0.5) / H
    field = torch.stack([u.expand(H, W), v[:, None].expand(H, W),
                         torch.ones(H, W)], 0)
    field = field / field.sum(0, keepdim=True)
    hp, wp = 21, 70
    out = resample_to_lattice(field, (hp, wp))
    up = (torch.arange(wp).float() + 0.5) / wp
    vp = (torch.arange(hp).float() + 0.5) / hp
    ref = torch.stack([up.expand(hp, wp), vp[:, None].expand(hp, wp),
                       torch.ones(hp, wp)], 0)
    ref = ref / ref.sum(0, keepdim=True)
    inner = (slice(None), slice(1, -1), slice(1, -1))
    assert torch.allclose(out[inner], ref[inner], atol=2e-3)
    assert torch.allclose(out.sum(0), torch.ones(hp, wp), atol=1e-5)


def test_pixel_provenance_reconstructs_its_own_point():
    dep, conf, K, pose = _synthetic_clip()
    s = 27.3665
    pts, fr, cf, dp, vp, up, _ = lifting.points_with_pixels(
        dep, conf, K, pose, s, 1.5, 1.0, 60.0)
    d = s * dep.astype(np.float64)
    anchor = dep.shape[0] - 1
    m = fr == anchor
    rec = np.stack([(up[m] - K[anchor][0, 2]) * d[fr[m], vp[m], up[m]] / K[anchor][0, 0],
                    (vp[m] - K[anchor][1, 2]) * d[fr[m], vp[m], up[m]] / K[anchor][1, 1],
                    d[fr[m], vp[m], up[m]]], -1)
    assert np.allclose(pts[m], rec)
    assert np.array_equal(dp, d[fr, vp, up].astype(np.float32))


# =========================================================================== #
# Frozen geometry: fusion, scaling, anchor
# =========================================================================== #
def _synthetic_clip(T=5, H=17, W=23, seed=0):
    rng = np.random.default_rng(seed)
    dep = rng.uniform(0.02, 3.0, (T, H, W)).astype(np.float32)
    conf = rng.uniform(0.0, 3.0, (T, H, W)).astype(np.float32)
    K = np.tile(np.array([[200., 0, 11.], [0, 205., 8.], [0, 0, 1.]]), (T, 1, 1))
    pose = np.tile(np.eye(4), (T, 1, 1))
    pose[:, :3, 3] = rng.normal(0, 2, (T, 3))
    for t in range(T):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        pose[t, :3, :3] = q * np.sign(np.linalg.det(q))
    return dep, conf, K, pose


def test_lifting_reproduces_the_frozen_fusion_bit_for_bit():
    from gates.voxel_gate.c3 import c3_points
    dep, conf, K, pose = _synthetic_clip()
    a = c3_points(dep, conf, K, pose, 27.3665, 1.5, 1.0, 60.0)
    b = lifting.points_with_pixels(dep, conf, K, pose, 27.3665, 1.5, 1.0, 60.0)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert np.array_equal(a[2], b[2]) and np.array_equal(a[3], b[3])
    assert np.array_equal(a[4], b[6])


def test_the_same_scalar_scales_depth_and_translation_and_leaves_rotation_alone():
    """With the depth gate open, the fused cloud must be exactly ``s`` times the unit one."""
    dep, conf, K, pose = _synthetic_clip()
    p1 = lifting.points_with_pixels(dep, conf, K, pose, 1.0, 1.5, 0.0, np.inf)[0]
    for s in (2.0, 27.3665):
        ps = lifting.points_with_pixels(dep, conf, K, pose, s, 1.5, 0.0, np.inf)[0]
        assert np.allclose(ps, s * p1, rtol=1e-12, atol=1e-9)


def test_relative_rotation_is_independent_of_scale():
    from decompose_residual import scaled_relative_pose
    _, _, _, pose = _synthetic_clip()
    R0 = scaled_relative_pose(pose, 0, 4, 1.0)[:3, :3]
    for s in (0.5, 3.0, 27.3665):
        T = scaled_relative_pose(pose, 0, 4, s)
        assert np.allclose(T[:3, :3], R0, atol=1e-12)
        assert np.allclose(T[:3, 3], s * scaled_relative_pose(pose, 0, 4, 1.0)[:3, 3])


def test_voxeliser_matches_both_frozen_implementations():
    from prompted_lingbot.occupancy import voxelize_points, SEMANTICKITTI_GRID as G
    from occ3d_zeroshot.grid import points_to_canonical, CANONICAL
    p = np.random.default_rng(1).uniform(-60, 60, (20000, 3))
    i1, k1 = voxelize_points(p, G)
    i2, k2 = grids.voxelize(p, G)
    assert np.array_equal(i1, i2.astype(i1.dtype)) and np.array_equal(k1, k2)
    i3, k3 = points_to_canonical(p)
    i4, k4 = grids.voxelize(p, CANONICAL)
    assert np.array_equal(i3, i4) and np.array_equal(k3, k4)


@pytest.mark.parametrize("ds", list(vocab.DATASETS))
def test_clips_are_five_chronological_frames_with_the_anchor_last(ds):
    recs = F.read_manifest(ds, REPO_ROOT)
    assert recs, f"{ds} manifest is empty"
    for rec in recs[:200]:
        assert len(rec.rel_images) == 5
        r = rec.raw
        if "timestamps_ns" in r:
            ts = r["timestamps_ns"]
        elif "timestamps_s" in r:
            ts = r["timestamps_s"]
        elif "frame_ids" in r:
            ts = r["frame_ids"]
        else:
            ts = r["native_frames"]
        assert list(ts) == sorted(ts), "frames must be chronological"
        if "anchor" in r:
            assert int(r["sscbench_indices"][-1]) == int(r["anchor"])
        if "sample_tokens" in r:
            assert r["anchor_token"] == r["sample_tokens"][-1]
        if "frame_ids" in r and ds == "semantickitti":
            assert r["frame_ids"][-1] == max(r["frame_ids"])


# =========================================================================== #
# Semantic fusion and dilation propagation
# =========================================================================== #
def test_voxel_fusion_is_the_uniform_mean_of_contributing_vectors():
    dev = torch.device("cpu")
    flat = np.array([7, 7, 7, 3, 9, 9], np.int64)
    P = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                      [0.5, 0.5, 0.], [1., 0., 0.], [0., 0., 1.]])
    u, m, c = lifting.fuse_voxel_probs(flat, P, 16, dev)
    assert list(u) == [3, 7, 9]
    assert torch.allclose(m[0], torch.tensor([0.5, 0.5, 0.]))
    assert torch.allclose(m[1], torch.tensor([1 / 3, 1 / 3, 1 / 3]))
    assert torch.allclose(m[2], torch.tensor([0.5, 0., 0.5]))
    assert list(c) == [1, 3, 2]


def test_fusion_ignores_confidence_and_frame_count():
    """Uniform means uniform: reordering or reweighting inputs must not change it."""
    dev = torch.device("cpu")
    flat = np.array([5, 5, 5, 5], np.int64)
    P = torch.rand(4, 6)
    _, m, _ = lifting.fuse_voxel_probs(flat, P, 8, dev)
    assert torch.allclose(m[0], P.mean(0), atol=1e-6)
    perm = [3, 0, 2, 1]
    _, m2, _ = lifting.fuse_voxel_probs(flat, P[perm], 8, dev)
    assert torch.allclose(m, m2, atol=1e-6)


def test_dilation_only_voxels_take_the_nearest_source_and_average_exact_ties():
    dev = torch.device("cpu")
    dims = (9, 9, 9)

    def flat(x, y, z):
        return np.ravel_multi_index((x, y, z), dims)

    # one source at (4,4,4) -> the single neighbour at distance 1 wins outright
    src = np.array([flat(4, 4, 4)], np.int64)
    sp = torch.tensor([[1.0, 0.0]])
    dst = np.array([flat(4, 4, 5)], np.int64)
    out, ok = lifting.propagate_dilation(dims, src, sp, dst, 2, dev)
    assert ok.all() and torch.allclose(out[0], torch.tensor([1.0, 0.0]))

    # two sources equidistant (both at Euclidean distance 1) -> averaged
    src = np.array([flat(4, 4, 3), flat(4, 4, 5)], np.int64)
    sp = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    dst = np.array([flat(4, 4, 4)], np.int64)
    out, ok = lifting.propagate_dilation(dims, src, sp, dst, 2, dev)
    assert ok.all() and torch.allclose(out[0], torch.tensor([0.5, 0.5]))

    # a nearer source must beat a farther one, not be averaged with it
    src = np.array([flat(4, 4, 5), flat(4, 6, 4)], np.int64)   # distances 1 and 2
    sp = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    dst = np.array([flat(4, 4, 4)], np.int64)
    out, _ = lifting.propagate_dilation(dims, src, sp, dst, 2, dev)
    assert torch.allclose(out[0], torch.tensor([1.0, 0.0]))

    # diagonal (sqrt(2)) must lose to axial (1)
    src = np.array([flat(4, 4, 5), flat(4, 5, 5)], np.int64)
    sp = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    out, _ = lifting.propagate_dilation(dims, src, sp, dst, 2, dev)
    assert torch.allclose(out[0], torch.tensor([1.0, 0.0]))


def test_dilation_offsets_are_the_chebyshev_ball_ordered_by_euclidean_distance():
    g = lifting._offsets_by_distance(2)
    assert sum(len(o) for _, o in g) == 5 ** 3 - 1
    assert [d for d, _ in g] == sorted(d for d, _ in g)
    assert all(max(abs(x) for x in off) <= 2 for _, offs in g for off in offs)


def test_dilation_propagation_is_order_independent():
    dev = torch.device("cpu")
    dims = (7, 7, 7)
    rng = np.random.default_rng(3)
    src = rng.choice(np.prod(dims), 40, replace=False).astype(np.int64)
    sp = torch.rand(40, 5)
    dst = np.setdiff1d(rng.choice(np.prod(dims), 60, replace=False).astype(np.int64), src)
    o1, _ = lifting.propagate_dilation(dims, src, sp, dst, 2, dev)
    p = rng.permutation(len(src))
    o2, _ = lifting.propagate_dilation(dims, src[p], sp[p], dst, 2, dev)
    assert torch.allclose(o1, o2, atol=1e-6)


# =========================================================================== #
# Metrics
# =========================================================================== #
def _toy_counts(seed=0, dims=(8, 8, 4)):
    v = vocab.load("kitti360")
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 19, dims).astype(np.int32)
    t[rng.random(dims) < 0.2] = 255
    p = rng.integers(0, 19, dims).astype(np.int32)
    keep = t != 255

    class G:
        pass
    g = G(); g.dims = dims; g.voxel_size = 0.2; g.origin = (0., -0.8, -0.4)
    d, h = metrics.band_masks(g, ((0, 1), (1, 2)), ((-1, 0), (0, 1)))
    return v, metrics.clip_counts(p, None, t, keep, v.labels, 0, d, h, "x", "g", "B-D"), t, p, keep


def test_error_categories_partition_the_valid_occupied_voxels():
    v, c, t, p, keep = _toy_counts()
    s = metrics.summarize(metrics.aggregate([c]), v.names)["decomposition"]
    assert s["coverage_miss"] + s["naming_error"] + s["correct"] == s["n_valid_gt_occupied"]
    assert s["n_valid_gt_occupied"] == int(((t != 0) & keep).sum())
    assert s["sums_to_one"]
    assert abs(s["coverage_miss_fraction"] + s["naming_error_fraction"]
               + s["correct_fraction"] - 1.0) < 1e-12


def test_count_identities_hold_so_the_permutation_control_is_exact():
    _, c, _, _, _ = _toy_counts()
    assert np.array_equal(c.fp, c.conf.sum(0) - np.diag(c.conf) + c.fp_empty)
    assert np.array_equal(c.fn, c.conf.sum(1) - np.diag(c.conf) + c.decomp[:, 0])


def test_empty_prediction_outside_the_frozen_occupancy():
    """A voxel the frozen occupancy calls empty is never given a class."""
    v = vocab.load("kitti360")
    dims = (6, 6, 4)
    n = int(np.prod(dims))
    occ = np.zeros(n, bool)
    occ[[3, 17, 40]] = True
    pl = np.full(n, v.empty_label, np.int32)
    pl[occ] = 5
    assert (pl[~occ] == v.empty_label).all()
    t = np.full(n, 7, np.int32)
    keep = np.ones(n, bool)

    class G:
        pass
    g = G(); g.dims = dims; g.voxel_size = 0.2; g.origin = (0., 0., 0.)
    d, h = metrics.band_masks(g, ((0, 5),), ((-1, 5),))
    c = metrics.clip_counts(pl.reshape(dims), None, t.reshape(dims), keep.reshape(dims),
                            v.labels, v.empty_label, d, h, "x", "g", "B-R")
    assert c.n_pred_occupied == int(occ.sum())
    assert c.decomp[:, 0].sum() == n - int(occ.sum())     # the rest is coverage miss


def test_permutation_control_is_deterministic_and_identity_matches_the_evaluator():
    v, c, _, _, _ = _toy_counts()
    agg = metrics.aggregate([c])
    ref = metrics.summarize(agg, v.names)
    M, miss, fpe = agg["conf"], agg["decomp"][:, 0], agg["fp_empty"]
    miou, bal = permute.metrics_under(M, miss, fpe, np.arange(len(v)))
    assert abs(miou - ref["ssc_miou"]) < 1e-12, "closed form must reproduce the evaluator"
    assert abs(bal - ref["tp_conditioned"]["balanced_recall"]) < 1e-12
    a = permute.run_control(M, miss, fpe, 20, 0)
    b = permute.run_control(M, miss, fpe, 20, 0)
    assert a["values"] == b["values"]
    assert permute.run_control(M, miss, fpe, 20, 1)["values"] != a["values"]
    assert np.array_equal(permute.permutations(9, 5, 0), permute.permutations(9, 5, 0))


def test_permutation_of_a_perfect_prediction_destroys_it():
    """Sanity on the control itself: a diagonal confusion must beat every permutation."""
    C = 8
    M = np.diag(np.arange(10, 10 + C)).astype(np.int64)
    miss = np.zeros(C, np.int64)
    fpe = np.zeros(C, np.int64)
    r = permute.run_control(M, miss, fpe, 50, 0)
    assert r["ssc_miou"]["real"] == pytest.approx(1.0)
    assert r["ssc_miou"]["real_exceeds_p95"]
    assert r["tp_balanced_recall"]["real_exceeds_p95"]


def test_contiguous_blocks_absorb_the_remainder():
    b = stats.contiguous_blocks(163, 20)
    assert b.max() == 7 and len(set(b.tolist())) == 8
    assert (b == 7).sum() == 23
    assert (np.diff(b) >= 0).all()


# =========================================================================== #
# Leakage: the prediction phase never touches a target
# =========================================================================== #
PREDICTION_MODULES = ["gates/gate6/lifting.py", "gates/gate6/pipelines.py", "gates/gate6/frames.py",
                      "gates/gate6/grids.py", "gates/gate6/trident_adapter.py", "gates/gate6/vocab.py",
                      "tools/gate6/predict.py", "tools/gate6/cache_semantics.py"]


def test_no_prediction_module_imports_the_target_loader():
    for rel in PREDICTION_MODULES:
        tree = ast.parse(open(os.path.join(ROOT, rel)).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "targets" not in node.module, f"{rel} imports {node.module}"
                assert "occ_datasets" not in node.module or rel == "gates/gate6/pipelines.py"
            if isinstance(node, ast.Import):
                for n in node.names:
                    assert "gate6.targets" != n.name, rel
    # pipelines may reach occ_datasets, but only for the *calibration* it exposes
    src = open(os.path.join(ROOT, "gates/gate6/pipelines.py")).read()
    assert "SemanticKittiOccSpec" in src
    assert ".target(" not in src and "load_semantickitti_target" not in src


def test_the_auditor_rejects_every_target_family():
    for bad in ["a/000000.label", "a/000000.invalid", "x/000040_1_1.npy",
                "gts/scene-0003/tok/labels.npz", "s/08/velodyne/000000.bin",
                "s/08/voxels/000000.bin", "artifacts/oracle_scales.csv",
                "artifacts/scale_targets_val.csv"]:
        with pytest.raises(g6audit.ForbiddenAccess):
            with g6audit.Gate6Audit():
                open(os.path.join(ROOT, bad))


def test_the_auditor_allows_the_declared_inputs():
    with g6audit.Gate6Audit() as a:
        open(os.path.join(ROOT, "configs/gate6/trident_semantic_precommit.yaml")).close()
        open(os.path.join(ROOT, "artifacts/gate5_2/scales_B.csv")).close()
    assert a.violations == []
    assert a.summary()["n_opened"] >= 2


def test_the_auditor_intercepts_cv2_imread():
    import cv2
    with pytest.raises(g6audit.ForbiddenAccess):
        with g6audit.Gate6Audit():
            cv2.imread(os.path.join(ROOT, "s/08/velodyne/000000.bin"))
    assert cv2.imread.__module__ != __name__          # restored on exit


def test_prediction_run_recorded_a_clean_audit():
    for ds in vocab.DATASETS:
        p = os.path.join(ART, f"prediction_manifest_{ds}.json")
        if not os.path.exists(p):
            pytest.skip(f"{ds} predictions not produced yet")
        m = json.load(open(p))
        assert m["audit"]["violations"] == []
        assert m["audit"]["n_opened"] > 0, "the auditor observed nothing: it was not active"
        assert m["targets_opened"].startswith("none")


def test_prediction_hashes_exist_and_were_pinned_before_evaluation():
    for ds in vocab.DATASETS:
        mp = os.path.join(ART, f"prediction_manifest_{ds}.json")
        sp = os.path.join(ART, f"summary_{ds}.json")
        if not (os.path.exists(mp) and os.path.exists(sp)):
            pytest.skip(f"{ds} not evaluated yet")
        assert os.path.getmtime(mp) <= os.path.getmtime(sp), (
            "the prediction manifest must be written before the evaluator opens a target")
        m = json.load(open(mp))
        assert len(m["rollup_sha256"]) == 64 and m["n_files"] == len(m["per_file_sha256"])
        s = json.load(open(sp))
        assert s["prediction_rollup_sha256"] == m["rollup_sha256"]
        assert s["n_hash_mismatch"] == 0


def test_predictions_are_bit_identical_under_target_tampering(tmp_path):
    """Randomising targets in a scratch tree must not move a single predicted voxel.

    Nothing is written to a released dataset: the tampering happens in ``tmp_path`` and is
    proved to be real by loading the tampered copy and checking it differs.
    """
    ds = "kitti360"
    mp = os.path.join(ART, f"prediction_manifest_{ds}.json")
    if not os.path.exists(mp):
        pytest.skip("predictions not produced yet")
    from gates.gate6 import targets as g6t
    anchor = 40
    real = os.path.join(g6t.KITTI360_TARGET_ROOT, "preprocess", "labels",
                        F.KITTI360_SEQUENCE, f"{anchor:06d}_1_1.npy")
    if not os.path.exists(real):
        pytest.skip("target tree unavailable")
    orig = np.load(real)
    fake_root = tmp_path / "sscbench"
    d = fake_root / "preprocess" / "labels" / F.KITTI360_SEQUENCE
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    np.save(d / f"{anchor:06d}_1_1.npy", rng.integers(0, 19, orig.shape).astype(orig.dtype))
    tampered, _ = __import__("sscbench_kitti360.adapter", fromlist=["load_target"]) \
        .load_target(str(fake_root), anchor)
    assert not np.array_equal(tampered, orig), "the tampering must actually change the target"

    # the deployable prediction is a pure function of RGB + frozen caches
    man = json.load(open(mp))
    cid = f"{F.KITTI360_SEQUENCE}_{anchor:06d}"
    f = f"{cid}.npz"
    if f not in man["per_file_sha256"]:
        pytest.skip("clip not in the pinned set")
    path = os.path.join(man["prediction_root"], f)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert h == man["per_file_sha256"][f], "prediction changed after pinning"


# =========================================================================== #
# No training happened
# =========================================================================== #
GATE6_SOURCES = ([os.path.join("gates", "gate6", f) for f in os.listdir(os.path.join(ROOT, "gates", "gate6"))
                  if f.endswith(".py")]
                 + [os.path.join("tools", "gate6", f)
                    for f in os.listdir(os.path.join(ROOT, "tools", "gate6"))
                    if f.endswith(".py")])


def _code_only(path):
    """Source with docstrings and string literals removed, so prose cannot trip a scan."""
    tree = ast.parse(open(os.path.join(ROOT, path)).read())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                             ast.Module)) and ast.get_docstring(node):
            node.body = node.body[1:]
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = ""
    return ast.unparse(tree)


def test_no_optimizer_no_backward_no_parameter_update():
    banned = ("torch.optim", ".backward(", ".step()", "requires_grad_(True)",
              "loss.backward", "nn.Parameter(", "zero_grad")
    for rel in GATE6_SOURCES:
        code = _code_only(rel)
        for b in banned:
            assert b not in code, f"{rel} contains {b!r}"


def test_teacher_and_lifting_run_without_gradients():
    dep, conf, K, pose = _synthetic_clip()
    P = torch.rand(6, 4, requires_grad=False)
    u, m, _ = lifting.fuse_voxel_probs(np.array([1, 1, 2, 2, 3, 3], np.int64), P, 8,
                                       torch.device("cpu"))
    assert not m.requires_grad and m.grad_fn is None
    out, _ = lifting.propagate_dilation((5, 5, 5), np.array([62], np.int64),
                                        torch.rand(1, 4), np.array([63], np.int64), 2,
                                        torch.device("cpu"))
    assert not out.requires_grad


def test_predict_frame_is_declared_no_grad():
    src = open(os.path.join(ROOT, "gates/gate6/trident_adapter.py")).read()
    assert "@torch.no_grad()" in src
    assert "requires_grad_(False)" in src


# =========================================================================== #
# Official masks and the frozen binary reproduction
# =========================================================================== #
PINNED_BINARY = {("semantickitti", "B-D"): 0.1584, ("occ3d", "B-D"): 0.2070,
                 ("kitti360", "B-D"): 0.1342, ("kitti360", "B-R"): 0.0376}
BINARY_TOL = 5e-4


@pytest.mark.parametrize("key,expected", sorted(PINNED_BINARY.items()))
def test_frozen_binary_occupancy_is_reproduced(key, expected):
    ds, cond = key
    s = os.path.join(ART, f"summary_{ds}.json")
    if not os.path.exists(s):
        pytest.skip(f"{ds} not evaluated yet")
    # the reproduction gates report the mean of per-clip IoU; compare like with like
    got = json.load(open(s))["conditions"][cond]["binary_iou_mean_per_clip"]
    assert abs(got - expected) <= BINARY_TOL, (
        f"{ds} {cond}: {got:.4f} vs frozen {expected:.4f} "
        f"(|d|={abs(got-expected):.5f} > {BINARY_TOL})")


def test_official_invalid_mask_behaviour_on_a_real_target():
    """KITTI-360: ``.invalid`` alone is NOT the evaluation mask.

    An occupied voxel is kept even where ``.invalid`` marks it -- the asymmetry Gate 5.2
    verified elementwise. Re-asserted here because Gate 6 scores *classes* through the same
    mask, so getting it wrong would silently delete occupied voxels from every metric.
    """
    from gates.gate6 import targets as g6t
    from sscbench_kitti360.adapter import load_target, binary_target
    p = os.path.join(g6t.KITTI360_TARGET_ROOT, "preprocess", "labels",
                     F.KITTI360_SEQUENCE, "000040_1_1.npy")
    if not os.path.exists(p):
        pytest.skip("target tree unavailable")
    t, keep = load_target(g6t.KITTI360_TARGET_ROOT, 40)
    occ, valid = binary_target(t)
    assert set(np.unique(t)) <= set(range(19)) | {255}
    assert np.array_equal(keep, t != 255)
    assert (occ == ((t != 0) & keep)).all()
    assert not keep.all(), "some voxels must be ignored, or the mask is inert"
    assert occ.sum() > 0


def test_occ3d_evaluation_mask_is_the_frozen_single_camera_one():
    from gates.gate6 import targets as g6t
    assert g6t.OCC3D_APPLY_CAMERA_MASK is True
    assert g6t.OCC3D_APPLY_LIDAR_MASK is False
    assert g6t.OCC3D_SINGLE_CAMERA_X_CUT == 100
    recs = F.read_manifest("occ3d", REPO_ROOT)
    p = os.path.join(g6t.OCC3D_ROOT, recs[0].raw["anchor_gt_path"])
    if not os.path.exists(p):
        pytest.skip("Occ3D target tree unavailable")
    t, keep = g6t.semantic_target("occ3d", recs[0].raw, REPO_ROOT)
    assert t.shape == (200, 200, 16)
    assert not keep[:100].any(), "the rear half must be excluded in the single-camera setting"
    assert set(np.unique(t)) <= set(range(18)) | {255}


# =========================================================================== #
# Precommit integrity
# =========================================================================== #
def test_precommit_is_pinned_and_matches_the_code():
    pin = _json("precommit_pin.json")
    p = os.path.join(ROOT, pin["path"])
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == pin["sha256"], (
        "the precommit file changed after it was pinned")
    import yaml
    cfg = yaml.safe_load(open(p))
    for ds in vocab.DATASETS:
        v = vocab.load(ds)
        assert cfg["vocabulary"][ds]["phrases"] == list(v.phrases)
        assert cfg["vocabulary"][ds]["labels"] == list(v.labels)
    assert cfg["decision_rules"]["binary_sanity_tolerance_abs"] == BINARY_TOL
    assert cfg["decision_rules"]["binary_sanity_targets"] == {
        "semantickitti_B_D": 0.1584, "occ3d_B_D": 0.2070,
        "kitti360_B_D": 0.1342, "kitti360_B_R": 0.0376}
    assert cfg["negative_controls"]["vocabulary_permutation"]["n_permutations"] == 100
    assert cfg["statistics"]["n_resamples"] == 10000
    assert cfg["technical_fallback"]["cat_seg_used"] is False
    assert cfg["frozen_geometry"]["correction"]["physical_radius_m"] == 0.4
    assert pipelines.DILATE_RADIUS_VOXELS == 2
    assert pipelines.CONF_THRESHOLD == 1.5
    assert [pipelines.MIN_DEPTH_M, pipelines.MAX_DEPTH_M] == [1.0, 60.0]


# =========================================================================== #
# Float16 caching
# =========================================================================== #
def test_float16_cache_does_not_change_class_predictions_or_fused_results():
    """The brief permits float16 only once it is shown not to matter downstream.

    Pixel-level agreement is checked while caching; this asserts the stronger, fused
    claim, measured by re-running the teacher in float32 on whole clips.
    """
    d = _json("fp16_fidelity_kitti360.json")
    assert d["n_clips"] >= 3
    assert d["occupancy_identical"], "occupancy must not depend on the teacher at all"
    assert d["max_abs_prob_err"] < 1e-3
    assert d["raw_disagreement_fraction"] < 1e-3, d["raw_disagreement_fraction"]
    assert d["dil_disagreement_fraction"] < 1e-3, d["dil_disagreement_fraction"]
    assert d["total_raw_voxels"] > 10000, "the check must cover a meaningful number of voxels"


def test_cached_semantics_are_probability_simplices_on_the_lingbot_lattice():
    from gates.gate6 import frames as _F
    for ds in vocab.DATASETS:
        u = _F.unique_frames(ds, REPO_ROOT)
        p = _F.semantic_cache_path(ds, u[0][0])
        if not os.path.exists(p):
            pytest.skip(f"{ds} semantics not cached yet")
        with np.load(p) as z:
            pr = z["probs"].astype(np.float32)
            hp, wp = [int(x) for x in z["proc_hw"]]
            assert pr.shape == (OFFICIAL_COUNTS[ds], hp, wp)
            assert np.isfinite(pr).all() and (pr >= 0).all()
            assert np.abs(pr.sum(0) - 1).max() < 5e-3
            assert int(z["label"].max()) < OFFICIAL_COUNTS[ds]
        # the cached lattice must be the lattice the depth map lives on
        rec = _F.read_manifest(ds, REPO_ROOT)[0]
        lp = _F.lingbot_cache_path(ds, rec.clip_id, REPO_ROOT)
        if os.path.exists(lp):
            with np.load(lp) as z2:
                assert tuple(int(x) for x in z2["proc_hw"]) == (hp, wp)
                assert z2["pred_depth"].shape[1:] == (hp, wp)


# =========================================================================== #
# Reported results are internally consistent
# =========================================================================== #
@pytest.mark.parametrize("ds", list(vocab.DATASETS))
def test_reported_decomposition_and_counts_are_consistent(ds):
    s = os.path.join(ART, f"summary_{ds}.json")
    if not os.path.exists(s):
        pytest.skip(f"{ds} not evaluated yet")
    j = json.load(open(s))
    v = vocab.load(ds)
    for cond, c in j["conditions"].items():
        d = c["decomposition"]
        assert d["coverage_miss"] + d["naming_error"] + d["correct"] == d["n_valid_gt_occupied"]
        assert d["sums_to_one"]
        # the TP-conditioned population is exactly the correct + naming voxels
        assert c["tp_conditioned"]["n"] == d["correct"] + d["naming_error"]
        # the support split partitions it
        assert (c["tp_conditioned_reconstruction_support"]["n"]
                + c["tp_conditioned_dilation_only"]["n"] == c["tp_conditioned"]["n"])
        # raw reconstruction has no dilation-only voxels by construction
        if cond == "B-R":
            assert c["tp_conditioned_dilation_only"]["n"] == 0
        assert len(c["per_class"]) == len(v)
        assert 0.0 <= c["ssc_miou"] <= 1.0 and 0.0 <= c["binary_iou"] <= 1.0


@pytest.mark.parametrize("ds", list(vocab.DATASETS))
def test_permutation_control_and_bootstrap_are_reported(ds):
    p = os.path.join(ART, f"analysis_{ds}.json")
    if not os.path.exists(p):
        pytest.skip(f"{ds} not analysed yet")
    a = json.load(open(p))
    for cond in ("B-R", "B-D"):
        pc = a["permutation_control"][cond]
        assert pc["n_permutations"] == 100 and pc["seed"] == 0
        assert len(pc["values"]) == 100
    for m in ("ssc_miou", "tp_accuracy", "coverage_miss_fraction", "naming_error_fraction"):
        b = a["bootstrap"][m]
        assert b["n_boot"] == 10000 and b["seed"] == 0
        assert b["ci"]["B-D"][0] <= b["point"]["B-D"] <= b["ci"]["B-D"][1]
    assert a["n_units"] >= 8
    if ds != "occ3d":
        assert a["blocks_are_not_independent_scenes"] is True
        assert "block" in a["bootstrap_unit"] and "scene" not in a["bootstrap_unit"]


# =========================================================================== #
# Optional comparator
# =========================================================================== #
def test_comparator_is_scored_on_the_same_clips_and_never_selected_anything():
    d = _json("occany_comparator.json")
    c, t = d["comparator"], d["primary"]
    assert c["n_clips"] == t["n_clips"] > 0, "both readouts must be scored on the same clips"
    # the comparator ran after the primary was pinned
    mp = os.path.join(ART, "prediction_manifest_semantickitti.json")
    assert os.path.getmtime(mp) <= os.path.getmtime(_art("occany_comparator.json"))
    # the primary is what the precommit predeclared, regardless of the comparator's score
    import yaml
    cfg = yaml.safe_load(open(os.path.join(ROOT, _json("precommit_pin.json")["path"])))
    assert cfg["teacher"]["name"] == "Trident-H"
    assert cfg["optional_comparator"]["may_influence_primary"] is False
    assert cfg["technical_fallback"]["cat_seg_used"] is False
    # the comparator declines to name some voxels; that must be reported, not hidden
    assert 0.0 < c["named_fraction_of_frozen_occupancy"] <= 1.0
    assert t["named_fraction_of_frozen_occupancy"] == pytest.approx(1.0)
    for s in (c, t):
        dd = s["decomposition"]
        assert dd["coverage_miss"] + dd["naming_error"] + dd["correct"] == dd["n_valid_gt_occupied"]


def test_bottleneck_diagnosis_holds_for_the_comparator_too():
    """The headline conclusion must not depend on which frozen teacher was used."""
    d = _json("occany_comparator.json")
    for k in ("comparator", "primary"):
        dd = d[k]["decomposition"]
        assert dd["coverage_miss_fraction"] > dd["naming_error_fraction"], k


def test_diagnosis_matches_the_predeclared_rules():
    d = _json("diagnosis.json")
    n = d["n_datasets_passing"]
    expected = ("FROZEN_TRIDENT_SEMANTICS_TRANSFER" if n == 3 else
                "FROZEN_TRIDENT_SEMANTICS_PARTIAL" if n >= 1 else
                "FROZEN_TRIDENT_SEMANTICS_FAIL")
    assert d["transfer_diagnosis"] == expected
    cov = d["coverage_dominates_on"]
    exp_b = ("COVERAGE_DOMINATES" if cov >= 2 else
             "NAMING_DOMINATES" if d["naming_dominates_on"] >= 2 else "MIXED_BOTTLENECK")
    assert d["bottleneck_diagnosis"] == exp_b
    # and the two branches cannot both fire
    assert not (cov >= 2 and d["naming_dominates_on"] >= 2)


def test_report_exists_and_carries_the_diagnosis():
    p = os.path.join(ROOT, "reports", "gate6", "frozen_trident_semantic_lifting.md")
    if not os.path.exists(p):
        pytest.skip("report not generated yet")
    text = open(p).read()
    d = _json("diagnosis.json")
    assert d["transfer_diagnosis"] in text and d["bottleneck_diagnosis"] in text
    assert "{{" not in text, "an unsubstituted table placeholder is left in the report"
    for section in ("## 6. Binary sanity checks", "## 13. Vocabulary-permutation control",
                    "## 15. Leakage audit", "## 18. Recommended next experiment",
                    "## 22. Deviations and blockers"):
        assert section in text, section
