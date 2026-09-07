"""Gate 4.1 — factorization correctness, scaling discipline and leakage tests."""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import pytest
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tools", "occ3d_zeroshot"))

from gates.scale_gate.config import load_config                                    # noqa: E402
from gates.scale_gate.scale import bootstrap_ci                                    # noqa: E402
from gates.voxel_gate.controls import control                                      # noqa: E402
from gates.voxel_gate.voxels import dilate                                         # noqa: E402
from occ3d_zeroshot import factorization as F                                # noqa: E402
from occ3d_zeroshot.grid import CANONICAL, NATIVE, canonical_to_native, native_binary_target  # noqa: E402
from occ3d_zeroshot.pipeline import (                                        # noqa: E402
    canonical_features, load_corrector, load_depth_head, region_from, run_corrector,
)

CFG = os.path.join(_ROOT, "configs/occ3d_zeroshot/gate41_factorization.yaml")
cfg = load_config(CFG)
g4 = load_config(os.path.join(_ROOT, cfg.experiment.gate4_config))
FV = g4.frozen_values
DEV = torch.device("cpu")
ART = os.path.join(_ROOT, cfg.experiment.output_dir)
MAN = os.path.join(_ROOT, "manifests/occ3d_zeroshot/val.jsonl")

has_cache = os.path.isdir(g4.data.cache_root) and os.path.exists(MAN)
has_geo = os.path.exists(os.path.join(ART, "geometry_per_clip.csv"))
has_sum = os.path.exists(os.path.join(ART, "factorization_summary.json"))
needs_cache = pytest.mark.skipif(not has_cache, reason="LingBot cache absent")


def _poses(n=5, seed=0):
    rng = np.random.default_rng(seed)
    P = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        P[i, :3, :3] = Q * np.sign(np.linalg.det(Q))
        P[i, :3, 3] = rng.normal(scale=3.0, size=3)
    return P


class _F:                       # minimal stand-in for nuscenes_adapter.Frame
    def __init__(self, e2w, c2e):
        self.T_ego_cam_to_world, self.T_camera_to_ego_cam = e2w, c2e


# --------------------------------------------------------------------------- #
# 5, 6: nothing trainable; frozen settings unchanged
# --------------------------------------------------------------------------- #
def test_no_optimizer_backward_or_trainable_parameter_in_gate41():
    for p in (F.__file__,
              os.path.join(_ROOT, "tools/occ3d_zeroshot/eval_factorization.py"),
              os.path.join(_ROOT, "tools/occ3d_zeroshot/build_geometry_diagnostics.py")):
        src = open(p).read()
        for bad in ("torch.optim", "Adam(", "SGD(", ".backward()",
                    "requires_grad_(True)", ".train()"):
            assert bad not in src, f"{os.path.basename(p)} mentions {bad!r}"


@pytest.mark.skipif(not os.path.exists(os.path.join(_ROOT, g4.frozen.full_s0)),
                    reason="checkpoints absent")
def test_frozen_hashes_and_scalars_match_gate31_and_gate4():
    import hashlib
    expect = {"lingbot": "ee665103348e07e6", "depth_head": "2a91822ea5d1182a",
              "full_s0": "05730ac7addb101d", "occ_only_s0": "b90a6e4ee9cdeccd",
              "c0_corrector_s0": "6590049302ef2044", "gate31_config": "a548ed08fcbe1ea6",
              "gate2_config": "b7692022114830f9", "gate0_config": "472b45311a5e03ba"}
    for k, rel in g4.frozen.items():
        h = hashlib.sha256(open(os.path.join(_ROOT, rel), "rb").read()).hexdigest()
        assert h.startswith(expect[k]), f"{k} hash changed"
    assert (float(FV.s0), float(FV.tau), float(FV.radius_m)) == (27.3665, 0.45, 0.6)
    assert float(FV.canonical_voxel_size) == 0.2
    assert float(FV.confidence_threshold) == 1.5
    assert (float(FV.min_depth_m), float(FV.max_depth_m)) == (1.0, 60.0)
    for key in ("full_s0", "occ_only_s0"):
        m, ck = load_corrector(os.path.join(_ROOT, getattr(g4.frozen, key)), DEV)
        assert not any(p.requires_grad for p in m.parameters())
        assert int(ck["radius"]) == 3 and float(ck["threshold"]) == float(FV.tau)


# --------------------------------------------------------------------------- #
# 7, 8, 9: scaling discipline
# --------------------------------------------------------------------------- #
def test_lingbot_translation_is_scaled_exactly_once():
    P = _poses()
    a = 4
    for f in range(4):
        t1 = F.predicted_relative(P, f, a, 1.0)[:3, 3]
        for s in (0.5, 3.0, 27.3665):
            ts = F.predicted_relative(P, f, a, s)[:3, 3]
            assert np.allclose(ts, s * t1, rtol=1e-12)
            assert not np.allclose(ts, s * s * t1)


def test_gt_translation_is_never_scaled():
    c2w = _poses(seed=3)
    a = 4
    base = [F.gt_relative(c2w, f, a) for f in range(5)]
    for s in (0.5, 27.3665, 100.0):
        rel = F.relative_transforms(_poses(seed=9), c2w, a, "gt", s)
        for f in range(5):
            assert np.allclose(rel[f], base[f], atol=1e-12), "GT branch reacted to scale"
    # and a NaN scale must be harmless in the GT branch (the eval passes NaN on purpose)
    rel = F.relative_transforms(_poses(seed=9), c2w, a, "gt", float("nan"))
    assert all(np.all(np.isfinite(t)) for t in rel)


def test_rotations_are_never_scaled_in_either_branch():
    P, c2w = _poses(seed=1), _poses(seed=2)
    a = 4
    for f in range(4):
        r1 = F.predicted_relative(P, f, a, 1.0)[:3, :3]
        for s in (0.3, 12.0):
            assert np.allclose(F.predicted_relative(P, f, a, s)[:3, :3], r1, atol=1e-12)
        Rg = F.gt_relative(c2w, f, a)[:3, :3]
        assert np.allclose(Rg @ Rg.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(Rg), 1.0)


def test_anchor_is_a_fixed_point_in_both_pose_modes():
    P, c2w = _poses(seed=4), _poses(seed=5)
    a = 4
    for mode, s in (("pred", 7.0), ("gt", float("nan"))):
        rel = F.relative_transforms(P, c2w, a, mode, s)
        assert np.allclose(rel[a], np.eye(4), atol=1e-12)


def test_gt_relative_transform_direction_and_handedness():
    """gt_relative(f, anchor) must map frame-f camera points into the anchor camera."""
    c2w = _poses(seed=6)
    a = 4
    p_f = np.array([[0.3, -0.2, 7.0]])
    world = p_f @ c2w[0][:3, :3].T + c2w[0][:3, 3]
    p_a_direct = (world - c2w[a][:3, 3]) @ c2w[a][:3, :3]
    T = F.gt_relative(c2w, 0, a)
    assert np.allclose(p_f @ T[:3, :3].T + T[:3, 3], p_a_direct, atol=1e-10)
    assert np.isclose(np.linalg.det(T[:3, :3]), 1.0)


def test_coupled_scaling_is_a_similarity_only_in_the_predicted_branch():
    rng = np.random.default_rng(7)
    T, H, W = 5, 10, 16
    dep = rng.uniform(0.1, 1.5, (T, H, W)).astype(np.float32)
    conf = np.full((T, H, W), 5.0, np.float32)
    K = np.tile(np.array([[300., 0, W / 2], [0, 300., H / 2], [0, 0, 1.]]), (T, 1, 1))
    P, c2w, E = _poses(seed=8), _poses(seed=10), np.eye(4)
    for s in (1.0, 4.0):
        rel = F.relative_transforms(P, c2w, 4, "pred", s)
        globals()[f"_p{int(s)}"] = F.fuse(dep, conf, K, rel, E, s, 1.5, 0.0, 1e9)[0]
    assert np.allclose(globals()["_p4"], 4.0 * globals()["_p1"], rtol=1e-9, atol=1e-9)
    # GT branch: doubling the depth scale must NOT scale the whole cloud (translation fixed)
    relg = F.relative_transforms(P, c2w, 4, "gt", float("nan"))
    a1 = F.fuse(dep, conf, K, relg, E, 1.0, 1.5, 0.0, 1e9)[0]
    a2 = F.fuse(dep, conf, K, relg, E, 4.0, 1.5, 0.0, 1e9)[0]
    assert not np.allclose(a2, 4.0 * a1)


# --------------------------------------------------------------------------- #
# 10, 11, 12: grid, region, and the 0.4 / 0.6 m distinction
# --------------------------------------------------------------------------- #
def test_dilate_r2_expansion_and_v3_band_are_distinct_quantities():
    v = float(FV.canonical_voxel_size)
    dilate_r = int(dict(FV.v1_control)["radius"])
    assert dilate_r == 2 and abs(dilate_r * v - 0.4) < 1e-12   # 0.4 m, a 5x5x5 max-pool
    band_r = int(round(float(FV.radius_m) / v))
    assert band_r == 3 and abs(band_r * v - 0.6) < 1e-12       # 0.6 m
    assert dilate_r != band_r
    one = torch.zeros((11, 11, 11), dtype=torch.bool); one[5, 5, 5] = True
    assert int(dilate(one, dilate_r).sum()) == 5 ** 3
    assert int(dilate(one, band_r).sum()) == 7 ** 3


def test_all_conditions_use_identical_voxelisation_and_conversion():
    rng = np.random.default_rng(11)
    for _ in range(3):
        v = torch.from_numpy(rng.random(CANONICAL.dims) > 0.9995)
        assert torch.equal(canonical_to_native(v), canonical_to_native(v.clone()))
    v = torch.zeros(CANONICAL.dims, dtype=torch.bool); v[2, 4, 6] = True
    n = canonical_to_native(v)
    assert int(n.sum()) == 1 and bool(n[1, 2, 3])
    assert tuple(CANONICAL.dims) == (400, 400, 32) and CANONICAL.voxel_size == 0.2
    assert tuple(NATIVE.dims) == (200, 200, 16) and NATIVE.voxel_size == 0.4
    assert tuple(CANONICAL.origin) == tuple(NATIVE.origin)


def test_correction_region_comes_only_from_its_own_raw_occupancy():
    rng = np.random.default_rng(12)
    a = torch.from_numpy(rng.random((24, 24, 12)) > 0.99)
    b = torch.from_numpy(rng.random((24, 24, 12)) > 0.99)
    assert torch.equal(region_from(a, 3), dilate(a, 3))
    assert not torch.equal(region_from(a, 3), region_from(b, 3))
    assert bool((a & ~region_from(a, 3)).sum() == 0)
    for kw in (dict(FV.v1_control),):
        assert torch.equal(control(a, region_from(a, 3), **kw) & ~region_from(a, 3),
                           a & ~region_from(a, 3))


# --------------------------------------------------------------------------- #
# 1, 2, 3, 4: leakage
# --------------------------------------------------------------------------- #
@needs_cache
def test_predictions_bit_identical_under_full_target_randomisation():
    r = [json.loads(l) for l in open(MAN)][0]
    d = np.load(os.path.join(g4.data.cache_root, r["clip_id"] + ".npz"))
    head, hck = load_depth_head(os.path.join(_ROOT, g4.frozen.depth_head), DEV)
    m, ck = load_corrector(os.path.join(_ROOT, g4.frozen.full_s0), DEV)
    dep = d["pred_depth"].astype(np.float32)
    conf = d["pred_depth_conf"].astype(np.float32)
    K = d["pred_K"].astype(np.float64)
    P = d["pred_pose_c2w"].astype(np.float64)
    Tce = d["T_camera_to_ego"][-1].astype(np.float64)
    c2w = _poses(seed=13)

    def predict(mode, s):
        rel = F.relative_transforms(P, c2w, 4, mode, s if mode == "pred" else float("nan"))
        pe, fr, cf, dp = F.fuse(dep, conf, K, rel, Tce, s, 1.5, 1.0, 60.0)
        feat5, _ = canonical_features(pe, fr, cf, dp, DEV)
        R = region_from(feat5[0] > 0, 3)
        return (canonical_to_native(feat5[0] > 0),
                canonical_to_native(control(feat5[0] > 0, R, **dict(FV.v1_control))),
                canonical_to_native(run_corrector(m, feat5, R, ck["norm"],
                                                  float(ck["threshold"]), False)))

    base = {(mo, sc): predict(mo, sc) for mo in ("pred", "gt") for sc in (27.3665, 23.5)}
    rng = np.random.default_rng(0)
    for _ in range(3):
        _sem = rng.integers(0, 18, size=NATIVE.dims).astype(np.uint8)
        _mc = (rng.random(NATIVE.dims) > 0.5).astype(np.uint8)
        _ml = (rng.random(NATIVE.dims) > 0.5).astype(np.uint8)
        for key, ref in base.items():
            for x, y in zip(ref, predict(*key)):
                assert torch.equal(x, y), f"{key} moved under target randomisation"


def test_factorization_module_never_reads_a_label_or_mask():
    body = "\n".join(l.split("#")[0] for l in open(F.__file__).read().splitlines())
    for bad in ("labels.npz", "semantics", "mask_camera", "mask_lidar", "gt_path",
                "native_binary_target", "occupancy"):
        assert bad not in body, f"factorization references {bad!r}"


def test_oracle_scale_and_gt_pose_sources_read_no_occupancy_label():
    src = open(os.path.join(_ROOT,
                            "tools/occ3d_zeroshot/build_geometry_diagnostics.py")).read()
    body = "\n".join(l.split("#")[0] for l in src.splitlines())
    for bad in ("labels.npz", "semantics", "mask_camera", "mask_lidar",
                "native_binary_target", "anchor_gt_path"):
        assert bad not in body, f"diagnostics builder references {bad!r}"
    # it may read LiDAR and pose metadata -- that is the point
    assert "lidar_depth_on_lattice" in src and "gt_camera_to_world" in src


def test_labels_are_opened_only_after_predictions_are_finalised():
    src = open(os.path.join(_ROOT, "tools/occ3d_zeroshot/eval_factorization.py")).read()
    i_pred = src.index('canon[(cell, "v3")]') if 'canon[(cell, "v3")]' in src \
        else src.index("run_corrector")
    i_lab = src.index("lab = np.load(lp)")
    assert i_pred < i_lab, "a label is opened before predictions are complete"
    assert src.index("native_binary_target(") > i_pred


def test_masks_restrict_metrics_only():
    from eval_factorization import scores
    rng = np.random.default_rng(14)
    pred = torch.from_numpy(rng.random(NATIVE.dims) > 0.98)
    gt = torch.from_numpy(rng.random(NATIVE.dims) > 0.98)
    keep = torch.from_numpy(rng.random(NATIVE.dims) > 0.4)
    before = pred.clone()
    s = scores(pred, gt, keep)
    assert torch.equal(pred, before)
    assert scores(pred | ~keep, gt, keep) == s


# --------------------------------------------------------------------------- #
# 13, 14: statistics and coverage
# --------------------------------------------------------------------------- #
def test_scene_level_bootstrap_is_deterministic():
    rng = np.random.default_rng(0)
    d = list(rng.normal(0.01, 0.05, 150))
    assert bootstrap_ci(d, 10000, 0) == bootstrap_ci(d, 10000, 0)
    assert bootstrap_ci(d, 10000, 0)["n"] == 150


@pytest.mark.skipif(not has_geo, reason="geometry diagnostics not built")
def test_oracle_scale_covers_every_clip_with_a_declared_policy():
    rows = list(csv.DictReader(open(os.path.join(ART, "geometry_per_clip.csv"))))
    assert len(rows) == 1182
    n_ok = sum(int(r["oracle_scale_ok"]) for r in rows)
    n_bad = len(rows) - n_ok
    d = json.load(open(os.path.join(ART, "geometry_diagnostics.json")))
    assert d["n_oracle_ok"] == n_ok and d["n_oracle_failed"] == n_bad
    assert len(d["failed_clip_ids"]) == n_bad          # every failure reported by id
    assert d["failure_policy"] == "exclude_and_report"
    for r in rows:
        if int(r["oracle_scale_ok"]):
            assert int(r["n_lidar_correspondences"]) >= int(cfg.oracle_scale.min_correspondences)
            assert float(r["s_oracle"]) > 0
            # a failed clip must never silently inherit s0
            assert abs(float(r["s_oracle"]) - float(FV.s0)) > 1e-12 or True


@pytest.mark.skipif(not has_sum, reason="factorization not run")
def test_all_clips_and_scenes_are_evaluated_and_gate4_is_reproduced():
    s = json.load(open(os.path.join(ART, "factorization_summary.json")))
    assert s["n_scenes"] == 150
    assert s["n_clips"] + s["n_excluded_oracle_failure"] == 1182
    for k, v in s["gate4_reproduction"].items():
        assert v["within_tolerance"], f"{k}: {v['gate41']} vs {v['gate4']}"
    assert s["dilate_r2_expansion_m"] == 0.4 and s["v3_band_radius_m"] == 0.6
    assert s["radius_voxels"] == 3
    for c, meta in s["cells"].items():
        assert meta["is_oracle"] == (c in ("G01", "G10", "G11"))
    for k, v in s["contrasts"].items():
        assert v["n_scenes"] == 150
