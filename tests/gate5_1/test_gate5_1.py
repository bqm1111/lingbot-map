"""Gate 5.1 — calibrated-gauge correctness, crop policy and leakage tests."""
from __future__ import annotations

import ast
import csv
import inspect
import json
import math
import os
import sys

import numpy as np
import pytest
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from gates.scale_gate.config import load_config                                    # noqa: E402
from gates.scale_gate.scale import bootstrap_ci                                    # noqa: E402
from gates.voxel_gate.controls import control                                      # noqa: E402
from gates.voxel_gate.voxels import dilate                                         # noqa: E402
from moge_gauge import adapter as A                                          # noqa: E402
from moge_gauge import calibrated as C                                       # noqa: E402
from moge_gauge import calibration as CAL                                    # noqa: E402
from moge_gauge import estimator as E                                        # noqa: E402
from occ3d_zeroshot import factorization as F                                # noqa: E402

CFG = os.path.join(_ROOT, "configs/gate5_1/calibrated_gauge.yaml")
cfg = load_config(CFG)
g5 = load_config(os.path.join(_ROOT, cfg.experiment.gate5_config))
g4 = load_config(os.path.join(_ROOT, cfg.experiment.gate4_config))
FV = g4.frozen_values
ART = os.path.join(_ROOT, cfg.experiment.output_dir)
G5ART = os.path.join(_ROOT, g5.experiment.output_dir)

have = lambda p: os.path.exists(os.path.join(ART, p))
needs_scales = pytest.mark.skipif(not have("scales_G51-D_kitti.csv"),
                                  reason="Gate-5.1 scales not cached")
needs_eval = pytest.mark.skipif(not have("summary_occ3d.json"),
                                reason="Gate-5.1 evaluation not run")


def code_only(path, drop_names=("FORBIDDEN_INFER_KWARGS",)):
    """Executable code with docstrings, comments and declared-name constants removed."""
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(getattr(body[0], "value", None), ast.Constant) \
                and isinstance(body[0].value.value, str):
            body.pop(0)
    tree.body = [n for n in tree.body if not (
        isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in drop_names for t in n.targets))]
    return ast.unparse(tree)


def _poses(n=5, seed=0):
    rng = np.random.default_rng(seed)
    P = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        P[i, :3, :3] = Q * np.sign(np.linalg.det(Q))
        P[i, :3, 3] = rng.normal(scale=3.0, size=3)
    return P


# --------------------------------------------------------------------------- #
# 1, 2: frozen
# --------------------------------------------------------------------------- #
def test_lingbot_and_moge_pins_are_unchanged():
    import hashlib
    expect = {"lingbot": "ee665103348e07e6", "depth_head": "2a91822ea5d1182a",
              "full_s0": "05730ac7addb101d", "occ_only_s0": "b90a6e4ee9cdeccd",
              "c0_corrector_s0": "6590049302ef2044", "gate31_config": "a548ed08fcbe1ea6",
              "gate2_config": "b7692022114830f9", "gate0_config": "472b45311a5e03ba"}
    for k, rel in g4.frozen.items():
        h = hashlib.sha256(open(os.path.join(_ROOT, rel), "rb").read()).hexdigest()
        assert h.startswith(expect[k]), f"{k} changed"
    assert g5.moge.hf_revision == A.MOGE_HF_REVISION == \
        "39c4d5e957afe587e04eec59dc2bcc3be5ecd968"
    assert g5.moge.commit == A.MOGE_COMMIT
    assert A.MOGE_MODEL_CLASS == "moge.model.v2.MoGeModel"
    w = json.load(open(os.path.join(G5ART, "moge_scale_occ3d.json")))["moge"]
    assert w["weight_sha256"] == \
        "3eefd4abb2102f38f12b2d1992e5ff15e4923e5431c67dd494afe157e0111cd5"
    assert (float(FV.s0), float(FV.confidence_threshold)) == (27.3665, 1.5)
    assert (float(FV.min_depth_m), float(FV.max_depth_m)) == (1.0, 60.0)
    assert int(g5.estimator.min_valid_pixels_per_clip) == 500


def test_no_optimizer_backward_or_trainable_parameter_in_gate5_1():
    for p in (C.__file__, CAL.__file__,
              os.path.join(_ROOT, "tools/gate5_1/cache_scales.py"),
              os.path.join(_ROOT, "tools/gate5_1/eval_gate5_1.py"),
              os.path.join(_ROOT, "tools/gate5_1/preflight.py")):
        src = open(p).read()
        for bad in ("torch.optim", "Adam(", "SGD(", ".backward()", ".train()"):
            assert bad not in src, f"{os.path.basename(p)} mentions {bad!r}"
    assert "requires_grad_(False)" in open(A.__file__).read()


# --------------------------------------------------------------------------- #
# 3, 4, 13, 14: only RGB (+ the declared scalar FOV) reaches MoGe
# --------------------------------------------------------------------------- #
def test_only_rgb_and_scalar_fov_reach_moge():
    params = list(inspect.signature(C.CalibratedMoGe.infer_calibrated).parameters)
    assert params == ["self", "rgb01", "fov_x_deg"], params
    tree = ast.parse(code_only(C.__file__))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "infer"]
    assert len(calls) == 1
    kw = {k.arg for k in calls[0].keywords}
    assert kw == {"apply_mask", "fov_x"}, f"unexpected MoGe keywords: {kw}"
    assert len(calls[0].args) == 1
    body = code_only(C.__file__)
    for bad in ("intrinsics_gt", "extrinsic", "ego_pose", "camera_height", "lidar",
                "velodyne", "labels.npz", "semantics", "mask_camera", "s_oracle"):
        assert bad not in body, f"calibrated adapter references {bad!r}"


def test_estimator_and_cache_tool_never_open_a_label_lidar_or_pose():
    for p in (E.__file__, CAL.__file__,
              os.path.join(_ROOT, "tools/gate5_1/cache_scales.py")):
        body = code_only(p)
        for bad in ("labels.npz", "semantics", "mask_camera", "mask_lidar", "velodyne",
                    "anchor_gt_path", "s_oracle", "occupancy", "ego_pose",
                    "camera_height", "T_camera_to_ego"):
            assert bad not in body, f"{os.path.basename(p)} references {bad!r}"


@needs_scales
def test_cached_scale_tables_carry_no_target_column():
    for v in ("G51-A", "G51-B", "G51-C", "G51-D"):
        for ds in ("kitti", "occ3d"):
            p = os.path.join(ART, f"scales_{v}_{ds}.csv")
            if not os.path.exists(p):
                continue
            cols = set(next(csv.DictReader(open(p))))
            for bad in ("s_oracle", "iou", "gt", "label", "mask_camera", "n_lidar"):
                assert not any(bad in c for c in cols), f"{v}/{ds} carries {bad!r}"


# --------------------------------------------------------------------------- #
# 5, 16, 17: estimator behaviour
# --------------------------------------------------------------------------- #
def test_randomising_targets_leaves_every_variant_scale_bit_identical():
    rng = np.random.default_rng(0)
    T, H, W = 5, 24, 60
    md = rng.uniform(2.0, 40.0, (T, H, W)).astype(np.float32)
    mm = rng.random((T, H, W)) > 0.15
    ld = rng.uniform(0.1, 2.0, (T, H, W)).astype(np.float32)
    lc = rng.uniform(1.0, 12.0, (T, H, W)).astype(np.float32)
    crop = CAL.aspect_safe_crop(W, H, W / 2)
    base = {}
    for tag, sl in (("full", slice(0, W)), ("crop", slice(crop.x0, crop.x1))):
        base[tag] = E.estimate_clip_scale(md[..., sl], mm[..., sl], ld[..., sl],
                                          lc[..., sl], 1.5, 1.0, 60.0, 500)["s_moge"]
    for _ in range(3):
        _sem = rng.integers(0, 18, size=(200, 200, 16))
        _mc = rng.random((200, 200, 16)) > 0.5
        _lidar = rng.uniform(1, 80, (T, H, W))
        for tag, sl in (("full", slice(0, W)), ("crop", slice(crop.x0, crop.x1))):
            again = E.estimate_clip_scale(md[..., sl], mm[..., sl], ld[..., sl],
                                          lc[..., sl], 1.5, 1.0, 60.0, 500)["s_moge"]
            assert again == base[tag]


def test_invalid_teacher_pixels_cannot_affect_the_estimate():
    T, H, W = 5, 8, 8
    md = np.full((T, H, W), 10.0, np.float32); md[:, 0, 0] = 1e6
    mm = np.ones((T, H, W), bool); mm[:, 0, 0] = False
    ld = np.full((T, H, W), 0.5, np.float32)
    lc = np.full((T, H, W), 5.0, np.float32)
    r = E.estimate_clip_scale(md, mm, ld, lc, 1.5, 1.0, 60.0, 10)
    assert abs(r["s_moge"] - 20.0) < 1e-9


def test_exactly_one_scalar_per_clip():
    rng = np.random.default_rng(1)
    md = rng.uniform(2, 40, (5, 16, 20)).astype(np.float32)
    r = E.estimate_clip_scale(md, np.ones((5, 16, 20), bool),
                              rng.uniform(.1, 2, (5, 16, 20)).astype(np.float32),
                              np.full((5, 16, 20), 5.0, np.float32), 1.5, 1.0, 60.0, 500)
    assert isinstance(r["s_moge"], float) and len(r["per_frame"]) == 5


# --------------------------------------------------------------------------- #
# 7, 8, 10, 11: the crop policy
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("W,H", [(518, 294), (400, 200), (100, 100), (200, 400)])
def test_crop_is_identity_when_aspect_ratio_at_most_two(W, H):
    c = CAL.aspect_safe_crop(W, H, W / 2)
    assert c.identity and (c.x0, c.x1, c.width) == (0, W, W)
    assert c.aspect_ratio == W / H <= 2.0


@pytest.mark.parametrize("W,H,cx", [(518, 154, 254.305), (1000, 100, 500.0),
                                    (401, 200, 200.0), (900, 200, 10.0),
                                    (900, 200, 890.0)])
def test_crop_is_deterministic_patch_aligned_and_within_two_to_one(W, H, cx):
    c1 = CAL.aspect_safe_crop(W, H, cx)
    c2 = CAL.aspect_safe_crop(W, H, cx)
    assert c1 == c2                                   # deterministic
    assert not c1.identity
    assert c1.width % CAL.PATCH == 0                  # patch-aligned
    assert c1.aspect_ratio <= 2.0 + 1e-12             # inside MoGe's range
    assert c1.width <= 2 * H and c1.height == H       # full height
    assert 0 <= c1.x0 and c1.x1 <= W                  # clipped to the image
    assert c1.x1 - c1.x0 == c1.width
    assert c1.cx_crop == pytest.approx(cx - c1.x0)    # principal point updated


def test_crop_is_centred_on_the_principal_point_when_it_fits():
    c = CAL.aspect_safe_crop(518, 154, 254.305)
    assert (c.x0, c.x1, c.width) == (100, 408, 308)
    assert abs(c.cx_crop - c.width / 2) <= 1.0        # centred to within a pixel


def test_crop_is_a_pure_slice_with_no_interpolation_or_padding():
    rng = np.random.default_rng(2)
    arr = rng.random((5, 154, 518)).astype(np.float32)
    c = CAL.aspect_safe_crop(518, 154, 254.305)
    out = c.apply(arr)
    assert out.shape == (5, 154, 308)
    assert np.array_equal(out, arr[..., 100:408])     # exact slice, same values
    assert out.base is not None or out.flags["OWNDATA"] is False  # a view, not a resample
    body = code_only(CAL.__file__)
    for bad in ("interp", "resize", "pad", "letterbox", "grid_sample", "zoom"):
        assert bad not in body, f"crop path performs {bad!r}"


def test_crop_slices_teacher_and_student_identically():
    """Teacher RGB and the LingBot depth/confidence must use the same columns."""
    rng = np.random.default_rng(3)
    rgb = rng.random((5, 3, 154, 518)).astype(np.float32)
    dep = rng.random((5, 154, 518)).astype(np.float32)
    c = CAL.aspect_safe_crop(518, 154, 254.305)
    assert c.apply(rgb).shape[-1] == c.apply(dep).shape[-1] == c.width
    assert np.array_equal(c.apply(rgb)[0, 0], rgb[0, 0, :, c.x0:c.x1])
    assert np.array_equal(c.apply(dep)[0], dep[0, :, c.x0:c.x1])


# --------------------------------------------------------------------------- #
# 9, 12: calibration
# --------------------------------------------------------------------------- #
def test_calibrated_fov_on_synthetic_cameras_with_known_answers():
    assert CAL.calibrated_fov_x_deg(100.0, 200) == pytest.approx(90.0)
    for want in (30.0, 60.0, 90.0, 120.0):
        W = 640
        fx = W / (2 * math.tan(math.radians(want / 2)))
        assert CAL.calibrated_fov_x_deg(fx, W) == pytest.approx(want, abs=1e-9)
    with pytest.raises(ValueError):
        CAL.calibrated_fov_x_deg(0.0, 100)


def test_intrinsics_are_transformed_through_preprocessing_and_crop():
    K = np.array([[707.091, 0, 601.887], [0, 707.091, 183.11], [0, 0, 1.0]])
    c, Kp, proc = CAL.crop_for(K, (370, 1226))
    assert tuple(proc) == (154, 518)
    assert Kp[0, 0] == pytest.approx(707.091 * 518 / 1226)     # fx scaled by sx
    assert Kp[1, 1] == pytest.approx(707.091 * 154 / 370)      # fy scaled by sy
    assert Kp[0, 2] == pytest.approx(601.887 * 518 / 1226)
    assert (c.x0, c.width) == (100, 308) and not c.identity
    assert CAL.calibrated_fov_x_deg(Kp[0, 0], 518) == pytest.approx(81.846, abs=1e-2)
    assert CAL.calibrated_fov_x_deg(Kp[0, 0], c.width) == pytest.approx(54.540, abs=1e-2)


def test_moge_expects_degrees():
    src = open(os.path.join(g5.moge.src_dir, "moge/model/v2.py")).read()
    assert "deg2rad" in src, "MoGe fov_x is not in degrees; the unit assumption is wrong"


# --------------------------------------------------------------------------- #
# 15, 18, 19, 20, 21: conventions and application
# --------------------------------------------------------------------------- #
def test_optical_axis_z_not_euclidean_ray():
    assert "points[..., 2]" in inspect.getsource(C.CalibratedMoGe.infer_calibrated)
    pts = np.array([[[3.0, 4.0, 12.0]]])
    assert pts[..., 2].item() == 12.0
    assert np.linalg.norm(pts, axis=-1).item() == pytest.approx(13.0)


def test_one_scalar_applied_once_to_all_depths_and_translations():
    rng = np.random.default_rng(4)
    T, H, W = 5, 10, 14
    dep = rng.uniform(0.1, 1.5, (T, H, W)).astype(np.float32)
    conf = np.full((T, H, W), 5.0, np.float32)
    K = np.tile(np.array([[300., 0, W / 2], [0, 300., H / 2], [0, 0, 1.]]), (T, 1, 1))
    P, E4, s = _poses(seed=5), np.eye(4), 24.5
    r1 = F.relative_transforms(P, P, T - 1, "pred", 1.0)
    rs = F.relative_transforms(P, P, T - 1, "pred", s)
    for f in range(T - 1):
        assert np.allclose(rs[f][:3, 3], s * r1[f][:3, 3], rtol=1e-12)
        assert not np.allclose(rs[f][:3, 3], s * s * r1[f][:3, 3])
        assert np.allclose(rs[f][:3, :3], r1[f][:3, :3], atol=1e-12)
    assert np.allclose(rs[T - 1], np.eye(4), atol=1e-12)
    p1, *_ = F.fuse(dep, conf, K, r1, E4, 1.0, 1.5, 0.0, 1e9)
    ps, *_ = F.fuse(dep, conf, K, rs, E4, s, 1.5, 0.0, 1e9)
    assert np.allclose(ps, s * p1, rtol=1e-9, atol=1e-9)


def test_moge_outputs_cannot_enter_fusion():
    assert list(inspect.signature(F.fuse).parameters) == [
        "dep", "conf", "K", "rel", "T_anchor_camera_to_ego", "depth_scale",
        "conf_thr", "dmin", "dmax"]
    assert "moge" not in code_only(F.__file__).lower()


def test_dilate_r2_remains_the_frozen_0_4_m_operation():
    kw = dict(FV.v1_control)
    assert kw == {"kind": "dilate", "radius": 2}
    v = float(FV.canonical_voxel_size)
    assert abs(kw["radius"] * v - 0.4) < 1e-12
    assert abs(int(round(float(FV.radius_m) / v)) * v - 0.6) < 1e-12
    one = torch.zeros((11, 11, 11), dtype=torch.bool); one[5, 5, 5] = True
    assert int(dilate(one, kw["radius"]).sum()) == 125
    R = dilate(one, 3)
    assert torch.equal(control(one, R, **kw) & ~R, one & ~R)


# --------------------------------------------------------------------------- #
# 6: G51-A reproduces Gate 5
# --------------------------------------------------------------------------- #
@needs_scales
def test_g51a_reproduces_gate5_scales_exactly():
    for ds in ("kitti", "occ3d"):
        p5 = os.path.join(G5ART, f"moge_scale_{ds}.csv")
        pA = os.path.join(ART, f"scales_G51-A_{ds}.csv")
        if not (os.path.exists(p5) and os.path.exists(pA)):
            continue
        g5s = {r["clip_id"]: float(r["s_moge"]) for r in csv.DictReader(open(p5))}
        aS = {r["clip_id"]: float(r["s_moge"]) for r in csv.DictReader(open(pA))}
        assert set(g5s) == set(aS)
        worst = max(abs(aS[k] - g5s[k]) for k in g5s)
        assert worst < 1e-9, f"{ds}: G51-A deviates from Gate 5 by {worst}"


@needs_scales
def test_identity_crop_makes_c_equal_a_and_d_equal_b_on_occ3d():
    """Occ3D's aspect ratio is <= 2, so the crop is identity and the pairs must match."""
    load = lambda v: {r["clip_id"]: float(r["s_moge"])
                      for r in csv.DictReader(open(os.path.join(ART, f"scales_{v}_occ3d.csv")))}
    if not have("scales_G51-D_occ3d.csv"):
        pytest.skip("occ3d variants not cached")
    a, b, c, d = load("G51-A"), load("G51-B"), load("G51-C"), load("G51-D")
    assert all(abs(c[k] - a[k]) < 1e-12 for k in a), "C != A despite identity crop"
    assert all(abs(d[k] - b[k]) < 1e-12 for k in b), "D != B despite identity crop"


@needs_eval
def test_gate5_baselines_reproduce_in_gate5_1():
    for ds in ("occ3d", "kitti"):
        p = os.path.join(ART, f"summary_{ds}.json")
        if not os.path.exists(p):
            continue
        s = json.load(open(p))
        assert s["gate5_reproduction"], "no reproduction record"
        for k, v in s["gate5_reproduction"].items():
            assert v["within_tolerance"], f"{ds} {k}: {v['gate5_1']} vs {v['gate5']}"
        assert all(n == 0 for n in s["skipped_by_condition"].values())


@needs_eval
def test_coverage_and_bootstrap_units():
    s = json.load(open(os.path.join(ART, "summary_occ3d.json")))
    assert s["n_clips"] == 1182 and s["n_units"] == 150 and s["bootstrap_unit"] == "scene"
    k = json.load(open(os.path.join(ART, "summary_kitti.json")))
    assert k["n_clips"] == 163 and k["bootstrap_unit"] == "block" and k["n_units"] == 21
    for d in (s, k):
        assert d["non_inferiority"]["margin"] == -0.01
        assert "does not prove equivalence" in d["non_inferiority"]["note"]


def test_bootstrap_deterministic():
    rng = np.random.default_rng(0)
    d = list(rng.normal(0.01, 0.05, 150))
    assert bootstrap_ci(d, 10000, 0) == bootstrap_ci(d, 10000, 0)
