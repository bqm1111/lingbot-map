"""Gate 4 — frozen-transfer correctness, coordinate and target-leakage tests."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from gates.scale_gate.config import load_config                                    # noqa: E402
from gates.scale_gate.scale import bootstrap_ci                                    # noqa: E402
from prompted_lingbot.occupancy import OCC3D_NUSCENES_GRID                   # noqa: E402
from gates.voxel_gate.controls import control                                      # noqa: E402
from gates.voxel_gate.models import apply_region                                   # noqa: E402
from gates.voxel_gate.voxels import dilate                                         # noqa: E402
from occ3d_zeroshot.grid import (                                            # noqa: E402
    CANONICAL, NATIVE, RATIO, canonical_to_native, native_binary_target,
    native_distance_bands, occupancy_canonical, points_to_canonical,
)
from occ3d_zeroshot.nuscenes_adapter import (                                # noqa: E402
    build_clips, load_annotations, quat_to_R, rt_to_T, scene_frames, val_scenes,
)
from occ3d_zeroshot.pipeline import (                                        # noqa: E402
    clip_scale, fuse_to_ego, load_corrector, load_depth_head, region_from, run_corrector,
)

CFG = os.path.join(_ROOT, "configs/occ3d_zeroshot/frozen_transfer.yaml")
cfg = load_config(CFG)
FV = cfg.frozen_values
DEV = torch.device("cpu")
MAN = os.path.join(_ROOT, "manifests/occ3d_zeroshot/val.jsonl")
ART = os.path.join(_ROOT, cfg.experiment.output_dir)

has_data = os.path.exists(os.path.join(cfg.data.occ3d_root, "annotations.json"))
has_man = os.path.exists(MAN)
has_cache = os.path.isdir(cfg.data.cache_root)
needs_data = pytest.mark.skipif(not has_data, reason="Occ3D not installed")
needs_cache = pytest.mark.skipif(not (has_cache and has_man), reason="LingBot cache absent")


def _clip_npz():
    r = [json.loads(l) for l in open(MAN)][0]
    return r, np.load(os.path.join(cfg.data.cache_root, r["clip_id"] + ".npz"))


# --------------------------------------------------------------------------- #
# Frozen values
# --------------------------------------------------------------------------- #
def test_physical_correction_radius_is_exactly_0_6_m():
    r = int(round(float(FV.radius_m) / float(FV.canonical_voxel_size)))
    assert r == 3
    assert abs(r * float(FV.canonical_voxel_size) - 0.6) < 1e-12
    # applying radius 3 on the NATIVE grid would double the physical radius -- guard it
    assert abs(3 * NATIVE.voxel_size - 1.2) < 1e-12


def test_frozen_scalars_match_gate31():
    assert float(FV.s0) == 27.3665
    assert float(FV.tau) == 0.45
    assert float(FV.confidence_threshold) == 1.5
    assert (float(FV.min_depth_m), float(FV.max_depth_m)) == (1.0, 60.0)
    assert float(FV.canonical_voxel_size) == 0.2
    g31 = load_config(cfg.frozen.gate31_config)
    assert float(g31.train.threshold) == float(FV.tau)
    assert int(g31.region.radius) == 3
    d = load_config(cfg.frozen.gate2_config)
    assert float(d.scale.constant) == float(FV.s0)


@pytest.mark.skipif(not os.path.exists(os.path.join(_ROOT, cfg.frozen.full_s0)),
                    reason="checkpoints absent")
def test_checkpoints_are_frozen_and_carry_gate31_settings():
    for key in ("full_s0", "occ_only_s0", "c0_corrector_s0"):
        m, ck = load_corrector(os.path.join(_ROOT, getattr(cfg.frozen, key)), DEV)
        assert not any(p.requires_grad for p in m.parameters())
        assert int(ck["radius"]) == 3
        assert float(ck["threshold"]) == float(FV.tau)
        assert int(ck["seed"]) == 0
        assert ck["n_params"] == 16577
    head, hck = load_depth_head(os.path.join(_ROOT, cfg.frozen.depth_head), DEV)
    assert not any(p.requires_grad for p in head.parameters())
    assert hck["use_rgb"] is False and hck["in_ch"] == 5
    assert abs(float(hck["metric_scale"]) - float(FV.s0)) < 1e-9


def test_no_optimizer_is_constructed_anywhere_in_gate4():
    import occ3d_zeroshot.pipeline as P
    for mod_path in (P.__file__,
                     os.path.join(_ROOT, "tools/occ3d_zeroshot/eval_transfer.py"),
                     os.path.join(_ROOT, "tools/occ3d_zeroshot/cache_lingbot.py")):
        src = open(mod_path).read()
        for bad in ("torch.optim", "Adam", "SGD", ".backward()", "requires_grad_(True)"):
            assert bad not in src, f"{os.path.basename(mod_path)} mentions {bad!r}"


# --------------------------------------------------------------------------- #
# Grid and the canonical -> native conversion
# --------------------------------------------------------------------------- #
def test_canonical_grid_covers_the_official_extent_exactly():
    assert tuple(NATIVE.dims) == (200, 200, 16) and NATIVE.voxel_size == 0.4
    assert tuple(NATIVE.origin) == (-40.0, -40.0, -1.0)
    assert tuple(CANONICAL.dims) == (400, 400, 32) and CANONICAL.voxel_size == 0.2
    assert tuple(CANONICAL.origin) == tuple(NATIVE.origin)
    assert np.allclose(CANONICAL.upper, NATIVE.upper)
    assert np.allclose(NATIVE.upper, [40.0, 40.0, 5.4])


def test_any_subvoxel_conversion_on_synthetic_patterns():
    v = torch.zeros(CANONICAL.dims, dtype=torch.bool)
    assert int(canonical_to_native(v).sum()) == 0
    for c, n in (((0, 0, 0), (0, 0, 0)), ((1, 1, 1), (0, 0, 0)),
                 ((2, 0, 0), (1, 0, 0)), ((399, 399, 31), (199, 199, 15))):
        v = torch.zeros(CANONICAL.dims, dtype=torch.bool); v[c] = True
        out = canonical_to_native(v)
        assert int(out.sum()) == 1 and bool(out[n])
    full = torch.ones(CANONICAL.dims, dtype=torch.bool)
    assert int(canonical_to_native(full).sum()) == int(np.prod(NATIVE.dims))


def test_conversion_is_monotone_and_method_agnostic():
    """More canonical occupancy can never yield less native occupancy."""
    rng = np.random.default_rng(0)
    a = torch.from_numpy(rng.random(CANONICAL.dims) > 0.999)
    b = a | torch.from_numpy(rng.random(CANONICAL.dims) > 0.999)
    na, nb = canonical_to_native(a), canonical_to_native(b)
    assert bool((na & ~nb).sum() == 0)
    # the rule is a pure function of the volume: no configuration identity enters it
    assert torch.equal(canonical_to_native(a), canonical_to_native(a.clone()))


def test_distance_bands_are_disjoint_on_the_native_grid():
    bands = native_distance_bands([(0, 10), (10, 20), (20, 30), (30, 40)])
    st = np.stack(list(bands.values()))
    assert st.sum(0).max() <= 1


# --------------------------------------------------------------------------- #
# Coordinates
# --------------------------------------------------------------------------- #
def test_quaternion_and_transform_helpers():
    R = quat_to_R([1, 0, 0, 0])
    assert np.allclose(R, np.eye(3))
    rng = np.random.default_rng(1)
    q = rng.normal(size=4)
    R = quat_to_R(q)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)      # orthonormal
    assert np.isclose(np.linalg.det(R), 1.0)                # right-handed
    T = rt_to_T([1, 2, 3], q)
    assert np.allclose(np.linalg.inv(T) @ T, np.eye(4), atol=1e-12)   # invertible
    assert np.allclose(T[:3, 3], [1, 2, 3])


def test_coupled_scaling_is_a_similarity_and_leaves_the_anchor_fixed():
    rng = np.random.default_rng(2)
    T, H, W = 5, 12, 18
    dep = rng.uniform(0.05, 1.5, (T, H, W)).astype(np.float32)
    conf = np.full((T, H, W), 5.0, np.float32)
    K = np.tile(np.array([[300., 0, W / 2], [0, 300., H / 2], [0, 0, 1.]]), (T, 1, 1))
    P = np.tile(np.eye(4), (T, 1, 1))
    for i in range(T):
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        P[i, :3, :3] = Q * np.sign(np.linalg.det(Q))
        P[i, :3, 3] = rng.normal(size=3)
    E = np.eye(4)
    p1, *_ = fuse_to_ego(dep, conf, K, P, E, 1.0, 1.5, 0.0, 1e9)
    p2, *_ = fuse_to_ego(dep, conf, K, P, E, 3.5, 1.5, 0.0, 1e9)
    assert np.allclose(p2, 3.5 * p1, rtol=1e-9, atol=1e-9)     # scale applied once
    # the anchor frame's own points never receive a pose transform
    only_anchor = np.zeros_like(conf); only_anchor[-1] = 9.0
    q1, *_ = fuse_to_ego(dep, only_anchor, K, P, E, 1.0, 1.5, 0.0, 1e9)
    q2, *_ = fuse_to_ego(dep, only_anchor, K, P * 1.0, E, 1.0, 1.5, 0.0, 1e9)
    assert np.allclose(q1, q2)


def test_camera_to_ego_places_points_in_front_and_upright():
    """A CAM_FRONT-like extrinsic must send camera +z to ego +x and +y to ego -z."""
    T = np.array([[0., 0., 1., 1.70],
                  [-1., 0., 0., 0.02],
                  [0., -1., 0., 1.51],
                  [0., 0., 0., 1.]])
    p_cam = np.array([[0., 0., 10.], [1., 0., 10.], [0., 1., 10.]])
    p_ego = p_cam @ T[:3, :3].T + T[:3, 3]
    assert p_ego[0, 0] > 10.0                       # forward
    assert p_ego[1, 1] < p_ego[0, 1]                # camera +x -> ego -y
    assert p_ego[2, 2] < p_ego[0, 2]                # camera +y (down) -> ego -z
    assert np.allclose(np.linalg.det(T[:3, :3]), 1.0)


def test_points_to_canonical_matches_the_floor_binning_rule():
    pts = np.array([[-40.0, -40.0, -1.0], [-39.9, -39.9, -0.9], [39.99, 39.99, 5.39],
                    [-40.1, 0.0, 0.0], [0.0, 0.0, 5.5]])
    idx, keep = points_to_canonical(pts)
    assert keep.tolist() == [True, True, True, False, False]
    assert idx[0].tolist() == [0, 0, 0]
    assert idx[2].tolist() == [399, 399, 31]


# --------------------------------------------------------------------------- #
# Target-label leakage — the central guarantee
# --------------------------------------------------------------------------- #
@needs_cache
def test_predictions_are_bit_identical_under_full_target_randomisation():
    """Randomise semantics, mask_camera and mask_lidar: every prediction must not move."""
    r, d = _clip_npz()
    head, hck = load_depth_head(os.path.join(_ROOT, cfg.frozen.depth_head), DEV)
    m, ck = load_corrector(os.path.join(_ROOT, cfg.frozen.full_s0), DEV)
    dep = d["pred_depth"].astype(np.float32)
    conf = d["pred_depth_conf"].astype(np.float32)
    K = d["pred_K"].astype(np.float64)
    pose = d["pred_pose_c2w"].astype(np.float64)
    Tce = d["T_camera_to_ego"][-1].astype(np.float64)

    def predict():
        _, s, _, _ = clip_scale(head, hck, dep, conf, float(FV.s0),
                                float(FV.confidence_threshold), float(FV.min_depth_m),
                                float(FV.max_depth_m), DEV)
        pe, fr, cf, dp = fuse_to_ego(dep, conf, K, pose, Tce, s,
                                     float(FV.confidence_threshold),
                                     float(FV.min_depth_m), float(FV.max_depth_m))
        from occ3d_zeroshot.pipeline import canonical_features
        feat5, _ = canonical_features(pe, fr, cf, dp, DEV)
        R = region_from(feat5[0] > 0, 3)
        learned = run_corrector(m, feat5, R, ck["norm"], float(ck["threshold"]), False)
        det = control(feat5[0] > 0, R, **dict(FV.v1_control))
        return (canonical_to_native(feat5[0] > 0), canonical_to_native(det),
                canonical_to_native(learned), R)

    base = predict()
    rng = np.random.default_rng(0)
    for _ in range(3):
        # mutate every target-side array the evaluator will later read
        _sem = rng.integers(0, 18, size=NATIVE.dims).astype(np.uint8)
        _mc = (rng.random(NATIVE.dims) > 0.5).astype(np.uint8)
        _ml = (rng.random(NATIVE.dims) > 0.5).astype(np.uint8)
        again = predict()
        for x, y in zip(base, again):
            assert torch.equal(x, y), "a prediction changed under target randomisation"


def test_pipeline_module_never_reads_a_label_or_mask():
    import occ3d_zeroshot.pipeline as P
    src = open(P.__file__).read()
    body = "\n".join(l.split("#")[0] for l in src.splitlines())
    for bad in ("labels.npz", "semantics", "mask_camera", "mask_lidar", "gt_path",
                "native_binary_target"):
        assert bad not in body, f"pipeline references {bad!r}"


def test_native_binary_target_reproduces_the_occany_reduction():
    """OccAny nuscenes.py:894-904 + ssc.py get_score_completion, on a synthetic label."""
    rng = np.random.default_rng(3)
    sem = rng.integers(0, 18, size=NATIVE.dims).astype(np.uint8)
    mc = (rng.random(NATIVE.dims) > 0.3).astype(np.uint8)
    ml = (rng.random(NATIVE.dims) > 0.3).astype(np.uint8)
    occ, keep = native_binary_target({"semantics": sem, "mask_camera": mc,
                                      "mask_lidar": ml},
                                     apply_camera_mask=True, apply_lidar_mask=False,
                                     single_camera_x_cut=100)
    ref = sem.astype(np.int32).copy()
    ref[mc == 0] = 255
    ref[:100, :, :] = 255
    assert np.array_equal(keep, ref != 255)
    assert np.array_equal(occ, (ref != 17) & (ref != 255))
    assert not keep[:100].any(), "the single-camera x-cut region must be excluded"


def test_masks_restrict_the_metric_but_never_the_prediction():
    rng = np.random.default_rng(4)
    pred = torch.from_numpy(rng.random(NATIVE.dims) > 0.98)
    gt = torch.from_numpy(rng.random(NATIVE.dims) > 0.98)
    keep = torch.from_numpy(rng.random(NATIVE.dims) > 0.4)
    before = pred.clone()
    sys.path.insert(0, os.path.join(_ROOT, "tools", "occ3d_zeroshot"))
    from eval_transfer import scores
    s = scores(pred, gt, keep)
    assert torch.equal(pred, before)
    assert s["tp"] + s["fp"] == int((pred & keep).sum())
    assert scores(pred | ~keep, gt, keep) == s          # outside keep cannot contribute


# --------------------------------------------------------------------------- #
# Region and gating
# --------------------------------------------------------------------------- #
def test_r_infer_is_a_pure_dilation_of_the_frozen_geometry():
    rng = np.random.default_rng(5)
    occ = torch.from_numpy(rng.random((30, 30, 12)) > 0.99)
    assert torch.equal(region_from(occ, 3), dilate(occ, 3))
    assert bool((occ & ~region_from(occ, 3)).sum() == 0)     # extensive


def test_output_gating_depends_only_on_r_infer():
    rng = np.random.default_rng(6)
    occ = torch.from_numpy(rng.random((20, 20, 10)) > 0.97)
    R = dilate(occ, 3)
    wild = torch.from_numpy(rng.random((20, 20, 10)) > 0.3)
    out = apply_region(wild, occ, R)
    assert torch.equal(out & R, wild & R)
    assert torch.equal(out & ~R, occ & ~R)


def test_controls_use_the_same_region_as_the_learned_model():
    rng = np.random.default_rng(7)
    occ = torch.from_numpy(rng.random((20, 20, 10)) > 0.97)
    R = dilate(occ, 3)
    for kw in (dict(FV.v1_control), dict(FV.v2_control)):
        out = control(occ, R, **kw)
        assert torch.equal(out & ~R, occ & ~R)


# --------------------------------------------------------------------------- #
# Manifest / protocol
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not has_man, reason="manifest not built")
def test_manifest_protocol_is_five_frames_one_camera_one_scene():
    recs = [json.loads(l) for l in open(MAN)]
    assert len(recs) > 0
    seen = set()
    for r in recs:
        assert len(r["sample_tokens"]) == 5
        assert len(set(r["sample_tokens"])) == 5
        assert r["camera"] == cfg.data.camera
        ts = r["timestamps_ns"]
        assert all(b > a for a, b in zip(ts, ts[1:]))          # strictly ordered
        assert r["anchor_token"] == r["sample_tokens"][-1]     # frozen anchor = last
        assert all(0.3 < g < 0.8 for g in r["spacings_s"])     # ~2 Hz keyframes
        assert r["clip_id"] not in seen
        seen.add(r["clip_id"])
        assert r["scene"] in r["clip_id"]


@needs_data
@pytest.mark.skipif(not has_man, reason="manifest not built")
def test_all_clips_are_official_val_scenes_only():
    ann = load_annotations(cfg.data.occ3d_root)
    val = set(val_scenes(ann))
    train = set(ann["train_split"])
    scenes = {json.loads(l)["scene"] for l in open(MAN)}
    assert scenes <= val
    assert not (scenes & train), "a training scene leaked into the evaluation manifest"


@needs_data
def test_clip_builder_opens_no_label():
    ann = load_annotations(cfg.data.occ3d_root)
    s = val_scenes(ann)[0]
    fr = scene_frames(ann, s, cfg.data.camera, cfg.data.nuscenes_root)
    clips, rej = build_clips(fr, 5, 1, 5, cfg.data.occ3d_root)
    assert clips and all(len(c.frames) == 5 for c in clips)
    assert all(c.anchor is c.frames[-1] for c in clips)
    src = open(os.path.join(_ROOT, "occ3d_zeroshot/nuscenes_adapter.py")).read()
    body = "\n".join(l.split("#")[0] for l in src.splitlines())
    for bad in ("np.load", "semantics", "mask_camera", "labels.npz"):
        assert bad not in body, f"adapter references {bad!r}"


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def test_scene_level_bootstrap_is_paired_and_deterministic():
    rng = np.random.default_rng(0)
    x = rng.random(150)
    d = list((x + 0.02) - x)
    a, b = bootstrap_ci(d, 10000, 0), bootstrap_ci(d, 10000, 0)
    assert a == b
    assert a["lo"] == pytest.approx(0.02) and a["hi"] == pytest.approx(0.02)


@pytest.mark.skipif(not os.path.exists(os.path.join(ART, "eval", "summary.json")),
                    reason="evaluation not run")
def test_reported_bootstrap_uses_scene_as_the_resampling_unit():
    s = json.load(open(os.path.join(ART, "eval", "summary.json")))
    assert s["n_scenes"] == 150
    for k, v in s["contrasts"].items():
        assert v["n_scenes"] == 150 and v["n"] == 150
    assert s["radius_m"] == 0.6 and s["radius_voxels"] == 3
    assert s["canonical_grid"]["voxel_size"] == 0.2
    assert s["native_grid"]["voxel_size"] == 0.4
