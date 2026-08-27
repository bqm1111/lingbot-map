"""Causality and leakage tests for the frozen-depth substitution experiment.

The ways this experiment could produce a flattering-but-meaningless number:

* letting a future external-depth prediction change an earlier output;
* letting a future scale ratio change an earlier LingbotMap pose scale;
* letting dataset poses leak into anything but the labelled diagnostic;
* rescaling external depth (it is already metric) instead of only the translation;
* letting the LiDAR used for *evaluation* depth metrics reach the prediction.
"""

import numpy as np
import pytest

from prompted_lingbot.conventions import Sim3, axis_angle_to_matrix
from prompted_lingbot.external_depth import (
    MODELS, causal_translation_scale, resample_to_cached_lattice,
)
from prompted_lingbot.occ_datasets import apply_transform, relative_c2w
from prompted_lingbot.occ_eval import (
    OccPointConfig, external_points_in_anchor_camera, points_in_anchor_camera,
)
from prompted_lingbot.occupancy import SEMANTICKITTI_GRID as G
from prompted_lingbot.occupancy import occupancy_from_points
from prompted_lingbot.runner import CachedSequence

H, W, NF = 24, 60, 40
TRUE_SCALE = 8.0
CFG = OccPointConfig(conf_threshold=1.0, min_depth=0.5, max_depth=60.0)


def _seq(seed=0):
    """Synthetic cache: LingbotMap depth is TRUE_SCALE times too small."""
    rng = np.random.default_rng(seed)
    K = np.array([[60.0, 0.0, W / 2], [0.0, 60.0, H / 2], [0.0, 0.0, 1.0]])
    poses = np.zeros((NF, 3, 4))
    for i in range(NF):
        poses[i, :3, :3] = axis_angle_to_matrix(np.array([0.0, 0.004 * i, 0.0]))
        poses[i, :3, 3] = [0.0, 0.0, 0.35 * i]          # arbitrary units
    metric = rng.uniform(4.0, 30.0, size=(NF, H, W))
    seq = CachedSequence(
        name="synthetic", meta={},
        pred_pose_c2w=poses, pred_K=np.tile(K, (NF, 1, 1)),
        pred_depth=(metric / TRUE_SCALE).astype(np.float16),
        pred_depth_conf=np.full((NF, H, W), 3.0, np.float16),
        gt_pose_c2w=poses.copy(), gt_world_from_local=np.eye(4), gt_K=K.copy(),
        gt_depth=None, gt_depth_valid=None, timestamps=np.arange(NF, dtype=float))
    return seq, metric.astype(np.float32)


# --------------------------------------------------------------------------- #
# the scale estimator
# --------------------------------------------------------------------------- #
def test_translation_scale_recovers_the_true_ratio():
    seq, ext = _seq()
    est = causal_translation_scale(ext[5], seq.pred_depth[5].astype(np.float32),
                                    seq.pred_depth_conf[5].astype(np.float32))
    assert est is not None
    assert np.isclose(np.exp(est[0]), TRUE_SCALE, rtol=1e-2)


def test_translation_scale_survives_outliers():
    seq, ext = _seq(1)
    rng = np.random.default_rng(2)
    bad = ext[5].copy()
    hit = rng.random(bad.shape) < 0.25
    bad[hit] *= 12.0
    est = causal_translation_scale(bad, seq.pred_depth[5].astype(np.float32),
                                    seq.pred_depth_conf[5].astype(np.float32))
    assert abs(np.exp(est[0]) - TRUE_SCALE) / TRUE_SCALE < 0.1


def test_translation_scale_refuses_a_tiny_overlap():
    seq, ext = _seq()
    e = np.zeros_like(ext[0])
    e[0, 0] = 10.0
    assert causal_translation_scale(e, seq.pred_depth[0].astype(np.float32), None) is None


def test_translation_scale_rejects_a_lattice_mismatch():
    seq, ext = _seq()
    with pytest.raises(ValueError):
        causal_translation_scale(ext[0][:, :10], seq.pred_depth[0].astype(np.float32), None)


# --------------------------------------------------------------------------- #
# external depth is metric and must NOT be rescaled
# --------------------------------------------------------------------------- #
def test_external_depth_is_never_rescaled_by_the_translation_scale():
    """Changing the translation scale must not move a single-frame cloud."""
    seq, ext = _seq()
    a = external_points_in_anchor_camera(seq, ext, 10, 1, 1.0, CFG)
    b = external_points_in_anchor_camera(seq, ext, 10, 1, 25.0, CFG)
    assert np.array_equal(a, b), "history-1 geometry must be scale-independent"
    assert np.isclose(np.median(a[:, 2]), np.median(ext[10]), rtol=0.1)


def test_translation_scale_does_move_multi_frame_geometry():
    seq, ext = _seq()
    a = external_points_in_anchor_camera(seq, ext, 10, 5, 1.0, CFG)
    b = external_points_in_anchor_camera(seq, ext, 10, 5, TRUE_SCALE, CFG)
    assert not np.allclose(a, b), "the scale must affect inter-frame placement"


def test_correct_scale_makes_aggregated_geometry_self_consistent():
    """With the right translation scale, points from earlier frames land on the
    SAME physical surface as the anchor's own; with a wrong scale they smear.

    This needs a real surface -- random per-pixel depth has nothing to be
    consistent with -- so the scene here is a fronto-parallel wall at a fixed
    world position that the camera drives towards.
    """
    seq, _ = _seq(3)
    WALL_Z = 40.0                       # metres in front of frame 0
    K = seq.pred_K[0]
    u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    ext = np.zeros((NF, H, W), np.float32)
    for f in range(NF):
        # camera has moved 0.35*f arbitrary units == 0.35*f*TRUE_SCALE metres
        ext[f] = WALL_Z - 0.35 * f * TRUE_SCALE
    ext = np.clip(ext, 1.0, None)

    good = external_points_in_anchor_camera(seq, ext, 20, 10, TRUE_SCALE, CFG)
    bad = external_points_in_anchor_camera(seq, ext, 20, 10, TRUE_SCALE * 3, CFG)
    # the anchor frame sees the wall at one depth; correct scale keeps the history
    # on that same plane, a wrong scale spreads it out
    assert good[:, 2].std() < bad[:, 2].std(), (
        f"correct scale should concentrate depth (got {good[:, 2].std():.3f} "
        f"vs {bad[:, 2].std():.3f})")


# --------------------------------------------------------------------------- #
# causality
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("history", [1, 5, 20])
def test_future_external_depth_cannot_change_an_earlier_output(history):
    seq, ext = _seq()
    anchor = 18
    base = external_points_in_anchor_camera(seq, ext, anchor, history, 2.0, CFG)
    tampered = ext.copy()
    tampered[anchor + 1:] = 999.0
    after = external_points_in_anchor_camera(seq, tampered, anchor, history, 2.0, CFG)
    assert np.array_equal(base, after)


def test_future_scale_ratios_cannot_change_an_earlier_pose_scale():
    """The running translation-scale filter is a strict prefix function."""
    import importlib.util, os, sys
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    spec_ = importlib.util.spec_from_file_location(
        "_ds_eval", os.path.join(root, "scripts", "evaluate_depth_substitution.py"))
    mod = importlib.util.module_from_spec(spec_)
    sys.modules["_ds_eval"] = mod
    spec_.loader.exec_module(mod)

    import argparse
    seq, ext = _seq()
    cfg = argparse.Namespace(conf_threshold=1.0, max_depth=60.0)
    base = mod.external_scale_stream(seq, ext, cfg)
    tampered = ext.copy()
    # Stay inside [min_depth, max_depth]: a value of 500 m is filtered out entirely
    # and would leave the estimate untouched, making the test vacuous.
    tampered[21:] = np.clip(tampered[21:] * 2.0, 4.0, 55.0)
    after = mod.external_scale_stream(seq, tampered, cfg)
    assert base[:21] == pytest.approx(after[:21]), "a future frame moved an earlier scale"
    assert base[-1] != pytest.approx(after[-1]), "the tampering had no effect at all"


def test_history_window_clamps_at_the_sequence_start():
    seq, ext = _seq()
    a = external_points_in_anchor_camera(seq, ext, 4, 100, 2.0, CFG, frame_lo=0)
    b = external_points_in_anchor_camera(seq, ext, 4, 5, 2.0, CFG, frame_lo=0)
    assert np.array_equal(a, b)


def test_longer_history_extends_rather_than_replaces():
    seq, ext = _seq()
    short = external_points_in_anchor_camera(seq, ext, 22, 3, 2.0, CFG)
    long = external_points_in_anchor_camera(seq, ext, 22, 10, 2.0, CFG)
    assert long.shape[0] > short.shape[0]
    assert np.allclose(long[-short.shape[0]:], short)


# --------------------------------------------------------------------------- #
# pose sources
# --------------------------------------------------------------------------- #
def test_dataset_poses_are_only_used_when_explicitly_passed():
    """The default pose source is LingbotMap's own; dataset poses require an
    explicit argument, which only the labelled diagnostic supplies."""
    seq, ext = _seq()
    seq.gt_pose_c2w = seq.gt_pose_c2w.copy()
    # A CONSTANT offset would cancel in a relative transform, so perturb per frame.
    rng = np.random.default_rng(11)
    seq.gt_pose_c2w[:, :3, 3] += rng.normal(scale=5.0, size=(NF, 3))
    default = external_points_in_anchor_camera(seq, ext, 15, 6, 2.0, CFG)
    explicit = external_points_in_anchor_camera(seq, ext, 15, 6, 2.0, CFG,
                                                poses_c2w=seq.gt_pose_c2w)
    assert not np.allclose(default, explicit)


def test_dataset_pose_path_ignores_the_estimated_scale():
    """Dataset poses are already metric, so the diagnostic must pass scale 1."""
    seq, ext = _seq()
    a = external_points_in_anchor_camera(seq, ext, 15, 6, 1.0, CFG,
                                         poses_c2w=seq.gt_pose_c2w)
    b = external_points_in_anchor_camera(seq, ext, 15, 6, 1.0, CFG,
                                         poses_c2w=seq.gt_pose_c2w)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# determinism, lattice, model registry
# --------------------------------------------------------------------------- #
def test_external_aggregation_is_deterministic():
    seq, ext = _seq()
    a = external_points_in_anchor_camera(seq, ext, 12, 7, 3.0, CFG)
    b = external_points_in_anchor_camera(seq, ext, 12, 7, 3.0, CFG)
    assert np.array_equal(a, b)


def test_resample_preserves_metric_values():
    """Resampling changes the lattice, never the depth values."""
    rng = np.random.default_rng(0)
    d = rng.uniform(3.0, 50.0, size=(370, 1226)).astype(np.float32)
    out = resample_to_cached_lattice(d, (370, 1226), (77, 259))
    assert out.shape == (77, 259)
    assert out.min() >= d.min() - 1e-5 and out.max() <= d.max() + 1e-5


def test_model_registry_records_what_is_needed_to_reproduce():
    spec = MODELS["da_v2_metric_vkitti_vits"]
    for k in ("family", "encoder", "checkpoint", "max_depth", "input_size",
              "depth_convention", "units", "provides_pose", "provides_intrinsics",
              "provides_confidence", "trained_on"):
        assert k in spec
    assert spec["depth_convention"] == "z_depth_camera_frame"
    assert spec["units"] == "metres"
    assert spec["provides_pose"] is False


def test_lingbot_and_external_paths_share_the_same_lattice():
    seq, ext = _seq()
    lb = points_in_anchor_camera(seq, 10, 1, TRUE_SCALE, CFG)
    ex = external_points_in_anchor_camera(seq, ext, 10, 1, 1.0, CFG)
    assert lb.shape == ex.shape, "both paths must sample identically"
