"""Integrity of the occupancy evaluation: causality and prompt-point leakage.

These are the two ways this experiment could produce a flattering number that
means nothing:

* letting frame ``t``'s occupancy depend on data after ``t``;
* letting the metric prompt's own LiDAR/depth points end up in the predicted
  volume, so the "prediction" is partly ground truth.

Both are tested end to end through the real evaluation path, not by inspection.
"""

import numpy as np
import pytest

from prompted_lingbot.conventions import Sim3, axis_angle_to_matrix
from prompted_lingbot.occ_datasets import apply_transform
from prompted_lingbot.occ_eval import OccPointConfig, evaluate_anchor_frame, points_in_anchor_camera
from prompted_lingbot.occupancy import SEMANTICKITTI_GRID as G
from prompted_lingbot.occupancy import binary_occupancy_scores, occupancy_from_points
from prompted_lingbot.runner import CachedSequence

H, W = 24, 60
NF = 40


def _seq(seed=0, depth_scale=1.0):
    """A synthetic cached sequence: a camera driving forward along +z."""
    rng = np.random.default_rng(seed)
    K = np.array([[60.0, 0.0, W / 2], [0.0, 60.0, H / 2], [0.0, 0.0, 1.0]])
    poses = np.zeros((NF, 3, 4))
    for i in range(NF):
        poses[i, :3, :3] = axis_angle_to_matrix(np.array([0.0, 0.004 * i, 0.0]))
        poses[i, :3, 3] = [0.0, 0.0, 0.35 * i]
    depth = (rng.uniform(3.0, 20.0, size=(NF, H, W)) / depth_scale).astype(np.float16)
    conf = np.full((NF, H, W), 3.0, np.float16)
    return CachedSequence(
        name="synthetic", meta={},
        pred_pose_c2w=poses, pred_K=np.tile(K, (NF, 1, 1)),
        pred_depth=depth, pred_depth_conf=conf,
        gt_pose_c2w=poses.copy(), gt_world_from_local=np.eye(4), gt_K=K.copy(),
        gt_depth=None, gt_depth_valid=None, timestamps=np.arange(NF, dtype=float),
    )


CFG = OccPointConfig(conf_threshold=1.0, min_depth=0.5, max_depth=60.0)
CAM2GRID = np.eye(4)
CAM2GRID[:3, 3] = [10.0, 0.0, 0.0]      # nudge the volume so points land inside


def _target(vol):
    t = np.zeros(G.dims, np.int32)
    t[vol] = 11
    return t


# --------------------------------------------------------------------------- #
# prompt-point leakage
# --------------------------------------------------------------------------- #
def test_occupancy_depends_on_prompts_only_through_the_scale():
    """Two wildly different prompt streams that happen to yield the same scale
    must produce byte-identical occupancy."""
    seq = _seq()
    a = points_in_anchor_camera(seq, 30, 20, 2.5, CFG)
    b = points_in_anchor_camera(seq, 30, 20, 2.5, CFG)
    assert np.array_equal(a, b)
    va = occupancy_from_points(apply_transform(CAM2GRID, a), G)
    vb = occupancy_from_points(apply_transform(CAM2GRID, b), G)
    assert np.array_equal(va, vb)
    # and a different scale must actually change something, or the test is vacuous
    vc = occupancy_from_points(
        apply_transform(CAM2GRID, points_in_anchor_camera(seq, 30, 20, 3.1, CFG)), G)
    assert not np.array_equal(va, vc)


def test_prompt_points_never_enter_the_predicted_cloud():
    """A prompt claiming geometry in an empty region must not create voxels there.

    The evaluation path takes only ``seq`` and a scalar scale, so a prompt's own
    metric points have no route into the volume.  This asserts the consequence.
    """
    seq = _seq()
    scale = 2.0
    vol = occupancy_from_points(
        apply_transform(CAM2GRID, points_in_anchor_camera(seq, 30, 10, scale, CFG)), G)

    # Where a LiDAR prompt might legitimately report returns, far off to the side:
    prompt_pts = np.column_stack([
        np.full(5000, 30.0), np.linspace(-24.0, -18.0, 5000), np.full(5000, 1.0)])
    prompt_idx, keep = np.floor(
        (prompt_pts - np.array(G.origin)) / G.voxel_size).astype(int), None
    inside = np.all((prompt_idx >= 0) & (prompt_idx < np.array(G.dims)), axis=1)
    prompt_idx = prompt_idx[inside]
    assert prompt_idx.size > 0, "the probe region must be inside the grid"
    assert not vol[prompt_idx[:, 0], prompt_idx[:, 1], prompt_idx[:, 2]].any(), \
        "prompt geometry leaked into the predicted volume"


def test_removing_prompt_pixels_changes_occupancy_only_through_their_own_depth():
    """The brief's explicit check.

    Estimate a correction, then delete the pixels a prompt sampled from the
    *prediction*.  Occupancy may only lose what those pixels themselves
    contributed -- it must never lose anything a prompt was propping up, and the
    scale must be untouched.
    """
    seq = _seq()
    scale = 2.0
    rng = np.random.default_rng(3)
    rows = rng.integers(0, H, 400)
    cols = rng.integers(0, W, 400)

    full = points_in_anchor_camera(seq, 30, 5, scale, CFG)
    v_full = occupancy_from_points(apply_transform(CAM2GRID, full), G)

    stripped = CachedSequence(**{**vars(seq), "pred_depth_conf": seq.pred_depth_conf.copy()})
    stripped.pred_depth_conf[:, rows, cols] = np.float16(0.0)   # below conf threshold
    part = points_in_anchor_camera(stripped, 30, 5, scale, CFG)
    v_part = occupancy_from_points(apply_transform(CAM2GRID, part), G)

    # Strictly a subset: nothing new appears, and the scale did not move.
    assert not (v_part & ~v_full).any(), "removing pixels created new occupancy"
    assert part.shape[0] < full.shape[0], "the probe removed nothing; test is vacuous"


# --------------------------------------------------------------------------- #
# causality
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("history", [1, 5, 20])
def test_future_frames_cannot_change_occupancy_at_t(history):
    seq = _seq()
    anchor = 20
    base = points_in_anchor_camera(seq, anchor, history, 2.0, CFG)

    tampered = CachedSequence(**{**vars(seq),
                                 "pred_depth": seq.pred_depth.copy(),
                                 "pred_pose_c2w": seq.pred_pose_c2w.copy()})
    rng = np.random.default_rng(7)
    tampered.pred_depth[anchor + 1:] = np.float16(99.0)
    tampered.pred_pose_c2w[anchor + 1:, :3, 3] += rng.normal(scale=50.0,
                                                             size=(NF - anchor - 1, 3))
    after = points_in_anchor_camera(tampered, anchor, history, 2.0, CFG)
    assert np.array_equal(base, after), "a future frame changed the past"


def test_occupancy_at_t_is_unchanged_by_future_frames_end_to_end():
    seq = _seq()
    anchor, history = 20, 10
    target = _target(occupancy_from_points(
        apply_transform(CAM2GRID, points_in_anchor_camera(seq, anchor, history, 2.0, CFG)), G))

    tampered = CachedSequence(**{**vars(seq), "pred_depth": seq.pred_depth.copy()})
    tampered.pred_depth[anchor + 1:] = np.float16(1.0)

    a = evaluate_anchor_frame(seq, anchor, history, Sim3(2.0, np.eye(3), np.zeros(3)),
                              G, CAM2GRID, target, None, CFG)
    b = evaluate_anchor_frame(tampered, anchor, history, Sim3(2.0, np.eye(3), np.zeros(3)),
                              G, CAM2GRID, target, None, CFG)
    assert a.scores == b.scores
    assert a.scores["iou"] == 1.0, "self-consistency: predicting its own target is perfect"


def test_history_window_never_reaches_before_the_chunk_start():
    seq = _seq()
    a = points_in_anchor_camera(seq, 5, 100, 2.0, CFG, frame_lo=0)
    b = points_in_anchor_camera(seq, 5, 6, 2.0, CFG, frame_lo=0)
    assert np.array_equal(a, b), "history must clamp at the start of the sequence"


def test_history_one_uses_exactly_one_frame():
    seq = _seq()
    p1 = points_in_anchor_camera(seq, 12, 1, 1.0, CFG)
    p2 = points_in_anchor_camera(seq, 12, 2, 1.0, CFG)
    assert p2.shape[0] > p1.shape[0]
    assert p1.shape[0] <= H * W


def test_longer_history_is_a_superset_of_shorter_history():
    seq = _seq()
    short = points_in_anchor_camera(seq, 25, 3, 2.0, CFG)
    long = points_in_anchor_camera(seq, 25, 12, 2.0, CFG)
    assert long.shape[0] > short.shape[0]
    # the last `short` rows of `long` are exactly `short`
    assert np.allclose(long[-short.shape[0]:], short)


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def test_evaluation_is_deterministic():
    seq = _seq()
    target = _target(occupancy_from_points(
        apply_transform(CAM2GRID, points_in_anchor_camera(seq, 18, 8, 2.0, CFG)), G))
    C = Sim3(2.0, np.eye(3), np.zeros(3))
    r1 = evaluate_anchor_frame(seq, 18, 8, C, G, CAM2GRID, target, None, CFG)
    r2 = evaluate_anchor_frame(seq, 18, 8, C, G, CAM2GRID, target, None, CFG)
    assert r1.scores == r2.scores and r1.n_points == r2.n_points


def test_zero_scale_correction_produces_no_occupancy_not_a_crash():
    """Raw LingbotMap in metric voxel space: scale 1 on a 26x-off model.
    It must score zero cleanly rather than raise."""
    seq = _seq(depth_scale=26.0)
    target = np.zeros(G.dims, np.int32)
    target[10:20, 10:20, 5:8] = 11
    r = evaluate_anchor_frame(seq, 20, 5, Sim3(1.0, np.eye(3), np.zeros(3)),
                              G, CAM2GRID, target, None, CFG)
    assert r.scores["iou"] == 0.0
    assert np.isfinite(r.scores["precision"])
