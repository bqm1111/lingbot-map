"""No-future-leakage: an anchor's output at frame t must not depend on any
prompt after t.  This is checked end to end, through the real runner, for every
causal anchor -- not by inspecting the code."""

import numpy as np
import pytest

from prompted_lingbot.anchors import BASELINES, CAUSAL_BASELINES, Prediction
from prompted_lingbot.conventions import axis_angle_to_matrix
from prompted_lingbot.prompts import Prompt, PromptConfig, make_prompt_stream

H = W = 16
K = np.array([[20.0, 0.0, W / 2], [0.0, 20.0, H / 2], [0.0, 0.0, 1.0]])
N = 60
SPLIT = 25
TRUE_SCALE = 7.0


def _synthetic():
    rng = np.random.default_rng(0)
    centres = np.cumsum(rng.normal(scale=0.5, size=(N, 3)), axis=0)
    centres -= centres[0]
    poses = np.zeros((N, 3, 4))
    for i in range(N):
        poses[i, :3, :3] = axis_angle_to_matrix(rng.normal(scale=0.06, size=3))
        poses[i, :3, 3] = centres[i]
    depth = rng.uniform(2.0, 25.0, size=(N, H, W))
    valid = np.ones((N, H, W), bool)
    return poses, depth, valid


def _predictions(poses, depth):
    out = []
    for i in range(N):
        p = poses[i].copy()
        p[:3, 3] /= TRUE_SCALE
        out.append(Prediction(frame=i, pose_c2w=p, depth=depth[i] / TRUE_SCALE,
                              depth_conf=np.full((H, W), 3.0), K=K))
    return out


def _drive(anchor, preds, prompts):
    anchor.reset()
    return [(_c := anchor.update(preds[t], prompts[t]), anchor.correction)[1] for t in range(N)]


@pytest.mark.parametrize("name", CAUSAL_BASELINES)
def test_future_prompts_cannot_change_earlier_corrections(name):
    poses, depth, valid = _synthetic()
    preds = _predictions(poses, depth)
    cfg = PromptConfig(name="c", depth_interval=3, pose_interval=3, seed=0)
    base = make_prompt_stream(cfg, "seq", N, depth, valid, poses)

    # Same stream up to SPLIT, wildly different after.
    rng = np.random.default_rng(1)
    tampered = list(base)
    for t in range(SPLIT + 1, N):
        p = Prompt(frame=t)
        if base[t].has_depth:
            p.has_depth = True
            p.depth_rows = base[t].depth_rows
            p.depth_cols = base[t].depth_cols
            p.depth_values = base[t].depth_values * 100.0
            p.depth_confidence = 1.0
        if base[t].has_pose:
            p.has_pose = True
            p.pose_c2w = base[t].pose_c2w + rng.normal(scale=50.0, size=(3, 4))
            p.pose_confidence = 1.0
        tampered[t] = p

    a = _drive(BASELINES[name](), preds, base)
    b = _drive(BASELINES[name](), preds, tampered)

    for t in range(SPLIT + 1):
        assert np.isclose(a[t].s, b[t].s), f"{name}: scale diverged at frame {t}"
        assert np.allclose(a[t].R, b[t].R), f"{name}: rotation diverged at frame {t}"
        assert np.allclose(a[t].t, b[t].t), f"{name}: translation diverged at frame {t}"

    # Sanity: the tampering must actually matter, otherwise the test proves
    # nothing.  "raw" ignores prompts and "first_depth_scale" freezes on the
    # first one, so for those two a null effect is the correct behaviour.
    if name not in ("raw", "first_depth_scale"):
        assert not np.isclose(a[N - 1].s, b[N - 1].s), (
            f"{name}: tampering with the future changed nothing at all -- "
            "the test is not exercising the anchor")


@pytest.mark.parametrize("name", CAUSAL_BASELINES)
def test_truncating_the_stream_reproduces_the_prefix(name):
    """Running only frames 0..T must give exactly the same corrections as
    running the whole sequence and reading off the first T."""
    poses, depth, valid = _synthetic()
    preds = _predictions(poses, depth)
    cfg = PromptConfig(name="c", depth_interval=4, pose_interval=4, seed=3)
    prompts = make_prompt_stream(cfg, "seq", N, depth, valid, poses)

    full = _drive(BASELINES[name](), preds, prompts)

    anchor = BASELINES[name]()
    anchor.reset()
    for t in range(SPLIT + 1):
        anchor.update(preds[t], prompts[t])
    assert np.isclose(anchor.correction.s, full[SPLIT].s)
    assert np.allclose(anchor.correction.t, full[SPLIT].t)


def test_the_offline_oracle_is_excluded_from_the_causal_set():
    assert "offline_oracle_sim3" not in CAUSAL_BASELINES
    assert "per_frame_oracle_scale" not in CAUSAL_BASELINES
    assert BASELINES["offline_oracle_sim3"]().is_causal is False


@pytest.mark.parametrize("name", CAUSAL_BASELINES)
def test_sequences_do_not_leak_into_each_other(name):
    """State from sequence A must not survive into sequence B after reset."""
    poses, depth, valid = _synthetic()
    preds = _predictions(poses, depth)
    cfg = PromptConfig(name="c", depth_interval=2, pose_interval=2, seed=5)
    prompts = make_prompt_stream(cfg, "seq", N, depth, valid, poses)

    fresh = _drive(BASELINES[name](), preds, prompts)

    reused = BASELINES[name]()
    # Poison it with a different sequence first.
    poison = make_prompt_stream(PromptConfig(name="p", depth_interval=1, pose_interval=1, seed=9),
                                "other", N, depth * 13.0, valid, poses * 13.0)
    _drive(reused, preds, poison)
    again = _drive(reused, preds, prompts)

    for t in range(N):
        assert np.isclose(fresh[t].s, again[t].s), f"{name}: state leaked at frame {t}"
