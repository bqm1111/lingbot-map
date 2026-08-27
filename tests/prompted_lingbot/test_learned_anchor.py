"""The learned corrector must obey the same causality and lifecycle contract as
the training-free anchors, and its untrained state must be a no-op."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from prompted_lingbot.anchors import BASELINES, Prediction
from prompted_lingbot.conventions import axis_angle_to_matrix
from prompted_lingbot.features import N_FEATURES, sequence_features
from prompted_lingbot.learned_anchor import LearnedCorrectorAnchor
from prompted_lingbot.model import CausalCorrector, CorrectorConfig, rollout
from prompted_lingbot.prompts import Prompt, PromptConfig, make_prompt_stream

H = W = 16
K = np.array([[20.0, 0.0, W / 2], [0.0, 20.0, H / 2], [0.0, 0.0, 1.0]])
N = 40
SCALE = 6.0


def _world(seed=0):
    rng = np.random.default_rng(seed)
    centres = np.cumsum(rng.normal(scale=0.4, size=(N, 3)), 0)
    centres -= centres[0]
    poses = np.zeros((N, 3, 4))
    for i in range(N):
        poses[i, :3, :3] = axis_angle_to_matrix(rng.normal(scale=0.05, size=3))
        poses[i, :3, 3] = centres[i]
    depth = rng.uniform(2.0, 20.0, size=(N, H, W))
    valid = np.ones((N, H, W), bool)
    preds = []
    for i in range(N):
        p = poses[i].copy()
        p[:3, 3] /= SCALE
        preds.append(Prediction(frame=i, pose_c2w=p, depth=depth[i] / SCALE,
                                depth_conf=np.full((H, W), 3.0), K=K))
    return poses, depth, valid, preds


def _model(base="anchor", seed=0):
    torch.manual_seed(seed)
    return CausalCorrector(CorrectorConfig(n_features=N_FEATURES, hidden=32, layers=1, base=base))


def _drive(anchor, preds, prompts):
    anchor.reset()
    out = []
    for t in range(N):
        anchor.update(preds[t], prompts[t])
        out.append(anchor.correction)
    return out


def test_untrained_corrector_reproduces_its_base_anchor():
    """The output head is zero-initialised, so the increment is the identity."""
    poses, depth, valid, preds = _world()
    prompts = make_prompt_stream(PromptConfig(name="c", depth_interval=3, pose_interval=3),
                                 "s", N, depth, valid, poses)
    learned = _drive(LearnedCorrectorAnchor(_model(), "depth_scale_pose_se3"), preds, prompts)
    base = _drive(BASELINES["depth_scale_pose_se3"](), preds, prompts)
    for t in range(N):
        assert np.isclose(learned[t].s, base[t].s, rtol=1e-5), f"frame {t}"
        assert np.allclose(learned[t].R, base[t].R, atol=1e-5)
        assert np.allclose(learned[t].t, base[t].t, atol=1e-5)


def test_learned_anchor_is_causal():
    poses, depth, valid, preds = _world()
    m = _model()
    # give the head non-zero weights so increments actually do something
    with torch.no_grad():
        for p in m.head[-1].parameters():
            p.normal_(0, 0.3)
    base_stream = make_prompt_stream(PromptConfig(name="c", depth_interval=3, pose_interval=3),
                                     "s", N, depth, valid, poses)
    split = 18
    rng = np.random.default_rng(2)
    tampered = list(base_stream)
    for t in range(split + 1, N):
        p = Prompt(frame=t)
        if base_stream[t].has_depth:
            p.has_depth = True
            p.depth_rows, p.depth_cols = base_stream[t].depth_rows, base_stream[t].depth_cols
            p.depth_values = base_stream[t].depth_values * 50.0
            p.depth_confidence = 1.0
        if base_stream[t].has_pose:
            p.has_pose = True
            p.pose_c2w = base_stream[t].pose_c2w + rng.normal(scale=20.0, size=(3, 4))
            p.pose_confidence = 1.0
        tampered[t] = p

    a = _drive(LearnedCorrectorAnchor(m, "depth_scale_pose_se3"), preds, base_stream)
    b = _drive(LearnedCorrectorAnchor(m, "depth_scale_pose_se3"), preds, tampered)
    for t in range(split + 1):
        assert np.isclose(a[t].s, b[t].s), f"scale diverged at {t}"
        assert np.allclose(a[t].t, b[t].t), f"translation diverged at {t}"
    assert not np.isclose(a[N - 1].s, b[N - 1].s)


def test_learned_anchor_reset_clears_the_recurrent_state():
    poses, depth, valid, preds = _world()
    m = _model()
    with torch.no_grad():
        for p in m.head[-1].parameters():
            p.normal_(0, 0.3)
    prompts = make_prompt_stream(PromptConfig(name="c", depth_interval=2, pose_interval=2),
                                 "s", N, depth, valid, poses)
    anchor = LearnedCorrectorAnchor(m, "depth_scale_pose_se3")
    first = _drive(anchor, preds, prompts)
    second = _drive(anchor, preds, prompts)     # reset() is called inside _drive
    for t in range(N):
        assert np.isclose(first[t].s, second[t].s), f"state leaked at frame {t}"


def test_learned_anchor_state_dict_roundtrip():
    poses, depth, valid, preds = _world()
    m = _model()
    with torch.no_grad():
        for p in m.head[-1].parameters():
            p.normal_(0, 0.3)
    prompts = make_prompt_stream(PromptConfig(name="c", depth_interval=2, pose_interval=2),
                                 "s", N, depth, valid, poses)
    a = LearnedCorrectorAnchor(m, "depth_scale_pose_se3")
    a.reset()
    for t in range(20):
        a.update(preds[t], prompts[t])
    state = a.state_dict()

    b = LearnedCorrectorAnchor(m, "depth_scale_pose_se3")
    b.reset()
    b.load_state_dict(state)
    for t in range(20, N):
        a.update(preds[t], prompts[t])
        b.update(preds[t], prompts[t])
    assert np.isclose(a.correction.s, b.correction.s)
    assert np.allclose(a.correction.t, b.correction.t)


def test_batched_rollout_matches_the_streaming_anchor():
    """Training (batched rollout) and evaluation (frame-by-frame) must agree."""
    poses, depth, valid, preds = _world()
    m = _model()
    with torch.no_grad():
        for p in m.head[-1].parameters():
            p.normal_(0, 0.2)
    prompts = make_prompt_stream(PromptConfig(name="c", depth_interval=3, pose_interval=3),
                                 "s", N, depth, valid, poses)
    feats, base = sequence_features(preds, prompts, BASELINES["depth_scale_pose_se3"]())
    s, R, t, _, _ = rollout(
        m, torch.from_numpy(feats)[None],
        torch.tensor([[b.s for b in base]], dtype=torch.float32),
        torch.tensor(np.stack([b.R for b in base])[None], dtype=torch.float32),
        torch.tensor(np.stack([b.t for b in base])[None], dtype=torch.float32))
    stream = _drive(LearnedCorrectorAnchor(m, "depth_scale_pose_se3"), preds, prompts)
    for i in range(N):
        assert np.isclose(float(s[0, i].detach()), stream[i].s, rtol=1e-4), f"scale mismatch at {i}"
        assert np.allclose(t[0, i].detach().numpy(), stream[i].t, atol=1e-4), f"translation mismatch at {i}"


def test_identity_base_mode_is_standalone():
    poses, depth, valid, preds = _world()
    m = _model(base="identity")
    prompts = make_prompt_stream(PromptConfig(name="c", depth_interval=3), "s", N,
                                 depth, valid, poses)
    out = _drive(LearnedCorrectorAnchor(m, "depth_scale_pose_se3"), preds, prompts)
    assert np.isclose(out[-1].s, 1.0)      # zero-init head => identity increments
