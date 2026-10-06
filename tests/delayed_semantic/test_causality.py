"""Causality, provenance, masks, transforms and occupancy invariance of the delayed-supervision pilot."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.delayed_semantic import config as C                                        # noqa: E402
from tools.delayed_semantic.baselines import geodesic_label                            # noqa: E402
from tools.delayed_semantic.memory import EvidenceMemory, PredictedSemantics           # noqa: E402
from tools.delayed_semantic.model import EvidenceAssociation, kl_to_target, voxel_centres  # noqa: E402
from tools.delayed_semantic.regions import build_regions                               # noqa: E402
from tools.delayed_semantic.stream import CausalHistory, DriveStream, FutureAccess, StreamFrame  # noqa: E402
from tools.delayed_semantic.targets import _voxel_of                                    # noqa: E402
from tools.semantic_completion.evidence.lift import grid_world_keys                     # noqa: E402

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA path")
X, Y, Z = C.GRID_DIMS


def _frames(n):
    return [StreamFrame(i, f"k{i:04d}", 5 * i, f"/img/{i}.png", f"/tok/{i}.npz", 0.5 * i) for i in range(n)]


def _flat(x, y, z):
    return (x * Y + y) * Z + z


# ------------------------------------------------------------------ causal history
def test_history_is_t_minus_4_to_t_and_rejects_future_frames():
    s = object.__new__(DriveStream); s.drive = "synthetic"; s.frames = _frames(30)
    h = s.history(10)
    assert h.indices == [6, 7, 8, 9, 10] and h.frame(10).index == 10
    with pytest.raises(FutureAccess):
        h.frame(11)
    assert [f.index for f in s.future(10)] == list(range(11, 30))[:C.TargetConfig().horizon_frames]


def test_history_rejects_a_window_that_is_not_t_minus_4_to_t():
    with pytest.raises(AssertionError):
        CausalHistory(_frames(12)[5:11], 10)


# ------------------------------------------------------------------ observed vs predicted provenance
def test_memory_rejects_predictions_and_future_frames():
    mem = EvidenceMemory(10, "cpu"); keys = torch.tensor([1, 2])
    p = PredictedSemantics(torch.tensor([1]), torch.zeros(1, C.TEACHER_DIM), 10, "B2")
    with pytest.raises(TypeError):
        mem.observe(4, 10, 5.0, keys, p)
    with pytest.raises(ValueError):
        mem.observe(4, 11, 5.5, keys, None)


def test_memory_records_frame_bits_times_and_the_gate15_feature_rule():
    T = np.eye(4); gk = grid_world_keys(np.linalg.inv(T), "cpu")
    a, b, c = _flat(10, 20, 5), _flat(11, 20, 5), _flat(40, 7, 3)
    fa, fb = torch.zeros(C.TEACHER_DIM), torch.zeros(C.TEACHER_DIM); fa[0] = 1.0; fb[1] = 1.0
    mem = EvidenceMemory(10, "cpu")
    mem.observe(0, 6, 3.0, gk[[a, b]], torch.stack([fa, fa]))
    mem.observe(4, 10, 5.0, gk[[b, c, b]], torch.stack([fb, fb, fb]))
    S = torch.ones(C.N_VOXELS, dtype=torch.bool); S[c] = False                       # c is not hard-map occupied: no evidence
    ev = mem.snapshot(S, T)
    assert ev.flat_index.tolist() == sorted([a, b]) and ev.provenance == "observed"
    row = {int(f): i for i, f in enumerate(ev.flat_index)}
    assert int(ev.frame_bits[row[a]]) == 1 and int(ev.frame_bits[row[b]]) == 1 | 16
    assert ev.current().tolist()[row[b]] and not ev.current().tolist()[row[a]]
    assert float(ev.first_time_s[row[b]]) == 3.0 and float(ev.last_time_s[row[b]]) == 5.0
    assert int(ev.count[row[b]]) == 3
    expect = torch.nn.functional.normalize(fa + 2 * fb, dim=0)                        # count-weighted mean, then unit norm
    assert torch.allclose(ev.feature[row[b]], expect, atol=1e-6)


# ------------------------------------------------------------------ never-observed masks
def test_region_masks_separate_current_past_never_and_geometric_observation():
    n = C.N_VOXELS; z = lambda: torch.zeros(n, dtype=torch.bool)
    gt = torch.zeros(n, dtype=torch.long); ids = [_flat(1, 1, 1), _flat(2, 2, 2), _flat(3, 3, 3), _flat(4, 4, 4), _flat(5, 5, 5)]
    gt[ids] = 7
    E, Ec, S, F_, O, R, Rh = z(), z(), z(), z(), z(), z(), z()
    E[ids[0]] = Ec[ids[0]] = True                    # observed now
    E[ids[1]] = True                                 # observed earlier only
    O[ids[2:]] = True; S[ids[2]] = True              # completed, geometry saw ids[2]
    R[ids[3]] = True; Rh[ids[3]] = True; R[ids[4]] = True
    r = build_regions(gt, E, Ec, S, F_, O, R, Rh)
    on = lambda k: set(r[k].nonzero(as_tuple=True)[0].tolist())
    assert on("observed_current") == {ids[0]} and on("observed_past_only") == {ids[1]}
    assert on("never_observed") == set(ids[2:]) and on("completed_never_observed") == set(ids[2:])
    assert on("completed_geo_observed") == {ids[2]} and on("completed_geo_unobserved") == {ids[3], ids[4]}
    assert on("completed_later_revealed") == {ids[3], ids[4]} and on("completed_revealed_unseen") == {ids[4]}
    assert ids[1] not in on("never_observed")        # "unobserved" is not "unseen in the current frame"


# ------------------------------------------------------------------ target coordinate transforms
def test_target_voxelisation_uses_the_evaluator_floor_rule():
    from core.datasets import kitti360_transforms as TF
    rng = np.random.default_rng(0)
    pts = rng.uniform([-2, -30, -3], [55, 30, 6], (5000, 3))
    ref_idx, ok = TF.voxelize(pts)
    ours = _voxel_of(torch.as_tensor(pts))
    assert (ours >= 0).numpy().tolist() == ok.tolist()
    assert np.array_equal(ours[ours >= 0].numpy(), TF.flat(ref_idx))
    assert torch.allclose(voxel_centres(torch.as_tensor(TF.flat(ref_idx[:50]))).double(), torch.as_tensor(TF.centres(ref_idx[:50])), atol=1e-5)


def test_pose_chain_round_trips_and_projection_inverts():
    from core.datasets import kitti360_splits as SRC, kitti360_transforms as TF
    geo = TF.DriveGeometry(C.DEV_DRIVE, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
    fr = sorted(geo.cam0_to_world)[100:130:10]
    a, b = fr[0], fr[-1]
    T_ab = TF.inv(geo.velo_to_world(a)) @ geo.velo_to_world(b)
    T_ba = TF.inv(geo.velo_to_world(b)) @ geo.velo_to_world(a)
    assert np.allclose(T_ab @ T_ba, np.eye(4), atol=1e-9)
    K = geo.calib.K; u, v, d = 700.3, 150.8, 17.0
    p_cam = np.array([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d])
    p_velo = TF.apply(geo.calib.rect_cam_to_velo, p_cam[None])
    q = TF.apply(geo.calib.velo_to_rect_cam, p_velo)[0]
    assert np.allclose([K[0, 0] * q[0] / q[2] + K[0, 2], K[1, 1] * q[1] / q[2] + K[1, 2]], [u, v], atol=1e-6)


# ------------------------------------------------------------------ fixed baseline, loss
def test_geodesic_label_follows_occupied_paths_not_free_space():
    n = C.N_VOXELS
    lab = torch.zeros(n, dtype=torch.long); E = torch.zeros(n, dtype=torch.bool); O = torch.zeros(n, dtype=torch.bool)
    a, b = _flat(10, 10, 5), _flat(10, 16, 5)                  # evidence a (class 1) and b (class 2)
    E[a] = E[b] = True; lab[a], lab[b] = 1, 2
    for x in range(10, 20):
        O[_flat(x, 10, 5)] = True                               # corridor from a along +x
    q = _flat(19, 10, 5)                                        # Euclidean-closer to b? |q-a| = 9, |q-b| = sqrt(81+36) > 9 -> use a point near b
    O[_flat(13, 15, 5)] = True                                  # isolated occupied voxel next to b, not connected to a
    fallback = torch.full((n,), 9, dtype=torch.long)
    out, reached = geodesic_label(lab, E, O, fallback, max_iter=64)
    assert int(out[q]) == 1 and bool(reached[q])
    assert int(out[_flat(13, 15, 5)]) == 9 and not bool(reached[_flat(13, 15, 5)])


def test_kl_is_zero_for_the_student_own_distribution():
    text = torch.nn.functional.normalize(torch.randn(C.N_CLASSES, C.TEACHER_DIM), dim=-1)
    f = torch.nn.functional.normalize(torch.randn(4, C.TEACHER_DIM), dim=-1)
    p = torch.softmax(C.LOGIT_SCALE * f @ text.T, -1)
    assert float(kl_to_target(p, f, text)) < 1e-5


# ------------------------------------------------------------------ occupancy invariance
@needs_cuda
def test_head_training_leaves_frozen_geometry_and_occupancy_bit_identical():
    from core.model.net import CompletionUNet
    from tools.semantic_completion.frozen_geometry import occupancy_and_trunk, state_digest
    dev = torch.device("cuda:0"); torch.manual_seed(0)
    net = CompletionUNet(cin=3, width=24, padding_mode="replicate").to(dev).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    x = (torch.rand(1, 3, 32, 32, 16, device=dev) > 0.9).float()
    occ0, trunk = occupancy_and_trunk(net, x); d0 = state_digest(net)
    head = EvidenceAssociation().to(dev); opt = torch.optim.AdamW(head.parameters(), lr=1e-2)
    N, Q = 40, 12
    ev = {"feature": torch.nn.functional.normalize(torch.randn(N, C.TEACHER_DIM, device=dev), dim=-1), "trunk": trunk.reshape(24, -1)[:, :N].T,
          "xyz": torch.rand(N, 3, device=dev) * 5, "count": torch.ones(N, device=dev), "current": torch.zeros(N, dtype=torch.bool, device=dev),
          "n_frames": torch.ones(N, device=dev)}
    masks = torch.zeros(3, *C.GRID_DIMS, dtype=torch.bool, device=dev)
    text = torch.nn.functional.normalize(torch.randn(C.N_CLASSES, C.TEACHER_DIM, device=dev), dim=-1)
    for _ in range(3):
        f, _w = head(trunk.reshape(24, -1)[:, N:N + Q].T, torch.rand(Q, 3, device=dev) * 5, torch.randint(N, (Q, 16), device=dev), ev, masks)
        loss = kl_to_target(torch.full((Q, C.N_CLASSES), 1 / C.N_CLASSES, device=dev), f, text)
        opt.zero_grad(); loss.backward(); opt.step()
    assert all(p.grad is None for p in net.parameters())
    occ1, _ = occupancy_and_trunk(net, x)
    assert state_digest(net) == d0 and torch.equal(occ0, occ1)


# ------------------------------------------------------------------ prefix invariance on a real anchor
def _real_caches_present(t=4):
    p = C.Paths()
    return os.path.exists(p.provenance_path(C.DEV_DRIVE, t)) and os.path.exists(p.g15.evidence_path(C.DEV_DRIVE, t))


@needs_cuda
@pytest.mark.skipif(not _real_caches_present(), reason="0006 caches not built")
def test_prediction_at_t_ignores_changed_and_appended_future_frames():
    from core.datasets.audit import FileAudit
    from tools.delayed_semantic.evaluate import predict_anchor
    from tools.semantic_completion.evidence import trident as TR
    from tools.semantic_completion.frozen_geometry import load_frozen_geometry
    dev = torch.device("cuda:0"); t = 40
    stream = DriveStream(C.DEV_DRIVE)
    net = load_frozen_geometry(C.GEOMETRY_SEED, dev)
    torch.manual_seed(0); heads = {"B2": EvidenceAssociation().to(dev).eval()}
    text, _, _ = TR.load_text_embeddings(); text = text.to(dev)
    with FileAudit(patterns=(), strict=False) as audit:
        ref = predict_anchor(stream, t, net, heads, text, None, dev)
    future_keys = [f.key for f in stream.frames[t + 1:]]
    assert not any(k in p for p in audit.opened for k in future_keys), "a future frame was opened"
    changed = object.__new__(DriveStream); changed.drive = stream.drive
    changed.frames = stream.frames[:t + 1] + [StreamFrame(f.index, f.key, f.native, "/nonexistent.png", "/nonexistent.npz", f.time_s + 99) for f in stream.frames[t + 1:]] \
        + [StreamFrame(len(stream) + i, f"x{i}", 10 ** 6 + i, "/x.png", "/x.npz", 1e6) for i in range(5)]
    truncated = object.__new__(DriveStream); truncated.drive = stream.drive; truncated.frames = stream.frames[:t + 1]
    for s in (changed, truncated):
        out = predict_anchor(s, t, net, heads, text, None, dev)
        assert torch.equal(out["O"], ref["O"])
        assert all(torch.equal(out["labels"][m], ref["labels"][m]) for m in ref["labels"])
