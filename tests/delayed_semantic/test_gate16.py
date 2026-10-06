"""Gate 16: candidate targets, evaluation-only masks, oracle ordering, causality, reproducibility."""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from tools.delayed_semantic import config as C                                          # noqa: E402
from tools.delayed_semantic.gate16 import config as G                                    # noqa: E402
from tools.delayed_semantic.gate16.candidate_regions import candidate_hit, oracle_label  # noqa: E402
from tools.delayed_semantic.gate16.candidate_targets import build_target                 # noqa: E402
from tools.delayed_semantic.gate16.candidates import CandidateSet                        # noqa: E402
from tools.delayed_semantic.gate16.selection_loss import selection_kl                    # noqa: E402

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA path")


def _probs(q, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.softmax(3 * torch.randn(q, k, 18, generator=g), -1), torch.softmax(3 * torch.randn(q, 18, generator=g), -1)


def _set(p):
    q, k, _ = p.shape
    return CandidateSet(torch.arange(q), torch.arange(k).expand(q, k), torch.arange(k).expand(q, k), torch.zeros(q, k), p, 1 + p.argmax(-1),
                        torch.ones(q, k, dtype=torch.bool))


def test_soft_candidate_target_is_a_distribution():
    p, q = _probs(50, G.K)
    ct = build_target(p, q)
    assert torch.allclose(ct.distribution.sum(-1), torch.ones(50)) and (ct.compatibility >= 0).all() and (ct.compatibility <= 1).all()
    assert torch.equal(ct.best_index, ct.compatibility.argmax(-1)) and G.TAU_TARGET == 1 / 40


def test_candidate_target_is_invariant_to_consistent_reordering():
    p, q = _probs(20, G.K, 1)
    perm = torch.randperm(G.K, generator=torch.Generator().manual_seed(3))
    a, b = build_target(p, q), build_target(p[:, perm], q)
    assert torch.allclose(a.distribution[:, perm], b.distribution, atol=1e-6)
    assert torch.equal(perm[b.best_index], a.best_index)


def test_no_semantic_gt_enters_target_construction_or_training():
    code = ("import sys; import tools.delayed_semantic.gate16.train, tools.delayed_semantic.gate16.candidate_targets, tools.delayed_semantic.gate16.selection_loss; "
            "print(any(m in sys.modules for m in ('tools.delayed_semantic.gate16.candidate_regions', 'tools.delayed_semantic.gate16.context', "
            "'tools.delayed_semantic.gate16.evaluate', 'tools.delayed_semantic.regions')))")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": REPO})
    assert out.stdout.strip() == "False", out.stderr[-500:]
    import inspect
    from tools.delayed_semantic.gate16 import candidate_targets, selection_loss
    src = inspect.getsource(candidate_targets) + inspect.getsource(selection_loss)
    assert "sem_gt" not in src and "eval_mask" not in src and "gt" not in inspect.signature(build_target).parameters


def test_candidate_hit_and_oracle_are_gt_functions_and_oracle_beats_nearest():
    p, _ = _probs(400, G.K, 2)
    cs = _set(p)
    gt = torch.randint(1, 19, (400,), generator=torch.Generator().manual_seed(5))
    hit = candidate_hit(cs, gt); orc = oracle_label(cs, gt)
    assert torch.equal(hit, (cs.classes == gt[:, None]).any(-1))
    assert ((orc == gt) == hit).all()                                   # the oracle is right exactly where a candidate carries GT
    near_ok = cs.classes[:, 0] == gt
    assert (near_ok <= (orc == gt)).all()                               # oracle >= nearest, voxel by voxel
    import inspect
    assert all("gt" in inspect.signature(f).parameters for f in (candidate_hit, oracle_label))


def test_selection_kl_is_zero_at_the_target_and_positive_elsewhere():
    p, q = _probs(10, G.K, 4)
    ct = build_target(p, q)
    assert float(selection_kl(ct.distribution, ct.distribution.log())) < 1e-6
    assert float(selection_kl(ct.distribution, torch.zeros(10, G.K))) > 0


def _synthetic_anchor(seed=0):
    g = torch.Generator().manual_seed(seed)
    flat = torch.randperm(C.N_VOXELS, generator=g)[:300].sort().values
    feat = torch.nn.functional.normalize(torch.randn(300, C.TEACHER_DIM, generator=g), dim=-1).half()
    D = torch.randperm(C.N_VOXELS, generator=g)[:120]
    return {"drive": "2013_05_28_drive_0000_sync", "t": 0, "flat": flat, "feature": feat, "count": torch.ones(300, dtype=torch.int32),
            "current": torch.zeros(300, dtype=torch.bool), "n_frames": torch.ones(300, dtype=torch.long), "trunk": torch.randn(300, 24, generator=g).half(),
            "masks_packed": torch.zeros(3, (C.N_VOXELS + 7) // 8, dtype=torch.uint8), "D_flat": D,
            "D_prob": torch.softmax(torch.randn(120, 18, generator=g), -1).half(), "D_trunk": torch.randn(120, 24, generator=g).half()}


def test_identical_seed_reproduces_draws_and_weights(tmp_path):
    from tools.delayed_semantic import train as PT
    from tools.delayed_semantic.model import EvidenceAssociation
    text = torch.nn.functional.normalize(torch.randn(18, C.TEACHER_DIM, generator=torch.Generator().manual_seed(9)), dim=-1)
    cfg = C.TrainConfig(hidden_queries=32, delayed_queries=32, anchors_per_step=2, log_every=10 ** 6, ckpt_every=10 ** 6)
    anchors = [_synthetic_anchor(0), _synthetic_anchor(1)]
    recs = []
    for seed in (0, 0, 1):
        torch.manual_seed(seed); m = EvidenceAssociation()
        r = PT.fit(m, anchors, text, cfg, 3, seed, True, "select", lambda it, s=seed: str(tmp_path / f"s{s}_{len(recs)}.pt"), "t", {})
        recs.append((r["draw_digest"], hashlib.sha256(b"".join(v.numpy().tobytes() for v in m.state_dict().values())).hexdigest()))
    assert recs[0] == recs[1] and recs[0][0] != recs[2][0]


@needs_cuda
def test_selection_objective_leaves_frozen_geometry_untouched():
    from core.model.net import CompletionUNet
    from tools.delayed_semantic.model import EvidenceAssociation
    from tools.semantic_completion.frozen_geometry import occupancy_and_trunk, state_digest
    dev = torch.device("cuda:0"); torch.manual_seed(0)
    net = CompletionUNet(cin=3, width=24, padding_mode="replicate").to(dev).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    x = (torch.rand(1, 3, 32, 32, 16, device=dev) > 0.9).float()
    occ0, trunk = occupancy_and_trunk(net, x); d0 = state_digest(net)
    head = EvidenceAssociation(semantic_projection=False).to(dev); N, Q = 40, 12
    ev = {"feature": torch.nn.functional.normalize(torch.randn(N, C.TEACHER_DIM, device=dev), dim=-1), "trunk": trunk.reshape(24, -1)[:, :N].T,
          "xyz": torch.rand(N, 3, device=dev) * 5, "count": torch.ones(N, device=dev), "current": torch.zeros(N, dtype=torch.bool, device=dev), "n_frames": torch.ones(N, device=dev)}
    _, _, s = head(trunk.reshape(24, -1)[:, N:N + Q].T, torch.rand(Q, 3, device=dev) * 5, torch.randint(N, (Q, 16), device=dev), ev,
                   torch.zeros(3, *C.GRID_DIMS, dtype=torch.bool, device=dev), return_logits=True)
    selection_kl(torch.full((Q, 16), 1 / 16, device=dev), s).backward()
    occ1, _ = occupancy_and_trunk(net, x)
    assert all(p.grad is None for p in net.parameters()) and state_digest(net) == d0 and torch.equal(occ0, occ1)


def _pilot_record(t=4):
    return os.path.join(C.Paths().predictions_root, C.DEV_DRIVE, f"{t:05d}.npz")


@needs_cuda
@pytest.mark.skipif(not os.path.exists(_pilot_record()), reason="pilot prediction records not built")
def test_student_context_is_prefix_causal_and_b0_reproduces_the_pilot():
    from tools.delayed_semantic.gate16.context import student_context
    from tools.delayed_semantic.stream import DriveStream, StreamFrame
    from tools.semantic_completion.evidence import trident as TR
    from tools.semantic_completion.frozen_geometry import load_frozen_geometry
    dev = torch.device("cuda:0"); t = 4
    text, _, _ = TR.load_text_embeddings(); text = text.to(dev)
    net = load_frozen_geometry(C.GEOMETRY_SEED, dev); s = DriveStream(C.DEV_DRIVE)
    ref = student_context(s, t, net, text, dev)
    cut = object.__new__(DriveStream); cut.drive = s.drive
    cut.frames = s.frames[:t + 1] + [StreamFrame(f.index, f.key, f.native, "/nonexistent", "/nonexistent", f.time_s) for f in s.frames[t + 1:]]
    other = student_context(cut, t, net, text, dev)
    assert all(torch.equal(ref[k], other[k]) for k in ("O", "E", "b0", "ev_label", "S", "F"))
    with np.load(_pilot_record(t)) as z:
        O = torch.from_numpy(np.unpackbits(z["O"])[:C.N_VOXELS].astype(bool)).to(dev)
        assert torch.equal(O, ref["O"])
        assert np.array_equal(z["label_B0"], ref["b0"][O].to(torch.uint8).cpu().numpy())


@needs_cuda
def test_delayed_targets_reproduce_the_pilot_records():
    from core.datasets import kitti360_splits as SRC, kitti360_transforms as TF
    from tools.delayed_semantic import targets as TG
    from tools.delayed_semantic.stream import DriveStream
    from tools.semantic_completion.evidence import trident as TR
    dev = torch.device("cuda:0"); P = C.Paths(); d, t = C.TRAIN_DRIVES[0], 50
    if not os.path.exists(P.target_path(d, t)):
        pytest.skip("targets not built")
    text, _, _ = TR.load_text_embeddings(); text = text.to(dev)
    s = DriveStream(d); geo = TF.DriveGeometry(d, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
    with np.load(P.g15.evidence_path(d, t)) as z:
        E = torch.zeros(C.N_VOXELS, dtype=torch.bool, device=dev); E[torch.from_numpy(z["flat_index"].astype(np.int64)).to(dev)] = True
    out = TG.build_anchor(s, geo, t, TG.ViewCache(geo, text, C.TargetConfig(), dev), E, C.TargetConfig(), dev)["arrays"]
    with np.load(P.target_path(d, t)) as st:
        assert all(np.array_equal(st[k], v, equal_nan=True) for k, v in out.items() if k != "prob")
        assert np.abs(st["prob"].astype(np.float32) - out["prob"].astype(np.float32)).max() <= 1e-4
