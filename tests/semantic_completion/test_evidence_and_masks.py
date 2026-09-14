"""Lifting, normalisation, crop/hiding masks, B0, metrics -- small deterministic checks."""
from __future__ import annotations
import os, sys
import numpy as np, pytest, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.semantic_completion import config as C
from tools.semantic_completion.dataset import crop_evidence, hidden_block_mask
from tools.semantic_completion.evaluation import baseline_nearest as B0, semantic_metrics as SM
from tools.semantic_completion.evaluation.regions import build_regions
from tools.semantic_completion.evidence import trident as TR
from tools.semantic_completion.evidence.lift import VoxelEvidence
from tools.semantic_completion.losses.feature_distillation import cosine_distillation


def test_sample_at_lattice_is_unit_norm_and_reproduces_constant_maps():
    tok = torch.zeros(C.TEACHER_DIM, 34, 128); tok[3] = 1.0
    out = TR.sample_at_lattice(tok, (140, 518))
    assert out.shape == (140, 518, C.TEACHER_DIM) and torch.allclose(out.norm(dim=-1), torch.ones(140, 518))
    assert torch.allclose(out[..., 3], torch.ones(140, 518))


def test_sample_at_lattice_left_right_halves_map_to_left_right_tokens():
    tok = torch.zeros(C.TEACHER_DIM, 34, 128); tok[0, :, :64] = 1.0; tok[1, :, 64:] = 1.0
    out = TR.sample_at_lattice(tok, (140, 518))
    assert out[70, 10].argmax() == 0 and out[70, 500].argmax() == 1


def test_classify_uses_cosine_argmax_and_label_offset():
    text = torch.eye(C.N_CLASSES, C.TEACHER_DIM)
    f = torch.zeros(2, C.TEACHER_DIM); f[0, 5] = 3.0; f[1, 0] = 0.1
    assert TR.classify(f, text).tolist() == [6, 1]


def test_crop_evidence_local_indices_round_trip():
    X, Y, Z = C.GRID_DIMS
    ijk = torch.tensor([[10, 20, 3], [200, 100, 30], [130, 140, 5]])
    flat = (ijk[:, 0] * Y + ijk[:, 1]) * Z + ijk[:, 2]
    ev = VoxelEvidence(flat, torch.nn.functional.normalize(torch.randn(3, C.TEACHER_DIM), dim=-1), torch.ones(3, dtype=torch.int32))
    idx, feat = crop_evidence(ev, (0, 10, 0), (128, 128, 32))
    assert len(idx) == 1 and int(idx[0]) == (10 * 128 + 10) * 32 + 3 and torch.equal(feat[0], ev.feature[0])


def test_hidden_block_mask_is_blockwise_and_deterministic():
    crop, block = (16, 16, 8), (8, 8, 4)
    idx = torch.arange(16 * 16 * 8)
    m1 = hidden_block_mask(idx, crop, block, 0.5, torch.Generator().manual_seed(3))
    m2 = hidden_block_mask(idx, crop, block, 0.5, torch.Generator().manual_seed(3))
    assert torch.equal(m1, m2)
    vol = m1.reshape(crop)
    for bx in range(2):
        for by in range(2):
            for bz in range(2):
                blk = vol[bx * 8:(bx + 1) * 8, by * 8:(by + 1) * 8, bz * 4:(bz + 1) * 4]
                assert bool(blk.all()) or not bool(blk.any())


def test_loss_supervises_only_given_voxels_and_splits_hidden():
    p = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1); t = p.clone(); t[2:] = -p[2:]
    out = cosine_distillation(p, t, torch.ones(4, dtype=torch.bool), torch.tensor([False, False, True, True]))
    assert abs(float(out["loss_visible"])) < 1e-6 and abs(float(out["loss_hidden"]) - 2.0) < 1e-6 and abs(float(out["loss"]) - 1.0) < 1e-6


def test_b0_nearest_label_and_distance():
    E = torch.zeros(C.N_VOXELS, dtype=torch.bool); lab = torch.zeros(C.N_VOXELS, dtype=torch.long)
    X, Y, Z = C.GRID_DIMS; a = (5 * Y + 5) * Z + 5; b = (50 * Y + 5) * Z + 5
    E[a] = E[b] = True; lab[a], lab[b] = 7, 11
    near, dist = B0.nearest_evidence(E)
    got = B0.nearest_label(lab, near)
    q = (10 * Y + 5) * Z + 5; r = (40 * Y + 5) * Z + 5
    assert int(got[q]) == 7 and int(got[r]) == 11 and abs(dist[q] - 5 * 0.2) < 1e-6 and dist[a] == 0


def test_regions_and_confusion():
    sem = torch.tensor([0, 1, 1, 7, 255, 7]); E = torch.tensor([0, 1, 0, 0, 1, 1], dtype=torch.bool); O = torch.tensor([1, 1, 1, 0, 1, 1], dtype=torch.bool)
    reg = build_regions(sem, E, O)
    assert reg["completed_unobserved"].tolist() == [False, False, True, False, False, False]
    assert reg["oracle_observed"].tolist() == [False, True, False, False, False, True] and not reg["ssc_full"][4]
    cm = SM.confusion(sem, torch.tensor([3, 1, 7, 7, 9, 7]), reg["oracle_unobserved"] | reg["oracle_observed"])
    s = SM.summarize(cm, True)
    assert cm[1, 1] == 1 and cm[1, 7] == 1 and cm[7, 7] == 2 and abs(s["per_class_iou"][7] - 2 / 3) < 1e-9 and abs(s["per_class_iou"][1] - 0.5) < 1e-9
