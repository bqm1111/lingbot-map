"""Semantic-field (B4) experiment: generative output, masking, feature conventions, firewalls, frozen geometry."""
from __future__ import annotations

import inspect
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from tools.semantic_field.config.schema import TRAIN_DRIVES, TRAIN_FORBIDDEN, MaskConfig          # noqa: E402
from tools.semantic_field.data.semantic_evidence import SparseSemanticEvidence, dense_semantic_channels, mask_evidence   # noqa: E402
from tools.semantic_field.losses.semantic_loss import centre, centered_feature_loss, text_distillation, text_log_probs   # noqa: E402
from tools.semantic_field.model.semantic_field_head import SemanticFieldHead                     # noqa: E402

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA path")
GRID = (64, 64, 16)


def _evidence(n=400, seed=0, dims=GRID, dim=1024):
    g = torch.Generator().manual_seed(seed)
    flat = torch.randperm(int(np.prod(dims)), generator=g)[:n].sort().values
    return SparseSemanticEvidence(flat, torch.nn.functional.normalize(torch.randn(n, dim, generator=g), dim=-1), torch.ones(n, dtype=torch.int32))


def test_output_is_not_restricted_to_a_convex_mixture_of_input_features():
    torch.manual_seed(0)
    model = SemanticFieldHead().eval()
    ev = _evidence(8, dims=(16, 16, 8))
    basis = torch.zeros(8, 1024); basis[torch.arange(8), torch.arange(8)] = 1.0             # every input lies in a 8-d subspace
    ev = SparseSemanticEvidence(ev.flat_index, basis, ev.observation_count)
    sem = dense_semantic_channels(ev, model.project_evidence, torch.zeros(1024), (16, 16, 8))
    x = torch.cat([torch.randn(24 + 3, 16, 16, 8), sem], 0)[None]
    with torch.no_grad():
        f = model.semantic_features(model.hidden_field(x)[0].reshape(32, -1).T[:20])
    outside = f[:, 8:].norm(dim=-1)                                                           # component orthogonal to the span of all inputs
    assert (outside > 0.5).all(), "B4 output must be able to leave the span (hence the convex hull) of the observed features"


def test_masking_is_deterministic_and_hidden_targets_are_removed_from_inputs():
    ev = _evidence(3000)
    a = mask_evidence(ev, GRID, MaskConfig(), torch.Generator().manual_seed(7)); b = mask_evidence(ev, GRID, MaskConfig(), torch.Generator().manual_seed(7))
    assert torch.equal(a[0].flat_index, b[0].flat_index) and torch.equal(a[1].flat_index, b[1].flat_index)
    visible, hidden = a
    assert len(set(visible.flat_index.tolist()) & set(hidden.flat_index.tolist())) == 0 and len(visible.flat_index) + len(hidden.flat_index) == 3000
    assert hidden.provenance == "masked_current" and 0.2 < len(hidden.flat_index) / 3000 < 0.8
    sem = dense_semantic_channels(visible, torch.nn.Linear(1024, 32), torch.zeros(1024), GRID)
    assert (sem[32].reshape(-1)[hidden.flat_index] == 0).all() and (sem[:32].reshape(32, -1)[:, hidden.flat_index] == 0).all()


def test_text_logits_use_the_frozen_bank_and_losses_behave():
    text = torch.nn.functional.normalize(torch.randn(18, 1024), dim=-1); f = torch.nn.functional.normalize(torch.randn(10, 1024), dim=-1)
    lp = text_log_probs(f, text, 40.0)
    assert torch.allclose(lp, torch.log_softmax(40.0 * f @ text.T, -1))
    assert float(text_distillation(lp.exp(), f, text, 40.0)) < 1e-5
    mu = torch.randn(1024) * 0.1
    assert float(centered_feature_loss(f, f, mu)) < 1e-6 and torch.allclose(centre(f, mu).norm(dim=-1), torch.ones(10))


def test_centering_uses_training_drives_only():
    from tools.semantic_field.data import feature_mean
    src = inspect.getsource(feature_mean.compute_feature_mean)
    assert "TRAIN_DRIVES" in src and "DEV_DRIVE" not in src and "drive_0006" in src
    assert "2013_05_28_drive_0006_sync" not in TRAIN_DRIVES


def test_training_cannot_open_labels_or_target_datasets():
    for token in ("eval_masks", "preprocess/labels", "data_3d_semantics"):
        assert token in TRAIN_FORBIDDEN
    code = ("import sys, tools.semantic_field.training.trainer; "
            "print(any(m in sys.modules for m in ('tools.delayed_semantic.gate16.candidate_regions', 'tools.zero_shot_eval.run_eval', 'tools.delayed_semantic.regions')))")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": REPO})
    assert out.stdout.strip() == "False", out.stderr[-400:]


def test_target_evaluation_is_locked_without_source_pass(tmp_path, monkeypatch):
    from tools.semantic_field import freeze
    monkeypatch.setattr(freeze, "ARTIFACT_DIR", str(tmp_path))
    with pytest.raises(RuntimeError):
        freeze.require_source_pass()


@needs_cuda
def test_b4_training_step_leaves_frozen_geometry_and_occupancy_bit_identical():
    from core.model.net import CompletionUNet
    from tools.semantic_completion.frozen_geometry import occupancy_and_trunk, state_digest
    dev = torch.device("cuda:0"); torch.manual_seed(0)
    net = CompletionUNet(cin=3, width=24, padding_mode="replicate").to(dev).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    x = (torch.rand(1, 3, 32, 32, 16, device=dev) > 0.9).float()
    occ0, trunk = occupancy_and_trunk(net, x); d0 = state_digest(net)
    model = SemanticFieldHead().to(dev); opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sem = torch.zeros(34, 32, 32, 16, device=dev)
    dense = torch.cat([trunk[0], x[0], sem], 0)[None]
    f = model.semantic_features(model.hidden_field(dense)[0].reshape(32, -1).T[:50])
    (1 - f[:, 0]).mean().backward(); opt.step()
    occ1, _ = occupancy_and_trunk(net, x)
    assert all(p.grad is None for p in net.parameters()) and state_digest(net) == d0 and torch.equal(occ0, occ1)


def test_parameter_budget():
    assert 500_000 <= SemanticFieldHead().n_params() <= 2_000_000
