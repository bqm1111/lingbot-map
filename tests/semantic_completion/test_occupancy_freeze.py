"""The semantic head cannot change occupancy or send gradient into the frozen geometry network."""
from __future__ import annotations
import os, sys
import pytest, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.model.net import CompletionUNet
from tools.semantic_completion import config as C
from tools.semantic_completion.frozen_geometry import occupancy_and_trunk, state_digest
from tools.semantic_completion.model.propagation_head import PropagationHead

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 autocast path runs on CUDA")


@needs_cuda
def test_head_training_leaves_geometry_bit_identical():
    dev = torch.device("cuda:0"); torch.manual_seed(0)
    net = CompletionUNet(cin=3, width=24, padding_mode="replicate").to(dev).eval()
    torch.nn.init.normal_(net.head_occ.weight, std=0.1)
    for p in net.parameters():
        p.requires_grad_(False)
    x = (torch.rand(1, 3, 32, 32, 16, device=dev) > 0.9).float()
    occ0, trunk = occupancy_and_trunk(net, x); d0 = state_digest(net)
    head = PropagationHead().to(dev); opt = torch.optim.AdamW(head.parameters(), lr=1e-2)
    dense = torch.zeros(1, head.cfg.evidence_dim, 32, 32, 16, device=dev); mask = torch.zeros(1, 1, 32, 32, 16, device=dev)
    for _ in range(3):
        out = head(trunk, dense, mask); loss = (1 - out[:, 0]).mean(); opt.zero_grad(); loss.backward(); opt.step()
    assert not trunk.requires_grad and all(p.grad is None for p in net.parameters())
    occ1, _ = occupancy_and_trunk(net, x)
    assert state_digest(net) == d0 and torch.equal(occ0, occ1)


def test_head_is_much_smaller_than_geometry():
    assert PropagationHead().n_params() < 0.35 * 967_322
