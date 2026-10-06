"""Train B3 (candidate-selection delayed term) or B3geom for every Gate 16 seed in one process.

Uses the pilot's anchors, observed-only term, TrainConfig and optimisation loop (``tools/delayed_semantic/train.fit``)
unchanged; only ``delayed_objective="select"`` differs from pilot B2. Seed s: ``torch.manual_seed(s)`` before building the
scorer, observed-term generator ``s``, delayed-subsample generator ``s + 1`` (seed 0 == pilot B2's draws).
"""
from __future__ import annotations

import json
import os
from typing import Dict, Sequence

import torch

from core.datasets.audit import FileAudit
from tools.delayed_semantic import config as PC, rules as PR, train as PT
from tools.delayed_semantic.gate16 import config as G, rules as R
from tools.delayed_semantic.model import EvidenceAssociation
from tools.semantic_completion.evidence import trident as TR


def train_variant(variant: str, device: str, seeds: Sequence[int] = G.SEEDS, steps: int | None = None) -> Dict[int, Dict]:
    assert variant in G.VARIANTS
    R.check("predeclaration"); PR.check("reliability")
    rel = json.load(open(os.path.join(PC.Paths().reliability, "reliability.json")))["gate"]
    assert rel["pass"]; tiers = rel["delayed_loss_tiers"]
    cfg = PC.TrainConfig(); steps = steps or cfg.steps; smoke = steps != cfg.steps
    dev = torch.device(device); torch.cuda.set_device(dev); G.ensure_dirs()
    text, _, _ = TR.load_text_embeddings(); text = text.to(dev)
    out = {}
    with FileAudit(patterns=PC.TRAIN_FORBIDDEN, strict=True) as audit:
        anchors = PT.load_training_anchors(dev, tiers, "")
        print(f"[{variant}] {len(anchors)} anchors loaded", flush=True)
        for s in seeds:
            torch.manual_seed(s)
            model = EvidenceAssociation(semantic_projection=G.VARIANTS[variant]["semantic_projection"]).to(dev)
            tag = f"{'smoke_' if smoke else ''}{variant}_seed{s}"
            ckpt = (lambda it, tag=tag: os.path.join(G.CHECKPOINTS, "smoke" if smoke else "", f"{tag.replace('smoke_', '')}_step{it:05d}.pt"))
            rec = PT.fit(model, anchors, text, cfg, steps, s, True, "select", ckpt, tag,
                         {"method": variant, "tiers": tiers, "tau_target": G.TAU_TARGET, "k_candidates": G.K})
            rec["firewall"] = {"opened": len(audit.opened), "violations": list(audit.violations)}
            json.dump(rec, open(os.path.join(G.RUNS, f"train_{tag}.json"), "w"), indent=1)
            out[s] = rec
    return out


__all__ = ["train_variant"]
