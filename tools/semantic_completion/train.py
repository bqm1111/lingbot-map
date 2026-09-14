"""Train the propagation head for one geometry seed. Geometry frozen; only PropagationHead parameters move.

Per step: 4 anchors drawn uniformly from 0003/0007/0010 -> frozen full-grid trunk -> uniform 128x128x32 crops
-> visible evidence scattered into the head input -> cosine distillation on every evidence voxel of the crop.
AdamW, cosine schedule, grad clip, final step checkpoint (no selection).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from core.datasets.audit import FileAudit
from tools.semantic_completion import config as C
from tools.semantic_completion.dataset import EvidenceDataset
from tools.semantic_completion.evidence.build import anchor_ids
from tools.semantic_completion.frozen_geometry import load_frozen_geometry, sha256_file, state_digest
from tools.semantic_completion.losses.feature_distillation import cosine_distillation
from tools.semantic_completion.model.propagation_head import PropagationHead


def head_loss(head: PropagationHead, samples, crop) -> dict:
    trunks, dense, masks, preds, teachers, hidden = [], [], [], [], [], []
    for s in samples:
        vis = ~s.hidden
        dense.append(head.scatter_evidence(s.local_index[vis], s.feature[vis], crop))
        m = torch.zeros(int(np.prod(crop)), device=s.trunk.device); m[s.local_index[vis]] = 1.0
        masks.append(m.reshape(1, *crop)); trunks.append(s.trunk)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = head(torch.stack(trunks), torch.stack(dense), torch.stack(masks))
    out = torch.nn.functional.normalize(out.float(), dim=1)
    for b, s in enumerate(samples):
        preds.append(out[b].reshape(C.TEACHER_DIM, -1).T[s.local_index]); teachers.append(s.feature); hidden.append(s.hidden)
    p, t, h = torch.cat(preds), torch.cat(teachers), torch.cat(hidden)
    return cosine_distillation(p, t, torch.ones(len(p), dtype=torch.bool, device=p.device), h)


def train_head(seed: int, device: str, steps: int | None = None, paths: C.Paths = C.Paths(), cfg: C.TrainConfig = C.TrainConfig()) -> dict:
    dev = torch.device(device); torch.cuda.set_device(dev)
    steps = steps or cfg.steps; tag = f"sem_seed{seed}" if steps == cfg.steps else f"smoke_seed{seed}"
    torch.manual_seed(seed); np.random.seed(seed)
    net = load_frozen_geometry(seed, dev, paths); geo_digest_before = state_digest(net)
    head = PropagationHead().to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    os.makedirs(paths.checkpoints, exist_ok=True); os.makedirs(paths.runs, exist_ok=True)
    digest = hashlib.sha256(); log, ckpts = [], []; t0 = time.time()
    with FileAudit(patterns=C.TRAIN_FORBIDDEN_TOKENS, strict=True) as audit:
        data = EvidenceDataset([(d, anchor_ids(d, paths)) for d in C.TRAIN_DRIVES], dev, seed, paths)
        print(f"[{tag}] {len(data)} training anchors, head params {head.n_params():,}, steps {steps}", flush=True)
        for it in range(1, steps + 1):
            head.train()
            parts = head_loss(head, data.batch(net, cfg, digest), cfg.crop)
            opt.zero_grad(set_to_none=True); parts["loss"].backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.grad_clip); opt.step(); sched.step()
            row = {k: float(v) for k, v in parts.items()}; row["step"] = it; log.append(row)
            if it % cfg.log_every == 0 or it == steps:
                print(f"  [{tag} {it}/{steps}] loss {row['loss']:.4f} visible {row['loss_visible']:.4f} hidden {row['loss_hidden']:.4f} "
                      f"{(time.time()-t0)/it:.2f}s/it", flush=True)
            if it % cfg.ckpt_every == 0 or it == steps:
                p = paths.head_checkpoint(seed, it) if tag.startswith("sem") else os.path.join(paths.checkpoints, "smoke", f"{tag}_step{it:05d}.pt")
                os.makedirs(os.path.dirname(p), exist_ok=True)
                torch.save({"state_dict": head.state_dict(), "head_config": asdict(head.cfg), "seed": seed, "step": it,
                            "geometry_checkpoint": os.path.relpath(paths.geometry_checkpoint(seed), C.REPO_ROOT)}, p)
                ckpts.append({"step": it, "path": os.path.relpath(p, C.REPO_ROOT), "sha256": sha256_file(p)})
    assert all(p.grad is None for p in net.parameters()), "a gradient reached the frozen geometry network"
    rec = {"tag": tag, "seed": seed, "steps": steps, "train_config": asdict(cfg), "head_config": asdict(head.cfg), "head_params": head.n_params(),
           "n_train_anchors": len(data), "draw_digest": digest.hexdigest(), "geometry_state_digest_before": geo_digest_before,
           "geometry_state_digest_after": state_digest(net), "geometry_grads_none": True, "checkpoints": ckpts, "log": log,
           "seconds": time.time() - t0, "firewall": {"opened": len(audit.opened), "violations": audit.violations}}
    json.dump(rec, open(os.path.join(paths.runs, f"train_{tag}.json"), "w"), indent=1)
    print(f"[{tag}] done {rec['seconds']/60:.1f} min; geometry unchanged {geo_digest_before == rec['geometry_state_digest_after']}", flush=True)
    return rec


__all__ = ["head_loss", "train_head"]
