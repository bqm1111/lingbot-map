"""Train B4 for one seed on KITTI-360 0003/0007/0010. Geometry frozen; only SemanticFieldHead parameters move.

Per step (``anchors_per_step`` anchors, generator seeded with the head seed):
    frozen geometry: hard map [S, 0, F] -> CompletionUNet (no_grad, bf16) -> occupancy O [N], trunk d1 [24, X, Y, Z]
    masking: Gate 15 grid-aligned 16x16x8 cuboids hidden with p = 0.5 -> visible evidence (input) / hidden evidence (target)
    crop: uniform 128x128x32 window; dense input = [trunk, S, F, O, projected centered visible evidence, mask, count]
    targets inside the crop: hidden evidence (<= 8,192, masked-current loss) and pilot delayed targets (<= 4,096, delayed loss)
Loss = L_masked + lambda_delayed * L_delayed. AdamW, cosine schedule, grad clip, final step kept.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import torch

from core.datasets.audit import FileAudit
from tools.delayed_semantic import config as PILOT, rules as PILOT_RULES
from tools.delayed_semantic.train import unpack
from tools.semantic_completion.evidence import build as G15B, trident as TR
from tools.semantic_completion.frozen_geometry import load_frozen_geometry, occupancy_and_trunk, state_digest
from tools.semantic_field import freeze
from tools.semantic_field.config.schema import (GEOMETRY_SEED, REPO_ROOT, TARGET_TOKENS, TRAIN_DRIVES, TRAIN_FORBIDDEN, FeatureSpace, MaskConfig,
                                                Paths, TrainConfig)
from tools.semantic_field.data.feature_mean import load_feature_mean
from tools.semantic_field.data.semantic_evidence import SparseSemanticEvidence, dense_semantic_channels, mask_evidence
from tools.semantic_field.losses.semantic_loss import delayed_loss, masked_reconstruction_loss
from tools.semantic_field.model.semantic_field_head import SemanticFieldHead

GRID = PILOT.GRID_DIMS


def load_training_anchor(drive: str, t: int, device, tiers: List[int]) -> Dict:
    g15 = PILOT.Paths().g15
    with np.load(g15.evidence_path(drive, t)) as z:
        ev = {"flat": torch.from_numpy(z["flat_index"].astype(np.int64)).to(device), "feature": torch.from_numpy(z["feature"]).to(device),
              "count": torch.from_numpy(z["count"]).to(device)}
    with np.load(g15.input_path(drive, t)) as z:
        ev["S_packed"] = torch.from_numpy(z["S_H"]).to(device); ev["F_packed"] = torch.from_numpy(z["F_H"]).to(device)
    with np.load(PILOT.Paths().target_path(drive, t)) as z:
        keep = ~z["in_evidence"] & np.isin(z["tier"], tiers)
        ev["D_flat"] = torch.from_numpy(z["flat_index"][keep].astype(np.int64)).to(device)
        ev["D_prob"] = torch.from_numpy(z["prob"][keep]).to(device)
    ev["drive"], ev["t"] = drive, t
    return ev


def to_crop(flat: torch.Tensor, origin: Tuple[int, int, int], crop: Tuple[int, int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Global flat index -> (inside mask [N], local flat index inside the crop [n])."""
    X, Y, Z = GRID
    ix, iy, iz = flat // (Y * Z), (flat // Z) % Y, flat % Z
    inside = (ix >= origin[0]) & (ix < origin[0] + crop[0]) & (iy >= origin[1]) & (iy < origin[1] + crop[1]) & (iz >= origin[2]) & (iz < origin[2] + crop[2])
    local = ((ix[inside] - origin[0]) * crop[1] + (iy[inside] - origin[1])) * crop[2] + (iz[inside] - origin[2])
    return inside, local


def subsample(n: int, k: int, gen: torch.Generator, device) -> torch.Tensor:
    return torch.randperm(n, generator=gen)[:k].to(device)


def train_seed(seed: int, device: str, steps: int | None = None, cfg: TrainConfig = TrainConfig(), mask_cfg: MaskConfig = MaskConfig(),
               space: FeatureSpace = FeatureSpace(), paths: Paths = Paths()) -> Dict:
    freeze.check("source_predeclaration"); freeze.check("centering"); PILOT_RULES.check("reliability")
    tiers = json.load(open(os.path.join(PILOT.Paths().reliability, "reliability.json")))["gate"]["delayed_loss_tiers"]
    dev = torch.device(device); torch.cuda.set_device(dev)
    steps = steps or cfg.steps; tag = f"B4_seed{seed}" if steps == cfg.steps else f"smoke_B4_seed{seed}"
    text, _, _ = TR.load_text_embeddings(); text = text.to(dev)
    mu = load_feature_mean(dev)
    net = load_frozen_geometry(GEOMETRY_SEED, dev); digest0 = state_digest(net)
    torch.manual_seed(seed); model = SemanticFieldHead().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator().manual_seed(seed); gen_d = torch.Generator().manual_seed(seed + 1)
    draws = hashlib.sha256(); log, ckpts = [], []; t0 = time.time()
    with FileAudit(patterns=TRAIN_FORBIDDEN + TARGET_TOKENS, strict=True) as audit:
        anchors = [load_training_anchor(d, t, dev, tiers) for d in TRAIN_DRIVES for t in G15B.anchor_ids(d)]
        print(f"[{tag}] {len(anchors)} anchors loaded in {(time.time() - t0) / 60:.1f} min; params {model.n_params():,}", flush=True)
        for it in range(1, steps + 1):
            picks = torch.randint(len(anchors), (cfg.anchors_per_step,), generator=gen).tolist()
            batch = [anchors[i] for i in picks]
            x = torch.stack([torch.stack([unpack(a["S_packed"]).reshape(GRID), torch.zeros(GRID, dtype=torch.bool, device=dev),
                                          unpack(a["F_packed"]).reshape(GRID)]).float() for a in batch])
            occ, trunk = occupancy_and_trunk(net, x)
            dense, hid_local, hid_feat, del_local, del_prob = [], [], [], [], []
            for b, a in enumerate(batch):
                origin = tuple(int(torch.randint(0, g - c + 1, (1,), generator=gen)) for g, c in zip(GRID, cfg.crop))
                evidence = SparseSemanticEvidence(a["flat"], a["feature"], a["count"])
                visible, hidden = mask_evidence(evidence, GRID, mask_cfg, gen)
                vin, vloc = to_crop(visible.flat_index, origin, cfg.crop)
                sem = dense_semantic_channels(SparseSemanticEvidence(vloc, visible.features[vin], visible.observation_count[vin]), model.project_evidence, mu, cfg.crop)
                sl = tuple(slice(o, o + c) for o, c in zip(origin, cfg.crop))
                geo = torch.stack([x[b, 0], x[b, 2], occ[b].reshape(GRID).float()])[(slice(None),) + sl]
                dense.append(torch.cat([trunk[b][(slice(None),) + sl], geo, sem], 0))
                hin, hloc = to_crop(hidden.flat_index, origin, cfg.crop)
                pick = subsample(len(hloc), cfg.max_masked_targets, gen, dev)
                hid_local.append(hloc[pick] + b * int(np.prod(cfg.crop))); hid_feat.append(hidden.features[hin][pick].float())
                din, dloc = to_crop(a["D_flat"], origin, cfg.crop)
                pick_d = subsample(len(dloc), cfg.max_delayed_targets, gen_d, dev)
                del_local.append(dloc[pick_d] + b * int(np.prod(cfg.crop))); del_prob.append(a["D_prob"][din][pick_d].float())
                draws.update(f"{a['drive'][17:21]}:{a['t']}:{origin}:{len(vloc)}:{len(hloc)};".encode())
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden_field = model.hidden_field(torch.stack(dense))
            flat_hidden = hidden_field.float().permute(0, 2, 3, 4, 1).reshape(-1, hidden_field.shape[1])
            hi, hf = torch.cat(hid_local), torch.cat(hid_feat); di, dp = torch.cat(del_local), torch.cat(del_prob)
            row, loss = {"step": it, "n_masked": len(hi), "n_delayed": len(di)}, torch.zeros((), device=dev)
            if len(hi):
                s = model.semantic_features(flat_hidden[hi])
                lm = masked_reconstruction_loss(hf, s, text, mu, space.text_logit_scale, cfg.lambda_text, cfg.lambda_feature)
                loss = loss + lm["loss"]; row.update({"masked": float(lm["loss"]), "masked_text": float(lm["text"]), "masked_feature": float(lm["feature"]),
                                                     "masked_teacher_agree": float(((s @ text.T).argmax(1) == (hf @ text.T).argmax(1)).float().mean())})
            if len(di):
                s = model.semantic_features(flat_hidden[di])
                ld = delayed_loss(dp, s, text, space.text_logit_scale)
                loss = loss + cfg.lambda_delayed * ld["loss"]
                row.update({"delayed": float(ld["loss"]), "delayed_agree": float(((s @ text.T).argmax(1) == dp.argmax(1)).float().mean())})
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); opt.step(); sched.step()
            row["loss"] = float(loss); log.append(row)
            if it % cfg.log_every == 0 or it == steps:
                print(f"  [{tag} {it}/{steps}] " + " ".join(f"{k} {v:.4f}" for k, v in row.items() if isinstance(v, float)) + f" {(time.time() - t0) / 60:.1f} min", flush=True)
            if it % cfg.ckpt_every == 0 or it == steps:
                p = paths.checkpoint(seed, it) if not tag.startswith("smoke") else os.path.join(paths.checkpoints, "smoke", f"{tag}_step{it:05d}.pt")
                os.makedirs(os.path.dirname(p), exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "config": model.config(), "seed": seed, "step": it}, p)
                ckpts.append({"step": it, "path": os.path.relpath(p, REPO_ROOT), "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()})
    assert all(p.grad is None for p in net.parameters()) and state_digest(net) == digest0, "the frozen geometry changed"
    rec = {"tag": tag, "seed": seed, "steps": steps, "params": model.n_params(), "train_config": asdict(cfg), "mask_config": asdict(mask_cfg),
           "model_config": model.config(), "tiers": tiers, "n_anchors": len(anchors), "draw_digest": draws.hexdigest(), "geometry_digest": digest0,
           "checkpoints": ckpts, "log": log, "seconds": time.time() - t0, "firewall": {"opened": len(audit.opened), "violations": audit.violations}}
    os.makedirs(paths.runs, exist_ok=True)
    json.dump(rec, open(os.path.join(paths.runs, f"train_{tag}.json"), "w"), indent=1)
    return rec


__all__ = ["load_training_anchor", "to_crop", "train_seed"]
