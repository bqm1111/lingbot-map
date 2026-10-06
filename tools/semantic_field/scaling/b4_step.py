"""One B4 training step, lifted unchanged from ``tools/semantic_field/training/trainer.py::train_seed`` (the loop body).

Generator consumption order per step (must never change -- it defines the sample sequence):
    gen:   randint(4 anchor picks); then per anchor: 3 x randint(crop origin), rand(block mask), randperm(masked subsample)
    gen_d: per anchor: randperm(delayed subsample)
Shapes: grid (X, Y, Z) = (256, 256, 32), velodyne frame of t, 0.2 m; crop (128, 128, 32); dense input [4, 61, 128, 128, 32].
Targets: hidden evidence (masked-current, <= 8,192 / anchor) and delayed targets (TRAINING-ONLY future observations, <= 4,096 / anchor).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch

from tools.delayed_semantic.train import unpack
from tools.semantic_completion.frozen_geometry import occupancy_and_trunk
from tools.semantic_field.config.schema import FeatureSpace, MaskConfig, TrainConfig
from tools.semantic_field.data.semantic_evidence import SparseSemanticEvidence, dense_semantic_channels, mask_evidence
from tools.semantic_field.losses.semantic_loss import delayed_loss, masked_reconstruction_loss
from tools.semantic_field.model.semantic_field_head import SemanticFieldHead
from tools.semantic_field.training.trainer import GRID, subsample, to_crop


@dataclass
class StepOutput:
    loss: torch.Tensor          # scalar, requires grad
    row: Dict[str, float]       # the original per-step log row without "step"
    draws: List[str]            # the original draw-digest records of this step


def b4_step(model: SemanticFieldHead, net: torch.nn.Module, anchors: Sequence[Dict], text: torch.Tensor, mu: torch.Tensor,
            gen: torch.Generator, gen_d: torch.Generator, cfg: TrainConfig = TrainConfig(), mask_cfg: MaskConfig = MaskConfig(),
            space: FeatureSpace = FeatureSpace()) -> StepOutput:
    dev = mu.device
    picks = torch.randint(len(anchors), (cfg.anchors_per_step,), generator=gen).tolist()
    batch = [anchors[i] for i in picks]
    x = torch.stack([torch.stack([unpack(a["S_packed"]).reshape(GRID), torch.zeros(GRID, dtype=torch.bool, device=dev),
                                  unpack(a["F_packed"]).reshape(GRID)]).float() for a in batch])
    occ, trunk = occupancy_and_trunk(net, x)
    dense, hid_local, hid_feat, del_local, del_prob, draws = [], [], [], [], [], []
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
        draws.append(f"{a['drive'][17:21]}:{a['t']}:{origin}:{len(vloc)}:{len(hloc)};")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden_field = model.hidden_field(torch.stack(dense))
    flat_hidden = hidden_field.float().permute(0, 2, 3, 4, 1).reshape(-1, hidden_field.shape[1])
    hi, hf = torch.cat(hid_local), torch.cat(hid_feat); di, dp = torch.cat(del_local), torch.cat(del_prob)
    row, loss = {"n_masked": len(hi), "n_delayed": len(di)}, torch.zeros((), device=dev)
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
    row["loss"] = float(loss)
    return StepOutput(loss, row, draws)


__all__ = ["StepOutput", "b4_step"]
