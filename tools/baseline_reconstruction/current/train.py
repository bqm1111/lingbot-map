"""Train one seed of the current geometry-only baseline with the frozen Gate 12/13 recipe.

The recipe is read from ``training_config.yaml`` (a verbatim copy of Gate 12's
``training_config_H.yaml``) and executed through the unchanged
``core.run._gate8a_train.step_loss``: CompletionUNet(cin=3, width 24, replicate padding),
focal + Dice, AdamW, cosine schedule, grad clip, bf16 autocast, uniform crops, final-step
checkpoint. RNG use mirrors ``tools/gate12_suf/train12.py`` exactly (seeded init, one
``random.Random(seed)`` for anchor draws and crop origins), so the run is a re-execution
of Gate 12 condition H / Gate 13 condition M, not a variation of it.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

from core.datasets import kitti360_splits as SRC
from core.datasets.audit import FileAudit
from core.model.net import CompletionUNet, save_checkpoint
from core.run._gate8a_train import step_loss
from tools.baseline_reconstruction import config as C
from tools.baseline_reconstruction.current.dataset import HInputDataset, anchor_ids

#: the training run may not open LiDAR, GT poses, semantics or any target dataset
TRAIN_FORBIDDEN = tuple(SRC.FORBIDDEN_PATH_TOKENS) + ("velodyne", "data_3d_raw", "data_poses", "trident", "semantics")


def sha256_file(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def state_digest(net: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in net.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def load_recipe(path: str) -> Dict:
    cfg = yaml.safe_load(open(path))
    assert cfg["input"] == "H" and cfg["w_sem"] == 0.0 and cfg["padding_mode"] == "replicate", "not the geometry-only H recipe"
    assert cfg["checkpoint_rule"].startswith("final step"), "checkpoint rule must be final-step"
    return cfg


@dataclass
class TrainRecord:
    seed: int
    steps: int
    n_train: int
    n_val: int
    n_params: int
    config_path: str
    config_sha256: str
    init_state_digest: str
    init_matches_gate12: Optional[bool]
    draw_digest: str
    target_digest: str
    final_checkpoint: str
    final_checkpoint_sha256: str
    checkpoints: List[Dict] = field(default_factory=list)
    log: List[Dict] = field(default_factory=list)
    seconds: float = 0.0
    firewall: Dict = field(default_factory=dict)


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def train_one_seed(seed: int, device: str, paths: C.CurrentPaths = C.CurrentPaths(), steps: Optional[int] = None,
                   tag_prefix: str = "geo") -> TrainRecord:
    cfg = load_recipe(paths.training_config)
    steps = steps or int(cfg["steps"]); tag = f"{tag_prefix}_seed{seed}"
    dev = torch.device(device); torch.cuda.set_device(dev)
    seed_everything(seed)
    net = CompletionUNet(cin=3, width=cfg["width"], padding_mode=cfg["padding_mode"]).to(dev)
    init_digest = state_digest(net)
    gate12_init = os.path.join(C.REPO_ROOT, "artifacts", "gate12_suf", "checkpoints", f"init_seed{seed}.pt")
    init_matches = None
    if os.path.exists(gate12_init):
        ref = CompletionUNet(cin=3, width=cfg["width"], padding_mode=cfg["padding_mode"])
        ref.load_state_dict(torch.load(gate12_init, map_location="cpu", weights_only=False)["state_dict"])
        init_matches = state_digest(ref) == init_digest
    os.makedirs(paths.checkpoints, exist_ok=True); os.makedirs(paths.runs, exist_ok=True)
    t0 = time.time(); log: List[Dict] = []; ckpts: List[Dict] = []
    draw_digest, target_digest = hashlib.sha256(), hashlib.sha256()
    with FileAudit(patterns=TRAIN_FORBIDDEN, strict=True) as audit:
        train = HInputDataset([(d, anchor_ids(d, paths)) for d in C.TRAIN_DRIVES], paths, dev, seed)
        val = HInputDataset([(C.VAL_DRIVE, anchor_ids(C.VAL_DRIVE, paths, cfg["val_every_nth_anchor"]))], paths, dev, seed + 1)
        print(f"[{tag}] train {len(train)} anchors, val {len(val)}, steps {steps}, params {net.n_params():,}, "
              f"init {init_digest[:12]} (matches Gate 12 init: {init_matches})", flush=True)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        bs, crop = cfg["batch_size"], tuple(cfg["crop"])
        for it in range(1, steps + 1):
            idx = [train.rng.randrange(len(train)) for _ in range(bs)]
            b = train.batch(idx, crop, draw_digest, target_digest); net.train()
            loss, m = step_loss(net, b, cfg)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"]); opt.step(); sched.step()
            m.update({"step": it, "loss": loss.item(), "lr": sched.get_last_lr()[0]}); log.append(m)
            if it % cfg["log_every"] == 0 or it == steps:
                print(f"  [{tag} {it}/{steps}] loss {m['loss']:.4f} focal {m['focal']:.4f} dice {m['dice']:.4f} "
                      f"iou {m['iou']:.4f} {(time.time()-t0)/it:.3f}s/it", flush=True)
            if it % cfg["val_every"] == 0 or it == steps:
                net.eval(); vs = []
                with torch.no_grad():
                    for j in range(0, len(val), bs):
                        vb = val.batch(list(range(j, min(j + bs, len(val)))), crop)
                        vl, vm = step_loss(net, vb, cfg); vm["loss"] = vl.item(); vs.append(vm)
                v = {k: float(np.nanmean([x[k] for x in vs])) for k in vs[0]}; v["step"] = it; log.append({"val": v})
                print(f"  VAL [{tag} {it}] loss {v['loss']:.4f} iou {v['iou']:.4f}", flush=True)
            if it % cfg["ckpt_every"] == 0 or it == steps:
                p = paths.checkpoint_path(seed, it) if tag_prefix == "geo" else os.path.join(paths.checkpoints, "smoke", f"{tag}_step{it:05d}.pt")
                os.makedirs(os.path.dirname(p), exist_ok=True)
                save_checkpoint(net, p, {"tag": tag, "step": it, "cond": "H", "seed": seed, "experiment": C.PROTOCOL_VERSION})
                ckpts.append({"step": it, "path": os.path.relpath(p, C.REPO_ROOT), "draw_digest": draw_digest.hexdigest()})
    final = os.path.join(C.REPO_ROOT, ckpts[-1]["path"])
    rec = TrainRecord(seed=seed, steps=steps, n_train=len(train), n_val=len(val), n_params=net.n_params(),
                      config_path=os.path.relpath(paths.training_config, C.REPO_ROOT), config_sha256=sha256_file(paths.training_config),
                      init_state_digest=init_digest, init_matches_gate12=init_matches, draw_digest=draw_digest.hexdigest(),
                      target_digest=target_digest.hexdigest(), final_checkpoint=os.path.relpath(final, C.REPO_ROOT),
                      final_checkpoint_sha256=sha256_file(final), checkpoints=ckpts, log=log, seconds=time.time() - t0,
                      firewall={"opened": len(audit.opened), "violations": audit.violations})
    json.dump(asdict(rec), open(os.path.join(paths.runs, f"train_{tag}.json"), "w"), indent=1)
    print(f"[{tag}] done {rec.seconds/60:.1f} min; final {rec.final_checkpoint_sha256[:12]}", flush=True)
    return rec


__all__ = ["TRAIN_FORBIDDEN", "sha256_file", "state_digest", "load_recipe", "TrainRecord", "train_one_seed"]
