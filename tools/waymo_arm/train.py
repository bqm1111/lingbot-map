"""Train the frozen recipe on the Waymo arm.

``tools/baseline_reconstruction/current/train.py`` is hash-frozen
(``artifacts/baseline_reconstruction/code_freeze.json``, enforced by
``freeze.require_source_freeze``), so it cannot be given a second scene list without
invalidating the KITTI-360 arm's own verification. This module therefore repeats its loop
with the scene list as an argument and **imports every substantive piece unchanged**: the
recipe yaml, ``CompletionUNet``, ``core.run._gate8a_train.step_loss``, the optimiser and
schedule construction order, ``HInputDataset`` and ``anchor_ids``, the RNG discipline and the
checkpoint rule. The loop body below is a transcription of that file's, with the drive
constants replaced by ``train_scenes`` / ``val_scenes``.

``cli/check_trainer_equivalence.py`` is the gate: run this loop and the frozen one on the
SAME KITTI-360 drives with the same seed and step budget, and every logged loss, every
metric and the final parameter digest must agree exactly. Without that gate a difference
between the two arms could be the trainer rather than the data.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from core.datasets.audit import FileAudit
from core.model.net import CompletionUNet, save_checkpoint
from core.run._gate8a_train import step_loss
from tools.baseline_reconstruction import config as BR
from tools.baseline_reconstruction.current.dataset import HInputDataset, anchor_ids
from tools.baseline_reconstruction.current.train import (TRAIN_FORBIDDEN, TrainRecord,
                                                         load_recipe, seed_everything,
                                                         sha256_file, state_digest)
from tools.waymo_arm import config as A


@dataclass(frozen=True)
class WaymoArmPaths:
    """The paths the frozen dataset code and this loop actually touch, pointed at this arm.

    ``BR.CurrentPaths`` is a frozen dataclass, so this is a standalone stand-in rather than a
    subclass. ``HInputDataset`` and ``anchor_ids`` touch only ``input_path`` and
    ``target_path``; the loop additionally reads ``training_config`` -- which stays the
    KITTI-360 arm's recipe file, byte for byte -- plus ``checkpoints`` and ``runs``.
    """
    inputs: str = field(default_factory=lambda: A.Paths().inputs)
    targets: str = field(default_factory=lambda: A.Paths().targets)
    checkpoints: str = field(default_factory=lambda: A.Paths().checkpoints)
    runs: str = os.path.join(A.ARTIFACT_DIR, "runs")
    #: the recipe is the frozen one; this arm changes the data, not the recipe
    training_config: str = field(default_factory=lambda: BR.CurrentPaths().training_config)
    #: deliberately the KITTI-360 arm's checkpoint NAME, in this arm's own directory, so the
    #: frozen evaluation harness can be pointed at it with
    #: ``dataclasses.replace(CurrentPaths(), checkpoints=...)`` and nothing else.
    checkpoint_prefix: str = "geo"

    def input_path(self, scene: str, t: int) -> str:
        return os.path.join(self.inputs, scene, f"{t:05d}.npz")

    def target_path(self, scene: str, t: int) -> str:
        return os.path.join(self.targets, scene, f"{t:05d}.npz")

    def checkpoint_path(self, seed: int, step: int) -> str:
        return os.path.join(self.checkpoints,
                            f"{self.checkpoint_prefix}_seed{seed}_step{step:05d}.pt")


@dataclass(frozen=True)
class K360ArmPaths(WaymoArmPaths):
    """The KITTI-360 arm's caches behind the same interface, for the trainer gate only."""
    inputs: str = field(default_factory=lambda: BR.CurrentPaths().inputs_root)
    targets: str = field(default_factory=lambda: BR.CurrentPaths().targets_root)
    checkpoint_prefix: str = "geo"


def scenes_with_data(paths, scenes: Sequence[str]) -> List[Tuple[str, List[int]]]:
    """``(scene, anchors)`` for scenes that have at least one input WITH a target."""
    out = []
    for s in scenes:
        d = os.path.dirname(paths.input_path(s, 0))
        if not os.path.isdir(d):
            continue
        ids = anchor_ids(s, paths)
        if ids:
            out.append((s, ids))
    return out


def train_one_seed(seed: int, device: str, train_scenes: Sequence[str],
                   val_scenes: Sequence[str], paths=None, steps: Optional[int] = None,
                   tag_prefix: str = "waymo_geo", val_every_nth: Optional[int] = None,
                   record_dir: Optional[str] = None) -> TrainRecord:
    paths = paths or WaymoArmPaths()
    cfg = load_recipe(paths.training_config)
    steps = steps or int(cfg["steps"])
    tag = f"{tag_prefix}_seed{seed}"
    dev = torch.device(device)
    torch.cuda.set_device(dev)
    seed_everything(seed)
    net = CompletionUNet(cin=3, width=cfg["width"], padding_mode=cfg["padding_mode"]).to(dev)
    init_digest = state_digest(net)
    os.makedirs(paths.checkpoints, exist_ok=True)
    os.makedirs(paths.runs, exist_ok=True)
    t0, log, ckpts = time.time(), [], []
    draw_digest, target_digest = hashlib.sha256(), hashlib.sha256()
    nth = int(val_every_nth if val_every_nth is not None else cfg["val_every_nth_anchor"])
    with FileAudit(patterns=TRAIN_FORBIDDEN, strict=True) as audit:
        train = HInputDataset(scenes_with_data(paths, train_scenes), paths, dev, seed)
        val_ids = [(s, anchor_ids(s, paths, nth)) for s in val_scenes
                   if os.path.isdir(os.path.dirname(paths.input_path(s, 0)))]
        val = HInputDataset([(s, i) for s, i in val_ids if i], paths, dev, seed + 1)
        print(f"[{tag}] train {len(train)} anchors / {len(train_scenes)} scenes, "
              f"val {len(val)} / {len(val_scenes)}, steps {steps}, params {net.n_params():,}, "
              f"init {init_digest[:12]}", flush=True)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        bs, crop = cfg["batch_size"], tuple(cfg["crop"])
        for it in range(1, steps + 1):
            idx = [train.rng.randrange(len(train)) for _ in range(bs)]
            b = train.batch(idx, crop, draw_digest, target_digest)
            net.train()
            loss, m = step_loss(net, b, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
            opt.step()
            sched.step()
            m.update({"step": it, "loss": loss.item(), "lr": sched.get_last_lr()[0]})
            log.append(m)
            if it % cfg["log_every"] == 0 or it == steps:
                print(f"  [{tag} {it}/{steps}] loss {m['loss']:.4f} focal {m['focal']:.4f} "
                      f"dice {m['dice']:.4f} iou {m['iou']:.4f} "
                      f"{(time.time() - t0) / it:.3f}s/it", flush=True)
            if (it % cfg["val_every"] == 0 or it == steps) and len(val):
                net.eval(); vs = []
                with torch.no_grad():
                    for j in range(0, len(val), bs):
                        vb = val.batch(list(range(j, min(j + bs, len(val)))), crop)
                        vl, vm = step_loss(net, vb, cfg)
                        vm["loss"] = vl.item()
                        vs.append(vm)
                v = {k: float(np.nanmean([x[k] for x in vs])) for k in vs[0]}
                v["step"] = it
                log.append({"val": v})
                print(f"  VAL [{tag} {it}] loss {v['loss']:.4f} iou {v['iou']:.4f}", flush=True)
            if it % cfg["ckpt_every"] == 0 or it == steps:
                p = (paths.checkpoint_path(seed, it) if tag_prefix == "waymo_geo"
                     else os.path.join(paths.checkpoints, "smoke",
                                       f"{tag}_step{it:05d}.pt"))
                os.makedirs(os.path.dirname(p), exist_ok=True)
                save_checkpoint(net, p, {"tag": tag, "step": it, "cond": "H", "seed": seed,
                                         "experiment": A.VERSION, "arm": "waymo",
                                         "n_train_anchors": len(train),
                                         "n_train_scenes": len(train_scenes)})
                ckpts.append({"step": it, "path": os.path.relpath(p, A.REPO_ROOT),
                              "draw_digest": draw_digest.hexdigest()})
    final = os.path.join(A.REPO_ROOT, ckpts[-1]["path"])
    rec = TrainRecord(seed=seed, steps=steps, n_train=len(train), n_val=len(val),
                      n_params=net.n_params(),
                      config_path=os.path.relpath(paths.training_config, A.REPO_ROOT),
                      config_sha256=sha256_file(paths.training_config),
                      init_state_digest=init_digest, init_matches_gate12=None,
                      draw_digest=draw_digest.hexdigest(),
                      target_digest=target_digest.hexdigest(),
                      final_checkpoint=os.path.relpath(final, A.REPO_ROOT),
                      final_checkpoint_sha256=sha256_file(final), checkpoints=ckpts,
                      log=log, seconds=time.time() - t0,
                      firewall={"opened": len(audit.opened), "violations": audit.violations})
    out_dir = record_dir or paths.runs
    os.makedirs(out_dir, exist_ok=True)
    json.dump(asdict(rec), open(os.path.join(out_dir, f"train_{tag}.json"), "w"), indent=1)
    print(f"[{tag}] done {rec.seconds / 60:.1f} min; "
          f"final {rec.final_checkpoint_sha256[:12]}", flush=True)
    return rec


__all__ = ["WaymoArmPaths", "scenes_with_data", "train_one_seed"]
