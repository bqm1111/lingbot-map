"""Train one (scale, regime, seed) of the Gate 19 learning curve.

The loop is the frozen Gate 12-H / Gate 13-M recipe, executed through the same imported pieces the frozen baseline
uses (`load_recipe`, `seed_everything`, `HInputDataset`, `step_loss`, `CompletionUNet`). **Only the training pool and
the step count change.** Nothing about architecture, optimiser, losses, crop, sampling or validation is touched.

Because the RNG use mirrors the frozen trainer exactly, running `D_small` under `fixed_steps` must reproduce the
frozen run's ``draw_digest``. `validate_small.py` checks that, and the whole curve should be distrusted if it fails.

    PYTHONPATH=$PWD python -m tools.data_scaling.train --scale D_small --regime fixed_steps --seed 0 --device cuda:0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np
import torch

from core.datasets.audit import FileAudit
from core.model.net import CompletionUNet, save_checkpoint
from core.run._gate8a_train import step_loss
from tools.baseline_reconstruction import config as BR
from tools.baseline_reconstruction.current.dataset import HInputDataset, anchor_ids
from tools.baseline_reconstruction.current.train import (TRAIN_FORBIDDEN, load_recipe, seed_everything, sha256_file,
                                                         state_digest)
from tools.data_scaling import config as G19
from tools.data_scaling.subsets import load_pool

#: fixed_compute is the control that makes fixed_epochs interpretable: every scale gets the SAME (large) number of
#: updates D_all needs for one fixed-epoch run, so a difference between scales is data, not optimisation budget.
REGIMES = ("fixed_steps", "fixed_epochs", "fixed_compute")
FIXED_COMPUTE_STEPS = 40056


def steps_of(scale: str, regime: str, n_anchors: int) -> int:
    if regime == "fixed_steps":
        return G19.FIXED_STEPS
    if regime == "fixed_compute":
        return FIXED_COMPUTE_STEPS
    return G19.steps_for_fixed_epochs(n_anchors, G19.SCALES["D_small"])


def train(scale: str, regime: str, seed: int, device: str, paths: BR.CurrentPaths = BR.CurrentPaths()) -> Dict:
    assert regime in REGIMES, regime
    pool = load_pool(scale)
    n_anchors = sum(len(a) for _, a in pool)
    steps = steps_of(scale, regime, n_anchors)
    tag = G19.run_tag(scale, regime, seed)
    cfg = load_recipe(paths.training_config)

    dev = torch.device(device); torch.cuda.set_device(dev)
    seed_everything(seed)
    net = CompletionUNet(cin=3, width=cfg["width"], padding_mode=cfg["padding_mode"]).to(dev)
    init_digest = state_digest(net)
    ckpt_dir = os.path.join(G19.checkpoint_dir(), tag); os.makedirs(ckpt_dir, exist_ok=True)
    runs_dir = os.path.join(G19.ARTIFACT_DIR, "runs"); os.makedirs(runs_dir, exist_ok=True)
    out_json = os.path.join(runs_dir, f"train_{tag}.json")
    if os.path.exists(out_json):
        print(f"[{tag}] already done", flush=True)
        return json.load(open(out_json))

    t0 = time.time(); log: List[Dict] = []; ckpts: List[Dict] = []
    draw_digest, target_digest = hashlib.sha256(), hashlib.sha256()
    with FileAudit(patterns=TRAIN_FORBIDDEN, strict=True) as audit:
        train_ds = HInputDataset(pool, paths, dev, seed)
        val_ds = HInputDataset([(G19.VAL_DRIVE, anchor_ids(G19.VAL_DRIVE, paths, cfg["val_every_nth_anchor"]))],
                               paths, dev, seed + 1)
        print(f"[{tag}] train {len(train_ds)} anchors, val {len(val_ds)}, steps {steps}, "
              f"epochs {G19.epochs_of(len(train_ds), steps):.2f}", flush=True)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        bs, crop = cfg["batch_size"], tuple(cfg["crop"])
        for it in range(1, steps + 1):
            idx = [train_ds.rng.randrange(len(train_ds)) for _ in range(bs)]
            b = train_ds.batch(idx, crop, draw_digest, target_digest); net.train()
            loss, m = step_loss(net, b, cfg)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"]); opt.step(); sched.step()
            m.update({"step": it, "loss": loss.item(), "lr": sched.get_last_lr()[0]}); log.append(m)
            if it % cfg["log_every"] == 0 or it == steps:
                print(f"  [{tag} {it}/{steps}] loss {m['loss']:.4f} iou {m['iou']:.4f} "
                      f"{(time.time()-t0)/it:.3f}s/it", flush=True)
            if it % cfg["val_every"] == 0 or it == steps:
                net.eval(); vs = []
                with torch.no_grad():
                    for j in range(0, len(val_ds), bs):
                        vb = val_ds.batch(list(range(j, min(j + bs, len(val_ds)))), crop)
                        vl, vm = step_loss(net, vb, cfg); vm["loss"] = vl.item(); vs.append(vm)
                v = {k: float(np.nanmean([x[k] for x in vs])) for k in vs[0]}; v["step"] = it
                log.append({"val": v})
                print(f"  VAL [{tag} {it}] loss {v['loss']:.4f} iou {v['iou']:.4f}", flush=True)
            if it == steps:
                p = os.path.join(ckpt_dir, f"{tag}_step{it:05d}.pt")
                save_checkpoint(net, p, {"tag": tag, "step": it, "cond": "H", "seed": seed,
                                         "scale": scale, "regime": regime, "n_anchors": len(train_ds),
                                         "experiment": G19.VERSION})
                ckpts.append({"step": it, "path": p, "draw_digest": draw_digest.hexdigest()})
    final = ckpts[-1]["path"]
    rec = {"version": G19.VERSION, "tag": tag, "scale": scale, "regime": regime, "seed": seed,
           "n_train": len(train_ds), "n_val": len(val_ds), "steps": steps,
           "epochs": G19.epochs_of(len(train_ds), steps), "n_params": net.n_params(),
           "config_sha256": sha256_file(paths.training_config), "init_state_digest": init_digest,
           "draw_digest": draw_digest.hexdigest(), "target_digest": target_digest.hexdigest(),
           "final_checkpoint": final, "final_checkpoint_sha256": sha256_file(final),
           "seconds": time.time() - t0, "gpu_hours": (time.time() - t0) / 3600.0,
           "firewall": {"opened": len(audit.opened), "violations": audit.violations}, "log": log}
    json.dump(rec, open(out_json, "w"), indent=1)
    print(f"[{tag}] done {rec['seconds']/60:.1f} min", flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", required=True, choices=list(G19.SCALES))
    ap.add_argument("--regime", required=True, choices=REGIMES)
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    r = train(a.scale, a.regime, a.seed, a.device)
    print(json.dumps({k: v for k, v in r.items() if k != "log"}, indent=1))


if __name__ == "__main__":
    main()
