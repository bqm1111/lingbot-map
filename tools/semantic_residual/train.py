"""Train the B5 residual head: cosine to the future teacher feature, nothing else.

One step = one anchor, up to ``queries_per_step`` of its supervised query voxels, cosine loss against the detached
future teacher feature. No class head, no KL, no contrastive term, no teacher or vocabulary tuning.

Resumable: the checkpoint carries the model, optimizer, scheduler, generator state and log, so resuming at step k
continues the same anchor and query sequence. Writes are atomic.

    PYTHONPATH=$PWD python -m tools.semantic_residual.train --seed 0 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from typing import Dict

import numpy as np
import torch

from tools.semantic_residual import config as B5
from tools.semantic_residual.data import AnchorKey, anchors_of, load_anchor, subsample
from tools.semantic_residual.model import ResidualHead


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Both are unit vectors; the target is detached teacher output and never receives gradient."""
    return (1.0 - (pred * target.detach()).sum(-1)).mean()


def resume_path(seed: int) -> str:
    return os.path.join(B5.CHECKPOINT_ROOT, f"b5_seed{seed}_resume.pt")


def _atomic_save(obj: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"; torch.save(obj, tmp); os.replace(tmp, path)


def train(seed: int, device: str, cfg: B5.TrainConfig = B5.TrainConfig(), stop_after: int | None = None) -> Dict:
    dev = torch.device(device); torch.cuda.set_device(dev); torch.cuda.reset_peak_memory_stats(dev)
    anchors = anchors_of(B5.TRAIN_DRIVES)
    if not anchors:
        raise RuntimeError("no training anchors: build the targets and the query cache first")

    torch.manual_seed(seed)
    model = ResidualHead().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps)
    gen = torch.Generator().manual_seed(seed)
    step, log, seconds = 0, [], 0.0

    rp = resume_path(seed)
    if os.path.exists(rp):
        ck = torch.load(rp, map_location="cpu", weights_only=False)
        if ck["steps"] != cfg.steps or ck["seed"] != seed:
            raise RuntimeError(f"{rp} belongs to steps={ck['steps']} seed={ck['seed']}, not {cfg.steps}/{seed}")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
        gen.set_state(ck["gen"]); step, log, seconds = ck["step"], ck["log"], ck["seconds"]
        print(f"[b5 seed{seed}] resumed at step {step}/{cfg.steps}", flush=True)

    last = min(cfg.steps, stop_after or cfg.steps)
    t0 = time.time(); seconds0 = seconds
    while step < last:
        step += 1
        a = anchors[int(torch.randint(len(anchors), (1,), generator=gen))]
        ctx, target, extras = load_anchor(a, dev)
        if len(target) == 0:
            continue
        ctx, target, extras = subsample(ctx, target, extras, cfg.queries_per_step, gen)
        pred = model.corrected_feature(ctx)
        loss = cosine_loss(pred, target)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); opt.step(); sched.step()

        with torch.no_grad():                       # B0's own cosine on the same voxels: the number to beat
            b0_cos = (ctx["b0_feature"] * target).sum(-1).mean()
        row = {"step": step, "loss": float(loss), "b0_cos": float(b0_cos),
               "b5_cos": float(1.0 - loss), "n_queries": int(len(target)), "anchor": str(a)}
        log.append(row); seconds = seconds0 + time.time() - t0
        if step % cfg.log_every == 0 or step == last:
            w = log[-cfg.log_every:]
            print(f"  [b5 seed{seed} {step}/{cfg.steps}] cos B5 {np.mean([r['b5_cos'] for r in w]):.4f} "
                  f"vs B0 {np.mean([r['b0_cos'] for r in w]):.4f}  {seconds / 60:.1f} min", flush=True)
        if step % cfg.ckpt_every == 0 or step == last:
            _atomic_save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                          "gen": gen.get_state(), "step": step, "log": log, "seconds": seconds,
                          "steps": cfg.steps, "seed": seed}, rp)

    rec = {"version": B5.VERSION, "seed": seed, "steps": step, "seconds": seconds,
           "n_train_anchors": len(anchors), "config": asdict(cfg), "model": model.config(),
           "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(dev)),
           "final_cos_b5": float(np.mean([r["b5_cos"] for r in log[-200:]])),
           "final_cos_b0": float(np.mean([r["b0_cos"] for r in log[-200:]]))}
    if step == cfg.steps:
        fp = B5.checkpoint_path(seed, cfg.steps)
        _atomic_save({"state_dict": model.state_dict(), "config": model.config(), "seed": seed, "step": step}, fp)
        rec["final_checkpoint"] = fp
    os.makedirs(B5.ARTIFACT_DIR, exist_ok=True)
    json.dump({**rec, "log": log}, open(os.path.join(B5.ARTIFACT_DIR, f"train_seed{seed}.json"), "w"), indent=1)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--stop-after", type=int)
    a = ap.parse_args()
    print(json.dumps(train(a.seed, a.device, stop_after=a.stop_after), indent=1))


if __name__ == "__main__":
    main()
