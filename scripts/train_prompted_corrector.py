#!/usr/bin/env python
"""Train the small causal corrector on cached LingbotMap predictions.

LingbotMap is never loaded here.  Gradients never leave the corrector.

    python scripts/train_prompted_corrector.py \
        --config configs/prompted_lingbot/corrector.yaml \
        --output-dir outputs/prompted_lingbot/corrector
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompted_lingbot.features import N_FEATURES
from prompted_lingbot.model import CausalCorrector, CorrectorConfig, rollout
from prompted_lingbot.training import (LossWeights, build_examples, collate, compute_losses,
                                       validation_metrics)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def paths_for(cache_dir, split_file, split):
    names = json.load(open(split_file))[split]
    return [os.path.join(cache_dir, f"{n}.npz") for n in names
            if os.path.isfile(os.path.join(cache_dir, f"{n}.npz"))]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg["optim"]["epochs"] = args.epochs
    if args.device is not None:
        cfg["device"] = args.device

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    d = cfg["data"]

    print("building training pool ...", flush=True)
    t0 = time.time()
    train = build_examples(paths_for(d["cache_dir"], d["split_file"], d["train_split"]),
                           n_prompt_samples=d["prompt_samples_per_sequence"],
                           base_anchor=d["base_anchor"],
                           n_depth_samples=d["depth_samples_per_frame"], seed=cfg["seed"])
    val = build_examples(paths_for(d["cache_dir"], d["split_file"], d["val_split"]),
                         n_prompt_samples=d["val_prompt_samples_per_sequence"],
                         base_anchor=d["base_anchor"],
                         n_depth_samples=d["depth_samples_per_frame"], seed=cfg["seed"] + 7717)
    print(f"  {len(train)} train / {len(val)} val examples in {time.time() - t0:.0f}s", flush=True)
    if not train or not val:
        raise SystemExit("empty train or val pool -- check the split file and cache dir")

    mcfg = CorrectorConfig(n_features=N_FEATURES, **cfg["model"])
    model = CausalCorrector(mcfg).to(device)
    print(f"corrector: {model.n_parameters:,} trainable parameters", flush=True)

    w = LossWeights(**cfg["loss"])
    o = cfg["optim"]
    opt = torch.optim.AdamW(model.parameters(), lr=o["lr"], weight_decay=o["weight_decay"])
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=o["epochs"])
             if o.get("scheduler") == "cosine" else None)

    val_batches = [collate(val[i:i + o["batch_size"]], device)
                   for i in range(0, len(val), o["batch_size"])]

    def run_val():
        model.eval()
        acc = {}
        with torch.no_grad():
            for b in val_batches:
                s, R, t, lv, inc = rollout(model, b["feats"], b["base_s"], b["base_R"], b["base_t"])
                _, parts = compute_losses(b, s, R, t, lv, inc, w)
                vm = validation_metrics(b, s, R, t)
                for k, v in {**parts, **vm}.items():
                    acc.setdefault(k, []).append(v)
        return {k: float(np.mean(v)) for k, v in acc.items()}

    baseline_val = None
    with torch.no_grad():   # the untrained model is exactly the base anchor
        baseline_val = run_val()
    print(f"base anchor on val: ATE {baseline_val['val_ate_m']:.4f} m, "
          f"log-scale err {baseline_val['val_log_scale_err']:.4f}", flush=True)

    select_on = o.get("select_on", "val_ate_m")
    best = float("inf")
    best_epoch = -1
    history = []
    rng = np.random.default_rng(cfg["seed"])

    for epoch in range(o["epochs"]):
        model.train()
        order = rng.permutation(len(train))
        tot = {}
        for i in range(0, len(order), o["batch_size"]):
            batch = collate([train[j] for j in order[i:i + o["batch_size"]]], device)
            s, R, t, lv, inc = rollout(model, batch["feats"], batch["base_s"],
                                       batch["base_R"], batch["base_t"])
            loss, parts = compute_losses(batch, s, R, t, lv, inc, w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), o["grad_clip"])
            opt.step()
            for k, v in parts.items():
                tot.setdefault(k, []).append(v)
        if sched:
            sched.step()

        vm = run_val()
        rec = {"epoch": epoch, "lr": opt.param_groups[0]["lr"],
               **{f"train_{k}": float(np.mean(v)) for k, v in tot.items()},
               **{f"val_{k}" if not k.startswith("val_") else k: v for k, v in vm.items()}}
        history.append(rec)
        score = vm[select_on]
        if score < best - 1e-6:
            best, best_epoch = score, epoch
            torch.save({"model_state": model.state_dict(), "model_config": mcfg.to_dict(),
                        "base_anchor": d["base_anchor"], "config": cfg, "epoch": epoch,
                        "val": vm, "baseline_val": baseline_val,
                        "feature_names_hash": N_FEATURES},
                       os.path.join(args.output_dir, "best.pt"))
        if epoch % 5 == 0 or epoch == o["epochs"] - 1:
            print(f"  epoch {epoch:3d} train {rec['train_loss']:.4f} | "
                  f"val ATE {vm['val_ate_m']:.4f} scale {vm['val_log_scale_err']:.4f}"
                  f"{'  *' if best_epoch == epoch else ''}", flush=True)
        if epoch - best_epoch >= o["early_stopping_patience"]:
            print(f"  early stop at epoch {epoch} (best {best_epoch}: {best:.4f})", flush=True)
            break

    json.dump({"history": history, "best_epoch": best_epoch, "best_score": best,
               "select_on": select_on, "baseline_val": baseline_val,
               "n_train": len(train), "n_val": len(val),
               "n_parameters": model.n_parameters},
              open(os.path.join(args.output_dir, "history.json"), "w"), indent=1)
    print(f"best epoch {best_epoch}: {select_on} = {best:.4f} "
          f"(base anchor {baseline_val[select_on]:.4f})")
    print(f"checkpoint: {os.path.join(args.output_dir, 'best.pt')}")


if __name__ == "__main__":
    main()
