#!/usr/bin/env python
"""Gate 3.1 step 2 — train a corrector on the clean, target-independent region.

Every methodological choice is frozen from Gate 3 (radius 3, three 3x3x3 convs at width
16, GroupNorm+GELU, zero-initialised 1x1x1 head, +/-4 prior logits, weighted BCE + 0.5
soft Dice, AdamW 1e-3/1e-4, batch 1, bf16, <=20 epochs, patience 4, tau = 0.45). The only
change is that the inference region no longer touches the SemanticKITTI valid mask.

    python tools/voxel_gate_validation/train_clean.py --run full_s0  --geometry c3 --seed 0
    python tools/voxel_gate_validation/train_clean.py --run occ_only --geometry c3 --seed 0 --occ-only
    python tools/voxel_gate_validation/train_clean.py --run c0_corrector --geometry c0 --seed 0

Sequence 08 is never loaded; asserted at start-up and recorded in ``train.json``.
"""
from __future__ import annotations

import argparse, json, os, random, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, collect_provenance, load_config, write_json
from gates.voxel_gate.models import VoxelCorrector3D, apply_region, count_params
from gates.voxel_gate.losses import compute_loss
from gates.voxel_gate_validation.data import compute_norm, sample, scores, select_clips


def seed_everything(seed: int) -> None:
    """All stochastic sources: python, numpy, torch CPU and every CUDA device."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, cfg, geom, clips, radius, norm, dev, tau, occ_only):
    model.eval()
    out = []
    with torch.no_grad():
        for c in clips:
            s = sample(cfg, geom, c["clip_id"], radius, norm, dev, occ_only)
            p = torch.sigmoid(model(s["x"][None])[0, 0])
            pred = apply_region(p >= tau, s["occupied"], s["R_infer"])
            out.append(scores(pred, s["gt"], s["keep"])["iou"])
    model.train()
    return float(np.mean(out))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/voxel_gate_validation/clean_infill.yaml")
    ap.add_argument("--run", required=True)
    ap.add_argument("--geometry", default="c3", choices=["c0", "c3"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--occ-only", action="store_true",
                    help="ablation A_occ_only: zero channels 1-4, keep the tensor shape")
    a = ap.parse_args()

    cfg = load_config(a.config)
    dcfg = load_config(cfg.data.depth_gate_config)
    scfg = load_config(dcfg.data.scale_gate_config)
    seed_everything(a.seed)
    dev = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    radius = int(cfg.region.radius)
    tau = float(cfg.train.threshold)
    assert cfg.region.intersect_with_valid_mask is False, "clean protocol requires this off"

    train = select_clips(cfg, a.geometry, "train", cfg.data.source_train_sequences)
    sel = select_clips(cfg, a.geometry, "train", cfg.data.source_select_sequences)
    val_ids = {c["clip_id"] for c in select_clips(cfg, a.geometry, "val",
                                                  cfg.data.val_sequences)}
    for c in train + sel:
        assert c["sequence"] not in cfg.data.val_sequences, "sequence 08 reached training"
    assert not ({c["clip_id"] for c in train + sel} & val_ids), "clip-id overlap"
    print(f"[{a.run}] geometry {a.geometry}  seed {a.seed}  occ_only {a.occ_only}  "
          f"radius {radius}  tau {tau}\n  train {len(train)}  select {len(sel)}")

    norm = compute_norm(cfg, a.geometry, [c["clip_id"] for c in train])

    pw = cfg.loss.pos_weight
    if pw is None:                       # class balance inside S_train, source-train only
        pos = neg = 0.0
        for c in train[::4]:
            s = sample(cfg, a.geometry, c["clip_id"], radius, None, dev)
            m = s["S_train"]
            pos += float((s["y"].bool() & m).sum()); neg += float((~s["y"].bool() & m).sum())
        pw = neg / max(pos, 1.0)
    pw = float(pw)

    model = VoxelCorrector3D(6, int(cfg.model.channels), int(cfg.model.n_blocks),
                             int(cfg.model.kernel)).to(dev)
    n_par = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.train.lr),
                            weight_decay=float(cfg.train.weight_decay))
    amp = bool(cfg.train.amp) and dev.type == "cuda"
    run_dir = os.path.join(REPO_ROOT, cfg.experiment.output_dir, "runs", a.run)
    os.makedirs(run_dir, exist_ok=True)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    print(f"  {n_par} params, pos_weight {pw:.3f}")

    best, best_ep, hist = -1.0, -1, []
    rng = np.random.default_rng(a.seed)
    t_start = time.time()
    for ep in range(int(cfg.train.epochs)):
        order = rng.permutation(len(train))
        tot, n, t0 = 0.0, 0, time.time()
        for j in order:
            s = sample(cfg, a.geometry, train[int(j)]["clip_id"], radius, norm, dev,
                       a.occ_only)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                z = model(s["x"][None])
            # loss lives on S_train = R_infer AND valid; the valid mask restricts the
            # *loss* only, never the input, the region or the output gate.
            out = compute_loss(z.float(), s["y"][None, None],
                               s["S_train"][None, None].float(), pw,
                               float(cfg.loss.dice_weight))
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(out["loss"]); n += 1
        iou = evaluate(model, cfg, a.geometry, sel, radius, norm, dev, tau, a.occ_only)
        hist.append({"epoch": ep, "train_loss": tot / n, "select_iou": iou,
                     "epoch_s": time.time() - t0})
        star = ""
        if iou > best:
            best, best_ep, star = iou, ep, "  *"
            torch.save({"state_dict": model.state_dict(), "norm": norm, "radius": radius,
                        "pos_weight": pw, "threshold": tau, "epoch": ep, "seed": a.seed,
                        "geometry": a.geometry, "occ_only": bool(a.occ_only),
                        "channels": int(cfg.model.channels),
                        "n_blocks": int(cfg.model.n_blocks), "kernel": int(cfg.model.kernel),
                        "n_params": n_par, "select_iou": iou},
                       os.path.join(run_dir, "best.pt"))
        print(f"  ep {ep:2d}  loss {tot/n:.4f}  select IoU {iou:.4f}  "
              f"{time.time()-t0:.0f}s{star}", flush=True)
        if ep - best_ep >= int(cfg.train.patience):
            print(f"  early stop (patience {cfg.train.patience})")
            break

    peak = (torch.cuda.max_memory_allocated(dev) / 2**30) if dev.type == "cuda" else 0.0
    write_json(os.path.join(run_dir, "train.json"), {
        "provenance": collect_provenance(scfg, "voxel_gate_validation.train"),
        "run": a.run, "geometry": a.geometry, "seed": a.seed,
        "occ_only": bool(a.occ_only), "n_params": n_par, "radius": radius,
        "threshold": tau, "pos_weight": pw, "norm": norm,
        "best_epoch": best_ep, "select_iou": best, "peak_gpu_gib": peak,
        "train_seconds": time.time() - t_start, "n_epochs_run": len(hist),
        "n_train_clips": len(train), "n_select_clips": len(sel),
        "source_train_sequences": list(cfg.data.source_train_sequences),
        "source_select_sequences": list(cfg.data.source_select_sequences),
        "sequence_08_used_in_training": False,
        "region_intersects_valid_mask": False, "history": hist})
    print(f"[{a.run}] best epoch {best_ep}  select IoU {best:.4f}  peak {peak:.2f} GiB  "
          f"{time.time()-t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
