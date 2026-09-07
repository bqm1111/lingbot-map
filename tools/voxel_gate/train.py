#!/usr/bin/env python
"""Gate 3 step 3 — train the minimal visible-voxel corrector.

Source sequences only: 00-07 for gradients and normalisation statistics, 09-10 for early
stopping and for choosing the output probability threshold. Sequence 08 is never loaded
by this tool; it asserts so at start-up.

    python tools/voxel_gate/train.py
"""
from __future__ import annotations

import argparse, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, collect_provenance, load_config, write_json
from gates.voxel_gate.data import compute_norm, sample, scores, select_clips
from gates.voxel_gate.losses import compute_loss
from gates.voxel_gate.models import VoxelCorrector3D, apply_region, count_params


def evaluate(model, cfg, clips, radius, norm, dev, thresholds):
    """Mean IoU on a split for every candidate threshold."""
    acc = {t: [] for t in thresholds}
    model.eval()
    with torch.no_grad():
        for c in clips:
            s = sample(cfg, c["clip_id"], radius, norm, dev)
            p = torch.sigmoid(model(s["x"][None])[0, 0])
            for t in thresholds:
                pred = apply_region(p >= t, s["occupied"], s["region"])
                acc[t].append(scores(pred, s["gt"], s["keep"])["iou"])
    model.train()
    return {t: float(np.mean(v)) for t, v in acc.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/voxel_gate/visible_correction.yaml")
    ap.add_argument("--run", default="voxel_cnn3d")
    a = ap.parse_args()
    cfg = load_config(a.config)
    dcfg = load_config(cfg.data.depth_gate_config)
    scfg = load_config(dcfg.data.scale_gate_config)   # provenance needs the frozen geometry cfg
    torch.manual_seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)
    dev = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    sel_path = os.path.join(REPO_ROOT, cfg.experiment.output_dir, "region_selection.json")
    radius = int(json.load(open(sel_path))["selected_radius"])

    train = select_clips(cfg, "train", cfg.data.source_train_sequences)
    sel = select_clips(cfg, "train", cfg.data.source_select_sequences)
    assert not (set(cfg.data.source_train_sequences) & set(cfg.data.val_sequences))
    assert not (set(cfg.data.source_select_sequences) & set(cfg.data.val_sequences))
    for c in train + sel:
        assert c["sequence"] not in cfg.data.val_sequences, "sequence 08 reached training"
    train_ids = {c["clip_id"] for c in train} | {c["clip_id"] for c in sel}
    val_ids = {c["clip_id"] for c in select_clips(cfg, "val", cfg.data.val_sequences)}
    assert not (train_ids & val_ids), "train/val clip-id overlap"
    print(f"train {len(train)} clips (seq {cfg.data.source_train_sequences}), "
          f"select {len(sel)} clips (seq {cfg.data.source_select_sequences}), radius {radius}")

    norm = compute_norm(cfg, [c["clip_id"] for c in train])
    print("normalisation (source-train only):",
          {k: round(v, 4) for k, v in norm.items() if k != "n_clips"})

    # class balance inside the supervision region, source-train only
    pw = cfg.loss.pos_weight
    if pw is None:
        pos = neg = 0.0
        for c in train[::4]:
            s = sample(cfg, c["clip_id"], radius, None, dev)
            m = s["region"]
            pos += float((s["y"].bool() & m).sum()); neg += float((~s["y"].bool() & m).sum())
        pw = neg / max(pos, 1.0)
        print(f"resolved pos_weight = {pw:.3f}  (positives {pos/(pos+neg):.4f} of region)")
    pw = float(pw)

    model = VoxelCorrector3D(6, int(cfg.model.channels), int(cfg.model.n_blocks),
                             int(cfg.model.kernel)).to(dev)
    n_par = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.train.lr),
                            weight_decay=float(cfg.train.weight_decay))
    amp = bool(cfg.train.amp) and dev.type == "cuda"
    thresholds = [float(t) for t in cfg.train.threshold_grid]
    run_dir = os.path.join(REPO_ROOT, cfg.experiment.output_dir, "runs", a.run)
    os.makedirs(run_dir, exist_ok=True)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    print(f"model {n_par} params, pos_weight {pw:.2f}, amp {amp}")
    best, best_ep, hist, rng = -1.0, -1, [], np.random.default_rng(cfg.experiment.seed)
    t_start = time.time()
    for ep in range(int(cfg.train.epochs)):
        order = rng.permutation(len(train))
        tot, n, t0 = 0.0, 0, time.time()
        for j in order:
            s = sample(cfg, train[int(j)]["clip_id"], radius, norm, dev)
            m = s["region"][None, None].float()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                z = model(s["x"][None])
            out = compute_loss(z.float(), s["y"][None, None], m, pw,
                               float(cfg.loss.dice_weight))
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(out["loss"]); n += 1
        ious = evaluate(model, cfg, sel, radius, norm, dev, thresholds)
        t_best = max(ious, key=ious.get)
        hist.append({"epoch": ep, "train_loss": tot / n, "select_iou": ious[t_best],
                     "select_threshold": t_best, "epoch_s": time.time() - t0,
                     "select_iou_by_threshold": ious})
        star = ""
        if ious[t_best] > best:
            best, best_ep, star = ious[t_best], ep, "  *"
            torch.save({"state_dict": model.state_dict(), "norm": norm, "radius": radius,
                        "pos_weight": pw, "threshold": t_best, "epoch": ep,
                        "channels": int(cfg.model.channels),
                        "n_blocks": int(cfg.model.n_blocks),
                        "kernel": int(cfg.model.kernel), "n_params": n_par,
                        "select_iou": ious[t_best]},
                       os.path.join(run_dir, "best.pt"))
        print(f"ep {ep:2d}  loss {tot/n:.4f}  select IoU {ious[t_best]:.4f} "
              f"@tau {t_best:.2f}  {time.time()-t0:.0f}s{star}", flush=True)
        if ep - best_ep >= int(cfg.train.patience):
            print(f"early stop (patience {cfg.train.patience})")
            break

    peak = (torch.cuda.max_memory_allocated(dev) / 2**30) if dev.type == "cuda" else 0.0
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu", weights_only=False)
    write_json(os.path.join(run_dir, "train.json"), {
        "provenance": collect_provenance(scfg, "voxel_gate.train"),
        "run": a.run, "n_params": n_par, "radius": radius, "pos_weight": pw,
        "norm": norm, "best_epoch": ck["epoch"], "selected_threshold": ck["threshold"],
        "select_iou": ck["select_iou"], "peak_gpu_gib": peak,
        "train_seconds": time.time() - t_start,
        "source_train_sequences": list(cfg.data.source_train_sequences),
        "source_select_sequences": list(cfg.data.source_select_sequences),
        "n_train_clips": len(train), "n_select_clips": len(sel),
        "sequence_08_used_in_training": False, "history": hist})
    print(f"\nbest epoch {ck['epoch']}  select IoU {ck['select_iou']:.4f}  "
          f"tau {ck['threshold']:.2f}  peak {peak:.2f} GiB  "
          f"{time.time()-t_start:.0f}s total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
