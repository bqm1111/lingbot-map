#!/usr/bin/env python
"""Gate 2 — train a per-pixel log-depth residual head on frozen LingBot depth.

    python tools/depth_gate/train.py --config configs/depth_gate/refine.yaml --arch rgbd_unet

LingBot is never loaded here: training reads the frozen cache, so no gradient can reach
it. Early stopping uses sequences 09-10; sequence 08 is untouched by training, model
selection, thresholds and normalisation statistics.
"""
from __future__ import annotations

import argparse, json, os, sys, time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, collect_provenance, load_config, set_seed, write_json
from gates.depth_gate.data import DepthRefineDataset, collate, compute_stats
from gates.depth_gate.losses import compute_loss
from gates.depth_gate.metrics import depth_metrics
from gates.depth_gate.models import build_inputs, build_model, refine


def evaluate(model, loader, cfg, stats, dev, use_rgb):
    model.eval()
    P, G, V = [], [], []
    with torch.no_grad():
        for b in loader:
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            r = model(build_inputs(b, stats, use_rgb))
            P.append(refine(b["base_depth"], r).cpu())
            G.append(b["projected_lidar_depth"].cpu())
            V.append(b["projected_lidar_valid_mask"].cpu())
    return depth_metrics(torch.cat(P), torch.cat(G), torch.cat(V))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/depth_gate/refine.yaml")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--confidence-weighted", action="store_true")
    ap.add_argument("--primary", default=None, choices=["log_smooth_l1", "relative"])
    ap.add_argument("--overfit", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.arch:
        cfg["model"]["arch"] = a.arch
        cfg["model"]["use_rgb"] = a.arch == "rgbd_unet"
    if a.confidence_weighted:
        cfg["loss"]["confidence_weighted"] = True
    if a.primary:
        cfg["loss"]["primary"] = a.primary
    if a.epochs:
        cfg["train"]["epochs"] = a.epochs
    scfg = load_config(cfg.data.scale_gate_config)
    set_seed(cfg.experiment.seed)
    dev = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    run = a.run_name or (cfg.model.arch + ("_cw" if cfg.loss.confidence_weighted else "")
                         + ("_rel" if cfg.loss.primary == "relative" else ""))
    out = os.path.join(REPO_ROOT, cfg.experiment.output_dir, "runs", run)
    os.makedirs(out, exist_ok=True)

    s_hat = float(cfg.scale.constant)
    tr = DepthRefineDataset(scfg, cfg, cfg.data.train_sequences, "train", s_hat)
    se = DepthRefineDataset(scfg, cfg, cfg.data.select_sequences, "train", s_hat)
    assert set(tr.sequences).isdisjoint(se.sequences), "train/select sequence overlap"
    assert "08" not in set(tr.sequences) | set(se.sequences), "sequence 08 leaked into training"
    stats = compute_stats(tr, seed=cfg.experiment.seed)      # training frames only
    print(f"{run}: {len(tr)} train frames ({tr.sequences}), {len(se)} select ({se.sequences})")

    if a.overfit:
        tr.frames = tr.frames[: a.overfit]
        se = tr
        print(f"OVERFIT MODE on {len(tr)} frames")
    bs = cfg.train.batch_size
    
    dl_tr = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=cfg.train.num_workers,
                       collate_fn=collate, drop_last=False, persistent_workers=False)
    dl_se = DataLoader(se, batch_size=bs, shuffle=False, num_workers=cfg.train.num_workers,
                       collate_fn=collate)

    in_ch = 8 if cfg.model.use_rgb else 5
    model = build_model(cfg.model.arch, in_ch, cfg.model.base_channels,
                        cfg.model.max_log_residual).to(dev)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  arch={cfg.model.arch} in_ch={in_ch} params={n_par:,} "
          f"loss={cfg.loss.primary} conf_weighted={cfg.loss.confidence_weighted}")
    if cfg.model.arch == "identity":
        m = evaluate(model, dl_se, cfg, stats, dev, cfg.model.use_rgb)
        write_json(os.path.join(out, "train.json"),
                   {"provenance": collect_provenance(scfg, "depth_gate.train"), "run": run,
                    "arch": "identity", "n_params": 0, "stats": stats, "select_metrics": m})
        print(f"  identity select AbsRel {m['abs_rel']:.4f}")
        return 0

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    epochs = 2 if a.smoke else cfg.train.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.train.amp))
    best, best_ep, bad, log = float("inf"), -1, 0, []
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats(dev)

    for ep in range(epochs):
        model.train(); tot = n = 0.0
        for b in dl_tr:
            b = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in b.items()}
            with torch.amp.autocast("cuda", enabled=bool(cfg.train.amp)):
                r = model(build_inputs(b, stats, cfg.model.use_rgb))
                terms = compute_loss(cfg, refine(b["base_depth"], r), r, b)
            opt.zero_grad(set_to_none=True)
            scaler.scale(terms["loss"]).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot += float(terms["loss"]) * len(b["base_depth"]); n += len(b["base_depth"])
        sched.step()
        m = evaluate(model, dl_se, cfg, stats, dev, cfg.model.use_rgb)
        log.append({"epoch": ep, "train_loss": tot / n, **m})
        if m["abs_rel"] < best - 1e-6:
            best, best_ep, bad = m["abs_rel"], ep, 0
            torch.save({"state_dict": model.state_dict(), "stats": stats,
                        "arch": cfg.model.arch, "in_ch": in_ch,
                        "base_channels": cfg.model.base_channels,
                        "max_log_residual": cfg.model.max_log_residual,
                        "use_rgb": cfg.model.use_rgb, "metric_scale": s_hat},
                       os.path.join(out, "best.pt"))
        else:
            bad += 1
            if bad >= cfg.train.patience and not (a.smoke or a.overfit):
                print(f"  early stop @ {ep} (best AbsRel {best:.4f} @ {best_ep})")
                break
        print(f"  ep {ep:2d} loss {tot/n:.5f}  select AbsRel {m['abs_rel']:.4f} "
              f"d1 {m['delta1']:.4f}", flush=True)

    torch.save({"state_dict": model.state_dict(), "stats": stats, "arch": cfg.model.arch,
                "in_ch": in_ch, "base_channels": cfg.model.base_channels,
                "max_log_residual": cfg.model.max_log_residual,
                "use_rgb": cfg.model.use_rgb, "metric_scale": s_hat},
               os.path.join(out, "last.pt"))
    peak = torch.cuda.max_memory_allocated(dev) / 2 ** 30
    if a.overfit:
        # Judge on AbsRel against the identity baseline, not on total loss: the loss
        # carries a residual regulariser that necessarily grows as the head learns a
        # real correction, so it cannot approach zero even on a perfectly fitted set.
        ident = build_model("identity", in_ch).to(dev)
        m0 = evaluate(ident, dl_se, cfg, stats, dev, cfg.model.use_rgb)
        # "Substantially reduce training error" (B6 step 3). The floor is not zero:
        # the residual is tanh-bounded and LiDAR supervision is sparse and does not
        # always correspond to what LingBot's depth represents at that pixel. Exact
        # residual recovery is proven separately in tests/depth_gate.
        ok = best < 0.75 * m0["abs_rel"]
        print(f"  overfit: AbsRel identity {m0['abs_rel']:.4f} -> refined {best:.4f} "
              f"({'PASS' if ok else 'FAIL'});  depth-term loss "
              f"{log[0]['train_loss']:.5f} -> {log[-1]['train_loss']:.5f}")
    write_json(os.path.join(out, "train.json"),
               {"provenance": collect_provenance(scfg, "depth_gate.train"), "run": run,
                "config": dict(cfg), "arch": cfg.model.arch, "n_params": n_par,
                "in_channels": in_ch, "stats": stats,
                "n_train_frames": len(tr), "n_select_frames": len(se),
                "train_sequences": tr.sequences, "select_sequences": se.sequences,
                "best_epoch": best_ep, "best_select_abs_rel": best,
                "seconds": time.time() - t0, "peak_gpu_gib": peak, "curve": log})
    print(f"  {run}: best select AbsRel {best:.4f} @ {best_ep} "
          f"({time.time()-t0:.0f}s, peak {peak:.2f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
