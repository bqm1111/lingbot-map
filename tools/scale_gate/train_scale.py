#!/usr/bin/env python
"""Train the non-oracle scale predictor (plan Phase 4).

    python tools/scale_gate/train_scale.py --config <cfg> --model combined_mlp

Models: ``global_median`` (baseline A, closed form), ``depth_mlp`` (B),
``image_mlp`` (C), ``combined_mlp`` (the proposed minimal model).

The split is **by sequence**, never by clip, so temporally overlapping clips cannot
straddle it. Targets are ``log(s*)`` from the joint oracle; unreliable clips are
excluded and counted, never silently dropped.
"""
from __future__ import annotations

import argparse, csv, json, math, os, sys, time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, collect_provenance, load_config, set_seed, write_json
from gates.scale_gate.features import build_matrix

MODELS = {"global_median": [], "depth_mlp": ["depth"], "image_mlp": ["image"],
          "combined_mlp": ["depth", "pose", "image"]}
#: Sequences held out of training to pick checkpoints without touching seq 08.
INNER_VAL_SEQUENCES = ["09", "10"]


class ScaleMLP(nn.Module):
    """Two hidden layers, normalisation and dropout; predicts one log-scale per clip."""

    def __init__(self, d_in: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_split(cfg, split):
    """Reliable clips only, with their joint-oracle targets."""
    p = os.path.join(REPO_ROOT, cfg.experiment.output_dir, f"scale_targets_{split}.csv")
    rows = [r for r in csv.DictReader(open(p))]
    good = [r for r in rows if r["reliable"] == "1" and r["log_s_joint"] not in ("", "nan")]
    return rows, good


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="combined_mlp", choices=list(MODELS))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--overfit", type=int, default=0, help="overfit N clips and exit")
    ap.add_argument("--smoke", action="store_true", help="2 epochs, no early stop")
    a = ap.parse_args()

    cfg = load_config(a.config)
    set_seed(cfg.experiment.seed)
    cache = os.path.join(REPO_ROOT, cfg.cache.root)
    out_dir = os.path.join(REPO_ROOT, cfg.experiment.output_dir, "scale_models")
    os.makedirs(out_dir, exist_ok=True)

    all_rows, good = load_split(cfg, "train")
    print(f"train clips: {len(all_rows)} total, {len(good)} reliable "
          f"({len(all_rows)-len(good)} excluded)")
    inner = [r for r in good if r["sequence"] in INNER_VAL_SEQUENCES]
    outer = [r for r in good if r["sequence"] not in INNER_VAL_SEQUENCES]
    print(f"  fit on {len(outer)} clips ({sorted({r['sequence'] for r in outer})}), "
          f"select on {len(inner)} clips ({INNER_VAL_SEQUENCES})")

    # ---- Baseline A: a single constant, closed form ---- #
    if a.model == "global_median":
        y = np.array([float(r["log_s_joint"]) for r in outer])
        log_s = float(np.median(y))
        write_json(os.path.join(out_dir, "global_median.json"),
                   {"provenance": collect_provenance(cfg, "train_scale:global_median"),
                    "model": "global_median", "log_s": log_s, "s": math.exp(log_s),
                    "n_fit_clips": len(y), "fit_sequences": sorted({r["sequence"] for r in outer})})
        print(f"global_median: s = {math.exp(log_s):.4f} from {len(y)} clips")
        return 0

    blocks = MODELS[a.model]
    Xo, ido = build_matrix(cache, [r["clip_id"] for r in outer], blocks)
    Xi, idi = build_matrix(cache, [r["clip_id"] for r in inner], blocks)
    tmap = {r["clip_id"]: float(r["log_s_joint"]) for r in good}
    yo = np.array([tmap[c] for c in ido], np.float32)
    yi = np.array([tmap[c] for c in idi], np.float32)
    if Xo.size == 0:
        raise SystemExit("no training features; is the LingBot cache built?")
    print(f"features: {Xo.shape[1]}-d from blocks {blocks}  "
          f"({len(ido)} fit / {len(idi)} select)")

    if a.overfit:
        Xo, yo, ido = Xo[: a.overfit], yo[: a.overfit], ido[: a.overfit]
        Xi, yi = Xo, yo
        print(f"OVERFIT MODE on {len(yo)} clips")

    mu, sd = Xo.mean(0, keepdims=True), Xo.std(0, keepdims=True) + 1e-6
    dev = torch.device(cfg.lingbot.device if torch.cuda.is_available() else "cpu")
    xo = torch.from_numpy((Xo - mu) / sd).float().to(dev)
    xi = torch.from_numpy((Xi - mu) / sd).float().to(dev)
    to_ = torch.from_numpy(yo).float().to(dev)
    ti = torch.from_numpy(yi).float().to(dev)

    model = ScaleMLP(Xo.shape[1], a.hidden, 0.0 if a.overfit else a.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    epochs = 2 if a.smoke else a.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    lossf = nn.SmoothL1Loss()

    best, best_ep, bad, log = float("inf"), -1, 0, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xo), device=dev)
        tot = 0.0
        for i in range(0, len(perm), 64):
            b = perm[i: i + 64]
            loss = lossf(model(xo[b]), to_[b])
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss) * len(b)
        sched.step()
        model.eval()
        with torch.no_grad():
            med_tr = float(torch.median(torch.abs(model(xo) - to_)))
            med_va = float(torch.median(torch.abs(model(xi) - ti)))
        log.append({"epoch": ep, "train_loss": tot / len(xo),
                    "median_abs_log_err_fit": med_tr, "median_abs_log_err_select": med_va})
        if med_va < best - 1e-5:
            best, best_ep, bad = med_va, ep, 0
            torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd,
                        "blocks": blocks, "d_in": Xo.shape[1], "hidden": a.hidden,
                        "model": a.model}, os.path.join(out_dir, f"{a.model}_best.pt"))
        else:
            bad += 1
            if bad >= a.patience and not a.smoke and not a.overfit:
                print(f"early stop at epoch {ep} (best {best:.4f} @ {best_ep})")
                break
        if ep % 25 == 0 or ep == epochs - 1:
            print(f"  ep {ep:3d} loss {tot/len(xo):.5f}  fit {med_tr:.4f}  select {med_va:.4f}")

    torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd, "blocks": blocks,
                "d_in": Xo.shape[1], "hidden": a.hidden, "model": a.model},
               os.path.join(out_dir, f"{a.model}_last.pt"))
    if a.overfit:
        ok = best < 0.02
        print(f"overfit median |log err| = {best:.5f} -> {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1

    write_json(os.path.join(out_dir, f"{a.model}_train.json"),
               {"provenance": collect_provenance(cfg, "train_scale"), "model": a.model,
                "blocks": blocks, "d_in": int(Xo.shape[1]),
                "n_params": sum(p.numel() for p in model.parameters()),
                "n_fit": len(ido), "n_select": len(idi),
                "fit_sequences": sorted({r["sequence"] for r in outer}),
                "select_sequences": INNER_VAL_SEQUENCES,
                "n_excluded": len(all_rows) - len(good),
                "best_epoch": best_ep, "best_median_abs_log_err": best,
                "seconds": time.time() - t0, "curve": log})
    print(f"{a.model}: best select median |log err| {best:.4f} @ epoch {best_ep} "
          f"({time.time()-t0:.0f}s, {sum(p.numel() for p in model.parameters()):,} params)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
