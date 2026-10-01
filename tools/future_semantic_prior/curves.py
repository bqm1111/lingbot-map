"""Training curves: E0/E1/E2 on a fixed 0006 subsample, swept over the saved checkpoints.

This runs AFTER training, never inside it. Measuring the dev drive between optimiser steps would mean the
training process opening 0006's ground truth, and the training firewall is stronger if it can say that
process never touched the dev drive at all (``config.TRAIN_FORBIDDEN`` contains ``drive_0006``). Sweeping
the checkpoints afterwards measures exactly the same quantity.

The curves are diagnostic only. No checkpoint is ever selected from them: the reported model is the final
step, as pre-registered.

    PYTHONPATH=$PWD python -m tools.future_semantic_prior.curves --control C3_full --seed 0 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from tools.delayed_semantic import config as C
from tools.future_semantic_prior import config as G, evaluate as EV
from tools.future_semantic_prior.model import build_head
from tools.semantic_completion.evaluation import semantic_metrics as SM
from tools.semantic_completion.evidence import trident as TR
from tools.semantic_completion.frozen_geometry import load_frozen_geometry


def checkpoint_steps(control: str, seed: int) -> List[int]:
    root = os.path.join(G.DATA_ROOT, "checkpoints")
    pre = f"{control}_seed{seed}_step"
    return sorted(int(f[len(pre):-3]) for f in os.listdir(root) if f.startswith(pre) and f.endswith(".pt"))


@torch.no_grad()
def sweep(control: str, seed: int, device: str, stride: Optional[int] = None,
          paths: C.Paths = C.Paths()) -> Dict:
    dev = torch.device(device)
    torch.cuda.set_device(dev)
    cfg = G.TrainConfig()
    stride = stride or cfg.curve_stride
    net = load_frozen_geometry(G.GEOMETRY_SEED, dev, paths.g15)
    text, _, _ = TR.load_text_embeddings(paths.g15.text_embeddings)
    text = text.to(dev)
    head = build_head(control).to(dev)
    rows: List[Dict] = []
    for step in checkpoint_steps(control, seed):
        ck = torch.load(G.checkpoint_path(control, seed, step), map_location=dev, weights_only=False)
        head.load_state_dict(ck["state_dict"])
        r = EV.evaluate(control, seed, device, step=step, stride=stride, with_ssc=False, head=head,
                        net=net, text=text, paths=paths, quiet=True)
        row = {"step": step}
        for p in ("E0", "E1", "E2"):
            s = SM.summarize(r["arrays"][f"conf__model__{p}"].sum(0), semantic_only=True)
            b = SM.summarize(r["arrays"][f"conf__B0__{p}"].sum(0), semantic_only=True)
            row[p] = {"miou": round(100.0 * s["miou"], 4), "top1": round(100.0 * s["overall_accuracy"], 4),
                      "b0_miou": round(100.0 * b["miou"], 4), "b0_top1": round(100.0 * b["overall_accuracy"], 4),
                      "delta_miou": round(100.0 * (s["miou"] - b["miou"]), 4)}
        rows.append(row)
        print(f"  [curve {control} s{seed} step {step}] E1 {row['E1']['miou']:.2f} "
              f"E2 {row['E2']['miou']:.2f} (B0 {row['E2']['b0_miou']:.2f}) "
              f"dE2 {row['E2']['delta_miou']:+.2f}", flush=True)
    # the training loss, read back from the run record so the curve carries both axes
    rec = json.load(open(os.path.join(G.seed_dir(seed), f"train_{control}.json")))
    loss_at = {r["step"]: r["loss"] for r in rec["log"]}
    tgt_at = {r["step"]: r["target_top1"] for r in rec["log"]}
    for row in rows:
        row["train_loss"] = loss_at.get(row["step"])
        row["train_target_top1"] = tgt_at.get(row["step"])
    out = {"control": control, "seed": seed, "stride": stride, "n_dev_anchors": len(EV.dev_anchor_ids(stride)),
           "note": "diagnostic only; no checkpoint is selected from these", "rows": rows}
    os.makedirs(os.path.join(G.ARTIFACT_DIR, "training_curves"), exist_ok=True)
    json.dump(out, open(os.path.join(G.ARTIFACT_DIR, "training_curves", f"{control}_seed{seed}.json"), "w"), indent=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--stride", type=int)
    a = ap.parse_args()
    r = sweep(a.control, a.seed, a.device, a.stride)
    print(json.dumps({k: v for k, v in r.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
