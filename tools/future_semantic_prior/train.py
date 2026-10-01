"""Train one (control, seed) on U^R: hidden voxels that a later frame happens to re-observe.

Per step, ``anchors_per_step`` anchors are drawn uniformly from 0003/0007/0010. For each one the frozen
causal state is computed, the head's channels are assembled, ONE forward runs over the whole grid, and the
soft distillation loss is gathered at that anchor's U^R rows. There is no crop: the head sees at training
time exactly the tensor it sees at evaluation time, so a result cannot be an artefact of a crop boundary.

Only the head moves. The geometry network is loaded frozen with ``requires_grad_(False)``, its state digest
is recorded before and after, and the run record asserts no gradient ever reached it.

**Where the future is allowed to be.** The target enters at exactly one place: ``soft_distillation`` on the
rows of ``U^R``. To make that a runtime guarantee rather than a reading of the code, each step digests the
head's input tensor before the target file is opened and again after the backward pass, and requires the
digest to be unchanged; the causal state is digested the same way. The input builder itself runs under a
file audit that forbids the Gate 21 target cache, Gate 20's cache, human labels, LiDAR and GT poses.

    PYTHONPATH=$PWD python -m tools.future_semantic_prior.train --control C3_full --seed 0 --device cuda:0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from core.datasets.audit import FileAudit
from tools.delayed_semantic import config as C
from tools.future_semantic_prior import config as G, features as FE, targets as TG
from tools.future_semantic_prior.model import build_head, soft_distillation
from tools.semantic_completion.evidence import trident as TR
from tools.semantic_completion.frozen_geometry import load_frozen_geometry, sha256_file, state_digest


def train_anchors() -> List[Tuple[str, int]]:
    """Every training anchor whose U^R is non-empty. An anchor with no revealed voxel carries no loss."""
    out: List[Tuple[str, int]] = []
    for d in G.TRAIN_DRIVES:
        root = os.path.join(G.DATA_ROOT, "targets", d)
        for f in sorted(os.listdir(root)):
            if not f.endswith(".npz"):
                continue
            with np.load(os.path.join(root, f)) as z:
                if len(z["flat_index"]):
                    out.append((d, int(f[:-4])))
    return out


def train(control: str, seed: int, device: str, steps: Optional[int] = None, paths: C.Paths = C.Paths(),
          cfg: G.TrainConfig = G.TrainConfig()) -> Dict:
    dev = torch.device(device)
    torch.cuda.set_device(dev)
    steps = steps or cfg.steps
    G.ensure_dirs()
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed * 100003 + 21)

    net = load_frozen_geometry(G.GEOMETRY_SEED, dev, paths.g15)
    geo_before = state_digest(net)
    assert not any(p.requires_grad for p in net.parameters()), "the frozen geometry network wants gradients"
    text, _, _ = TR.load_text_embeddings(paths.g15.text_embeddings)
    text = text.to(dev)
    head = build_head(control).to(dev)
    needs_input = head.cin > 0
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    items = train_anchors()
    draws = hashlib.sha256()
    log: List[Dict] = []
    ckpts: List[Dict] = []
    q_sum = torch.zeros(G.N_CLASSES, dtype=torch.float64, device=dev)
    q_n = 0
    t0 = time.time()
    with FileAudit(patterns=G.TRAIN_FORBIDDEN, strict=True) as audit:
        print(f"[{control} s{seed}] {len(items)} training anchors, {head.n_params():,} params, "
              f"{head.cin} input channels, {steps} steps", flush=True)
        for it in range(1, steps + 1):
            head.train()
            picks = [items[rng.randrange(len(items))] for _ in range(cfg.anchors_per_step)]
            states, xs = [], []
            for d, t in picks:
                if needs_input:
                    st = FE.frozen_causal_state(d, t, net, text, dev, paths)
                    states.append(st)
                    xs.append(FE.build_input(st, head, dev))
                else:
                    states.append(None)
                    xs.append(torch.zeros(1, 0, *[int(x) for x in G.GRID_DIMS], device=dev))
            dx = [FE.fingerprint(x) for x in xs] if needs_input else []
            ds = [FE.state_fingerprint(s) for s in states] if needs_input else []

            # ---- the future enters only below this line, and only as a target ----------------------
            tgts = [TG.load_targets(d, t, dev) for d, t in picks]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = head(torch.cat(xs))
            lg = torch.cat([out[b].reshape(G.N_CLASSES, -1).T[tg.flat_index] for b, tg in enumerate(tgts)])
            q = torch.cat([tg.q_future for tg in tgts])
            loss = soft_distillation(lg, q)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            # ---- and it changed nothing the model is allowed to see --------------------------------
            if needs_input:
                assert [FE.fingerprint(x) for x in xs] == dx, "the head input changed after the target was opened"
                assert [FE.state_fingerprint(s) for s in states] == ds, "the causal state changed after the target was opened"
            assert all(p.grad is None for p in net.parameters()), "a gradient reached the frozen geometry network"

            with torch.no_grad():
                q_sum += q.double().sum(0)
                q_n += len(q)
                hard = q.argmax(1)
                top1 = float((lg.float().argmax(1) == hard).float().mean()) if len(q) else float("nan")
            row = {"step": it, "loss": float(loss), "target_top1": round(top1, 5), "n_voxels": int(len(q)),
                   "grad_norm": float(gn), "lr": sched.get_last_lr()[0]}
            log.append(row)
            draws.update("".join(f"{d[17:21]}:{t};" for d, t in picks).encode())
            if it % cfg.log_every == 0 or it == steps:
                print(f"  [{control} s{seed} {it}/{steps}] loss {row['loss']:.4f} tgt_top1 {top1:.3f} "
                      f"n {row['n_voxels']} {(time.time()-t0)/it:.2f}s/it", flush=True)
            if it % cfg.ckpt_every == 0 or it == steps:
                p = G.checkpoint_path(control, seed, it)
                torch.save({"state_dict": head.state_dict(), "control": control, "seed": seed, "step": it,
                            "head_config": asdict(head.cfg), "describe": head.describe(),
                            "geometry_checkpoint": os.path.relpath(paths.g15.geometry_checkpoint(G.GEOMETRY_SEED),
                                                                   C.REPO_ROOT)}, p)
                ckpts.append({"step": it, "path": p, "sha256": sha256_file(p)})

    geo_after = state_digest(net)
    rec = {"version": G.VERSION, "control": control, "seed": seed, "steps": steps,
           "train_config": asdict(cfg), "head": head.describe(), "n_train_anchors": len(items),
           "draw_digest": draws.hexdigest(), "geometry_digest_before": geo_before,
           "geometry_digest_after": geo_after, "geometry_unchanged": geo_before == geo_after,
           "geometry_grads_none": True, "mean_target_distribution": (q_sum / max(q_n, 1)).cpu().tolist(),
           "n_target_voxels_seen": q_n, "checkpoints": ckpts, "log": log,
           "seconds": round(time.time() - t0, 1),
           "firewall": {"patterns": list(G.TRAIN_FORBIDDEN), "opened": len(audit.opened),
                        "violations": audit.violations}}
    out = os.path.join(G.seed_dir(seed), f"train_{control}.json")
    json.dump(rec, open(out, "w"), indent=1)
    print(f"[{control} s{seed}] done {rec['seconds']/60:.1f} min; geometry unchanged {rec['geometry_unchanged']}",
          flush=True)
    return {k: rec[k] for k in ("control", "seed", "steps", "seconds", "geometry_unchanged", "draw_digest")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--steps", type=int)
    a = ap.parse_args()
    print(json.dumps(train(a.control, a.seed, a.device, a.steps), indent=1))


if __name__ == "__main__":
    main()
