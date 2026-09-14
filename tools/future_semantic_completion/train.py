#!/usr/bin/env python
"""Train the semantic-completion head (B2) on the frozen geometry, one seed.

    python tools/future_semantic_completion/train.py --seed 0 --device cuda:0

Supervision is the key set only -- voxels the causal map has not observed at ``t``, that
the future mapper observes in ``t+1..t+20``, that the raw-LiDAR target confirms occupied,
and where the future mapper lifted a teacher vector. Loss is the repository's validated
teacher distillation, ``KL(teacher || student)``, averaged over those voxels. The trunk and
both original heads are the adopted checkpoint of the same seed, frozen and run without
gradient; the occupancy logits are never touched.

Full 256x256x32 grids rather than crops: the trunk is frozen, so a full grid costs no
trunk backward, and the head then trains under exactly the inference-time boundary
conditions. Batch 2, 6 000 steps, AdamW, cosine schedule -- the adopted optimiser and step
budget. The last step is the checkpoint; nothing is selected on the validation drive.
"""
from __future__ import annotations
import argparse, os, random, sys, time
import numpy as np, torch, yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.future_semantic_completion import study as ST                          # noqa: E402
from tools.future_semantic_completion import head as H                            # noqa: E402
from core.datasets.audit import FileAudit                                         # noqa: E402
from core.datasets import kitti360_splits as SRC, union_vocab as V8               # noqa: E402
from core.model.data import DriveSamples                                          # noqa: E402

CONFIG = os.path.join(ST.REPO_ROOT, "configs", "future_semantic_completion", "base.yaml")


def gather(logits, fut_p, mask):
    """``[B,U,...]`` logits and ``[B,...,U]`` targets at the masked voxels -> ``[n,U]`` each."""
    m = mask.reshape(-1)
    lg = logits.permute(0, 2, 3, 4, 1).reshape(-1, V8.U)[m]
    tg = fut_p.reshape(-1, V8.U)[m]
    return lg, tg


def batch_of(samples, idxs):
    ds = [samples.load(i) for i in idxs]
    return {"input": torch.stack([d["input"] for d in ds]),
            "fut_p": torch.stack([d["fut_p"] for d in ds]),
            "key": torch.stack([H.key_mask(d) for d in ds]),
            "obs": torch.stack([H.observed_mask(d) for d in ds])}


@torch.no_grad()
def metrics(logits, tgt, b0):
    if logits.shape[0] == 0:
        return {"cos": float("nan"), "top1": float("nan"), "cos_b0": float("nan")}
    p = torch.softmax(logits.float(), 1)
    return {"cos": float(H.cosine(p, tgt).mean()), "top1": float((p.argmax(1) == tgt.argmax(1)).float().mean()),
            "cos_b0": float(H.cosine(p, b0.expand_as(p)).mean())}


def run(cfg, seed, device, steps=None):
    tag = ST.run_tag(seed)
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    dev = torch.device(device); torch.cuda.set_device(dev)
    train = DriveSamples(cfg["train_drives"], cfg["crop"], dev, seed, None, sampler="uniform")
    val = DriveSamples([cfg["val_drive"]], cfg["crop"], dev, seed + 1, cfg["val_limit"], sampler="uniform")
    n_bad = train.verify() + val.verify()
    geo = H.FrozenGeometry(ST.frozen_checkpoint(seed), dev)
    geo_digest = geo.state_digest()
    with np.load(os.path.join(ST.ART, "b0_mean.npz")) as z:
        b0 = torch.from_numpy(z["mean"]).to(dev)
    torch.manual_seed(seed)                        # head init draws from a known state
    head = H.SemanticCompletionHead(width=cfg["head_width"], padding_mode=cfg["padding_mode"]).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = steps or cfg["steps"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    print(f"[{tag}] train " + ", ".join(f"{d[17:21]}:{len(train.by_drive[d])}" for d in train.drives)
          + f"  val {cfg['val_drive'][17:21]}:{len(val)}" + (f"  ({n_bad} bad)" if n_bad else "")
          + f"  head params {head.n_params()}  frozen {geo.path}", flush=True)
    log, t0 = [], time.time()
    bs = cfg["batch_size"]; trace = []
    n_frozen_grad = sum(1 for p in geo.net.parameters() if p.requires_grad)
    assert n_frozen_grad == 0
    for it in range(1, steps + 1):
        idx = [train.draw() for _ in range(bs)]
        trace.append(idx)
        b = batch_of(train, idx)
        _, sem_exist, trunk = geo.forward(b["input"])
        head.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = head(trunk, b["input"])
        lg, tg = gather(logits, b["fut_p"], b["key"])
        loss = H.loss_kl(lg, tg)
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(head.parameters(), cfg["grad_clip"]))
        opt.step(); sched.step()
        m = {"step": it, "loss": float(loss), "n_key": int(lg.shape[0]), "grad_norm": gn,
             "lr": sched.get_last_lr()[0], **metrics(lg, tg, b0)}
        log.append(m)
        if it % cfg["log_every"] == 0 or it == steps:
            print(f"  [{tag} {it}/{steps}] kl {m['loss']:.4f} cos {m['cos']:.4f} top1 {m['top1']:.4f} "
                  f"cos_b0 {m['cos_b0']:.4f} n {m['n_key']} {(time.time()-t0)/it:.2f}s/it", flush=True)
        if it % cfg["val_every"] == 0 or it == steps:
            head.eval(); vs = {"loss": 0.0, "n": 0, "cos": 0.0, "top1": 0.0, "cos_b0": 0.0,
                               "existing_head_loss": 0.0, "b0_loss": 0.0,
                               "obs_loss": 0.0, "obs_n": 0}
            with torch.no_grad():
                for j in range(0, len(val), bs):
                    vb = batch_of(val, list(range(j, min(j + bs, len(val)))))
                    _, sem_exist, trunk = geo.forward(vb["input"])
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        vl = head(trunk, vb["input"])
                    lg, tg = gather(vl, vb["fut_p"], vb["key"])
                    n = lg.shape[0]
                    if n:
                        p = torch.softmax(lg.float(), 1)
                        vs["loss"] += float(H.loss_kl(lg, tg)) * n; vs["n"] += n
                        vs["cos"] += float(H.cosine(p, tg).sum())
                        vs["top1"] += float((p.argmax(1) == tg.argmax(1)).sum())
                        vs["cos_b0"] += float(H.cosine(p, b0.expand_as(p)).sum())
                        le, _ = gather(sem_exist, vb["fut_p"], vb["key"])
                        vs["existing_head_loss"] += float(H.loss_kl(le, tg)) * n
                        vs["b0_loss"] += float(H.loss_kl(torch.log(b0.clamp_min(1e-6)).expand(n, -1), tg)) * n
                    lo, to = gather(vl, vb["fut_p"], vb["obs"])
                    if lo.shape[0]:
                        vs["obs_loss"] += float(H.loss_kl(lo, to)) * lo.shape[0]; vs["obs_n"] += lo.shape[0]
            n = max(vs["n"], 1)
            v = {"step": it, "loss": vs["loss"] / n, "cos": vs["cos"] / n, "top1": vs["top1"] / n,
                 "cos_b0": vs["cos_b0"] / n, "n_key": vs["n"],
                 "existing_head_loss": vs["existing_head_loss"] / n, "b0_loss": vs["b0_loss"] / n,
                 "observed_loss": vs["obs_loss"] / max(vs["obs_n"], 1), "n_observed": vs["obs_n"]}
            log.append({"val": v})
            print(f"  VAL [{tag} {it}] kl {v['loss']:.4f} (B0 {v['b0_loss']:.4f}, existing head "
                  f"{v['existing_head_loss']:.4f}) cos {v['cos']:.4f} top1 {v['top1']:.4f} "
                  f"cos_b0 {v['cos_b0']:.4f} n {v['n_key']} | observed kl {v['observed_loss']:.4f} "
                  f"n {v['n_observed']}", flush=True)
    assert geo.state_digest() == geo_digest, "frozen geometry changed during training"
    os.makedirs(os.path.dirname(ST.head_checkpoint(seed)), exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "head_width": cfg["head_width"],
                "padding_mode": cfg["padding_mode"], "frozen_checkpoint": os.path.relpath(geo.path, ST.REPO_ROOT),
                "frozen_checkpoint_sha256": ST.sha256_file(geo.path), "seed": seed, "steps": steps,
                "config": cfg}, ST.head_checkpoint(seed))
    return {"tag": tag, "seed": seed, "config": cfg, "steps": steps,
            "frozen_checkpoint": os.path.relpath(geo.path, ST.REPO_ROOT),
            "frozen_checkpoint_sha256": ST.sha256_file(geo.path),
            "frozen_state_digest_before": geo_digest, "frozen_state_digest_after": geo.state_digest(),
            "n_frozen_params_requiring_grad": n_frozen_grad,
            "head_params": head.n_params(),
            "n_train": {d: len(train.by_drive[d]) for d in train.drives}, "n_val": len(val),
            "trace_sha256": ST.sha256_json(trace), "seconds": time.time() - t0,
            "peak_gpu_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30,
            "checkpoint": os.path.relpath(ST.head_checkpoint(seed), ST.REPO_ROOT), "log": log}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()
    ST.require_data_root()
    cfg = yaml.safe_load(open(a.config))
    SRC.assert_no_target_access(cfg["train_drives"] + [cfg["val_drive"]])
    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        out = run(cfg, a.seed, a.device, a.steps)
    out["config_path"] = os.path.relpath(a.config, ST.REPO_ROOT)
    out["checkpoint_sha256"] = ST.sha256_file(ST.head_checkpoint(a.seed))
    out["firewall"] = audit.summary()
    out["selection_rule"] = "final-step head; validation drive monitored, never selected on"
    out["env"] = ST.environment()
    ST.write_json(os.path.join(ST.ART, f"train_{out['tag']}.json"), out)
    print(f"[{out['tag']}] done in {out['seconds']/60:.1f} min, firewall violations "
          f"{len(out['firewall']['violations'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
