# extracted from tools/gate8/train.py -- imports rewritten, logic unchanged.
# see core/README.md; the original is still in place and is the audited copy.
#!/usr/bin/env python
"""Train the completion module on cached (causal input, privileged target) samples.

Checkpoint selection uses **geometry** validation loss and the teacher-KL only -- never a
semantic ground-truth label, and never a held-out KITTI-360 number.

    python tools/gate8/train.py --config configs/gate8/completion.yaml --device cuda:1
"""
from __future__ import annotations
import argparse, glob, json, os, random, sys, time
import numpy as np, torch, yaml
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.datasets.config import REPO_ROOT, write_json                              # noqa: E402
from core.model import losses as L                        # noqa: E402
from core.supervision import targets as TG                        # noqa: E402
from core.datasets import union_vocab as V8                        # noqa: E402
from core.model.net import CompletionUNet, apply_residual, save_checkpoint            # noqa: E402
from core.run._gate8_build_samples import SAMPLE_ROOT                                # noqa: E402

ART = os.path.join(REPO_ROOT, "artifacts", "gate8")


class Samples:
    def __init__(self, sources, crop, device, seed=0, limit=None, stride=1):
        self.files = []
        for s in sources:
            fs = sorted(glob.glob(f"{SAMPLE_ROOT}/{s}/*.npz"))[::stride]
            self.files += fs
        if limit:
            self.files = self.files[:limit]
        self.crop = tuple(crop); self.device = device
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.files)

    def load(self, i):
        with np.load(self.files[i]) as z:
            return TG.unpack_sample(z, self.device)

    def verify(self) -> int:
        """Drop unreadable sample files before training rather than dying mid-run."""
        import zipfile
        ok, bad = [], []
        for f in self.files:
            try:
                with zipfile.ZipFile(f) as z:
                    z.getinfo("dims.npy")
                ok.append(f)
            except Exception:
                bad.append(f)
        self.files = ok
        if bad:
            print(f"  WARNING: skipping {len(bad)} unreadable sample files "
                  f"(first: {os.path.basename(bad[0])})", flush=True)
        return len(bad)

    def crop_of(self, d, centre_on_occ_p=0.7):
        X, Y, Z = d["gt_occ"].shape
        cx, cy, cz = self.crop
        if self.rng.random() < centre_on_occ_p:
            cand = (d["gt_occ"] > 0) & d["gt_valid"] & ~d["observed"]
            idx = cand.nonzero()
            if idx.numel() == 0:
                idx = (d["gt_occ"] > 0).nonzero()
            if idx.numel():
                c = idx[self.rng.randrange(idx.shape[0])].tolist()
                x0 = int(np.clip(c[0] - cx // 2, 0, X - cx)); y0 = int(np.clip(c[1] - cy // 2, 0, Y - cy))
                z0 = int(np.clip(c[2] - cz // 2, 0, Z - cz))
                return x0, y0, z0
        return (self.rng.randrange(X - cx + 1), self.rng.randrange(Y - cy + 1),
                self.rng.randrange(Z - cz + 1))

    def batch(self, idxs):
        out = {k: [] for k in ("input", "gt_occ", "gt_valid", "observed", "fut_p", "fut_valid",
                               "base_logodds")}
        for i in idxs:
            d = self.load(i)
            x0, y0, z0 = self.crop_of(d)
            cx, cy, cz = self.crop
            sl = (slice(x0, x0 + cx), slice(y0, y0 + cy), slice(z0, z0 + cz))
            out["input"].append(d["input"][(slice(None),) + sl])
            for k in ("gt_occ", "gt_valid", "observed", "fut_valid", "base_logodds"):
                out[k].append(d[k][sl])
            out["fut_p"].append(d["fut_p"][sl].permute(3, 0, 1, 2))
        return {k: torch.stack(v) for k, v in out.items()}


def step_loss(net, b, cfg):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        occ_res, sem_log = net(b["input"])
    occ_res = occ_res[:, 0].float(); sem_log = sem_log.float()
    final = apply_residual(b["base_logodds"], occ_res)
    w = L.balanced_weights(b["gt_occ"], b["gt_valid"], b["observed"], cfg["w_unknown"])
    n_pos = (b["gt_occ"] * b["gt_valid"].float()).sum(); n_neg = b["gt_valid"].float().sum() - n_pos
    pw = float(min(cfg["pos_weight_cap"], max(1.0, (n_neg / n_pos.clamp_min(1)).item())))
    l_focal = L.focal_bce(final, b["gt_occ"], w, cfg["focal_gamma"], pw)
    l_dice = L.soft_dice(final, b["gt_occ"], b["gt_valid"].float())
    l_sem = L.semantic_kl(sem_log, b["fut_p"], b["fut_valid"].float())
    loss = cfg["w_focal"] * l_focal + cfg["w_dice"] * l_dice + cfg["w_sem"] * l_sem
    with torch.no_grad():
        pred = (final > 0) & b["gt_valid"]; gt = (b["gt_occ"] > 0) & b["gt_valid"]
        tp = (pred & gt).sum().item(); fp = (pred & ~gt).sum().item(); fn = (~pred & gt).sum().item()
        base = (b["base_logodds"] > 0) & b["gt_valid"]
        btp = (base & gt).sum().item(); bfp = (base & ~gt).sum().item(); bfn = (~base & gt).sum().item()
    return loss, {"focal": l_focal.item(), "dice": l_dice.item(), "sem_kl": l_sem.item(),
                  "iou": tp / max(tp + fp + fn, 1), "base_iou": btp / max(btp + bfp + bfn, 1)}


def _default_device() -> str:
    """First GPU of the Gate-8 pool. ``GATE8_GPUS`` (default "1 2 3") reserves
    GPU 0 for other users; ``GATE8_DEVICE`` overrides outright."""
    d = os.environ.get("GATE8_DEVICE")
    if d:
        return d
    return f"cuda:{os.environ.get('GATE8_GPUS', '1 2 3').split()[0]}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/gate8/completion.yaml")
    ap.add_argument("--device", default=_default_device())
    ap.add_argument("--tag", default="completion")
    ap.add_argument("--overfit", type=int, default=0, help="one-batch overfit test: N steps")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(REPO_ROOT, a.config)))
    torch.manual_seed(cfg["seed"]); random.seed(cfg["seed"]); np.random.seed(cfg["seed"])
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    train = Samples(cfg["train_sources"], cfg["crop"], dev, cfg["seed"], a.limit,
                    cfg.get("train_stride", 1))
    val = Samples(cfg["val_sources"], cfg["crop"], dev, cfg["seed"] + 1, cfg.get("val_limit"),
                  cfg.get("val_stride", 1))
    n_bad = train.verify() + val.verify()
    print(f"train samples {len(train)}  val samples {len(val)}"
          + (f"  ({n_bad} unreadable files skipped)" if n_bad else ""))
    net = CompletionUNet(width=cfg["width"]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = a.steps or (a.overfit or cfg["steps"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    os.makedirs(os.path.join(ART, "checkpoints"), exist_ok=True)
    log, best, t0 = [], (float("inf"), -1), time.time()
    bs = cfg["batch_size"]
    fixed = None
    for it in range(1, steps + 1):
        if a.overfit:
            if fixed is None:
                fixed = train.batch(list(range(min(bs, len(train)))))
            b = fixed
        else:
            b = train.batch([train.rng.randrange(len(train)) for _ in range(bs)])
        net.train()
        loss, m = step_loss(net, b, cfg)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
        opt.step(); sched.step()
        m.update({"step": it, "loss": loss.item(), "lr": sched.get_last_lr()[0]})
        log.append(m)
        if it % cfg["log_every"] == 0 or it == steps:
            print(f"  [{it}/{steps}] loss {loss.item():.4f} focal {m['focal']:.4f} dice {m['dice']:.4f} "
                  f"kl {m['sem_kl']:.4f} iou {m['iou']:.4f} (base {m['base_iou']:.4f}) "
                  f"{(time.time()-t0)/it:.2f}s/it", flush=True)
        if (it % cfg["val_every"] == 0 or it == steps) and not a.overfit and len(val):
            net.eval(); vs = []
            with torch.no_grad():
                for j in range(0, len(val), bs):
                    vb = val.batch(list(range(j, min(j + bs, len(val)))))
                    vl, vm = step_loss(net, vb, cfg); vm["loss"] = vl.item(); vs.append(vm)
            v = {k: float(np.mean([x[k] for x in vs])) for k in vs[0]}
            v["step"] = it; log.append({"val": v})
            print(f"  VAL [{it}] loss {v['loss']:.4f} iou {v['iou']:.4f} (base {v['base_iou']:.4f}) "
                  f"kl {v['sem_kl']:.4f}", flush=True)
            if v["loss"] < best[0]:
                best = (v["loss"], it)
                save_checkpoint(net, os.path.join(ART, "checkpoints", f"{a.tag}_best.pt"),
                                {"step": it, "val": v, "config": cfg, "tag": a.tag})
        if it % cfg["ckpt_every"] == 0:
            save_checkpoint(net, os.path.join(ART, "checkpoints", f"{a.tag}_last.pt"),
                            {"step": it, "config": cfg, "tag": a.tag})
    save_checkpoint(net, os.path.join(ART, "checkpoints", f"{a.tag}_last.pt"),
                    {"step": steps, "config": cfg, "tag": a.tag})
    write_json(os.path.join(ART, f"train_{a.tag}.json"),
               {"tag": a.tag, "config": cfg, "steps": steps, "n_train": len(train),
                "n_val": len(val), "n_params": net.n_params(), "best_val_loss": best[0],
                "best_step": best[1], "seconds": time.time() - t0,
                "peak_gpu_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30,
                "log": log, "overfit": bool(a.overfit),
                "selection_rule": "lowest validation (focal+dice+teacher-KL) loss; no "
                                  "semantic ground truth, no KITTI-360"})
    print(f"done: best val loss {best[0]:.4f} at step {best[1]}, {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
