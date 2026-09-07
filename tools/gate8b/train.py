#!/usr/bin/env python
"""Train the completion module for one leave-one-dataset-out fold.

Gate 8A's frozen cell B, verbatim -- ``tools/gate8a/train.py``'s trainer, uniform crop
sampler, focal BCE + Dice, 0.5 x teacher KL, 6 000 steps, batch 4, crop 128x128x32,
seed 0, checkpoints {best, last} -- with two fold-specific data rules the brief fixes:

* the two training sources are drawn with **equal probability** (p = 1/2 per source, then
  uniform within the source), so neither source's sample count sets the mixture;
* the within-cell validation set takes the same number of samples from *each* source
  validation domain (Gate 8A's took the first 120 files of the concatenated list, which
  with two domains would have been one domain only).

The held-out target of the fold never appears in ``train_sources`` or ``val_sources``;
the config asserts it and a test asserts it.

    python tools/gate8b/train.py --config configs/gate8b/fold_semantickitti.yaml --device cuda:1
"""
from __future__ import annotations
import argparse, glob, os, random, sys, time
import numpy as np, torch, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, default_device                               # noqa: E402
from gates.scale_gate.config import write_json                                         # noqa: E402
from gates.gate8 import sources as S                                                   # noqa: E402
from gates.gate8.net import CompletionUNet, save_checkpoint                            # noqa: E402
from gates.gate8b.sources import G8B_ROOT                                              # noqa: E402
from tools.gate8.build_samples import SAMPLE_ROOT as G8_SAMPLES                  # noqa: E402
from tools.gate8a.train import AblationSamples, step_loss                        # noqa: E402

SAMPLE_ROOTS = {"sk_train": G8_SAMPLES, "occ3d_train": G8_SAMPLES, "semantickitti": G8_SAMPLES,
                "occ3d": G8_SAMPLES, "k360_train": f"{G8B_ROOT}/samples",
                "kitti360": f"{G8B_ROOT}/samples"}


class FoldSamples(AblationSamples):
    """Per-source file lists (each source under its own cache root), balanced draws."""

    def __init__(self, sources, crop, device, seed=0, per_source_limit=None, **kw):
        # bypass the parent's single-root glob; everything else is inherited
        AblationSamples.__init__(self, [], crop, device, seed, None, 1, **kw)
        self.by_source = {}
        for s in sources:
            fs = sorted(glob.glob(f"{SAMPLE_ROOTS[s]}/{s}/*.npz"))
            if per_source_limit:
                fs = fs[:per_source_limit]
            assert fs, f"no samples for {s} under {SAMPLE_ROOTS[s]}"
            self.by_source[s] = fs
        self.sources = list(sources)
        self.files = [f for s in self.sources for f in self.by_source[s]]
        self._offset = {}
        o = 0
        for s in self.sources:
            self._offset[s] = o; o += len(self.by_source[s])

    def verify(self) -> int:
        n_bad = super().verify()
        keep = set(self.files)
        self.by_source = {s: [f for f in fs if f in keep] for s, fs in self.by_source.items()}
        o = 0
        for s in self.sources:
            self._offset[s] = o; o += len(self.by_source[s])
        self.files = [f for s in self.sources for f in self.by_source[s]]
        return n_bad

    def draw(self) -> int:
        """Index of one sample: source chosen uniformly, then a file uniformly within."""
        s = self.sources[self.rng.randrange(len(self.sources))]
        return self._offset[s] + self.rng.randrange(len(self.by_source[s]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(REPO_ROOT, a.config)))
    tag = f"fold_{cfg['target']}"
    tgt = cfg["target"]
    for s in cfg["train_sources"] + cfg["val_sources"]:
        assert S.DATASET_OF[s] != tgt, f"target {tgt} leaks into the fold through {s}"
    assert cfg["sampler"] == "uniform" and cfg["occ_loss"] == "focal_dice"
    torch.manual_seed(cfg["seed"]); random.seed(cfg["seed"]); np.random.seed(cfg["seed"])
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    mk = lambda srcs, seed, lim: FoldSamples(srcs, cfg["crop"], dev, seed, lim,
                                             sampler=cfg["sampler"],
                                             centre_on_occ_p=cfg["centre_on_occ_p"])
    train = mk(cfg["train_sources"], cfg["seed"], None)
    val = mk(cfg["val_sources"], cfg["seed"] + 1, cfg["val_limit_per_source"])
    n_bad = train.verify() + val.verify()
    print(f"[{tag}] train " + ", ".join(f"{s}:{len(train.by_source[s])}" for s in train.sources)
          + "  val " + ", ".join(f"{s}:{len(val.by_source[s])}" for s in val.sources)
          + (f"  ({n_bad} bad)" if n_bad else ""), flush=True)
    net = CompletionUNet(width=cfg["width"]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = a.steps or cfg["steps"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    os.makedirs(os.path.join(ART, "checkpoints"), exist_ok=True)
    log, best, t0 = [], (float("inf"), -1), time.time()
    bs = cfg["batch_size"]
    drawn = {s: 0 for s in train.sources}
    meta = {"config": cfg, "config_path": a.config, "tag": tag, "target": tgt,
            "sampler": cfg["sampler"], "occ_loss": cfg["occ_loss"]}
    for it in range(1, steps + 1):
        idx = [train.draw() for _ in range(bs)]
        for i in idx:
            for s in train.sources:
                if train._offset[s] <= i < train._offset[s] + len(train.by_source[s]):
                    drawn[s] += 1
        b = train.batch(idx)
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
                  f"dens {m['density']:.4f} {(time.time()-t0)/it:.2f}s/it", flush=True)
        if it % cfg["val_every"] == 0 or it == steps:
            net.eval(); vs = []
            with torch.no_grad():
                for j in range(0, len(val), bs):
                    vb = val.batch(list(range(j, min(j + bs, len(val)))))
                    vl, vm = step_loss(net, vb, cfg); vm["loss"] = vl.item(); vs.append(vm)
            with np.errstate(invalid="ignore"):
                v = {k: float(np.nanmean([x[k] for x in vs]))
                     for k in vs[0] if not all(np.isnan(x[k]) for x in vs)}
            v["step"] = it; log.append({"val": v})
            print(f"  VAL [{it}] loss {v['loss']:.4f} iou {v['iou']:.4f} "
                  f"(base {v['base_iou']:.4f}) dens {v['density']:.4f}", flush=True)
            if v["loss"] < best[0]:
                best = (v["loss"], it)
                save_checkpoint(net, os.path.join(ART, "checkpoints", f"{tag}_best.pt"),
                                dict(meta, step=it, val=v))
    save_checkpoint(net, os.path.join(ART, "checkpoints", f"{tag}_last.pt"), dict(meta, step=steps))
    write_json(os.path.join(ART, f"train_{tag}.json"),
               {"tag": tag, "target": tgt, "config": cfg, "config_path": a.config, "steps": steps,
                "sampler": cfg["sampler"], "occ_loss": cfg["occ_loss"],
                "n_train": {s: len(train.by_source[s]) for s in train.sources},
                "n_val": {s: len(val.by_source[s]) for s in val.sources},
                "samples_drawn": drawn, "n_params": net.n_params(),
                "best_val_loss": best[0], "best_step": best[1], "seconds": time.time() - t0,
                "peak_gpu_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30, "log": log,
                "selection_rule": "within-fold: lowest validation loss (focal+dice+teacher KL) "
                                  "on the two source-validation domains. Across {best,last}: "
                                  "macro AP on source validation (tools/gate8b/select_fold.py). "
                                  "No semantic GT; the target is never read."})
    print(f"[{tag}] done: best val loss {best[0]:.4f} @ {best[1]}, {(time.time()-t0)/60:.1f} min",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
