# extracted from tools/gate8a/train.py -- imports rewritten, logic unchanged.
# see core/README.md; the original is still in place and is the audited copy.
#!/usr/bin/env python
"""Train one cell of the Gate 8A 2x2 ablation (crop sampler x occupancy loss).

Identical to ``tools/gate8/train.py`` in every respect except the two ablated factors:
the crop-origin sampler (``sampler: occ_centred | uniform``) and the occupancy objective
(``occ_loss: focal_dice | bce``). Model, data, budget, batch size, crop, seed, optimizer,
schedule and the 0.5 x future-teacher KL are unchanged, and the checkpoint kept during
training is still the lowest-validation-loss one -- geometry and teacher-KL only, no
semantic ground truth and no KITTI-360.

    python tools/gate8a/train.py --config configs/gate8a/cellD_uniform_bce.yaml \
        --tag cellD_uniform_bce --device cuda:1
"""
from __future__ import annotations
import argparse, os, random, sys, time
import numpy as np, torch, yaml
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.datasets.config import REPO_ROOT, write_json                              # noqa: E402
from core.model import losses as L                                                    # noqa: E402
from core.model.net import CompletionUNet, apply_residual, save_checkpoint            # noqa: E402
from core.model import losses_bce_ablation as L8A, sampler as SMP                                 # noqa: E402
from core.evaluation.regions import editable_native                                       # noqa: E402
from core.run._gate8_train import Samples as G8Samples, _default_device              # noqa: E402

ART = os.path.join(REPO_ROOT, "artifacts", "gate8a")


class AblationSamples(G8Samples):
    """Gate 8's sample reader with a switchable crop-origin sampler."""

    def __init__(self, *a, sampler: str = "occ_centred", centre_on_occ_p: float = 0.7, **kw):
        super().__init__(*a, **kw)
        assert sampler in SMP.SAMPLERS, sampler
        self.sampler = sampler
        self.centre_on_occ_p = float(centre_on_occ_p)

    def crop_of(self, d, centre_on_occ_p=None):
        if self.sampler == "uniform":
            # the origin is a function of the lattice shape and the RNG alone
            return SMP.uniform_origin(self.rng, d["gt_occ"].shape, self.crop)
        p = self.centre_on_occ_p if centre_on_occ_p is None else centre_on_occ_p
        return super().crop_of(d, p)


def step_loss(net, b, cfg):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        occ_res, sem_log = net(b["input"])
    occ_res = occ_res[:, 0].float(); sem_log = sem_log.float()
    final = apply_residual(b["base_logodds"], occ_res)
    l_sem = L.semantic_kl(sem_log, b["fut_p"], b["fut_valid"].float())
    parts = {"sem_kl": l_sem.item()}
    if cfg["occ_loss"] == "bce":
        edit = editable_native(b["base_logodds"], cfg["lock_logodds"])
        l_occ = L8A.bce_editable(final, b["gt_occ"], b["gt_valid"], edit)
        loss = cfg["w_bce"] * l_occ + cfg["w_sem"] * l_sem
        parts.update({"bce": l_occ.item(), "focal": float("nan"), "dice": float("nan")})
    else:
        w = L.balanced_weights(b["gt_occ"], b["gt_valid"], b["observed"], cfg["w_unknown"])
        n_pos = (b["gt_occ"] * b["gt_valid"].float()).sum()
        n_neg = b["gt_valid"].float().sum() - n_pos
        pw = float(min(cfg["pos_weight_cap"], max(1.0, (n_neg / n_pos.clamp_min(1)).item())))
        l_focal = L.focal_bce(final, b["gt_occ"], w, cfg["focal_gamma"], pw)
        l_dice = L.soft_dice(final, b["gt_occ"], b["gt_valid"].float())
        loss = cfg["w_focal"] * l_focal + cfg["w_dice"] * l_dice + cfg["w_sem"] * l_sem
        parts.update({"focal": l_focal.item(), "dice": l_dice.item(), "bce": float("nan")})
    with torch.no_grad():
        gt = (b["gt_occ"] > 0) & b["gt_valid"]
        pred = (final > 0) & b["gt_valid"]
        base = (b["base_logodds"] > 0) & b["gt_valid"]
        def iou(p):
            tp = (p & gt).sum().item(); fp = (p & ~gt).sum().item(); fn = (~p & gt).sum().item()
            return tp / max(tp + fp + fn, 1)
        parts.update({"iou": iou(pred), "base_iou": iou(base),
                      "prevalence": float(gt.sum().item() / max(b["gt_valid"].sum().item(), 1)),
                      "density": float(pred.sum().item() / max(b["gt_valid"].sum().item(), 1))})
    return loss, parts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default=_default_device())
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(REPO_ROOT, a.config)))
    torch.manual_seed(cfg["seed"]); random.seed(cfg["seed"]); np.random.seed(cfg["seed"])
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    mk = lambda srcs, seed, lim, st: AblationSamples(
        srcs, cfg["crop"], dev, seed, lim, st,
        sampler=cfg["sampler"], centre_on_occ_p=cfg["centre_on_occ_p"])
    train = mk(cfg["train_sources"], cfg["seed"], a.limit, cfg.get("train_stride", 1))
    val = mk(cfg["val_sources"], cfg["seed"] + 1, cfg.get("val_limit"), cfg.get("val_stride", 1))
    n_bad = train.verify() + val.verify()
    print(f"[{a.tag}] sampler={cfg['sampler']} loss={cfg['occ_loss']}  "
          f"train {len(train)}  val {len(val)}" + (f"  ({n_bad} bad)" if n_bad else ""), flush=True)
    net = CompletionUNet(width=cfg["width"]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = a.steps or cfg["steps"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    os.makedirs(os.path.join(ART, "checkpoints"), exist_ok=True)
    log, best, t0 = [], (float("inf"), -1), time.time()
    bs = cfg["batch_size"]
    meta = {"config": cfg, "config_path": a.config, "tag": a.tag,
            "sampler": cfg["sampler"], "occ_loss": cfg["occ_loss"]}
    for it in range(1, steps + 1):
        b = train.batch([train.rng.randrange(len(train)) for _ in range(bs)])
        net.train()
        loss, m = step_loss(net, b, cfg)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
        opt.step(); sched.step()
        m.update({"step": it, "loss": loss.item(), "lr": sched.get_last_lr()[0]})
        log.append(m)
        if it % cfg["log_every"] == 0 or it == steps:
            print(f"  [{it}/{steps}] loss {loss.item():.4f} occ "
                  f"{m['bce'] if cfg['occ_loss']=='bce' else m['focal']:.4f} kl {m['sem_kl']:.4f} "
                  f"iou {m['iou']:.4f} (base {m['base_iou']:.4f}) dens {m['density']:.4f} "
                  f"{(time.time()-t0)/it:.2f}s/it", flush=True)
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
                save_checkpoint(net, os.path.join(ART, "checkpoints", f"{a.tag}_best.pt"),
                                dict(meta, step=it, val=v))
    save_checkpoint(net, os.path.join(ART, "checkpoints", f"{a.tag}_last.pt"),
                    dict(meta, step=steps))
    write_json(os.path.join(ART, f"train_{a.tag}.json"),
               {"tag": a.tag, "config": cfg, "config_path": a.config, "steps": steps,
                "sampler": cfg["sampler"], "occ_loss": cfg["occ_loss"],
                "n_train": len(train), "n_val": len(val), "n_params": net.n_params(),
                "best_val_loss": best[0], "best_step": best[1],
                "seconds": time.time() - t0,
                "peak_gpu_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30, "log": log,
                "selection_rule": "within-cell: lowest validation loss (own objective + "
                                  "teacher KL). Across cells: macro AP on source validation "
                                  "(tools/gate8a/selection.py). No semantic GT, no KITTI-360."})
    print(f"[{a.tag}] done: best val loss {best[0]:.4f} @ {best[1]}, "
          f"{(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
