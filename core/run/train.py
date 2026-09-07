# extracted from tools/gate8c1/train.py -- imports rewritten, logic unchanged.
# see core/README.md; the original is still in place and is the audited copy.
#!/usr/bin/env python
"""Gate 8C-1: train one seed on KITTI-360 alone, behind a runtime dataset firewall.

Architecture, loss, sampler, crop, batch size, optimizer, schedule and the 6 000-step
budget are Gate 8A cell B's, unchanged. The data is KITTI-360 drives 0003/0007/0010 with
occupancy rebuilt from raw Velodyne (never SSCBench's `_1_1.npy`) and semantics distilled
from frozen Trident-H over the future window.

The firewall is enforced, not asserted: the whole run executes inside a
:class:`FileAudit` that intercepts ``open``, ``np.load`` and ``np.fromfile`` and raises the
moment a SemanticKITTI, Occ3D/nuScenes or SSCBench-completion-label path is touched. The
audit summary is written into the training record as the proof the manifest cites.

    python tools/gate8c1/train.py --config configs/gate8c1/seed0.yaml --device cuda:1
"""
from __future__ import annotations
import argparse, os, random, sys, time
import numpy as np, torch, yaml
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.run._common import ART, REPO_ROOT, default_device                               # noqa: E402
from core.datasets.config import write_json                                         # noqa: E402
from core.model.net import CompletionUNet, save_checkpoint                            # noqa: E402
from core.datasets import kitti360_splits as SRC                                               # noqa: E402
from core.model.data import DriveSamples                                            # noqa: E402
from core.datasets.audit import FileAudit                                    # noqa: E402
from core.run._gate8a_train import step_loss                                         # noqa: E402


def run(cfg, device, steps=None):
    seed = int(cfg["seed"]); tag = cfg.get("tag", f"seed{seed}")
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    dev = torch.device(device); torch.cuda.set_device(dev)
    mk = lambda drives, sd, lim: DriveSamples(drives, cfg["crop"], dev, sd, lim,
                                              sampler=cfg["sampler"],
                                              centre_on_occ_p=cfg["centre_on_occ_p"])
    train = mk(cfg["train_drives"], seed, None)
    val = mk([cfg["val_drive"]], seed + 1, cfg["val_limit"])
    n_bad = train.verify() + val.verify()
    print(f"[{tag}] train " + ", ".join(f"{d[17:21]}:{len(train.by_drive[d])}" for d in train.drives)
          + f"  val {cfg['val_drive'][17:21]}:{len(val)}"
          + (f"  ({n_bad} bad)" if n_bad else ""), flush=True)
    net = CompletionUNet(width=cfg["width"],
                         padding_mode=cfg.get("padding_mode", "zeros")).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = steps or cfg["steps"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    os.makedirs(os.path.join(ART, "checkpoints"), exist_ok=True)
    log, best, t0 = [], (float("inf"), -1), time.time()
    bs = cfg["batch_size"]
    drawn = {d: 0 for d in train.drives}
    meta = {"config": cfg, "tag": tag, "seed": seed, "train_drives": cfg["train_drives"],
            "val_drive": cfg["val_drive"]}
    for it in range(1, steps + 1):
        idx = [train.draw() for _ in range(bs)]
        for i in idx:
            drawn[train.drive_of(i)] += 1
        b = train.batch(idx)
        net.train()
        loss, m = step_loss(net, b, cfg)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
        opt.step(); sched.step()
        m.update({"step": it, "loss": loss.item(), "lr": sched.get_last_lr()[0]})
        log.append(m)
        if it % cfg["log_every"] == 0 or it == steps:
            print(f"  [{tag} {it}/{steps}] loss {loss.item():.4f} focal {m['focal']:.4f} "
                  f"dice {m['dice']:.4f} kl {m['sem_kl']:.4f} iou {m['iou']:.4f} "
                  f"(base {m['base_iou']:.4f}) dens {m['density']:.4f} "
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
            print(f"  VAL [{tag} {it}] loss {v['loss']:.4f} iou {v['iou']:.4f} "
                  f"(base {v['base_iou']:.4f}) dens {v['density']:.4f}", flush=True)
            if v["loss"] < best[0]:
                best = (v["loss"], it)
                save_checkpoint(net, os.path.join(ART, "checkpoints", f"{tag}_best.pt"),
                                dict(meta, step=it, val=v))
    save_checkpoint(net, os.path.join(ART, "checkpoints", f"{tag}_last.pt"), dict(meta, step=steps))
    return {"tag": tag, "seed": seed, "config": cfg, "steps": steps,
            "train_drives": cfg["train_drives"], "val_drive": cfg["val_drive"],
            "n_train": {d: len(train.by_drive[d]) for d in train.drives}, "n_val": len(val),
            "samples_drawn_per_drive": drawn, "n_params": net.n_params(),
            "best_val_loss": best[0], "best_step": best[1], "seconds": time.time() - t0,
            "gpu_hours": (time.time() - t0) / 3600.0,
            "peak_gpu_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30, "log": log}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(REPO_ROOT, a.config)))
    SRC.assert_no_target_access(cfg["train_drives"] + [cfg["val_drive"]])
    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        out = run(cfg, a.device, a.steps)
    out["config_path"] = a.config
    out["firewall"] = dict(audit.summary(),
                           note="training executed inside a file-access audit; a "
                                "SemanticKITTI / Occ3D / SSCBench-label open would have raised")
    out["selection_rule"] = ("within-seed: lowest KITTI-360 drive-0006 validation loss. "
                             "Across {best,last}: drive-0006 occupancy AP. Threshold: "
                             "drive-0006 IoU. Semantic threshold: teacher consistency. "
                             "No target dataset is read anywhere.")
    write_json(os.path.join(ART, f"train_{out['tag']}.json"), out)
    print(f"[{out['tag']}] done: best val loss {out['best_val_loss']:.4f} @ {out['best_step']}, "
          f"{out['seconds']/60:.1f} min, firewall violations "
          f"{len(out['firewall']['violations'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
