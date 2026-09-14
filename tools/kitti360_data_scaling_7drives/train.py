#!/usr/bin/env python
"""One run of the seven-drive data-scaling experiment.

    python tools/kitti360_data_scaling_7drives/train.py --run full --seed 0 --device cuda:0

Model, objective and optimiser are the adopted Gate 8C-1 configuration
(``configs/gate8c1_sky/seed0.yaml``: CompletionUNet width 24, replicate padding; focal +
Dice occupancy with w_unknown 2 plus 0.5 x KL to the future teacher; AdamW 1e-3 / wd 0.01,
cosine schedule, grad clip 1; batch 4, 128x128x32 uniform crops; drive-uniform then
anchor-uniform sampling) executed through ``core.run._gate8a_train.step_loss`` unchanged.
What varies: the training manifest (``manifests/{small,full}.json``) and the step budget,
which is common to all four runs (``budget.json``). Initial weights are drawn once per
seed, saved, and loaded byte-identically by both runs of that seed. Checkpoints every
1 000 steps; the final step is the primary checkpoint; nothing is selected.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, sys, time
import numpy as np, torch, yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drives7                                                                            # noqa: E402
drives7.apply(redirect_artifacts=False)
from core.datasets import kitti360_splits as SRC                                          # noqa: E402
from core.datasets.audit import FileAudit                                                 # noqa: E402
from core.model.net import CompletionUNet, save_checkpoint                                # noqa: E402
from core.run._gate8a_train import AblationSamples, step_loss                             # noqa: E402
from tools.kitti360_data_scaling import common as C0                                      # noqa: E402

BASE_CONFIG = os.path.join(drives7.REPO_ROOT, "configs", "gate8c1_sky", "seed0.yaml")
SAMPLES = os.path.join(drives7.DATA_ROOT, "samples_gtsem")
CKPT_EVERY = 1000


class ManifestSamples(AblationSamples):
    """The adopted crop sampler and batch layout over an explicit anchor manifest."""

    def __init__(self, manifest: dict, crop, device, seed, **kw):
        AblationSamples.__init__(self, [], crop, device, seed, None, 1, **kw)
        self.drives = [d for d in drives7.TRAIN_DRIVES7 if manifest.get(d)]
        self.by_drive = {d: [os.path.join(SAMPLES, d, f"{i:05d}.npz") for i in manifest[d]] for d in self.drives}
        self.files, self._off = [], {}
        for d in self.drives:
            self._off[d] = len(self.files); self.files += self.by_drive[d]
        missing = [f for f in self.files if not os.path.exists(f)]
        assert not missing, f"{len(missing)} manifest samples missing, first {missing[:2]}"
        self.exposure = np.zeros(len(self.files), np.int64)

    def draw(self) -> int:
        d = self.drives[self.rng.randrange(len(self.drives))]
        i = self._off[d] + self.rng.randrange(len(self.by_drive[d])); self.exposure[i] += 1
        return i

    def drive_of(self, i):
        for d in self.drives:
            if self._off[d] <= i < self._off[d] + len(self.by_drive[d]):
                return d


def state_digest(net):
    h = hashlib.sha256()
    for k, v in net.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, choices=["small", "full"]); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--steps", type=int, default=None, help="smoke tests only")
    a = ap.parse_args()
    budget = json.load(open(os.path.join(drives7.ART, "budget.json")))
    steps = a.steps or int(budget["steps"])
    cfg = yaml.safe_load(open(BASE_CONFIG)); cfg["seed"] = a.seed; tag = f"kds7_{a.run}_seed{a.seed}"; cfg["tag"] = tag
    man = json.load(open(os.path.join(drives7.ART, "manifests", f"{a.run}.json")))
    assert man["sha256"] == C0.sha256_json(man["drives"]), "manifest altered after freezing"
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    ck_dir = os.path.join(drives7.ART, "checkpoints" if not a.steps else "smoke"); os.makedirs(ck_dir, exist_ok=True)
    init_path = os.path.join(drives7.ART, "checkpoints", f"init_seed{a.seed}.pt")
    net = CompletionUNet(width=cfg["width"], padding_mode=cfg["padding_mode"]).to(dev)
    if os.path.exists(init_path):
        net.load_state_dict(torch.load(init_path, map_location=dev, weights_only=False)["state_dict"])
    else:
        os.makedirs(os.path.dirname(init_path), exist_ok=True)
        save_checkpoint(net, init_path, {"seed": a.seed, "note": "shared initial weights of both runs of this seed"})
    init_digest = state_digest(net); init_sha = C0.sha256_file(init_path)
    t0 = time.time(); log = []
    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        train = ManifestSamples(man["drives"], cfg["crop"], dev, a.seed, sampler=cfg["sampler"], centre_on_occ_p=cfg["centre_on_occ_p"])
        val_files = sorted(os.listdir(os.path.join(SAMPLES, drives7.VAL_DRIVE)))[:cfg["val_limit"]]
        # validation reuses the adopted reader over the first val_limit 0006 samples (fixed crops per call order)
        val = AblationSamples([], cfg["crop"], dev, a.seed + 1, None, 1, sampler=cfg["sampler"], centre_on_occ_p=cfg["centre_on_occ_p"])
        val.files = [os.path.join(SAMPLES, drives7.VAL_DRIVE, f) for f in val_files]
        print(f"[{tag}] {len(train)} training anchors over {len(train.drives)} drives, val {len(val)}, {steps} steps, init {init_digest[:12]}", flush=True)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        bs = cfg["batch_size"]; drawn = {d: 0 for d in train.drives}
        for it in range(1, steps + 1):
            idx = [train.draw() for _ in range(bs)]
            for i in idx:
                drawn[train.drive_of(i)] += 1
            b = train.batch(idx); net.train()
            loss, m = step_loss(net, b, cfg)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"]); opt.step(); sched.step()
            m.update({"step": it, "loss": loss.item(), "lr": sched.get_last_lr()[0]}); log.append(m)
            if it % cfg["log_every"] == 0 or it == steps:
                print(f"  [{tag} {it}/{steps}] loss {m['loss']:.4f} focal {m['focal']:.4f} dice {m['dice']:.4f} kl {m['sem_kl']:.4f} iou {m['iou']:.4f} "
                      f"unique {int((train.exposure > 0).sum())} {(time.time()-t0)/it:.3f}s/it", flush=True)
            if it % cfg["val_every"] == 0 or it == steps:
                net.eval(); vs = []
                with torch.no_grad():
                    for j in range(0, len(val), bs):
                        vb = val.batch(list(range(j, min(j + bs, len(val)))))
                        vl, vm = step_loss(net, vb, cfg); vm["loss"] = vl.item(); vs.append(vm)
                v = {k: float(np.nanmean([x[k] for x in vs])) for k in vs[0]}; v["step"] = it; log.append({"val": v})
                print(f"  VAL [{tag} {it}] loss {v['loss']:.4f} kl {v['sem_kl']:.4f} iou {v['iou']:.4f}", flush=True)
            if it % CKPT_EVERY == 0 or it == steps:
                save_checkpoint(net, os.path.join(ck_dir, f"{tag}_step{it:05d}.pt"), {"tag": tag, "step": it, "run": a.run, "seed": a.seed})
    ex = train.exposure
    out = {"tag": tag, "run": a.run, "seed": a.seed, "steps": steps, "config": cfg, "config_path": os.path.relpath(BASE_CONFIG, drives7.REPO_ROOT),
           "manifest_sha256": man["sha256"], "n_train_anchors": len(train), "init_checkpoint": os.path.relpath(init_path, drives7.REPO_ROOT),
           "init_sha256": init_sha, "init_state_digest": init_digest, "samples_drawn_per_drive": drawn,
           "exposure": {"unique_anchors_seen": int((ex > 0).sum()), "fraction_seen": float((ex > 0).mean()), "mean_exposures_per_anchor": float(ex.mean()),
                        "max_exposures": int(ex.max()), "effective_epochs": float(steps * cfg["batch_size"] / len(train))},
           "checkpoints": sorted(os.path.relpath(os.path.join(ck_dir, f), drives7.REPO_ROOT) for f in os.listdir(ck_dir) if f.startswith(tag)),
           "final_checkpoint": os.path.relpath(os.path.join(ck_dir, f"{tag}_step{steps:05d}.pt"), drives7.REPO_ROOT),
           "final_checkpoint_sha256": C0.sha256_file(os.path.join(ck_dir, f"{tag}_step{steps:05d}.pt")),
           "seconds": time.time() - t0, "gpu_hours": (time.time() - t0) / 3600, "peak_gpu_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30,
           "firewall": audit.summary(), "env": C0.environment(), "log": log}
    C0.write_json(os.path.join(drives7.ART, "runs" if not a.steps else "smoke", f"train_{tag}.json"), out)
    print(f"[{tag}] done in {out['seconds']/60:.1f} min; unique anchors {out['exposure']['unique_anchors_seen']}/{len(train)}, epochs {out['exposure']['effective_epochs']:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
