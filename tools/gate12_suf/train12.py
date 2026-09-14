#!/usr/bin/env python
"""One Gate 12 training run: condition H or SUF, one seed.

    python tools/gate12_suf/train12.py --cond H --seed 0 --device cuda:0 [--steps N --smoke]

Adopted Gate 8C-1 completion recipe (configs/gate8c1_sky/seed0.yaml) executed through the unchanged
core.run._gate8a_train.step_loss: CompletionUNet width 24, replicate padding; focal + Dice occupancy
(w_unknown 2, pos-weight cap 8, gamma 2); AdamW 1e-3 / wd 0.01; cosine schedule; grad clip 1;
batch 4; uniform 128x128x32 crops; anchors drawn uniformly over all training anchors; bf16 autocast;
final-step checkpoint. Geometry only: w_sem = 0 and no semantic input. The input is the three-channel
[S, U, F] volume (H: [S_H, 0, F_H]); the base log-odds is zero, so the frozen residual lock passes the
network output unchanged. The loss's observed/unknown weighting uses S_H | F_H in BOTH conditions.

Identical per seed for H and SUF: initial weights (saved once), anchor draws, crop origins, target
tensors, loss weights, validation crops -- only the input channels differ. Digests of every draw and
every target crop are recorded and compared in verification.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, sys, time
import numpy as np, torch, yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g12                                                                                  # noqa: E402
sys.path.insert(0, g12.REPO_ROOT)
from core.datasets import kitti360_splits as SRC                                            # noqa: E402
from core.datasets.audit import FileAudit                                                   # noqa: E402
from core.model import sampler as SMP                                                       # noqa: E402
from core.model.net import CompletionUNet, save_checkpoint                                  # noqa: E402
from core.run._gate8a_train import step_loss                                                # noqa: E402

DIMS = (256, 256, 32); N = int(np.prod(DIMS))


def config_path(cond):
    return os.path.join(g12.ART, f"training_config_{cond}.yaml")


class Paired:
    """Packed volumes held in RAM; dense tensors built per batch."""

    def __init__(self, drive_ids, cond, device, seed):
        self.cond, self.dev, self.rng = cond, device, random.Random(seed)
        self.items, self.inp, self.tgt = [], [], []
        need = ("S_H", "F_H") + (("S", "U", "F") if cond == "SUF" else ())
        for d, ids in drive_ids:
            for t in ids:
                zi = np.load(g12.input_path(d, t)); zt = np.load(g12.target_path(d, t))
                self.items.append((d, t)); self.inp.append({k: zi[k] for k in need})
                self.tgt.append((zt["valid_packed"], zt["occ_packed"]))

    def __len__(self):
        return len(self.items)

    def dense(self, i):
        u = lambda p: torch.from_numpy(np.unpackbits(p)[:N].reshape(DIMS)).to(self.dev)
        I = self.inp[i]; SH, FH = u(I["S_H"]).bool(), u(I["F_H"]).bool()
        x = torch.stack([SH, torch.zeros_like(SH), FH]) if self.cond == "H" else torch.stack([u(I["S"]), u(I["U"]), u(I["F"])]).bool()
        return x.float(), u(self.tgt[i][1]).float(), u(self.tgt[i][0]).bool(), SH | FH

    def batch(self, idxs, crop, h_draw=None, h_tgt=None):
        out = {k: [] for k in ("input", "gt_occ", "gt_valid", "observed")}
        for i in idxs:
            x, occ, val, obs = self.dense(i)
            x0, y0, z0 = SMP.uniform_origin(self.rng, DIMS, crop)
            sl = (slice(x0, x0 + crop[0]), slice(y0, y0 + crop[1]), slice(z0, z0 + crop[2]))
            out["input"].append(x[(slice(None),) + sl]); out["gt_occ"].append(occ[sl]); out["gt_valid"].append(val[sl]); out["observed"].append(obs[sl])
            if h_draw is not None:
                h_draw.update(f"{self.items[i][0][17:21]}:{self.items[i][1]}:{x0},{y0},{z0};".encode())
                h_tgt.update(occ[sl].to(torch.uint8).cpu().numpy().tobytes()); h_tgt.update(val[sl].cpu().numpy().tobytes()); h_tgt.update(obs[sl].cpu().numpy().tobytes())
        b = {k: torch.stack(v) for k, v in out.items()}
        B = len(idxs); X, Y, Z = crop
        b["base_logodds"] = torch.zeros((B, X, Y, Z), device=self.dev)
        b["fut_p"] = torch.zeros((B, 1, X, Y, Z), device=self.dev); b["fut_valid"] = torch.zeros((B, X, Y, Z), dtype=torch.bool, device=self.dev)
        return b


def ids_of(d, every=1):
    ids = sorted(int(f[:-4]) for f in os.listdir(os.path.dirname(g12.input_path(d, 0))))
    return [t for t in ids if os.path.exists(g12.target_path(d, t))][::every]


def state_digest(net):
    h = hashlib.sha256()
    for k, v in net.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cond", required=True, choices=g12.CONDITIONS); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--steps", type=int, default=None); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(config_path(a.cond))); assert cfg["input"] == a.cond
    steps = a.steps or int(cfg["steps"]); tag = f"g12_{a.cond}_seed{a.seed}"
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    sub = "smoke" if a.smoke else "checkpoints"; ck_dir = os.path.join(g12.ART, sub); os.makedirs(ck_dir, exist_ok=True)
    init_path = os.path.join(g12.ART, "checkpoints", f"init_seed{a.seed}.pt"); os.makedirs(os.path.dirname(init_path), exist_ok=True)
    net = CompletionUNet(cin=3, width=cfg["width"], padding_mode=cfg["padding_mode"]).to(dev)
    if os.path.exists(init_path):
        net.load_state_dict(torch.load(init_path, map_location=dev, weights_only=False)["state_dict"])
    else:
        save_checkpoint(net, init_path, {"seed": a.seed, "note": "shared initial weights of H and SUF for this seed"})
    init_digest = state_digest(net)
    audit_patterns = tuple(SRC.FORBIDDEN_PATH_TOKENS) + ("semantickitti", "occ3d", "nuscenes", "trident", "semantics")
    t0 = time.time(); log = []; h_draw, h_tgt = hashlib.sha256(), hashlib.sha256(); digests = []
    with FileAudit(patterns=audit_patterns, strict=True) as audit:
        train = Paired([(d, ids_of(d)) for d in g12.TRAIN_DRIVES], a.cond, dev, a.seed)
        val = Paired([(g12.VAL_DRIVE, ids_of(g12.VAL_DRIVE, cfg["val_every_nth_anchor"]))], a.cond, dev, a.seed + 1)
        print(f"[{tag}] train {len(train)} anchors, val {len(val)}, steps {steps}, params {net.n_params():,}, init {init_digest[:12]}", flush=True)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        bs = cfg["batch_size"]; crop = tuple(cfg["crop"])
        for it in range(1, steps + 1):
            idx = [train.rng.randrange(len(train)) for _ in range(bs)]
            b = train.batch(idx, crop, h_draw, h_tgt); net.train()
            loss, m = step_loss(net, b, cfg)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"]); opt.step(); sched.step()
            m.update({"step": it, "loss": loss.item(), "lr": sched.get_last_lr()[0]}); log.append(m)
            if it % cfg["log_every"] == 0 or it == steps:
                print(f"  [{tag} {it}/{steps}] loss {m['loss']:.4f} focal {m['focal']:.4f} dice {m['dice']:.4f} iou {m['iou']:.4f} dens {m['density']:.4f} {(time.time()-t0)/it:.3f}s/it", flush=True)
            if it % cfg["val_every"] == 0 or it == steps:
                net.eval(); vs = []
                with torch.no_grad():
                    for j in range(0, len(val), bs):
                        vb = val.batch(list(range(j, min(j + bs, len(val)))), crop)
                        vl, vm = step_loss(net, vb, cfg); vm["loss"] = vl.item(); vs.append(vm)
                v = {k: float(np.nanmean([x[k] for x in vs])) for k in vs[0]}; v["step"] = it; log.append({"val": v})
                print(f"  VAL [{tag} {it}] loss {v['loss']:.4f} iou {v['iou']:.4f} dens {v['density']:.4f}", flush=True)
            if it % cfg["ckpt_every"] == 0 or it == steps:
                save_checkpoint(net, os.path.join(ck_dir, f"{tag}_step{it:05d}.pt"), {"tag": tag, "step": it, "cond": a.cond, "seed": a.seed})
                digests.append({"step": it, "draw": h_draw.hexdigest(), "target": h_tgt.hexdigest()})
    fin = os.path.join(ck_dir, f"{tag}_step{steps:05d}.pt")
    out = {"tag": tag, "cond": a.cond, "seed": a.seed, "steps": steps, "config": cfg, "config_path": os.path.relpath(config_path(a.cond), g12.REPO_ROOT),
           "config_sha256": g12.sha256_file(config_path(a.cond)), "n_train": len(train), "n_val": len(val), "n_params": net.n_params(),
           "train_items_sha256": hashlib.sha256(json.dumps(train.items).encode()).hexdigest(), "init_state_digest": init_digest,
           "draw_digest": h_draw.hexdigest(), "target_digest": h_tgt.hexdigest(), "digests_by_checkpoint": digests,
           "optimizer": {"type": type(opt).__name__, "lr": cfg["lr"], "weight_decay": cfg["weight_decay"]}, "scheduler": type(sched).__name__,
           "final_checkpoint": os.path.relpath(fin, g12.REPO_ROOT), "final_checkpoint_sha256": g12.sha256_file(fin), "checkpoint_rule": cfg["checkpoint_rule"],
           "seconds": time.time() - t0, "peak_gpu_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30,
           "firewall": {"opened": len(audit.opened), "violations": audit.violations}, "log": log}
    od = os.path.join(g12.ART, "runs" if not a.smoke else "smoke"); os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, f"train_{tag}.json"), "w"), indent=1)
    print(f"[{tag}] done {(time.time()-t0)/60:.1f} min; draw {out['draw_digest'][:12]} target {out['target_digest'][:12]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
