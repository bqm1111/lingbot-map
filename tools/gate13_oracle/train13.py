#!/usr/bin/env python
"""One Gate 13 training run: condition M (monocular Gate 12 H input) or O (oracle LiDAR input), one seed.

    python tools/gate13_oracle/train13.py --cond O --seed 0 --device cuda:0 [--steps N --smoke]

Gate 12's frozen training loop, unchanged except for the input source: the same Paired reader
(anchor list = Gate 12 inputs with targets, uniform anchor draws, uniform crops from the per-seed RNG),
the same loss weighting mask (Gate 12 H observed S_H | F_H) in both arms, the same config (only
`input` differs), and Gate 12's saved per-seed initial weights. Channels [S, 0, F] from the M or O cache.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, sys, time
import numpy as np, torch, yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g13                                                                                   # noqa: E402
from g13 import g12                                                                          # noqa: E402
sys.path.insert(0, g13.REPO_ROOT)
import train12 as T12                                                                        # noqa: E402  (Gate 12, frozen)
from core.datasets import kitti360_splits as SRC                                             # noqa: E402
from core.datasets.audit import FileAudit                                                    # noqa: E402
from core.model.net import CompletionUNet, save_checkpoint                                   # noqa: E402
from core.run._gate8a_train import step_loss                                                 # noqa: E402

N, DIMS = T12.N, T12.DIMS


class Paired13(T12.Paired):
    def __init__(self, drive_ids, cond, device, seed):
        self.cond, self.dev, self.rng = cond, device, random.Random(seed)
        self.items, self.inp, self.tgt = [], [], []
        for d, ids in drive_ids:
            for t in ids:
                zm = np.load(g13.m_path(d, t)); zt = np.load(g12.target_path(d, t)); I = {"S_H": zm["S_H"], "F_H": zm["F_H"]}
                if cond == "O":
                    zo = np.load(g13.o_path(d, t)); I.update({"S_O": zo["S_O"], "F_O": zo["F_O"]})
                self.items.append((d, t)); self.inp.append(I); self.tgt.append((zt["valid_packed"], zt["occ_packed"]))

    def dense(self, i):
        u = lambda p: torch.from_numpy(np.unpackbits(p)[:N].reshape(DIMS)).to(self.dev).bool()
        I = self.inp[i]; SH, FH = u(I["S_H"]), u(I["F_H"])
        S, F = (SH, FH) if self.cond == "M" else (u(I["S_O"]), u(I["F_O"]))
        return torch.stack([S, torch.zeros_like(S), F]).float(), u(self.tgt[i][1]).float(), u(self.tgt[i][0]), SH | FH


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cond", required=True, choices=g13.CONDITIONS); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--steps", type=int, default=None); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg_path = os.path.join(g13.ART, f"training_config_{a.cond}.yaml"); cfg = yaml.safe_load(open(cfg_path)); assert cfg["input"] == a.cond
    steps = a.steps or int(cfg["steps"]); tag = f"g13_{a.cond}_seed{a.seed}"
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    ck_dir = os.path.join(g13.ART, "smoke" if a.smoke else "checkpoints"); os.makedirs(ck_dir, exist_ok=True)
    init_path = os.path.join(g13.G12_ART, "checkpoints", f"init_seed{a.seed}.pt"); assert os.path.exists(init_path), "Gate 12 initial weights missing"
    net = CompletionUNet(cin=3, width=cfg["width"], padding_mode=cfg["padding_mode"]).to(dev)
    net.load_state_dict(torch.load(init_path, map_location=dev, weights_only=False)["state_dict"]); init_digest = T12.state_digest(net)
    audit_patterns = tuple(SRC.FORBIDDEN_PATH_TOKENS) + ("semantickitti", "occ3d", "nuscenes", "trident", "semantics")
    t0 = time.time(); log = []; h_draw, h_tgt = hashlib.sha256(), hashlib.sha256(); digests = []
    with FileAudit(patterns=audit_patterns, strict=True) as audit:
        train = Paired13([(d, T12.ids_of(d)) for d in g13.TRAIN_DRIVES], a.cond, dev, a.seed)
        val = Paired13([(g13.VAL_DRIVE, T12.ids_of(g13.VAL_DRIVE, cfg["val_every_nth_anchor"]))], a.cond, dev, a.seed + 1)
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
    out = {"tag": tag, "cond": a.cond, "seed": a.seed, "steps": steps, "config": cfg, "config_path": os.path.relpath(cfg_path, g13.REPO_ROOT), "config_sha256": g12.sha256_file(cfg_path),
           "n_train": len(train), "n_val": len(val), "n_params": net.n_params(), "train_items_sha256": hashlib.sha256(json.dumps(train.items).encode()).hexdigest(),
           "init_checkpoint": os.path.relpath(init_path, g13.REPO_ROOT), "init_state_digest": init_digest, "draw_digest": h_draw.hexdigest(), "target_digest": h_tgt.hexdigest(),
           "digests_by_checkpoint": digests, "optimizer": {"type": type(opt).__name__, "lr": cfg["lr"], "weight_decay": cfg["weight_decay"]}, "scheduler": type(sched).__name__,
           "final_checkpoint": os.path.relpath(fin, g13.REPO_ROOT), "final_checkpoint_sha256": g12.sha256_file(fin), "checkpoint_rule": cfg["checkpoint_rule"],
           "seconds": time.time() - t0, "peak_gpu_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30, "firewall": {"opened": len(audit.opened), "violations": audit.violations}, "log": log}
    od = os.path.join(g13.ART, "smoke" if a.smoke else "runs"); os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, f"train_{tag}.json"), "w"), indent=1)
    print(f"[{tag}] done {(time.time()-t0)/60:.1f} min; draw {out['draw_digest'][:12]} target {out['target_digest'][:12]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
