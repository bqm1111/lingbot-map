#!/usr/bin/env python
"""Train one condition x seed of the feature-supervision study.

    python tools/feature_study/train.py --condition C --seed 0 --device cuda:0

Conditions differ in exactly one term:

    A   L = L_occ
    B   L = L_occ + w_aux * L_KL          (the adopted objective, retrained here)
    C   L = L_occ + w_aux * L_feature

``L_occ`` is the adopted masked focal + Dice objective, unchanged. The feature head is
constructed in **all three** conditions so the shared trunk and the two original heads draw
identical initial values from the RNG; in A and B it receives no gradient, so AdamW never
steps it (PyTorch skips parameters whose ``.grad`` is ``None``) and it stays at
initialisation. The same is true of the semantic head in A and C.

The run is executed inside the gate's file-access audit, so a SemanticKITTI, Occ3D or
SSCBench-label read would raise rather than be discovered later.
"""
from __future__ import annotations
import argparse, os, random, sys, time
import numpy as np, torch, yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# study first: it pins G8C1_DATA_ROOT, which core.datasets reads at import time
from tools.feature_study import study as ST                                       # noqa: E402
from core.datasets.audit import FileAudit                                         # noqa: E402
from core.datasets import kitti360_splits as SRC                                  # noqa: E402
from core.model import losses as L                                                # noqa: E402
from core.model.net import CompletionUNet, apply_residual, save_checkpoint        # noqa: E402
from tools.feature_study.losses import feature_cosine, features_at                # noqa: E402
from tools.feature_study.samples import StudySamples                              # noqa: E402

BASE_CONFIG = os.path.join(ST.REPO_ROOT, "configs", "feature_study", "base.yaml")


def step_loss(net, b, cfg, condition: str, probe: bool = False):
    """Returns ``(loss, metrics)``. Only the condition's own auxiliary term is built."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        occ_res, sem_log, trunk = net(b["input"], return_trunk=True)
    occ_res = occ_res[:, 0].float(); sem_log = sem_log.float()
    final = apply_residual(b["base_logodds"], occ_res)

    w = L.balanced_weights(b["gt_occ"], b["gt_valid"], b["observed"], cfg["w_unknown"])
    n_pos = (b["gt_occ"] * b["gt_valid"].float()).sum()
    n_neg = b["gt_valid"].float().sum() - n_pos
    pw = float(min(cfg["pos_weight_cap"], max(1.0, (n_neg / n_pos.clamp_min(1)).item())))
    l_focal = L.focal_bce(final, b["gt_occ"], w, cfg["focal_gamma"], pw)
    l_dice = L.soft_dice(final, b["gt_occ"], b["gt_valid"].float())
    l_occ = cfg["w_focal"] * l_focal + cfg["w_dice"] * l_dice

    n_kl = int(b["fut_valid"].sum())
    n_feat = int(b["aux_index"].numel())
    l_aux = None
    if condition == "B":
        l_aux = L.semantic_kl(sem_log, b["fut_p"], b["fut_valid"].float())
    elif condition == "C":
        l_aux = feature_cosine(features_at(net, trunk, b["aux_index"]), b["aux_target"])
    loss = l_occ if l_aux is None else l_occ + cfg["w_aux"] * l_aux

    m = {"focal": l_focal.item(), "dice": l_dice.item(), "occ": l_occ.item(),
         "aux": (float("nan") if l_aux is None else l_aux.item()),
         "n_kl_voxels": n_kl, "n_feature_voxels": n_feat}
    if probe:
        g_occ = torch.autograd.grad(l_occ, trunk, retain_graph=True)[0]
        m["occ_grad_norm"] = float(g_occ.detach().float().norm())
        if l_aux is not None and l_aux.requires_grad:
            g_aux = torch.autograd.grad(cfg["w_aux"] * l_aux, trunk, retain_graph=True)[0]
            m["aux_grad_norm"] = float(g_aux.detach().float().norm())
        else:
            m["aux_grad_norm"] = 0.0
    with torch.no_grad():
        gt = (b["gt_occ"] > 0) & b["gt_valid"]
        pred = (final > 0) & b["gt_valid"]
        base = (b["base_logodds"] > 0) & b["gt_valid"]

        def iou(p):
            tp = (p & gt).sum().item(); fp = (p & ~gt).sum().item(); fn = (~p & gt).sum().item()
            return tp / max(tp + fp + fn, 1)
        m.update({"iou": iou(pred), "base_iou": iou(base),
                  "density": float(pred.sum().item() / max(b["gt_valid"].sum().item(), 1))})
    return loss, m


def run(cfg, condition: str, seed: int, device: str, steps=None):
    tag = ST.run_tag(condition, seed)
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    dev = torch.device(device); torch.cuda.set_device(dev)
    mk = lambda drives, sd, lim: StudySamples(drives, cfg["crop"], dev, sd, lim,
                                              sampler=cfg["sampler"],
                                              centre_on_occ_p=cfg["centre_on_occ_p"])
    train = mk(cfg["train_drives"], seed, None)
    val = mk([cfg["val_drive"]], seed + 1, cfg["val_limit"])
    n_bad = train.verify() + val.verify()
    print(f"[{tag}] train " + ", ".join(f"{d[17:21]}:{len(train.by_drive[d])}"
                                        for d in train.drives)
          + f"  val {cfg['val_drive'][17:21]}:{len(val)}"
          + (f"  ({n_bad} bad)" if n_bad else ""), flush=True)

    # every condition builds the feature head, so the shared initialisation is identical
    net = CompletionUNet(width=cfg["width"], padding_mode=cfg["padding_mode"],
                         n_feat=cfg["feature_dim"]).to(dev)
    shared = [p for n_, p in net.named_parameters() if not n_.startswith("head_feat")]
    n_shared = sum(p.numel() for p in shared)
    active = {"A": n_shared - sum(p.numel() for p in net.head_sem.parameters()),
              "B": n_shared,
              "C": n_shared - sum(p.numel() for p in net.head_sem.parameters())
                   + sum(p.numel() for p in net.head_feat.parameters())}[condition]
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = steps or cfg["steps"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    os.makedirs(os.path.join(ST.ART, "checkpoints"), exist_ok=True)
    log, t0 = [], time.time()
    bs = cfg["batch_size"]
    drawn = {d: 0 for d in train.drives}
    meta = {"config": cfg, "tag": tag, "condition": condition, "seed": seed}
    for it in range(1, steps + 1):
        idx = [train.draw() for _ in range(bs)]
        for i in idx:
            drawn[train.drive_of(i)] += 1
        b = train.batch(idx)
        net.train()
        loss, m = step_loss(net, b, cfg, condition, probe=(it % cfg["probe_every"] == 0))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
        opt.step(); sched.step()
        m.update({"step": it, "loss": loss.item(), "lr": sched.get_last_lr()[0]})
        log.append(m)
        if it % cfg["log_every"] == 0 or it == steps:
            print(f"  [{tag} {it}/{steps}] loss {loss.item():.4f} occ {m['occ']:.4f} "
                  f"aux {m['aux']:.4f} iou {m['iou']:.4f} (base {m['base_iou']:.4f}) "
                  f"dens {m['density']:.4f} {(time.time()-t0)/it:.2f}s/it", flush=True)
        if it % cfg["val_every"] == 0 or it == steps:
            net.eval(); vs = []
            with torch.no_grad():
                for j in range(0, len(val), bs):
                    vb = val.batch(list(range(j, min(j + bs, len(val)))), record=False)
                    vl, vm = step_loss(net, vb, cfg, condition)
                    vm["loss"] = vl.item(); vs.append(vm)
            with np.errstate(invalid="ignore"):
                v = {k: float(np.nanmean([x[k] for x in vs]))
                     for k in vs[0] if not all(np.isnan(float(x[k])) for x in vs)}
            v["step"] = it; log.append({"val": v})
            print(f"  VAL [{tag} {it}] loss {v['loss']:.4f} occ {v['occ']:.4f} "
                  f"aux {v.get('aux', float('nan')):.4f} iou {v['iou']:.4f}", flush=True)
    # the final-step checkpoint is THE checkpoint; nothing earlier is kept or considered
    save_checkpoint(net, ST.checkpoint_path(condition, seed), dict(meta, step=steps))
    np.savez_compressed(os.path.join(ST.ART, f"trace_{tag}.npz"),
                        trace=np.asarray(train.trace, np.int32))
    return {"tag": tag, "condition": condition, "seed": seed, "config": cfg,
            "steps": steps, "train_drives": cfg["train_drives"],
            "val_drive": cfg["val_drive"],
            "n_train": {d: len(train.by_drive[d]) for d in train.drives},
            "n_val": len(val), "samples_drawn_per_drive": drawn,
            "n_params_total": net.n_params(), "n_params_shared": n_shared,
            "n_params_active": active,
            "n_missing_feature_target": train.n_missing_feature_target,
            "trace_sha256": ST.sha256_json(train.trace),
            "checkpoint": os.path.relpath(ST.checkpoint_path(condition, seed), ST.REPO_ROOT),
            "seconds": time.time() - t0,
            "peak_gpu_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30, "log": log}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", required=True, choices=list(ST.CONDITIONS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--config", default=BASE_CONFIG)
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()
    ST.require_data_root()
    cfg = yaml.safe_load(open(a.config))
    SRC.assert_no_target_access(cfg["train_drives"] + [cfg["val_drive"]])
    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        out = run(cfg, a.condition, a.seed, a.device, a.steps)
    out["config_path"] = os.path.relpath(a.config, ST.REPO_ROOT)
    out["checkpoint_sha256"] = ST.sha256_file(ST.checkpoint_path(a.condition, a.seed))
    out["firewall"] = dict(audit.summary(),
                           note="training ran inside a file-access audit; a SemanticKITTI / "
                                "Occ3D / SSCBench-label open would have raised")
    out["selection_rule"] = ("final-step checkpoint, no within-run selection. Occupancy "
                             "threshold is chosen afterwards on KITTI-360 drive 0006 by the "
                             "gate's own rule. No target dataset is read anywhere.")
    out["env"] = ST.environment()
    ST.write_json(os.path.join(ST.ART, f"train_{out['tag']}.json"), out)
    print(f"[{out['tag']}] done in {out['seconds']/60:.1f} min, firewall violations "
          f"{len(out['firewall']['violations'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
