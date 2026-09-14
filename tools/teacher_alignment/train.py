#!/usr/bin/env python
"""Train one condition x seed of the teacher-alignment control.

    python tools/teacher_alignment/train.py --condition S --seed 0 --device cuda:0

Identical to the feature study's B run in data, crops, initialisation, architecture,
optimizer, schedule and budget. Two things differ, and they are the whole experiment:

* the auxiliary target: **R** keeps the future-teacher class vector at its own voxel; **S**
  permutes those vectors within (z, observed) strata of every crop (``shuffle.py``);
* the trunk gradient of ``0.5 * L_KL`` is rescaled every step to ``rho`` times the norm of
  the occupancy gradient (``gradmatch.py``), with ``rho`` frozen by ``calibrate_rho.py``.

Both arms use the same rule. Runs inside the gate's file-access audit.
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
import numpy as np, torch, yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.teacher_alignment import study as ST                                   # noqa: E402
from core.datasets.audit import FileAudit                                         # noqa: E402
from core.datasets import kitti360_splits as SRC                                  # noqa: E402
from core.model.net import CompletionUNet, save_checkpoint                        # noqa: E402
from tools.feature_study.samples import StudySamples                              # noqa: E402
from tools.teacher_alignment import gradmatch as GM, shuffle as SH                # noqa: E402

BASE_CONFIG = os.path.join(ST.REPO_ROOT, "configs", "feature_study", "base.yaml")
SHUFFLE_KEYS = ("n_eligible", "n_changed", "n_argmax_changed", "n_in_homogeneous_strata")


def aux_targets(b, condition: str, seed: int, step: int):
    """The common mask (tiny strata dropped) and, for S, the permuted targets."""
    B = b["fut_valid"].shape[0]
    keeps, targets, st = [], [], {"n_valid": 0, "n_kept": 0, "n_dropped_tiny_strata": 0,
                                  **{k: 0 for k in SHUFFLE_KEYS}, "l1_sum": 0.0}
    for k in range(B):
        keep, es = SH.eligible_mask(b["fut_valid"][k], b["observed"][k])
        for kk in ("n_valid", "n_kept", "n_dropped_tiny_strata"):
            st[kk] += es[kk]
        t = b["fut_p"][k]
        if condition == "S":
            t, ss = SH.shuffle_targets(t, keep, b["observed"][k], seed, step, k)
            for kk in SHUFFLE_KEYS:
                st[kk] += ss[kk]
            st["l1_sum"] += ss["mean_l1"] * ss["n_eligible"]
        keeps.append(keep); targets.append(t)
    return torch.stack(keeps), torch.stack(targets), st


def run(cfg, condition: str, seed: int, rho: float, device: str, steps=None):
    tag = ST.run_tag(condition, seed)
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    dev = torch.device(device); torch.cuda.set_device(dev)
    mk = lambda drives, sd, lim: StudySamples(drives, cfg["crop"], dev, sd, lim,
                                              sampler=cfg["sampler"],
                                              centre_on_occ_p=cfg["centre_on_occ_p"],
                                              want_features=False)
    train = mk(cfg["train_drives"], seed, None)
    val = mk([cfg["val_drive"]], seed + 1, cfg["val_limit"])
    n_bad = train.verify() + val.verify()
    print(f"[{tag}] rho {rho:.4f}  train " + ", ".join(f"{d[17:21]}:{len(train.by_drive[d])}"
                                                      for d in train.drives)
          + f"  val {cfg['val_drive'][17:21]}:{len(val)}" + (f"  ({n_bad} bad)" if n_bad else ""),
          flush=True)
    # same constructor call as historical B, so the initial state is bit-identical to it
    net = CompletionUNet(width=cfg["width"], padding_mode=cfg["padding_mode"],
                         n_feat=cfg["feature_dim"]).to(dev)
    groups = GM.param_groups(net)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = steps or cfg["steps"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    os.makedirs(os.path.join(ST.ART, "checkpoints"), exist_ok=True)
    log, t0 = [], time.time()
    bs = cfg["batch_size"]
    drawn = {d: 0 for d in train.drives}
    n_degenerate = 0
    meta = {"config": cfg, "tag": tag, "condition": condition, "seed": seed, "rho": rho}
    for it in range(1, steps + 1):
        idx = [train.draw() for _ in range(bs)]
        for i in idx:
            drawn[train.drive_of(i)] += 1
        b = train.batch(idx)
        keep, tgt, st = aux_targets(b, condition, seed, it)
        net.train()
        l_occ, l_kl, m = GM.losses(net, b, cfg, tgt, keep)
        tg = GM.task_gradients(l_occ, l_kl, groups, cfg["w_aux"])
        opt.zero_grad(set_to_none=True)
        gm = GM.assemble(groups, tg, rho)
        n_degenerate += gm["degenerate"]
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
        opt.step(); sched.step()
        m.update(gm); m.update({k: st[k] for k in st if k != "l1_sum"})
        m["shuffle_mean_l1"] = (st["l1_sum"] / max(st["n_eligible"], 1)
                                if condition == "S" else 0.0)
        m.update({"step": it, "lr": sched.get_last_lr()[0],
                  "loss": float(l_occ) + cfg["w_aux"] * float(l_kl)})
        log.append(m)
        if it % cfg["log_every"] == 0 or it == steps:
            print(f"  [{tag} {it}/{steps}] occ {m['occ']:.4f} kl {m['kl']:.4f} "
                  f"agree {m['aux_top1_agree']:.3f} | |g_occ| {m['occ_grad_norm']:.3f} "
                  f"|g_aux| {m['aux_grad_norm']:.3f} raw {m['raw_ratio']:.2f} "
                  f"c {m['norm_factor']:.3f} cos {m['grad_cosine']:+.3f} "
                  f"| iou {m['iou']:.4f} dens {m['density']:.4f} "
                  f"{(time.time()-t0)/it:.2f}s/it", flush=True)
        if it % cfg["val_every"] == 0 or it == steps:
            net.eval(); vs = []
            with torch.no_grad():
                for j in range(0, len(val), bs):
                    vb = val.batch(list(range(j, min(j + bs, len(val)))), record=False)
                    # validation always scores against the ALIGNED targets, both arms
                    vkeep, vtgt, _ = aux_targets(vb, "R", seed, 0)
                    vo, vkl, vm = GM.losses(net, vb, cfg, vtgt, vkeep)
                    vm["loss"] = float(vo) + cfg["w_aux"] * float(vkl); vs.append(vm)
            v = {k: float(np.mean([x[k] for x in vs])) for k in vs[0]}
            v["step"] = it; log.append({"val": v})
            print(f"  VAL [{tag} {it}] occ {v['occ']:.4f} kl(aligned) {v['kl']:.4f} "
                  f"agree(aligned) {v['aux_top1_agree']:.3f} iou {v['iou']:.4f}", flush=True)
    save_checkpoint(net, ST.checkpoint_path(condition, seed), dict(meta, step=steps))
    np.savez_compressed(os.path.join(ST.ART, f"trace_{tag}.npz"),
                        trace=np.asarray(train.trace, np.int32))
    steps_log = [x for x in log if "val" not in x]
    agg = {k: float(np.mean([x[k] for x in steps_log])) for k in
           ("n_valid", "n_kept", "n_dropped_tiny_strata", "n_eligible", "n_changed",
            "n_argmax_changed", "n_in_homogeneous_strata", "shuffle_mean_l1",
            "achieved_ratio", "raw_ratio", "grad_cosine", "norm_factor")}
    tot_el = sum(x["n_eligible"] for x in steps_log)
    manip = {"fraction_supervision_dropped_tiny_strata":
                 sum(x["n_dropped_tiny_strata"] for x in steps_log)
                 / max(sum(x["n_valid"] for x in steps_log), 1),
             "fraction_vectors_changed": (sum(x["n_changed"] for x in steps_log) / max(tot_el, 1)
                                          if condition == "S" else 0.0),
             "fraction_argmax_changed": (sum(x["n_argmax_changed"] for x in steps_log)
                                         / max(tot_el, 1) if condition == "S" else 0.0),
             "mean_l1_before_after": agg["shuffle_mean_l1"],
             "fraction_in_homogeneous_strata": (sum(x["n_in_homogeneous_strata"] for x in steps_log)
                                                / max(tot_el, 1) if condition == "S" else 0.0)}
    return {"tag": tag, "condition": condition, "seed": seed, "rho": rho, "config": cfg,
            "steps": steps, "train_drives": cfg["train_drives"], "val_drive": cfg["val_drive"],
            "n_train": {d: len(train.by_drive[d]) for d in train.drives}, "n_val": len(val),
            "samples_drawn_per_drive": drawn,
            "n_params_total": net.n_params(),
            "n_params_shared": sum(p.numel() for p in groups["shared"]),
            "n_params_active": sum(p.numel() for g in ("shared", "head_occ", "head_sem")
                                   for p in groups[g]),
            "n_degenerate_steps": n_degenerate,
            "gradient_matching": {"rho": rho, "rule": "g_shared = g_occ + rho*|g_occ|/|g_aux| * g_aux; "
                                  "heads unscaled", "achieved_ratio_mean": agg["achieved_ratio"],
                                  "raw_ratio_mean": agg["raw_ratio"],
                                  "raw_ratio_median": float(np.median([x["raw_ratio"] for x in steps_log])),
                                  "grad_cosine_median": float(np.median([x["grad_cosine"] for x in steps_log])),
                                  "norm_factor_median": float(np.median([x["norm_factor"] for x in steps_log]))},
            "manipulation": manip, "trace_sha256": ST.sha256_json(train.trace),
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
    rho = float(json.load(open(ST.rho_path()))["rho"])
    SRC.assert_no_target_access(cfg["train_drives"] + [cfg["val_drive"]])
    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        out = run(cfg, a.condition, a.seed, rho, a.device, a.steps)
    out["config_path"] = os.path.relpath(a.config, ST.REPO_ROOT)
    out["rho_record"] = os.path.relpath(ST.rho_path(), ST.REPO_ROOT)
    out["checkpoint_sha256"] = ST.sha256_file(ST.checkpoint_path(a.condition, a.seed))
    out["firewall"] = dict(audit.summary(), note="training ran inside a file-access audit")
    out["selection_rule"] = ("final-step checkpoint, no within-run selection; thresholds "
                             "chosen afterwards on KITTI-360 drive 0006 by the gate's rule")
    out["env"] = ST.environment()
    ST.write_json(os.path.join(ST.ART, f"train_{out['tag']}.json"), out)
    print(f"[{out['tag']}] done in {out['seconds']/60:.1f} min, degenerate steps "
          f"{out['n_degenerate_steps']}, firewall violations {len(out['firewall']['violations'])}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
