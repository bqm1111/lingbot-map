#!/usr/bin/env python
"""Fix ``rho`` -- the auxiliary-to-occupancy trunk-gradient ratio -- before any run.

The brief prefers the pooled median ratio from historical B's own training logs. Those logs
exist (``feature_completion_study/train_fcs_B_seed*.json``, ``aux_grad_norm`` /
``occ_grad_norm`` every 250 steps, with the 0.5 coefficient included) but they were taken
with respect to the **decoder activation** the heads read, not with respect to the shared
parameters. That is a different vector space, so under the brief's own rule they are not
compatible and are not used. This tool takes the fallback: historical B seed 0's final
checkpoint, 32 deterministic source-training minibatches, no parameter updates,

    rho = median_i  ||d(0.5 L_KL)/d theta_shared||_i / ||d L_occ/d theta_shared||_i

with theta_shared the encoder/decoder parameters excluding all output heads -- the same set
``train.py`` matches on. Runs inside the file-access audit.

    python tools/teacher_alignment/calibrate_rho.py --device cuda:0
"""
from __future__ import annotations
import argparse, os, random, sys, time
import numpy as np, torch, yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.teacher_alignment import study as ST                                   # noqa: E402
from core.datasets.audit import FileAudit                                         # noqa: E402
from core.datasets import kitti360_splits as SRC                                  # noqa: E402
from core.model.net import load_checkpoint                                        # noqa: E402
from tools.feature_study.samples import StudySamples                              # noqa: E402
from tools.teacher_alignment import gradmatch as GM                               # noqa: E402
from tools.teacher_alignment import shuffle as SH                                 # noqa: E402

N_BATCHES = 32
SAMPLER_SEED = 20260911          # only the calibration's minibatch draws; never a training seed
BASE_CONFIG = os.path.join(ST.REPO_ROOT, "configs", "feature_study", "base.yaml")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    ST.require_data_root()
    if os.path.exists(ST.rho_path()):
        raise SystemExit(f"{ST.rho_path()} exists; rho is frozen. Delete it to recalibrate.")
    cfg = yaml.safe_load(open(BASE_CONFIG))
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    t0 = time.time()
    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        net = load_checkpoint(ST.HISTORICAL_B_SEED0, dev).net
        net.train()                                       # GroupNorm has no running stats;
        groups = GM.param_groups(net)                     # train-mode forward as in training
        train = StudySamples(cfg["train_drives"], cfg["crop"], dev, SAMPLER_SEED, None,
                             sampler=cfg["sampler"], centre_on_occ_p=cfg["centre_on_occ_p"],
                             want_features=False)
        train.verify()
        rows = []
        for i in range(N_BATCHES):
            idx = [train.draw() for _ in range(cfg["batch_size"])]
            b = train.batch(idx)
            # the common auxiliary mask R and S will train on (tiny strata dropped)
            keep = torch.stack([SH.eligible_mask(b["fut_valid"][k], b["observed"][k])[0]
                                for k in range(b["fut_valid"].shape[0])])
            l_occ, l_kl, m = GM.losses(net, b, cfg, b["fut_p"], keep)
            tg = GM.task_gradients(l_occ, l_kl, groups, cfg["w_aux"])
            go, ga = tg["shared"]
            n_o, n_a = GM._norm(go), GM._norm(ga)
            rows.append({"batch": i, "files": [int(x) for x in idx],
                         "crops": train.trace[-len(idx):],
                         "occ_grad_norm": n_o, "aux_grad_norm": n_a,
                         "ratio": n_a / max(n_o, 1e-12),
                         "cosine": GM._dot(go, ga) / max(n_o * n_a, 1e-24),
                         "l_occ": float(l_occ), "l_kl": float(l_kl),
                         "n_aux_voxels": m["n_aux_voxels"]})
            net.zero_grad(set_to_none=True)
    ratios = np.asarray([r["ratio"] for r in rows])
    rho = float(np.median(ratios))
    rec = {"rho": rho, "rule": "pooled median over minibatches of "
                               "||d(0.5*L_KL)/d theta_shared|| / ||d L_occ/d theta_shared||",
           "includes_w_aux_coefficient": True, "w_aux": cfg["w_aux"],
           "shared_parameter_set": "all trainable parameters except head_occ, head_sem, "
                                   "head_feat (n = %d tensors, %d values)"
                                   % (len(groups["shared"]),
                                      sum(p.numel() for p in groups["shared"])),
           "checkpoint": os.path.relpath(ST.HISTORICAL_B_SEED0, ST.REPO_ROOT),
           "checkpoint_sha256": ST.sha256_file(ST.HISTORICAL_B_SEED0),
           "n_batches": N_BATCHES, "batch_size": cfg["batch_size"],
           "sampler_seed": SAMPLER_SEED, "crop": cfg["crop"],
           "why_not_historical_logs": "feature_completion_study/train_fcs_B_seed*.json log "
               "aux_grad_norm/occ_grad_norm w.r.t. the decoder ACTIVATION the heads read, not "
               "w.r.t. the shared parameters; different vector space, not compatible. For the "
               "record their pooled median (0.5 included) was 4.28/4.52/4.22 by seed.",
           "ratio_summary": {"median": rho, "mean": float(ratios.mean()),
                             "min": float(ratios.min()), "max": float(ratios.max()),
                             "q25": float(np.percentile(ratios, 25)),
                             "q75": float(np.percentile(ratios, 75))},
           "cosine_summary": {"median": float(np.median([r["cosine"] for r in rows]))},
           "batches": rows, "seconds": time.time() - t0,
           "firewall": dict(audit.summary()), "env": ST.environment()}
    ST.write_json(ST.rho_path(), rec)
    print(f"rho = {rho:.4f}  (ratio q25 {rec['ratio_summary']['q25']:.3f}, q75 "
          f"{rec['ratio_summary']['q75']:.3f}; cosine median "
          f"{rec['cosine_summary']['median']:+.3f}) over {N_BATCHES} batches, "
          f"{time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
