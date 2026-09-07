#!/usr/bin/env python
"""Gate 8D Phase 6: choose the deployment checkpoint and thresholds on KITTI-360 alone.

Gate 8C-1's neighbourhood experiment raised source AUROC from 0.68 to 0.74 while Occ3D IoU
fell from 33.28 to 28.12. Source *ranking* is therefore not a transfer proxy, and this
selector deliberately does not use it. Instead each candidate is scored on **source stress
variants** built only from drive 0006 -- generic perturbations of voxel size, history
length, LiDAR density and metric scale, none of them a copy of any target dataset's
settings -- and combined with the weights preregistered in ``gate8d/protocol.py``.

Runs inside the file-access audit. Nothing here opens a target benchmark.

    python tools/gate8d/selection.py --device cuda:1
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, default_device                               # noqa: E402
from gates.scale_gate.config import write_json                                         # noqa: E402
from gates.gate8 import targets as TG                                                  # noqa: E402
from gates.gate8d import protocol as P, sources as SRC                                 # noqa: E402
from gates.gate8d.net import apply_residual, load_checkpoint                           # noqa: E402
from sscbench_kitti360.audit import FileAudit                                    # noqa: E402

TAUS = np.round(np.arange(-2.0, 2.0001, 0.0625), 4)


def stress_variants():
    """Generic source-side perturbations, expressed as transforms on a loaded sample.

    These are deliberately *not* modelled on any target benchmark's grid or sensor: they
    are ranges justified by KITTI-360 itself, so a checkpoint that survives them is robust
    rather than tuned.
    """
    out = [{"name": "identity"}]
    for k in (2, 3):
        out.append({"name": f"voxel_pool_{k}", "pool": k})
    for h in (5, 10):
        out.append({"name": f"history_{h}", "age_cap": h})
    out.append({"name": "lidar_half", "drop_obs": 0.5})
    out.append({"name": "scale_noise", "scale_log": 0.05})
    out.append({"name": "fov_crop", "fov": 0.9})
    return out


def apply_stress(d, v, rng):
    """Perturb one sample in place-ish; returns (input, gt_occ, gt_valid, base)."""
    x = d["input"].clone()
    occ, val, base = d["gt_occ"], d["gt_valid"], d["base_logodds"]
    if v.get("pool"):
        k = int(v["pool"])
        X, Y, Z = occ.shape
        X2, Y2 = (X // k) * k, (Y // k) * k
        sl = (slice(0, X2), slice(0, Y2), slice(None))
        f = torch.nn.functional.avg_pool3d
        x = f(x[(slice(None),) + sl][None], (k, k, 1))[0]
        occ = (torch.nn.functional.max_pool3d(occ[sl][None, None].float(), (k, k, 1))[0, 0] > 0)
        val = (torch.nn.functional.max_pool3d(val[sl][None, None].float(), (k, k, 1))[0, 0] > 0)
        base = f(base[sl][None, None], (k, k, 1))[0, 0]
    if v.get("age_cap"):
        x[5] = x[5].clamp(max=float(v["age_cap"]) / 100.0)
    if v.get("drop_obs"):
        m = (torch.rand_like(x[2]) < float(v["drop_obs"]))
        x[2] = torch.where(m, torch.zeros_like(x[2]), x[2])
        x[3] = 1.0 - x[2]
        x[0] = torch.where(m, torch.zeros_like(x[0]), x[0])
        base = torch.where(m if base.shape == m.shape else torch.zeros_like(base, dtype=bool),
                           torch.zeros_like(base), base)
    if v.get("scale_log"):
        s = float(np.exp(rng.normal(0.0, v["scale_log"])))
        x[0] = x[0] * s
        base = base * s
    if v.get("fov"):
        X = occ.shape[0]
        cut = int(X * (1.0 - float(v["fov"])))
        if cut:
            x = x[:, cut:]; occ = occ[cut:]; val = val[cut:]; base = base[cut:]
    # two stride-2 downsamples: every spatial extent must stay divisible by 4, so a
    # perturbation that changes the shape is trimmed rather than padded (padding would
    # invent the impossible observed/unobserved state the Gate 8C-1 padding bug came from)
    X, Y, Z = occ.shape
    X4, Y4, Z4 = (X // 4) * 4, (Y // 4) * 4, (Z // 4) * 4
    if (X4, Y4, Z4) != (X, Y, Z):
        sl = (slice(0, X4), slice(0, Y4), slice(0, Z4))
        x = x[(slice(None),) + sl]
        occ, val, base = occ[sl], val[sl], base[sl]
    return x, occ, val, base


def score(ckpt, files, dev, rng):
    comp = load_checkpoint(ckpt, dev)
    variants = stress_variants()
    acc = {v["name"]: np.zeros((len(TAUS), 3), np.int64) for v in variants}
    thick, thick_n = 0.0, 0
    sem_agree, sem_n = 0, 0
    sem_new, sem_new_n = 0, 0
    for f in files:
        with np.load(f) as z:
            d = TG.unpack_sample(z, dev)
        for v in variants:
            x, occ, val, base = apply_stress(d, v, rng)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o, sem_log, dist = comp.net(x[None])
            final = apply_residual(base, o[0, 0].float())
            fv, gv = final[val], (occ > 0)[val]
            for j, t in enumerate(TAUS):
                p = fv >= float(t)
                a = acc[v["name"]]
                a[j, 0] += int((p & gv).sum()); a[j, 1] += int((p & ~gv).sum())
                a[j, 2] += int((~p & gv).sum())
            if v["name"] == "identity":
                # surface thickness proxy: mean predicted distance at true surfaces, which
                # a thick prediction inflates
                m = (occ > 0) & val
                if m.any():
                    thick += float(dist[0, 0][m].mean().item()); thick_n += 1
                # semantic teacher agreement, split observed vs newly completed
                fp, fvd = d["fut_p"], d["fut_valid"]
                obs = d["observed"]
                pred_c = sem_log[0].float().argmax(0)
                tgt_c = fp.argmax(-1)
                ok = fvd & (occ > 0)
                if ok.any():
                    sem_agree += int((pred_c[ok] == tgt_c[ok]).sum()); sem_n += int(ok.sum())
                    nw = ok & ~obs
                    if nw.any():
                        sem_new += int((pred_c[nw] == tgt_c[nw]).sum())
                        sem_new_n += int(nw.sum())
    per = {}
    for name, a in acc.items():
        iou = a[:, 0] / np.maximum(a.sum(1), 1)
        j = int(np.argmax(iou))
        tp, fp_, fn = a[j]
        per[name] = {"best_tau": float(TAUS[j]), "iou": float(iou[j]),
                     "precision": float(tp / max(tp + fp_, 1)),
                     "recall": float(tp / max(tp + fn, 1)),
                     "pred_over_gt": float((tp + fp_) / max(tp + fn, 1))}
    ious = [v["iou"] for v in per.values()]
    ident = per["identity"]
    w = P.SELECTION_SCORE
    vol_pen = abs(ident["pred_over_gt"] - 1.0)
    thickness = thick / max(thick_n, 1)
    agree = sem_agree / max(sem_n, 1)
    new_acc = sem_new / max(sem_new_n, 1)
    total = (w["mean_stress_sc_iou"] * float(np.mean(ious))
             + w["worst_stress_sc_iou"] * float(np.min(ious))
             - w["volume_ratio_penalty"] * vol_pen
             + w["semantic_teacher_agreement"] * agree
             + w["completed_semantic_accuracy"] * new_acc
             - w["surface_thickness_penalty"] * thickness)
    return {"per_variant": per, "mean_stress_iou": float(np.mean(ious)),
            "worst_stress_iou": float(np.min(ious)),
            "identity_tau": ident["best_tau"], "identity_iou": ident["iou"],
            "pred_over_gt": ident["pred_over_gt"],
            "volume_penalty": vol_pen, "surface_thickness_m": thickness,
            "semantic_teacher_agreement": agree,
            "completed_semantic_accuracy": new_acc,
            "selection_score": float(total)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--device", default=default_device())
    a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    rng = np.random.default_rng(0)
    val_dir = os.path.dirname(SRC.sample_path(SRC.VAL_DRIVE, 0))
    files = sorted(glob.glob(os.path.join(val_dir, "*.npz")))[:a.limit]
    assert files, "no Gate 8D validation samples"
    res = {"val_drive": SRC.VAL_DRIVE, "n_val_samples": len(files),
           "protocol_sha256": P.source_digest(), "weights": dict(P.SELECTION_SCORE),
           "rule": ("preregistered weighted score over source stress variants; source "
                    "AUROC is deliberately excluded"),
           "candidates": {}}
    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        for s in P.SEEDS:
            for which in ("best", "last"):
                p = os.path.join(ART, "checkpoints", f"g8d_seed{s}_{which}.pt")
                if not os.path.exists(p):
                    continue
                m = score(p, files, dev, rng)
                m["checkpoint"] = os.path.relpath(p, REPO_ROOT)
                res["candidates"][f"seed{s}_{which}"] = m
                print(f"  seed{s} {which:4s} score {m['selection_score']:.4f}  "
                      f"mean {100*m['mean_stress_iou']:5.2f}  worst "
                      f"{100*m['worst_stress_iou']:5.2f}  tau {m['identity_tau']:+.4f}  "
                      f"pred/GT {m['pred_over_gt']:.2f}  thick {m['surface_thickness_m']:.3f}m",
                      flush=True)
    win = max(res["candidates"], key=lambda k: res["candidates"][k]["selection_score"])
    res["selected"] = win
    res["occupancy_threshold"] = res["candidates"][win]["identity_tau"]
    res["firewall"] = audit.summary()
    write_json(os.path.join(ART, "selection.json"), res)
    print(f"\nSELECTED {win}: tau {res['occupancy_threshold']:+.4f}, "
          f"score {res['candidates'][win]['selection_score']:.4f}, "
          f"firewall violations {len(res['firewall']['violations'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
