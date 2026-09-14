#!/usr/bin/env python
"""Source measurement on KITTI-360 drive 0006: the map-observability partition, and the
feature diagnostics.

Scored from the cached validation samples, which carry the full-grid causal export at each
anchor, so the network sees exactly what a streaming evaluator would hand it.

Three scored populations, all intersected with the rebuilt target's ``valid`` mask:

* ``full``      -- every supervised voxel (the gate's own region, and the region its
                   threshold rule is defined on);
* ``observed``  -- the causal map reached it;
* ``later``     -- **the primary**: the causal map did not, and a future sweep measured it
                   occupied or free. Built by ``build_eval_masks.py`` before any prediction
                   exists, from measurement only -- sky-rule inference and cross-sweep
                   conflicts are excluded.

The occupancy threshold is the gate's: the final-logit threshold maximising ``full``-grid
IoU on this drive, over the frozen 1 024-bin candidate grid of ``core.evaluation.scores``.
It is chosen here and frozen into the manifest; every number in the partition is then
reported at it.

    python tools/feature_study/evaluate_source.py --condition C --seed 0 --device cuda:0
"""
from __future__ import annotations
import argparse, glob, os, sys, time
import numpy as np, torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# study first: it pins G8C1_DATA_ROOT, which core.datasets reads at import time
from tools.feature_study import study as ST                                       # noqa: E402
from core.datasets.config import REPO_ROOT                                        # noqa: E402
from core.datasets import kitti360_splits as SRC, union_vocab as V8               # noqa: E402
from core.datasets.audit import FileAudit                                         # noqa: E402
from core.evaluation import scores as SC                                          # noqa: E402
from core.model.net import apply_residual, load_checkpoint                        # noqa: E402
from core.supervision import targets as TG                                        # noqa: E402
from tools.feature_study import feature_targets as FT                             # noqa: E402
from tools.feature_study.losses import feature_diagnostics, features_at           # noqa: E402

SEM_TAUS = np.round(np.arange(0.0, 0.96, 0.05), 2)
REGIONS = ("full", "observed", "later")


def train_mean_feature(device) -> torch.Tensor:
    """Mean unit target feature over the training drives -- the collapse reference.

    Cached, because it is a property of the frozen targets and must be identical for every
    condition and seed. Every 16th training anchor contributes; the mean of ~90 000 000
    unit vectors is not sensitive to the stride.
    """
    p = os.path.join(ST.ART, "train_mean_feature.npz")
    if os.path.exists(p):
        with np.load(p) as z:
            return torch.from_numpy(z["mean"]).to(device)
    acc = np.zeros(ST.FEATURE_DIM, np.float64); n = 0
    files = []
    for d in SRC.TRAIN_DRIVES:
        files += sorted(glob.glob(os.path.dirname(ST.feature_target_path(d, 0)) + "/*.npz"))
    for f in files[::16]:
        a = FT.load(f)
        acc += a["feat"].astype(np.float64).sum(0); n += a["feat"].shape[0]
    mean = (acc / max(n, 1)).astype(np.float32)
    os.makedirs(ST.ART, exist_ok=True)
    np.savez(p, mean=mean, n_vectors=np.int64(n), n_files=np.int64(len(files[::16])))
    return torch.from_numpy(mean).to(device)


@torch.no_grad()
def run(condition: str, seed: int, device: str, limit=None):
    dev = torch.device(device); torch.cuda.set_device(dev)
    ckpt = ST.checkpoint_path(condition, seed)
    comp = load_checkpoint(ckpt, dev)
    net = comp.net
    t0 = time.time()
    files = sorted(glob.glob(os.path.dirname(SRC.sample_path(SRC.VAL_DRIVE, 0)) + "/*.npz"))
    if limit:
        files = files[:limit]
    mean_feat = train_mean_feature(dev)

    acc = {r: SC.ScoreAccumulator() for r in REGIONS}
    part = {"n_later_occupied": 0, "n_later_free": 0, "n_observed": 0, "n_still_unknown": 0}
    sem = {float(t): [0, 0] for t in SEM_TAUS}
    fd, n_aux, n_missing_mask = [], 0, 0
    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        for p in files:
            idx = int(os.path.basename(p)[:-4])
            mp = ST.eval_mask_path(SRC.VAL_DRIVE, idx)
            if not os.path.exists(mp):
                n_missing_mask += 1
                continue
            with np.load(p) as z:
                d = TG.unpack_sample(z, dev)
            n = d["gt_occ"].numel()
            with np.load(mp) as z:
                def um(k):
                    return torch.from_numpy(
                        np.unpackbits(z[k])[:n].astype(bool)).to(dev)
                later = um("later_measured"); obs_m = um("map_observed")
                m_occ = um("measured_occ"); m_free = um("measured_free")
            # bf16 autocast: the adopted inference convention (core.model.net.Completer.raw),
            # so the source measurement and the transfer evaluation see the same arithmetic
            with torch.autocast("cuda", dtype=torch.bfloat16):
                occ_res, sem_log, trunk = net(d["input"][None], return_trunk=True)
            sem_log = sem_log.float()
            final = apply_residual(d["base_logodds"], occ_res[0, 0].float()).reshape(-1)
            valid = d["gt_valid"].reshape(-1)
            gt = (d["gt_occ"] > 0).reshape(-1)
            reg = {"full": valid, "observed": valid & obs_m, "later": valid & later}
            cid = os.path.basename(p)[:-4]
            for r, mk in reg.items():
                acc[r].add(final, gt, mk, cid, "0006")
            part["n_later_occupied"] += int((later & m_occ).sum())
            part["n_later_free"] += int((later & m_free).sum())
            part["n_observed"] += int(reg["observed"].sum())
            part["n_still_unknown"] += int((valid & ~obs_m & ~later).sum())
            # ---- feature diagnostics on held-out source data -------------------------
            fp = ST.feature_target_path(SRC.VAL_DRIVE, idx)
            if net.head_feat is not None and os.path.exists(fp):
                a = FT.load(fp)
                if len(a["feat_rows"]):
                    ix = torch.from_numpy(a["feat_rows"]).to(dev)
                    pr = features_at(net, trunk, ix)
                    tg = torch.from_numpy(a["feat"]).to(dev)
                    fd.append(feature_diagnostics(pr, tg, mean_feat))
                    n_aux += int(len(a["feat_rows"]))
            # ---- teacher-consistency curve (the gate's semantic-threshold rule) -------
            if condition in getattr(ST, "SEMANTIC_CONDITIONS", ("B",)):
                with np.load(p) as z:
                    ag = torch.from_numpy(
                        np.unpackbits(z["fut_half_agree"])[:n].astype(bool)).to(dev)
                mk = d["fut_valid"].reshape(-1) & ag
                if mk.any():
                    tp_ = d["fut_p"].reshape(-1, V8.U)[mk]
                    sp = torch.softmax(sem_log[0].reshape(V8.U, -1).T[mk], dim=1)
                    conf, arg = sp.max(1)
                    ok = arg == tp_.argmax(1)
                    for t in SEM_TAUS:
                        s = conf >= float(t)
                        sem[float(t)][0] += int(s.sum()); sem[float(t)][1] += int((s & ok).sum())

    os.makedirs(os.path.join(ST.ART, "source"), exist_ok=True)
    blk = {r: a.block() for r, a in acc.items()}
    sw = {r: SC.sweep(b["pos"], b["neg"]) for r, b in blk.items()}
    j = int(np.argmax(sw["full"]["iou"]))
    tau = float(sw["full"]["tau"][j])                 # the gate's rule, on the gate's region
    out = {"condition": condition, "seed": seed, "tag": ST.run_tag(condition, seed),
           "checkpoint": os.path.relpath(ckpt, ST.REPO_ROOT),
           "checkpoint_sha256": ST.sha256_file(ckpt),
           "val_drive": SRC.VAL_DRIVE, "n_val_samples": len(files),
           "n_missing_eval_mask": n_missing_mask,
           "occupancy_threshold": tau,
           "threshold_rule": "final-logit threshold maximising full-grid IoU on KITTI-360 "
                             "drive 0006, over the frozen 1024-bin candidate grid",
           "partition_counts": part, "regions": {}}
    for r in REGIONS:
        s = sw[r]
        at = SC.at_threshold(s, tau)
        n_tot, n_pos = float(s["n_tot"]), float(s["n_pos"])
        at["tn"] = int(n_tot - n_pos - at["fp"])
        out["regions"][r] = {
            "n_voxels": int(n_tot), "n_occupied": int(n_pos),
            "prevalence": n_pos / max(n_tot, 1),
            "average_precision": SC.average_precision(s), "auroc": SC.auroc(s),
            "at_threshold": at,
            "best_iou": {"threshold": float(s["tau"][int(np.argmax(s["iou"]))]),
                         "iou": float(np.max(s["iou"]))}}
        np.savez_compressed(os.path.join(ST.ART, "source",
                                         f"clipcounts_{ST.run_tag(condition, seed)}_{r}.npz"),
                            counts=SC.per_clip_counts(blk[r], tau),
                            clip_id=blk[r]["clip_id"], n_scored=blk[r]["n_scored"])
    if fd:
        w = np.asarray([x["n"] for x in fd], np.float64)
        out["feature_diagnostics"] = {
            k: float(np.average([x[k] for x in fd], weights=w))
            for k in ("cosine_mean", "cosine_median", "cosine_mean_predictor",
                      "pred_dispersion", "target_dispersion")}
        out["feature_diagnostics"]["n_voxels"] = int(n_aux)
        out["feature_diagnostics"]["head_trained"] = (condition == "C")
    if condition in getattr(ST, "SEMANTIC_CONDITIONS", ("B",)):
        curve = [{"tau": float(t), "coverage": sem[float(t)][0] / max(sem[0.0][0], 1),
                  "agreement": sem[float(t)][1] / max(sem[float(t)][0], 1)} for t in SEM_TAUS]
        for c in curve:
            c["score"] = c["coverage"] * c["agreement"]
        best = max(curve, key=lambda x: x["score"])
        out["semantic_threshold"] = best["tau"]
        out["semantic_consistency"] = {"curve": curve, "selected": best}
    else:
        out["semantic_threshold"] = None
        out["semantic_note"] = ("the 25-class head receives no gradient in this condition, "
                                "so it is at initialisation and no semantic threshold is "
                                "meaningful; the scoring path does not consume this field")
    out["seconds"] = time.time() - t0
    out["firewall"] = dict(audit.summary())
    out["env"] = ST.environment()
    ST.write_json(os.path.join(ST.ART, "source", f"source_{ST.run_tag(condition, seed)}.json"),
                  out)
    lr = out["regions"]["later"]["at_threshold"]; fu = out["regions"]["full"]["at_threshold"]
    print(f"[{ST.run_tag(condition, seed)}] tau {tau:+.4f} | full IoU {fu['iou']:.4f} "
          f"| LATER IoU {lr['iou']:.4f} P {lr['precision']:.4f} R {lr['recall']:.4f} "
          f"| obs IoU {out['regions']['observed']['at_threshold']['iou']:.4f} "
          f"| AP/prev {out['regions']['later']['average_precision']/max(out['regions']['later']['prevalence'],1e-9):.2f}"
          + (f" | cos {out['feature_diagnostics']['cosine_mean']:.4f} "
             f"(mean-pred {out['feature_diagnostics']['cosine_mean_predictor']:.4f})"
             if "feature_diagnostics" in out else ""), flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", required=True, choices=list(ST.CONDITIONS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    ST.require_data_root()
    run(a.condition, a.seed, a.device, a.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
