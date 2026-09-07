#!/usr/bin/env python
"""Gate 8C-1: select checkpoint and thresholds per seed, on KITTI-360 drive 0006 only.

Scored directly from the cached validation samples, which carry the full-grid causal
export at each anchor -- so the network sees exactly what a streaming evaluator would hand
it, without rebuilding the map. The rebuilt raw-LiDAR target supplies occupancy and the
valid mask; SSCBench's completion label is never opened.

Three selections, all label-free and all on the source drive:

* **checkpoint** -- highest full-grid occupancy AP over {best, last};
* **occupancy threshold** -- the final-logit threshold maximising full-grid IoU;
* **semantic confidence threshold** -- chosen by *teacher consistency*: on voxels where the
  two halves of the future window agree, the value maximising ``coverage x agreement``
  between the model's top-1 class and the teacher's. No human semantic label is involved.

    python tools/gate8c1/selection.py --device cuda:1
"""
from __future__ import annotations
import argparse, glob, os, sys, time
import numpy as np, torch, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, SEEDS, default_device, sha256                # noqa: E402
from gates.scale_gate.config import write_json                                         # noqa: E402
from gates.gate8 import targets as TG, vocab as V8                                     # noqa: E402
from gates.gate8.net import CompletionUNet, apply_residual, load_checkpoint            # noqa: E402
from gates.gate8a import scores as SC                                                  # noqa: E402
from gates.gate8a.regions import editable_native                                       # noqa: E402
from gates.gate8c1 import sources as SRC                                               # noqa: E402
from sscbench_kitti360.audit import FileAudit                                    # noqa: E402

SEM_TAUS = np.round(np.arange(0.0, 0.96, 0.05), 2)


@torch.no_grad()
def score_checkpoint(ckpt, files, dev, lock=2.0):
    net = load_checkpoint(ckpt, dev)
    acc = {r: SC.ScoreAccumulator() for r in ("full", "edit")}
    sem = {float(t): [0, 0] for t in SEM_TAUS}       # [n_covered, n_agree]
    for k, p in enumerate(files):
        with np.load(p) as z:
            d = TG.unpack_sample(z, dev)
            agree_half = torch.from_numpy(
                np.unpackbits(z["fut_half_agree"])[:d["gt_occ"].numel()].astype(bool)).to(dev)
        X, Y, Z = d["gt_occ"].shape
        occ_res, sem_log = net.net(d["input"][None])
        final = apply_residual(d["base_logodds"], occ_res[0, 0].float()).reshape(-1)
        valid = d["gt_valid"].reshape(-1)
        gt = (d["gt_occ"] > 0).reshape(-1)
        edit = editable_native(d["base_logodds"], lock).reshape(-1)
        acc["full"].add(final, gt, valid, os.path.basename(p)[:-4], "0006")
        acc["edit"].add(final, gt, valid & edit, os.path.basename(p)[:-4], "0006")
        # ---- teacher-consistency curve for the semantic threshold -------------------
        fv = d["fut_valid"].reshape(-1)
        m = fv & agree_half.reshape(-1)
        if m.any():
            tp = d["fut_p"].reshape(-1, V8.U)[m]
            sp = torch.softmax(sem_log[0].reshape(V8.U, -1).T[m], dim=1)
            conf, arg = sp.max(1)
            ok = arg == tp.argmax(1)
            for t in SEM_TAUS:
                sel = conf >= float(t)
                sem[float(t)][0] += int(sel.sum()); sem[float(t)][1] += int((sel & ok).sum())
    blk = {r: a.block() for r, a in acc.items()}
    out = {}
    for r, b in blk.items():
        sw = SC.sweep(b["pos"], b["neg"])
        j = int(np.argmax(sw["iou"]))
        out[r] = {"average_precision": SC.average_precision(sw), "auroc": SC.auroc(sw),
                  "prevalence": float(sw["n_pos"] / sw["n_tot"]),
                  "best_threshold": float(sw["tau"][j]), "best_iou": float(sw["iou"][j]),
                  "at_zero": SC.at_threshold(sw, 0.0)}
        out[r]["ap_over_prevalence"] = out[r]["average_precision"] / max(out[r]["prevalence"], 1e-9)
    n_tot = max(sum(v[0] for v in sem.values()), 1)
    curve = []
    for t in SEM_TAUS:
        cov_n, agr_n = sem[float(t)]
        cov = cov_n / max(sem[0.0][0], 1)
        agr = agr_n / max(cov_n, 1)
        curve.append({"tau": float(t), "coverage": cov, "agreement": agr, "score": cov * agr})
    best_sem = max(curve, key=lambda x: x["score"])
    out["semantic_consistency_curve"] = curve
    out["semantic_threshold"] = best_sem
    return out, blk


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prefix", default="seed",
                    help="checkpoint name prefix, e.g. 'padfix_seed' for the boundary fix")
    ap.add_argument("--out", default="selection.json",
                    help="filename under artifacts/gate8c1/ for the selection record")
    a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev); t0 = time.time()
    val_dir = os.path.dirname(SRC.sample_path(SRC.VAL_DRIVE, 0))
    files = sorted(glob.glob(os.path.join(val_dir, "*.npz")))[: a.limit]
    assert files, "no KITTI-360 drive-0006 validation samples"
    res = {"val_drive": SRC.VAL_DRIVE, "n_val_samples": len(files), "seeds": {},
           "rule": {"checkpoint": "highest full-grid occupancy AP on KITTI-360 drive 0006",
                    "occupancy_threshold": "final-logit threshold maximising full-grid IoU "
                                           "on drive 0006",
                    "semantic_threshold": "maximises coverage x agreement with the frozen "
                                          "teacher on voxels where both halves of the future "
                                          "window agree; no human semantic label",
                    "forbidden": "SemanticKITTI and Occ3D/nuScenes are not read"}}
    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        for s in SEEDS:
            cands = {}
            for which in ("best", "last"):
                p = os.path.join(ART, "checkpoints", f"{a.prefix}{s}_{which}.pt")
                if not os.path.exists(p):
                    continue
                m, blk = score_checkpoint(p, files, dev)
                np.savez_compressed(
                    os.path.join(ART, f"srcscores_{a.prefix}{s}_{which}.npz"),
                                    **{f"{r}_{k}": v for r, b in blk.items() for k, v in b.items()})
                cands[which] = dict(m, checkpoint=os.path.relpath(p, REPO_ROOT),
                                    checkpoint_sha256=sha256(p))
                print(f"  seed{s} {which:4s} AP {m['full']['average_precision']:.4f} "
                      f"(prev {m['full']['prevalence']:.4f}, x{m['full']['ap_over_prevalence']:.2f}) "
                      f"AUROC {m['full']['auroc']:.4f} bestIoU {m['full']['best_iou']:.4f}"
                      f"@{m['full']['best_threshold']:+.3f} sem_tau "
                      f"{m['semantic_threshold']['tau']:.2f} "
                      f"(agree {m['semantic_threshold']['agreement']:.3f})", flush=True)
            assert cands, f"no checkpoints for seed {s}"
            win = max(cands, key=lambda k: cands[k]["full"]["average_precision"])
            res["seeds"][str(s)] = {
                "candidates": cands, "selected": win,
                "checkpoint": cands[win]["checkpoint"],
                "checkpoint_sha256": cands[win]["checkpoint_sha256"],
                "source_ap": cands[win]["full"]["average_precision"],
                "source_ap_over_prevalence": cands[win]["full"]["ap_over_prevalence"],
                "source_auroc": cands[win]["full"]["auroc"],
                "occupancy_threshold": cands[win]["full"]["best_threshold"],
                "source_iou_at_threshold": cands[win]["full"]["best_iou"],
                "semantic_threshold": cands[win]["semantic_threshold"]["tau"],
                "semantic_agreement_at_threshold": cands[win]["semantic_threshold"]["agreement"],
                "semantic_coverage_at_threshold": cands[win]["semantic_threshold"]["coverage"]}
            print(f"  seed{s} SELECTED {win}: tau {res['seeds'][str(s)]['occupancy_threshold']:+.4f}, "
                  f"sem_tau {res['seeds'][str(s)]['semantic_threshold']:.2f}", flush=True)
    res["firewall"] = dict(audit.summary(),
                           note="selection ran inside a file-access audit; opening a "
                                "SemanticKITTI / Occ3D / SSCBench-label path would have raised")
    res["seconds"] = time.time() - t0
    write_json(os.path.join(ART, a.out), res)
    print(f"selection done in {(time.time()-t0)/60:.1f} min, firewall violations "
          f"{len(res['firewall']['violations'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
