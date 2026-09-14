#!/usr/bin/env python
"""Source measurement for R/S: the feature study's evaluator, re-pointed at this study.

Adds one thing the brief asks for beyond the inherited metrics: **agreement with the
original aligned teacher targets** on the drive-0006 anchors, overall and restricted to the
map-unobserved/later-measured population -- the region the occupancy metric is about.

    python tools/teacher_alignment/evaluate_source.py --condition S --seed 0 --device cuda:0
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np, torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.teacher_alignment import study as ST                                   # noqa: E402
from core.datasets import kitti360_splits as SRC, union_vocab as V8               # noqa: E402
from core.datasets.audit import FileAudit                                         # noqa: E402
from core.model.net import load_checkpoint                                        # noqa: E402
from core.supervision import targets as TG                                        # noqa: E402
from tools.feature_study import evaluate_source as ES                             # noqa: E402


@torch.no_grad()
def teacher_agreement(condition, seed, device, limit=None):
    dev = torch.device(device)
    net = load_checkpoint(ST.checkpoint_path(condition, seed), dev).net
    files = sorted(glob.glob(os.path.dirname(SRC.sample_path(SRC.VAL_DRIVE, 0)) + "/*.npz"))
    if limit:
        files = files[:limit]
    tot = {"all": [0, 0], "later": [0, 0], "observed": [0, 0]}
    kl = {"all": 0.0, "later": 0.0}
    conf = torch.zeros(V8.U, V8.U, dtype=torch.long, device=dev)
    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True):
        for p in files:
            idx = int(os.path.basename(p)[:-4])
            with np.load(p) as z:
                d = TG.unpack_sample(z, dev)
            n = d["gt_occ"].numel()
            with np.load(ST.eval_mask_path(SRC.VAL_DRIVE, idx)) as z:
                later = torch.from_numpy(np.unpackbits(z["later_measured"])[:n].astype(bool)).to(dev)
                obs = torch.from_numpy(np.unpackbits(z["map_observed"])[:n].astype(bool)).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, sem_log = net(d["input"][None])
            lg = sem_log[0].float().reshape(V8.U, -1).T
            tp = d["fut_p"].reshape(-1, V8.U); fv = d["fut_valid"].reshape(-1)
            pred = lg.argmax(1); ref = tp.argmax(1)
            logq = torch.log_softmax(lg, 1); t = tp.clamp_min(1e-6)
            klv = (t * (t.log() - logq)).sum(1)
            for name, m in (("all", fv), ("later", fv & later), ("observed", fv & obs)):
                tot[name][0] += int(m.sum()); tot[name][1] += int(((pred == ref) & m).sum())
                if name in kl:
                    kl[name] += float(klv[m].sum())
            m = fv
            conf += torch.bincount(ref[m] * V8.U + pred[m], minlength=V8.U * V8.U).reshape(V8.U, V8.U)
    out = {k: {"n": v[0], "top1_agreement": v[1] / max(v[0], 1)} for k, v in tot.items()}
    out["all"]["mean_kl"] = kl["all"] / max(tot["all"][0], 1)
    out["later"]["mean_kl"] = kl["later"] / max(tot["later"][0], 1)
    c = conf.cpu().numpy()
    inter = np.diag(c); union = c.sum(0) + c.sum(1) - inter
    out["per_class_agreement_iou"] = {str(u): (float(inter[u] / union[u]) if union[u] else None)
                                      for u in range(V8.U)}
    out["class_support_teacher"] = {str(u): int(c.sum(1)[u]) for u in range(V8.U)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", required=True, choices=list(ST.CONDITIONS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    ST.require_data_root()
    out = ES.run(a.condition, a.seed, a.device, a.limit)       # writes source_<tag>.json
    agr = teacher_agreement(a.condition, a.seed, a.device, a.limit)
    p = os.path.join(ST.ART, "source", f"source_{ST.run_tag(a.condition, a.seed)}.json")
    rec = json.load(open(p)); rec["teacher_agreement"] = agr
    ST.write_json(p, rec)
    print(f"[{ST.run_tag(a.condition, a.seed)}] aligned-teacher top-1 agreement: all "
          f"{agr['all']['top1_agreement']:.4f} later {agr['later']['top1_agreement']:.4f} "
          f"observed {agr['observed']['top1_agreement']:.4f}  KL(all) {agr['all']['mean_kl']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
