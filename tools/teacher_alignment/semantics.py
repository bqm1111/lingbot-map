#!/usr/bin/env python
"""Semantic completion on the targets: per-class IoU and support for R and S, and naming
accuracy on the *common* set -- ground-truth occupied voxels that both R and S predicted
occupied -- so the naming comparison is not confounded by which voxels each arm filled.

    python tools/teacher_alignment/semantics.py
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.teacher_alignment import study as ST                                   # noqa: E402
from core.datasets import vocab                                                   # noqa: E402


def per_class(ds, cond, seed):
    j = json.load(open(os.path.join(ST.ART, "transfer", cond, f"eval_{ds}_past5_seed{seed}.json")))
    v = vocab.load(ds)
    p = os.path.join(ST.ART, "transfer", cond, f"counts_{ds}_past5_seed{seed}_completion.npz")
    sup, iou = {}, {}
    if os.path.exists(p):
        with np.load(p) as z:
            tp = z["tp"].sum(0); fn = z["fn"].sum(0); fp = z["fp"].sum(0)
            names = [str(x) for x in z["names"]]
            for k, n in enumerate(names):
                sup[n] = int(tp[k] + fn[k])
                den = tp[k] + fp[k] + fn[k]
                iou[n] = float(tp[k] / den) if den else None
    return {"ssc_miou": j["semantic"]["completion"]["ssc_miou"],
            "per_class_iou": j["semantic"]["completion"]["per_class_iou"],
            "per_class_iou_from_counts": iou, "class_support": sup}


def common_set(ds, seed):
    dR = os.path.join(ST.ART, "transfer", "R", f"semlab_{ds}_seed{seed}")
    dS = os.path.join(ST.ART, "transfer", "S", f"semlab_{ds}_seed{seed}")
    files = sorted(os.path.basename(f) for f in glob.glob(dR + "/*.npz"))
    n_gt = n_common = okR = okS = 0
    n_predR = n_predS = 0
    for f in files:
        with np.load(os.path.join(dR, f)) as zr, np.load(os.path.join(dS, f)) as zs:
            gt_idx = zr["gt_occ_idx"]; gt_lab = zr["gt_occ_label"]
            n_gt += len(gt_idx)
            r = dict(zip(zr["idx"].tolist(), zr["label"].tolist()))
            s_ = dict(zip(zs["idx"].tolist(), zs["label"].tolist()))
            n_predR += len(r); n_predS += len(s_)
            for i, g in zip(gt_idx.tolist(), gt_lab.tolist()):
                lr = r.get(i); ls = s_.get(i)
                if lr is not None and ls is not None:
                    n_common += 1; okR += (lr == g); okS += (ls == g)
    return {"n_gt_occupied": n_gt, "n_common": n_common,
            "coverage": n_common / max(n_gt, 1),
            "naming_accuracy_R": okR / max(n_common, 1), "naming_accuracy_S": okS / max(n_common, 1),
            "n_clips": len(files)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    a = ap.parse_args()
    out = {}
    for ds in ("semantickitti", "occ3d"):
        out[ds] = {}
        for s in ST.SEEDS:
            row = {c: per_class(ds, c, s) for c in ST.CONDITIONS}
            row["common"] = common_set(ds, s)
            out[ds][f"seed{s}"] = row
            c = row["common"]
            print(f"  {ds} seed{s}: mIoU R {row['R']['ssc_miou']:.4f} S {row['S']['ssc_miou']:.4f} | "
                  f"common {c['n_common']}/{c['n_gt_occupied']} ({100*c['coverage']:.1f}%) naming "
                  f"R {c['naming_accuracy_R']:.4f} S {c['naming_accuracy_S']:.4f}")
    ST.write_json(os.path.join(ST.ART, "semantics.json"), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
