#!/usr/bin/env python
"""Gate 3 step 2 — freeze the correction-band radius and the deterministic controls.

Both decisions are made on **source data only**. Sequence 08 is never loaded here; the
tool refuses to run if a sequence-08 clip reaches it.

    radius   chosen on the source-*training* sequences (00-07) by a model-free rule:
             the smallest candidate whose in-band oracle IoU reaches 90 % of the best
             candidate's. "In-band oracle" is the IoU a perfect corrector confined to the
             band would score: the occupancy of (LiDAR-occupied AND band). Preferring the
             smallest sufficient band is the anti-densification choice -- a wider band
             gives any method, learned or deterministic, more room to inflate volume.
             The visible-ceiling-restricted variant is reported alongside as a diagnostic.

    control  chosen on the source-*selection* sequences (09, 10), the same split Gate 2
             used for early stopping, by best mean IoU over the frozen candidate grid.

    python tools/voxel_gate/select_region.py
"""
from __future__ import annotations

import argparse, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, load_config, write_json
from gates.voxel_gate.controls import control, enumerate_controls
from gates.voxel_gate.data import sample, scores, select_clips
from gates.voxel_gate.voxels import correction_region

FRACTION_OF_BEST = 0.90


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/voxel_gate/visible_correction.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    dev = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    train = select_clips(cfg, "train", cfg.data.source_train_sequences)
    sel = select_clips(cfg, "train", cfg.data.source_select_sequences)
    for c in train + sel:
        assert c["sequence"] not in cfg.data.val_sequences, \
            f"sequence-08 clip {c['clip_id']} reached the selection tool"
    print(f"source train {len(train)} clips, source select {len(sel)} clips")

    # ---- 1. band radius, on source TRAIN ----------------------------------- #
    radii = [int(r) for r in cfg.region.candidate_radii]
    table = {r: {"oracle_iou": [], "oracle_iou_vc_restricted": [], "c3_iou": [],
                 "vc_iou": [], "region_frac": [], "gt_in_region": [],
                 "vc_in_region": []} for r in radii}
    for i, c in enumerate(train):
        s = sample(cfg, c["clip_id"], 0, None, dev)
        occ, keep, gt, vc = s["occupied"], s["keep"], s["gt"], s["vc"]
        for r in radii:
            R = correction_region(occ, r) & keep
            t = table[r]
            t["oracle_iou"].append(scores(gt & R, gt, keep)["iou"])
            t["oracle_iou_vc_restricted"].append(scores(gt & vc & R, gt, keep)["iou"])
            t["c3_iou"].append(scores(occ, gt, keep)["iou"])
            t["vc_iou"].append(scores(vc, gt, keep)["iou"])
            t["region_frac"].append(float(R.sum()) / R.numel())
            t["gt_in_region"].append(float((gt & keep & R).sum()) / max(float((gt & keep).sum()), 1))
            t["vc_in_region"].append(float((vc & R).sum()) / max(float(vc.sum()), 1))
        if (i + 1) % 100 == 0:
            print(f"  radius scan {i+1}/{len(train)}", flush=True)
    agg = {r: {k: float(np.mean(v)) for k, v in t.items()} for r, t in table.items()}
    best = max(agg[r]["oracle_iou"] for r in radii)
    radius = min(r for r in radii if agg[r]["oracle_iou"] >= FRACTION_OF_BEST * best)
    print(f"\n{'radius':>7} {'|R|/grid':>9} {'GT in R':>8} {'VC in R':>8} "
          f"{'oracle IoU':>11} {'oracle|VC':>10}")
    for r in radii:
        m = "  <-- selected" if r == radius else ""
        print(f"{r:>7} {agg[r]['region_frac']:9.4f} {agg[r]['gt_in_region']:8.3f} "
              f"{agg[r]['vc_in_region']:8.3f} {agg[r]['oracle_iou']:11.4f} "
              f"{agg[r]['oracle_iou_vc_restricted']:10.4f}{m}")
    print(f"source-train C3 IoU {agg[radii[0]]['c3_iou']:.4f}, "
          f"full visible ceiling {agg[radii[0]]['vc_iou']:.4f}")

    # ---- 2. deterministic controls, on source SELECT ------------------------ #
    cands = enumerate_controls(cfg)
    per = {k: {"iou": [], "precision": [], "recall": [], "n_pred": []} for k in cands}
    base = {"iou": [], "n_pred": []}
    for i, c in enumerate(sel):
        s = sample(cfg, c["clip_id"], radius, None, dev)
        occ, R, keep, gt = s["occupied"], s["region"], s["keep"], s["gt"]
        b = scores(occ, gt, keep)
        base["iou"].append(b["iou"]); base["n_pred"].append(b["n_pred_occupied"])
        for k, kw in cands.items():
            sc = scores(control(occ, R, **kw), gt, keep)
            for f, g in (("iou", "iou"), ("precision", "precision"),
                         ("recall", "recall"), ("n_pred", "n_pred_occupied")):
                per[k][f].append(sc[g])
        if (i + 1) % 40 == 0:
            print(f"  control scan {i+1}/{len(sel)}", flush=True)
    cagg = {k: {f: float(np.mean(v)) for f, v in d.items()} for k, d in per.items()}
    best_ctrl = max(cagg, key=lambda k: cagg[k]["iou"])
    print(f"\nsource-select C3 baseline IoU {np.mean(base['iou']):.4f} "
          f"({np.mean(base['n_pred']):.0f} occupied)")
    print(f"{'control':16s} {'IoU':>8} {'P':>7} {'R':>7} {'occupied':>9}")
    for k in sorted(cagg, key=lambda k: -cagg[k]["iou"])[:10]:
        m = "  <-- selected" if k == best_ctrl else ""
        v = cagg[k]
        print(f"{k:16s} {v['iou']:8.4f} {v['precision']:7.3f} {v['recall']:7.3f} "
              f"{v['n_pred']:9.0f}{m}")

    out = os.path.join(REPO_ROOT, cfg.experiment.output_dir, "region_selection.json")
    write_json(out, {"rule_radius": f"smallest radius with in-band oracle IoU >= "
                                    f"{FRACTION_OF_BEST} * best",
                     "selected_radius": radius, "radius_table": agg,
                     "source_train_clips": len(train), "source_select_clips": len(sel),
                     "source_train_sequences": list(cfg.data.source_train_sequences),
                     "source_select_sequences": list(cfg.data.source_select_sequences),
                     "selected_control": best_ctrl, "control_params": cands[best_ctrl],
                     "control_table": cagg,
                     "source_select_c3_iou": float(np.mean(base["iou"])),
                     "source_select_c3_n_pred": float(np.mean(base["n_pred"]))})
    print(f"\nwrote {os.path.relpath(out, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
