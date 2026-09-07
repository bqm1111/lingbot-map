#!/usr/bin/env python
"""Gate 3.1 step 3 — freeze the two deterministic controls on source data only.

    V1_clean  strongest deterministic control by mean IoU on sequences 09-10
    V2_clean  the candidate whose mean occupied count on sequences 09-10 is closest to
              the reference-seed clean V3's mean occupied count **on 09-10**

Neither uses a sequence-08 statistic. This replaces the Gate-3 V2, which was matched to
V3's sequence-08 occupied count and is retained in the report only as a non-clean
historical diagnostic.

    python tools/voxel_gate_validation/select_controls.py --run full_s0
"""
from __future__ import annotations

import argparse, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, load_config, write_json
from gates.voxel_gate.controls import control, enumerate_controls
from gates.voxel_gate.models import VoxelCorrector3D, apply_region
from gates.voxel_gate_validation.data import sample, scores, select_clips


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/voxel_gate_validation/clean_infill.yaml")
    ap.add_argument("--run", default="full_s0", help="reference-seed clean V3 run")
    a = ap.parse_args()
    cfg = load_config(a.config)
    dev = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    radius = int(cfg.region.radius)

    sel = select_clips(cfg, "c3", "train", cfg.data.source_select_sequences)
    for c in sel:
        assert c["sequence"] not in cfg.data.val_sequences, "sequence 08 reached selection"
    print(f"source-selection clips: {len(sel)} (sequences "
          f"{cfg.data.source_select_sequences})")

    run_dir = os.path.join(REPO_ROOT, cfg.experiment.output_dir, "runs", a.run)
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu", weights_only=False)
    model = VoxelCorrector3D(6, ck["channels"], ck["n_blocks"], ck["kernel"])
    model.load_state_dict(ck["state_dict"]); model.to(dev).eval()
    tau, norm = float(ck["threshold"]), ck["norm"]

    cands = enumerate_controls(cfg)
    per = {k: {"iou": [], "precision": [], "recall": [], "n_pred": []} for k in cands}
    base = {"iou": [], "n_pred": []}
    v3 = {"iou": [], "n_pred": []}
    for i, c in enumerate(sel):
        s = sample(cfg, "c3", c["clip_id"], radius, norm, dev)
        occ, R, keep, gt = s["occupied"], s["R_infer"], s["keep"], s["gt"]
        b = scores(occ, gt, keep)
        base["iou"].append(b["iou"]); base["n_pred"].append(b["n_pred_occupied"])
        with torch.no_grad():
            p = torch.sigmoid(model(s["x"][None])[0, 0])
        sv = scores(apply_region(p >= tau, occ, R), gt, keep)
        v3["iou"].append(sv["iou"]); v3["n_pred"].append(sv["n_pred_occupied"])
        for k, kw in cands.items():
            sc = scores(control(occ, R, **kw), gt, keep)
            per[k]["iou"].append(sc["iou"]); per[k]["precision"].append(sc["precision"])
            per[k]["recall"].append(sc["recall"]); per[k]["n_pred"].append(sc["n_pred_occupied"])
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(sel)}", flush=True)

    agg = {k: {f: float(np.mean(v)) for f, v in d.items()} for k, d in per.items()}
    v1 = max(agg, key=lambda k: agg[k]["iou"])
    v3_count = float(np.mean(v3["n_pred"]))
    v2 = min(agg, key=lambda k: abs(agg[k]["n_pred"] - v3_count))

    print(f"\nsource-select C3 base IoU {np.mean(base['iou']):.4f} "
          f"({np.mean(base['n_pred']):.0f} occ)")
    print(f"source-select clean V3 IoU {np.mean(v3['iou']):.4f} ({v3_count:.0f} occ)")
    print(f"{'control':16s} {'IoU':>8} {'P':>7} {'R':>7} {'occupied':>9}")
    for k in sorted(agg, key=lambda k: -agg[k]["iou"])[:8]:
        m = ("  <-- V1_clean" if k == v1 else "") + ("  <-- V2_clean" if k == v2 else "")
        v = agg[k]
        print(f"{k:16s} {v['iou']:8.4f} {v['precision']:7.3f} {v['recall']:7.3f} "
              f"{v['n_pred']:9.0f}{m}")
    if v2 != v1:
        v = agg[v2]
        print(f"{v2:16s} {v['iou']:8.4f} {v['precision']:7.3f} {v['recall']:7.3f} "
              f"{v['n_pred']:9.0f}  <-- V2_clean (count-matched on 09-10)")

    out = os.path.join(REPO_ROOT, cfg.experiment.output_dir, "control_selection.json")
    write_json(out, {"radius": radius, "reference_run": a.run,
                     "source_select_sequences": list(cfg.data.source_select_sequences),
                     "selection_used_sequence_08": False,
                     "source_select_c3_iou": float(np.mean(base["iou"])),
                     "source_select_c3_n_pred": float(np.mean(base["n_pred"])),
                     "source_select_v3_iou": float(np.mean(v3["iou"])),
                     "source_select_v3_n_pred": v3_count,
                     "v1_clean": {"name": v1, **cands[v1], **agg[v1]},
                     "v2_clean": {"name": v2, **cands[v2], **agg[v2]},
                     "control_table": agg})
    print(f"\nwrote {os.path.relpath(out, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
