#!/usr/bin/env python
"""Gate 2 — depth metrics on sequence 08, before and after refinement.

Reports AbsRel, log RMSE, RMSE (m), delta-1, median absolute error and valid-pixel count,
overall and split by ground-truth distance, with per-clip clip-level bootstrap CIs.
"""
from __future__ import annotations

import argparse, csv, os, sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, collect_provenance, load_config, write_json
from gates.scale_gate.scale import bootstrap_ci
from gates.depth_gate.data import DepthRefineDataset, collate
from gates.depth_gate.metrics import binned_metrics, depth_metrics
from gates.depth_gate.models import build_inputs, build_model, refine


def load_head(ckpt_path, dev):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = build_model(ck["arch"], ck["in_ch"], ck["base_channels"], ck["max_log_residual"])
    m.load_state_dict(ck["state_dict"]); m.to(dev).eval()
    return m, ck


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/depth_gate/refine.yaml")
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    scfg = load_config(cfg.data.scale_gate_config)
    dev = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(REPO_ROOT, cfg.experiment.output_dir, "runs", a.run)
    ck_path = os.path.join(run_dir, "best.pt")

    ds = DepthRefineDataset(scfg, cfg, cfg.data.val_sequences, "val", cfg.scale.constant)
    assert ds.sequences == ["08"], f"validation must be sequence 08, got {ds.sequences}"
    dl = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False,
                    num_workers=cfg.train.num_workers, collate_fn=collate)

    if os.path.exists(ck_path):
        model, ck = load_head(ck_path, dev)
        stats, use_rgb = ck["stats"], ck["use_rgb"]
    else:                                          # identity baseline has no checkpoint
        import json
        stats = json.load(open(os.path.join(run_dir, "train.json")))["stats"]
        use_rgb = False
        model = build_model("identity", 5).to(dev).eval()

    Pb, Pr, G, V, clips = [], [], [], [], []
    with torch.no_grad():
        for b in dl:
            bb = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            r = model(build_inputs(bb, stats, use_rgb))
            Pr.append(refine(bb["base_depth"], r).cpu())
            Pb.append(bb["base_depth"].cpu())
            G.append(b["projected_lidar_depth"]); V.append(b["projected_lidar_valid_mask"])
            clips += b["clip_id"]
    Pb, Pr, G, V = torch.cat(Pb), torch.cat(Pr), torch.cat(G), torch.cat(V)
    bins = [tuple(x) for x in cfg.eval.distance_bins_m]

    res = {"base": {"overall": depth_metrics(Pb, G, V),
                    "by_distance": binned_metrics(Pb, G, V, bins)},
           "refined": {"overall": depth_metrics(Pr, G, V),
                       "by_distance": binned_metrics(Pr, G, V, bins)}}

    # per-clip AbsRel and its paired bootstrap
    rows, deltas = [], []
    for cid in sorted(set(clips)):
        idx = [i for i, c in enumerate(clips) if c == cid]
        b0 = depth_metrics(Pb[idx], G[idx], V[idx])
        b1 = depth_metrics(Pr[idx], G[idx], V[idx])
        rows.append({"clip_id": cid, "abs_rel_base": b0["abs_rel"],
                     "abs_rel_refined": b1["abs_rel"],
                     "delta_abs_rel": b1["abs_rel"] - b0["abs_rel"],
                     "delta1_base": b0["delta1"], "delta1_refined": b1["delta1"],
                     "rmse_base": b0["rmse_m"], "rmse_refined": b1["rmse_m"],
                     "n_valid": b0["n_valid"]})
        deltas.append(b1["abs_rel"] - b0["abs_rel"])
    res["abs_rel_delta_bootstrap"] = bootstrap_ci(deltas, cfg.eval.bootstrap_n,
                                                  cfg.eval.bootstrap_seed)
    res["n_clips"] = len(rows)
    res["improved_clip_fraction"] = float(np.mean([d < 0 for d in deltas]))

    out = os.path.join(REPO_ROOT, cfg.experiment.output_dir, "depth_eval")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"per_clip_{a.run}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    write_json(os.path.join(out, f"summary_{a.run}.json"),
               {"provenance": collect_provenance(scfg, "depth_gate.eval_depth"),
                "run": a.run, **res})

    b0, b1 = res["base"]["overall"], res["refined"]["overall"]
    print(f"{a.run}:  AbsRel {b0['abs_rel']:.4f} -> {b1['abs_rel']:.4f} | "
          f"logRMSE {b0['log_rmse']:.4f} -> {b1['log_rmse']:.4f} | "
          f"RMSE {b0['rmse_m']:.3f} -> {b1['rmse_m']:.3f} m | "
          f"d1 {b0['delta1']:.4f} -> {b1['delta1']:.4f} | "
          f"medAE {b0['median_abs_err_m']:.3f} -> {b1['median_abs_err_m']:.3f} m")
    ci = res["abs_rel_delta_bootstrap"]
    print(f"  dAbsRel {ci['mean']:+.4f} [{ci['lo']:+.4f},{ci['hi']:+.4f}]"
          f"{'*' if ci['excludes_zero'] else ''}  improved {res['improved_clip_fraction']:.1%} of clips")
    for k in res["refined"]["by_distance"]:
        x, y = res["base"]["by_distance"][k], res["refined"]["by_distance"][k]
        print(f"  {k:>8}  AbsRel {x['abs_rel']:.4f} -> {y['abs_rel']:.4f}   "
              f"d1 {x['delta1']:.4f} -> {y['delta1']:.4f}   n={y['n_valid']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
