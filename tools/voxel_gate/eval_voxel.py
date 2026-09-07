#!/usr/bin/env python
"""Gate 3 step 4 — frozen evaluation of V0/V1/V2/V3/VC on sequence 08.

Nothing is selected here. The band radius, the deterministic control, the model weights
and the probability threshold all arrive frozen from source-only tools. The occupancy
protocol is the frozen one: 163 five-frame clips, stride 5, pixel_stride 1, confidence
1.5, unchanged grid, voxeliser and evaluation mask.

    python tools/voxel_gate/eval_voxel.py --run voxel_cnn3d
"""
from __future__ import annotations

import argparse, csv, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, collect_provenance, load_config, write_json
from gates.scale_gate.scale import bootstrap_ci
from gates.voxel_gate.controls import control, enumerate_controls
from gates.voxel_gate.data import sample, scores, select_clips
from gates.voxel_gate.models import VoxelCorrector3D, apply_region
from gates.voxel_gate.voxels import distance_bins

from prompted_lingbot.occupancy import SEMANTICKITTI_GRID as G

VISIBLE_CEILING = 0.1255          # Gate-1 configuration A, frozen
C3_TARGET, C3_TOL = 0.0778, 5e-4

CONTRASTS = [("V0", "V1"), ("V0", "V2"), ("V0", "V3"), ("V1", "V3"), ("V2", "V3")]


def binned(pred, gt, keep, bins):
    return {k: scores(pred & torch.as_tensor(m, device=pred.device), gt, keep)
            for k, m in bins.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/voxel_gate/visible_correction.yaml")
    ap.add_argument("--run", default="voxel_cnn3d")
    a = ap.parse_args()
    cfg = load_config(a.config)
    dcfg = load_config(cfg.data.depth_gate_config)
    scfg = load_config(dcfg.data.scale_gate_config)   # provenance needs the frozen geometry cfg
    dev = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(REPO_ROOT, cfg.experiment.output_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)

    sel = json.load(open(os.path.join(REPO_ROOT, cfg.experiment.output_dir,
                                      "region_selection.json")))
    radius = int(sel["selected_radius"])
    ctrl_name, ctrl_kw = sel["selected_control"], sel["control_params"]

    run_dir = os.path.join(REPO_ROOT, cfg.experiment.output_dir, "runs", a.run)
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu", weights_only=False)
    assert int(ck["radius"]) == radius, "checkpoint radius != frozen selection"
    model = VoxelCorrector3D(6, ck["channels"], ck["n_blocks"], ck["kernel"])
    model.load_state_dict(ck["state_dict"]); model.to(dev).eval()
    norm, tau = ck["norm"], float(ck["threshold"])
    print(f"V1 = {ctrl_name} {ctrl_kw}   V3 = {a.run} ({ck['n_params']} params) "
          f"tau {tau:.2f}  radius {radius}")

    clips = select_clips(cfg, "val", cfg.data.val_sequences)
    bins = {k: torch.from_numpy(m).to(dev) for k, m in
            distance_bins(tuple(tuple(b) for b in cfg.eval.distance_bins_m)).items()}

    # ---- pass 1: V0, V1, V3 (V2 needs V3's occupied count) ------------------ #
    cands = enumerate_controls(cfg)
    rows, per_bin, extra = [], [], []
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    t_inf = 0.0
    for i, c in enumerate(clips):
        cid = c["clip_id"]
        s = sample(cfg, cid, radius, norm, dev)
        occ, R, keep, gt, vc = s["occupied"], s["region"], s["keep"], s["gt"], s["vc"]
        t0 = time.time()
        with torch.no_grad():
            p = torch.sigmoid(model(s["x"][None])[0, 0])
        t_inf += time.time() - t0
        P = {"V0": occ,
             "V1": control(occ, R, **ctrl_kw),
             "V3": apply_region(p >= tau, occ, R),
             "VC": vc,
             # diagnostic reference rows, not methods: VR is the best any band-restricted
             # method could do (a perfect corrector confined to R); VRV additionally
             # restricts it to the five-frame visible set.
             "VR": gt & R & keep,
             "VRV": gt & vc & R & keep}
        for name, pred in P.items():
            sc = scores(pred, gt, keep)
            added = int((pred & ~occ & keep).sum()); removed = int((~pred & occ & keep).sum())
            rows.append({"clip_id": cid, "config": name, **sc,
                         "added_vs_c3": added, "removed_vs_c3": removed,
                         "n_points": c["n_points"], "n_c3_voxels": c["n_c3_voxels"],
                         "in_region_frac": float((pred & R & keep).sum()) /
                                           max(float((pred & keep).sum()), 1.0)})
            for bn, bm in bins.items():
                per_bin.append({"clip_id": cid, "config": name, "bin": bn,
                                **scores(pred & bm, gt & bm, keep)})
        extra.append({"clip_id": cid,
                      "v3_added_inside_vc": int((P["V3"] & ~occ & vc & keep).sum()),
                      "v3_added_outside_vc": int((P["V3"] & ~occ & ~vc & keep).sum()),
                      "v1_added_inside_vc": int((P["V1"] & ~occ & vc & keep).sum()),
                      "v1_added_outside_vc": int((P["V1"] & ~occ & ~vc & keep).sum())})
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(clips)}", flush=True)

    by = {}
    for r in rows:
        by.setdefault(r["config"], {})[r["clip_id"]] = r
    c3_iou = float(np.mean([v["iou"] for v in by["V0"].values()]))
    if abs(c3_iou - C3_TARGET) > C3_TOL:
        print(f"\nC3_REPRODUCTION_FAILED at evaluation: {c3_iou:.5f} vs {C3_TARGET}")
        return 2
    print(f"\nC3 (V0) reproduced at evaluation: {c3_iou:.5f}")

    # ---- pass 2: V2, the occupied-count-matched deterministic control ------- #
    v3_count = float(np.mean([v["n_pred_occupied"] for v in by["V3"].values()]))
    match = {k: [] for k in cands}
    samples = []
    for c in clips:
        s = sample(cfg, c["clip_id"], radius, None, dev)
        samples.append(s)
        for k, kw in cands.items():
            match[k].append(float((control(s["occupied"], s["region"], **kw)
                                   & s["keep"]).sum()))
    match = {k: float(np.mean(v)) for k, v in match.items()}
    v2_name = min(match, key=lambda k: abs(match[k] - v3_count))
    v2_kw = cands[v2_name]
    print(f"V2 = {v2_name} {v2_kw}: {match[v2_name]:.0f} occupied vs V3 {v3_count:.0f}")
    for s in samples:
        cid, occ, R, keep, gt = (s["clip_id"], s["occupied"], s["region"],
                                 s["keep"], s["gt"])
        pred = control(occ, R, **v2_kw)
        sc = scores(pred, gt, keep)
        rows.append({"clip_id": cid, "config": "V2", **sc,
                     "added_vs_c3": int((pred & ~occ & keep).sum()),
                     "removed_vs_c3": int((~pred & occ & keep).sum()),
                     "n_points": 0, "n_c3_voxels": int(occ.sum()),
                     "in_region_frac": float((pred & R & keep).sum()) /
                                       max(float((pred & keep).sum()), 1.0)})
        for bn, bm in bins.items():
            per_bin.append({"clip_id": cid, "config": "V2", "bin": bn,
                            **scores(pred & bm, gt & bm, keep)})
        by.setdefault("V2", {})[cid] = rows[-1]
    del samples

    # ---- aggregate ---------------------------------------------------------- #
    order = ["V0", "V1", "V2", "V3", "VC", "VR", "VRV"]
    mean = lambda n, k: float(np.mean([v[k] for v in by[n].values()]))
    agg = {n: {k: mean(n, k) for k in
               ("iou", "precision", "recall", "n_pred_occupied", "tp", "fp", "fn",
                "added_vs_c3", "removed_vs_c3", "in_region_frac")} for n in order}
    for n in order:
        agg[n]["n_clips"] = len(by[n])
        agg[n]["iou_p25"], agg[n]["iou_median"], agg[n]["iou_p75"] = [
            float(np.quantile([v["iou"] for v in by[n].values()], q)) for q in (.25, .5, .75)]
    bb = {}
    for r in per_bin:
        bb.setdefault(r["config"], {}).setdefault(r["bin"], []).append(r)
    binned_agg = {n: {b: {k: float(np.mean([x[k] for x in v])) for k in
                          ("iou", "precision", "recall", "n_pred_occupied")}
                      for b, v in d.items()} for n, d in bb.items()}

    def paired(x, y):
        ids = sorted(set(by[x]) & set(by[y]))
        d = [by[y][i]["iou"] - by[x][i]["iou"] for i in ids]
        ci = bootstrap_ci(d, cfg.eval.bootstrap_n, cfg.eval.bootstrap_seed)
        ci["improved_clips"] = int(np.sum([v > 0 for v in d]))
        ci["improved_clip_fraction"] = float(np.mean([v > 0 for v in d]))
        ci["n_pairs"] = len(d)
        return ci
    contrasts = {f"{x}->{y}": paired(x, y) for x, y in CONTRASTS}

    gap = VISIBLE_CEILING - agg["V0"]["iou"]
    recovered = {n: (agg[n]["iou"] - agg["V0"]["iou"]) / gap for n in ("V1", "V2", "V3")}
    band_gap = agg["VR"]["iou"] - agg["V0"]["iou"]
    recovered_band = {n: (agg[n]["iou"] - agg["V0"]["iou"]) / band_gap
                      for n in ("V1", "V2", "V3")}
    ex = {k: float(np.mean([e[k] for e in extra])) for k in extra[0] if k != "clip_id"}

    with open(os.path.join(out_dir, f"per_clip_{a.run}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(out_dir, f"per_bin_{a.run}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_bin[0])); w.writeheader(); w.writerows(per_bin)

    peak = (torch.cuda.max_memory_allocated(dev) / 2**30) if dev.type == "cuda" else 0.0
    write_json(os.path.join(out_dir, f"summary_{a.run}.json"), {
        "provenance": collect_provenance(scfg, "voxel_gate.eval_voxel"),
        "run": a.run, "n_clips": len(clips), "radius": radius,
        "c3_iou_reproduced": c3_iou, "c3_target": C3_TARGET,
        "visible_ceiling": VISIBLE_CEILING,
        "v1_control": {"name": ctrl_name, **ctrl_kw},
        "v2_control": {"name": v2_name, **v2_kw, "mean_occupied": match[v2_name]},
        "v3": {"n_params": int(ck["n_params"]), "threshold": tau,
               "best_epoch": int(ck["epoch"]), "select_iou": float(ck["select_iou"]),
               "inference_seconds_total": t_inf,
               "inference_seconds_per_clip": t_inf / len(clips),
               "peak_gpu_gib_eval": peak},
        "aggregate": agg, "by_distance": binned_agg, "contrasts": contrasts,
        "gap_to_visible_ceiling": gap, "fraction_of_gap_recovered": recovered,
        "gap_to_in_band_oracle": band_gap,
        "fraction_of_in_band_headroom_recovered": recovered_band,
        "vc_diagnostics": ex, "control_count_match": match,
        "grid": {"dims": list(G.dims), "voxel_size": G.voxel_size}})

    print(f"\n{'cfg':4s} {'IoU':>8} {'P':>7} {'R':>7} {'occupied':>9} {'TP':>8} {'FP':>8} "
          f"{'FN':>8} {'added':>8} {'removed':>8}")
    for n in order:
        s = agg[n]
        print(f"{n:4s} {s['iou']:8.4f} {s['precision']:7.3f} {s['recall']:7.3f} "
              f"{s['n_pred_occupied']:9.0f} {s['tp']:8.0f} {s['fp']:8.0f} {s['fn']:8.0f} "
              f"{s['added_vs_c3']:8.0f} {s['removed_vs_c3']:8.0f}")
    print(f"\n{'contrast':10s} {'dIoU':>9}  95% CI               improved")
    for k, v in contrasts.items():
        print(f"{k:10s} {v['mean']:+9.4f}  [{v['lo']:+.4f},{v['hi']:+.4f}]"
              f"{'*' if v['excludes_zero'] else ' '} {v['improved_clips']:4d}/{v['n_pairs']}")
    print("\nfraction of the C3 -> visible-ceiling gap recovered: "
          + ", ".join(f"{k} {v:+.1%}" for k, v in recovered.items()))
    print("fraction of the C3 -> in-band-oracle headroom recovered: "
          + ", ".join(f"{k} {v:+.1%}" for k, v in recovered_band.items()))
    print(f"V3 additions inside/outside the visible ceiling: "
          f"{ex['v3_added_inside_vc']:.0f} / {ex['v3_added_outside_vc']:.0f}   "
          f"V1: {ex['v1_added_inside_vc']:.0f} / {ex['v1_added_outside_vc']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
