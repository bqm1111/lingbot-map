#!/usr/bin/env python
"""Gate 3.1 step 4 — the single frozen sequence-08 evaluation.

Nothing is selected here. Region radius, controls, thresholds, checkpoints and seeds all
arrive frozen from source-only tools. Both baselines are re-asserted before any learned
number is reported.

    python tools/voxel_gate_validation/eval_clean.py
"""
from __future__ import annotations

import argparse, csv, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, collect_provenance, load_config, write_json
from gates.scale_gate.scale import bootstrap_ci
from gates.voxel_gate.controls import control, enumerate_controls
from gates.voxel_gate.models import VoxelCorrector3D, apply_region
from gates.voxel_gate.voxels import distance_bins
from gates.voxel_gate_validation.data import sample, scores, select_clips

from prompted_lingbot.occupancy import SEMANTICKITTI_GRID as G


def load_run(cfg, run, dev):
    d = os.path.join(REPO_ROOT, cfg.experiment.output_dir, "runs", run)
    ck = torch.load(os.path.join(d, "best.pt"), map_location="cpu", weights_only=False)
    m = VoxelCorrector3D(6, ck["channels"], ck["n_blocks"], ck["kernel"])
    m.load_state_dict(ck["state_dict"]); m.to(dev).eval()
    ck["train_json"] = json.load(open(os.path.join(d, "train.json")))
    return m, ck


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/voxel_gate_validation/clean_infill.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    dcfg = load_config(cfg.data.depth_gate_config)
    scfg = load_config(dcfg.data.scale_gate_config)
    dev = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    radius = int(cfg.region.radius)
    art = os.path.join(REPO_ROOT, cfg.experiment.output_dir)
    out_dir = os.path.join(art, "eval"); os.makedirs(out_dir, exist_ok=True)

    sel = json.load(open(os.path.join(art, "control_selection.json")))
    v1_kw = {k: sel["v1_clean"][k] for k in ("kind", "radius", "min_neighbors")
             if k in sel["v1_clean"]}
    v2_kw = {k: sel["v2_clean"][k] for k in ("kind", "radius", "min_neighbors")
             if k in sel["v2_clean"]}
    seeds = [int(s) for s in cfg.experiment.seeds]
    ref_seed = int(cfg.experiment.seed)

    runs = {f"V3_s{s}": ("c3", f"full_s{s}", False) for s in seeds}
    runs["A_occ_only"] = ("c3", "occ_only_s0", True)
    runs["A_c0_corrector"] = ("c0", "c0_corrector_s0", False)
    models = {n: load_run(cfg, r, dev) for n, (_, r, _) in runs.items()}
    print("loaded:", ", ".join(f"{n}({runs[n][1]})" for n in runs))
    print(f"V1_clean {sel['v1_clean']['name']}  V2_clean {sel['v2_clean']['name']}  "
          f"radius {radius}")

    clips = select_clips(cfg, "c3", "val", cfg.data.val_sequences)
    bins = {k: torch.from_numpy(m).to(dev) for k, m in
            distance_bins(tuple(tuple(b) for b in cfg.eval.distance_bins_m)).items()}
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    rows, per_bin, vis_rows = [], [], []
    t_inf = {n: 0.0 for n in runs}
    for i, c in enumerate(clips):
        cid = c["clip_id"]
        s3 = sample(cfg, "c3", cid, radius, None, dev)
        s0 = sample(cfg, "c0", cid, radius, None, dev)
        occ3, R3, keep, gt, vc = s3["occupied"], s3["R_infer"], s3["keep"], s3["gt"], s3["vc"]
        occ0, R0 = s0["occupied"], s0["R_infer"]

        P = {"C0": (occ0, occ0, R0), "C3": (occ3, occ3, R3),
             "V1_clean": (control(occ3, R3, **v1_kw), occ3, R3),
             "V2_clean": (control(occ3, R3, **v2_kw), occ3, R3),
             "VC": (vc, occ3, R3),
             "VR": (gt & R3, occ3, R3)}            # in-band oracle, diagnostic reference
        for name, (geom, run, occ_only) in runs.items():
            model, ck = models[name]
            g = runs[name][0]
            sx = sample(cfg, g, cid, radius, ck["norm"], dev, occ_only)
            t0 = time.time()
            with torch.no_grad():
                p = torch.sigmoid(model(sx["x"][None])[0, 0])
            t_inf[name] += time.time() - t0
            P[name] = (apply_region(p >= float(ck["threshold"]), sx["occupied"],
                                    sx["R_infer"]), sx["occupied"], sx["R_infer"])

        for name, (pred, base_occ, R) in P.items():
            sc = scores(pred, gt, keep)
            rows.append({"clip_id": cid, "config": name, **sc,
                         "added_vs_base": int((pred & ~base_occ & keep).sum()),
                         "removed_vs_base": int((~pred & base_occ & keep).sum()),
                         "outside_region": int((pred & ~R & ~base_occ).sum()),
                         "region_size": int(R.sum()),
                         "region_has_invalid": int((R & ~keep).sum())})
            for bn, bm in bins.items():
                per_bin.append({"clip_id": cid, "config": name, "bin": bn,
                                **scores(pred & bm, gt & bm, keep)})
            if name in ("V1_clean", "V2_clean", *runs):
                add = pred & ~base_occ & keep
                nonvis = R & ~vc & keep
                vis_rows.append({
                    "clip_id": cid, "config": name,
                    "added_inside_vc": int((add & vc).sum()),
                    "added_outside_vc": int((add & ~vc).sum()),
                    **{f"vis_{k}": v for k, v in scores(pred & vc, gt & vc, keep).items()},
                    **{f"nonvis_{k}": v for k, v in
                       scores(pred & nonvis, gt & nonvis, keep).items()}})
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(clips)}", flush=True)

    by = {}
    for r in rows:
        by.setdefault(r["config"], {})[r["clip_id"]] = r

    # ---- baselines must reproduce before any learned number is accepted ----- #
    repro = {}
    for name, want in (("C0", float(cfg.reference.c0_iou)),
                       ("C3", float(cfg.reference.c3_iou))):
        got = float(np.mean([v["iou"] for v in by[name].values()]))
        ok = abs(got - want) <= float(cfg.reference.tolerance)
        repro[name] = {"target": want, "reproduced": got, "abs_diff": abs(got - want),
                       "within_tolerance": bool(ok)}
        print(f"{name} baseline {'OK' if ok else 'FAILED'}: {got:.5f} vs {want}")
    if not all(v["within_tolerance"] for v in repro.values()):
        write_json(os.path.join(out_dir, "reproduction_failure.json"), repro)
        print("\nBASELINE_REPRODUCTION_FAILED")
        return 2

    order = ["C0", "C3", "V1_clean", "V2_clean", *[f"V3_s{s}" for s in seeds],
             "A_occ_only", "A_c0_corrector", "VC", "VR"]
    mean = lambda n, k: float(np.mean([v[k] for v in by[n].values()]))
    agg = {n: {**{k: mean(n, k) for k in
                  ("iou", "precision", "recall", "n_pred_occupied", "tp", "fp", "fn",
                   "added_vs_base", "removed_vs_base", "region_size",
                   "region_has_invalid")},
               "n_clips": len(by[n]),
               **{f"iou_{q}": float(np.quantile([v["iou"] for v in by[n].values()], p))
                  for q, p in (("p25", .25), ("median", .5), ("p75", .75))}}
           for n in order}

    bb = {}
    for r in per_bin:
        bb.setdefault(r["config"], {}).setdefault(r["bin"], []).append(r)
    binned = {n: {b: {k: float(np.mean([x[k] for x in v])) for k in
                      ("iou", "precision", "recall", "n_pred_occupied")}
                  for b, v in d.items()} for n, d in bb.items()}

    vb = {}
    for r in vis_rows:
        vb.setdefault(r["config"], []).append(r)
    vis = {n: {k: float(np.mean([x[k] for x in v])) for k in v[0] if k != "clip_id"
               and k != "config"} for n, v in vb.items()}
    for n, d in vis.items():
        tot = d["added_inside_vc"] + d["added_outside_vc"]
        d["pct_added_outside_vc"] = 100.0 * d["added_outside_vc"] / max(tot, 1e-9)

    def paired(x, y):
        ids = sorted(set(by[x]) & set(by[y]))
        d = [by[y][i]["iou"] - by[x][i]["iou"] for i in ids]
        ci = bootstrap_ci(d, cfg.eval.bootstrap_n, cfg.eval.bootstrap_seed)
        ci.update({"improved_clips": int(np.sum([v > 0 for v in d])),
                   "improved_clip_fraction": float(np.mean([v > 0 for v in d])),
                   "n_pairs": len(d)})
        return ci
    ref = f"V3_s{ref_seed}"
    pairs = ([("C3", "V1_clean"), ("C3", "V2_clean")]
             + [("C3", f"V3_s{s}") for s in seeds]
             + [("V1_clean", ref), ("V2_clean", ref), ("A_occ_only", ref),
                ("A_c0_corrector", ref), ("C0", "A_c0_corrector"), ("C0", "C3")])
    contrasts = {f"{x}->{y}": paired(x, y) for x, y in pairs}

    seed_ious = [agg[f"V3_s{s}"]["iou"] for s in seeds]
    seed_stats = {"mean": float(np.mean(seed_ious)), "std": float(np.std(seed_ious, ddof=1)),
                  "min": float(np.min(seed_ious)), "max": float(np.max(seed_ious)),
                  "per_seed": {str(s): agg[f"V3_s{s}"]["iou"] for s in seeds}}
    band_gap = agg["VR"]["iou"] - agg["C3"]["iou"]
    headroom = {n: (agg[n]["iou"] - agg["C3"]["iou"]) / band_gap
                for n in ("V1_clean", "V2_clean", *[f"V3_s{s}" for s in seeds],
                          "A_occ_only")}

    for nm, data in (("per_clip", rows), ("per_bin", per_bin), ("visible_split", vis_rows)):
        with open(os.path.join(out_dir, f"{nm}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)

    peak = (torch.cuda.max_memory_allocated(dev) / 2**30) if dev.type == "cuda" else 0.0
    write_json(os.path.join(out_dir, "summary.json"), {
        "provenance": collect_provenance(scfg, "voxel_gate_validation.eval"),
        "n_clips": len(clips), "radius": radius, "reference_seed": ref_seed,
        "seeds": seeds, "reproduction": repro,
        "region_intersects_valid_mask": False,
        "v1_clean": sel["v1_clean"], "v2_clean": sel["v2_clean"],
        "runs": {n: {"geometry": runs[n][0], "run": runs[n][1], "occ_only": runs[n][2],
                     "n_params": int(models[n][1]["n_params"]),
                     "threshold": float(models[n][1]["threshold"]),
                     "seed": int(models[n][1]["seed"]),
                     "best_epoch": int(models[n][1]["epoch"]),
                     "select_iou": float(models[n][1]["select_iou"]),
                     "peak_gpu_gib_train": models[n][1]["train_json"]["peak_gpu_gib"],
                     "train_seconds": models[n][1]["train_json"]["train_seconds"],
                     "inference_seconds_per_clip": t_inf[n] / len(clips)}
                 for n in runs},
        "aggregate": agg, "by_distance": binned, "visible_split": vis,
        "contrasts": contrasts, "seed_study": seed_stats,
        "in_band_oracle_gap": band_gap, "fraction_of_in_band_headroom": headroom,
        "visible_ceiling_diagnostic": float(cfg.reference.visible_ceiling),
        "peak_gpu_gib_eval": peak,
        "grid": {"dims": list(G.dims), "voxel_size": G.voxel_size}})

    print(f"\n{'cfg':16s} {'IoU':>8} {'P':>7} {'R':>7} {'occupied':>9} {'TP':>8} "
          f"{'FP':>8} {'FN':>8} {'added':>8} {'removed':>8}")
    for n in order:
        s = agg[n]
        print(f"{n:16s} {s['iou']:8.4f} {s['precision']:7.3f} {s['recall']:7.3f} "
              f"{s['n_pred_occupied']:9.0f} {s['tp']:8.0f} {s['fp']:8.0f} {s['fn']:8.0f} "
              f"{s['added_vs_base']:8.0f} {s['removed_vs_base']:8.0f}")
    print(f"\nseed study V3: mean {seed_stats['mean']:.4f}  sd {seed_stats['std']:.4f}  "
          f"min {seed_stats['min']:.4f}  max {seed_stats['max']:.4f}")
    print(f"\n{'contrast':32s} {'dIoU':>9}  95% CI               improved")
    for k, v in contrasts.items():
        print(f"{k:32s} {v['mean']:+9.4f}  [{v['lo']:+.4f},{v['hi']:+.4f}]"
              f"{'*' if v['excludes_zero'] else ' '} {v['improved_clips']:4d}/{v['n_pairs']}")
    print("\nin-band headroom recovered: "
          + ", ".join(f"{k} {v:.1%}" for k, v in headroom.items()))
    print(f"\n{'cfg':16s} {'added in VC':>12} {'added out VC':>13} {'% out':>7} "
          f"{'vis IoU':>8} {'nonvis IoU':>11}")
    for n, d in vis.items():
        print(f"{n:16s} {d['added_inside_vc']:12.0f} {d['added_outside_vc']:13.0f} "
              f"{d['pct_added_outside_vc']:6.1f}% {d['vis_iou']:8.4f} {d['nonvis_iou']:11.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
