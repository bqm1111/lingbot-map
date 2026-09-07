#!/usr/bin/env python
"""Gate 7A — turn the sharded count blocks into the reported numbers.

Pooled voxel counts are the primary aggregation throughout; anything mean-per-clip is
labelled as such and never mixed into a pooled table. Confidence intervals reuse Gate 6's
resampling units and its seed.

    python tools/gate7a/aggregate.py --dataset kitti360
"""
from __future__ import annotations

import argparse, glob, json, os, sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT, write_json                       # noqa: E402
from gates.gate6 import grids as G6G, vocab                                     # noqa: E402
from gates.gate7a import config as C, distance as D, frustum as FR, stats as S   # noqa: E402
from tools.gate6.analyze import unit_ids                                   # noqa: E402

ART = os.path.join(REPO_ROOT, "artifacts", "gate7a")
QUANTILES = (0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def load_shards(ds):
    paths = sorted(glob.glob(os.path.join(ART, f"shard_{ds}_*of*.npz")))
    if not paths:
        raise SystemExit(f"no Gate-7A shards for {ds}; run tools/gate7a/reachability.py")
    zs = [np.load(p, allow_pickle=False) for p in paths]
    metas = [json.load(open(p[:-4] + ".json")) for p in paths]
    per_clip = {k: np.concatenate([z[k] for z in zs], axis=0)
                for k in ("clip_id", "group", "binary", "perclass", "volume",
                          "base_binary", "base_perclass", "trans", "trans_conf")}
    summed = {k: np.sum([z[k] for z in zs], axis=0)
              for k in ("miss_hist", "miss_in_r", "band_gt", "band_tp", "frust", "resid",
                        "resid_hist", "tie")}
    order = np.argsort(per_clip["clip_id"], kind="stable")
    for k in per_clip:
        per_clip[k] = per_clip[k][order]
    return per_clip, summed, zs[0], metas


def pooled(b3):
    tp, fp, fn = (float(x) for x in b3)
    den = tp + fp + fn
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn),
            "binary_iou": tp / den if den else 0.0,
            "binary_precision": tp / (tp + fp) if tp + fp else 0.0,
            "binary_recall": tp / (tp + fn) if tp + fn else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=list(C.DATASETS))
    ap.add_argument("--n-boot", type=int, default=C.BOOTSTRAP["n_boot"])
    a = ap.parse_args()
    ds = a.dataset
    pc, sm, z0, metas = load_shards(ds)
    v = vocab.load(ds)
    G = G6G.EVAL_GRID[ds]
    vs = float(G.voxel_size)
    R = list(C.RADII_M)
    n = len(pc["clip_id"])
    band_names = ["all"] + [f"{int(lo)}-{int(hi)}m" for lo, hi in C.RANGE_BANDS]

    res = {"dataset": ds, "grid": G.name, "voxel_size_m": vs, "n_clips": n,
           "radii_m": R, "range_bands_m": [list(b) for b in C.RANGE_BANDS],
           "conditions": list(C.CONDITIONS), "constructions": list(C.CONSTRUCTIONS),
           "primary_condition": C.PRIMARY_CONDITION,
           "class_names": list(v.names),
           "precommit_sha256": metas[0]["precommit_sha256"],
           "gate6_rollup_sha256": metas[0]["gate6_rollup_sha256"],
           "n_verified_against_pinned": sum(m["n_verified_against_pinned"] for m in metas),
           "max_projection_roundtrip_pixel_error":
               max(m["max_projection_roundtrip_pixel_error"] for m in metas),
           "empty_base_clips": {c: sorted(sum((m["empty_base_clips"][c] for m in metas),
                                              [])) for c in C.CONDITIONS},
           "seconds": sum(m["seconds"] for m in metas),
           "seconds_wall": max(m["seconds"] for m in metas),
           "n_shards": len(metas),
           "pinned_channel_agreement": {
               k: sum(m["pinned_channel_agreement"][k] for m in metas)
               for k in ("n_raw_voxels", "n_raw_channel_mismatch", "n_dil_voxels",
                         "n_dil_channel_mismatch")},
           "max_top1_top2_gap_at_mismatch":
               max(m["pinned_channel_agreement"]["max_top1_top2_gap_at_mismatch"]
                   for m in metas),
           "peak_gpu_gib": max(m["peak_gpu_gib"] for m in metas)}

    # ------------------------------------------------------------------ base
    base_b = pc["base_binary"].sum(0)
    base_pcl = pc["base_perclass"].sum(0).astype(np.float64)
    res["base"] = {}
    for ic, cond in enumerate(C.CONDITIONS):
        d = pooled(base_b[ic])
        d["ssc_miou"] = float(S.miou(base_pcl[ic]))
        den = pc["base_binary"][:, ic].sum(axis=1)
        d["binary_iou_mean_per_clip"] = float(np.mean(
            np.where(den > 0, pc["base_binary"][:, ic, 0] / np.maximum(den, 1), 0.0)))
        d["n_clips_with_empty_base"] = len(res["empty_base_clips"][cond])
        res["base"][cond] = d

    # ------------------------------------------------- envelopes at every radius
    bin_s = pc["binary"].sum(0).astype(np.float64)          # [2,2,R,3]
    pcl_s = pc["perclass"].sum(0).astype(np.float64)        # [2,2,R,C,3]
    vol_s = pc["volume"].sum(0).astype(np.float64)          # [2,2,R,4]
    res["envelopes"] = {}
    for ic, cond in enumerate(C.CONDITIONS):
        base = res["base"][cond]
        res["envelopes"][cond] = {}
        for ik, constr in enumerate(C.CONSTRUCTIONS):
            rows = []
            for ir, r in enumerate(R):
                d = pooled(bin_s[ic, ik, ir])
                d["radius_m"] = float(r)
                d["ssc_miou"] = float(S.miou(pcl_s[ic, ik, ir]))
                d["delta_binary_iou"] = d["binary_iou"] - base["binary_iou"]
                d["delta_ssc_miou"] = d["ssc_miou"] - base["ssc_miou"]
                nb, na, ntp, nfp = vol_s[ic, ik, ir]
                d["n_base_voxels"] = int(nb)
                d["n_added"] = int(na)
                d["n_added_true_positive"] = int(ntp)
                d["n_added_false_positive"] = int(nfp)
                d["added_volume_ratio"] = float(na / nb) if nb else 0.0
                d["added_precision"] = float(ntp / na) if na else 0.0
                d["tp_recovered"] = d["tp"] - base["tp"]
                d["fn_recovered"] = base["fn"] - d["fn"]
                d["fp_unchanged_vs_base"] = bool(d["fp"] == base["fp"])
                rows.append(d)
            mono_r = all(rows[i + 1]["binary_recall"] >= rows[i]["binary_recall"] - 1e-12
                         for i in range(len(rows) - 1))
            mono_i = all(rows[i + 1]["binary_iou"] >= rows[i]["binary_iou"] - 1e-12
                         for i in range(len(rows) - 1))
            res["envelopes"][cond][constr] = {
                "by_radius": rows,
                "recall_monotonic_non_decreasing": mono_r,
                "iou_monotonic_non_decreasing": mono_i,
                "false_positives_constant": bool(all(x["fp_unchanged_vs_base"]
                                                     for x in rows))
                if constr == "oracle" else None}

    # ------------------------------------------------------- miss distances
    res["miss_distance"] = {}
    for ic, cond in enumerate(C.CONDITIONS):
        per_band = {}
        for ib, bn in enumerate(band_names):
            h = sm["miss_hist"][ic, ib]
            tot = int(h.sum())
            gt = int(sm["band_gt"][ic, ib])
            tp = int(sm["band_tp"][ic, ib])
            q = D.quantiles_from_histogram(h, QUANTILES, C.MISS_HIST_BIN_M)
            within = {f"{r:g}": int(sm["miss_in_r"][ic, ib, ir]) for ir, r in enumerate(R)}
            per_band[bn] = {
                "n_gt_occupied": gt, "n_true_positive": tp, "n_miss": tot,
                "quantiles_m": q,
                "miss_fraction_within_radius": {k: (x / tot if tot else 0.0)
                                                for k, x in within.items()},
                "gt_recoverable_fraction_within_radius": {
                    k: ((tp + x) / gt if gt else 0.0) for k, x in within.items()},
                "n_miss_within_radius": within}
            if tot:
                per_band[bn]["mean_m"] = float(
                    (h * (np.arange(len(h)) + 0.5) * C.MISS_HIST_BIN_M).sum() / tot)
        res["miss_distance"][cond] = per_band

    # ------------------------------------------------- semantic transport
    tr = pc["trans"].sum(0).astype(np.float64)              # [2, I, C, 2]
    tc = pc["trans_conf"].sum(0)                            # [2, I, 3]
    res["semantic_transport"] = {}
    for ic, cond in enumerate(C.CONDITIONS):
        rows = []
        for ii, (lo, hi) in enumerate(C.TRANSPORT_INTERVALS_M):
            row, cor = tr[ic, ii, :, 0], tr[ic, ii, :, 1]
            nvox = float(row.sum())
            rec = np.where(row > 0, cor / np.maximum(row, 1), np.nan)
            rows.append({
                "interval_m": [float(lo), float(hi)],
                "n_voxels": int(nvox),
                "top1_accuracy": float(cor.sum() / nvox) if nvox else 0.0,
                "balanced_recall": (float(np.nanmean(rec)) if (row > 0).any() else 0.0),
                "n_classes_present": int((row > 0).sum()),
                "mean_max_probability": float(tc[ic, ii, 1] / tc[ic, ii, 0])
                                        if tc[ic, ii, 0] else 0.0,
                "mean_entropy_nats": float(tc[ic, ii, 2] / tc[ic, ii, 0])
                                     if tc[ic, ii, 0] else 0.0,
                "per_class_support": {v.names[i]: int(row[i]) for i in range(len(row))},
                "per_class_recall": {v.names[i]: (None if row[i] == 0 else float(rec[i]))
                                     for i in range(len(row))}})
        res["semantic_transport"][cond] = {
            "note": ("the oracle-added and morphology-added TRUE POSITIVE voxels are the "
                     "same set at every radius -- morphology adds every valid voxel in "
                     "range and the oracle adds exactly its ground-truth-occupied subset "
                     "-- so this table applies to both constructions"),
            "by_interval": rows}

    # ------------------------------------------------------------- frustum
    edges = list(C.RADII_M[1:]) + [None]
    dist_band_names = ([f"<= {C.RADII_M[1]:g} m"]
                       + [f"{C.RADII_M[i]:g}-{C.RADII_M[i+1]:g} m"
                          for i in range(1, len(C.RADII_M) - 1)]
                       + [f"> {C.RADII_M[-1]:g} m"])
    res["frustum"] = {}
    for ic, cond in enumerate(C.CONDITIONS):
        per_band = {}
        for ib, bn in enumerate(band_names):
            f = sm["frust"][ic, ib]
            tot = int(f.sum())
            rc = sm["resid"][ic, ib]
            rh = sm["resid_hist"][ic, ib]
            bin_m = float(z0["resid_bin_m"])
            max_m = float(z0["resid_max_m"])
            centres = (np.arange(len(rh)) + 0.5) * bin_m - max_m
            nrh = int(rh.sum())
            per_band[bn] = {
                "n_coverage_miss": tot,
                "in_any_input_frustum": int(f[:, 0].sum()),
                "outside_all_input_frusta": int(f[:, 1].sum()),
                "in_frustum_fraction": float(f[:, 0].sum() / tot) if tot else 0.0,
                "by_distance_to_base": {
                    dist_band_names[i]: {"in_frustum": int(f[i, 0]),
                                         "outside": int(f[i, 1]),
                                         "in_frustum_fraction":
                                             float(f[i, 0] / max(f[i].sum(), 1))}
                    for i in range(len(dist_band_names))},
                "residual_classes": {FR.RESIDUAL_CLASSES[i]: int(rc[i])
                                     for i in range(len(rc))},
                "residual_class_fraction": {FR.RESIDUAL_CLASSES[i]:
                                            float(rc[i] / tot) if tot else 0.0
                                            for i in range(len(rc))},
                "residual_median_m": (float(centres[int(np.searchsorted(np.cumsum(rh),
                                                                       0.5 * nrh))])
                                      if nrh else None),
                "n_residual_defined": nrh}
        res["frustum"][cond] = per_band
    res["frustum"]["note"] = (
        "'in-frustum' is projection plus positive camera depth plus the frozen metric "
        "depth range. It is NOT visibility: occlusion is not tested. With a tolerance of "
        "one voxel diagonal the 'near the predicted surface' class is empty for B-D by "
        "construction, because B-D is exactly a 0.4 m dilation and has already absorbed "
        "that band; B-R is reported alongside for that reason.")

    # ----------------------------------------------------------- tie statistics
    res["propagation_ties"] = {
        C.CONDITIONS[ic]: {"n_propagated": int(sm["tie"][ic, 0]),
                           "mean_tied_sources": float(sm["tie"][ic, 1] / sm["tie"][ic, 0])
                                                if sm["tie"][ic, 0] else 0.0,
                           "fraction_with_a_tie": float(sm["tie"][ic, 2] / sm["tie"][ic, 0])
                                                  if sm["tie"][ic, 0] else 0.0}
        for ic in range(len(C.CONDITIONS))}

    # ------------------------------------------------------------- bootstrap
    units, unit_name = unit_ids(ds, list(pc["clip_id"]), list(pc["group"]))
    uids = sorted(set(units))
    uix = {u: i for i, u in enumerate(uids)}
    U = len(uids)
    ub = np.zeros((U,) + pc["binary"].shape[1:], np.float64)
    up = np.zeros((U,) + pc["perclass"].shape[1:], np.float64)
    ubb = np.zeros((U,) + pc["base_binary"].shape[1:], np.float64)
    ubp = np.zeros((U,) + pc["base_perclass"].shape[1:], np.float64)
    for i, u in enumerate(units):
        j = uix[u]
        ub[j] += pc["binary"][i]
        up[j] += pc["perclass"][i]
        ubb[j] += pc["base_binary"][i]
        ubp[j] += pc["base_perclass"][i]
    mult = S.multiplicities(U, a.n_boot, C.BOOTSTRAP["seed"])
    rb = S.resample(ub, mult)          # [B,2,2,R,3]
    rp = S.resample(up, mult)
    rbb = S.resample(ubb, mult)
    rbp = S.resample(ubp, mult)
    iou_b, miou_b = S.binary_iou(rb), S.miou(rp)
    iou_base, miou_base = S.binary_iou(rbb), S.miou(rbp)
    res["bootstrap"] = {"n_units": U, "unit": unit_name, "n_boot": a.n_boot,
                        "seed": C.BOOTSTRAP["seed"],
                        "independence_caveat": C.BOOTSTRAP["independence_caveat"],
                        "deltas": {}}
    for ic, cond in enumerate(C.CONDITIONS):
        for ik, constr in enumerate(C.CONSTRUCTIONS):
            for ir, r in enumerate(R):
                if r == 0.0:
                    continue
                key = f"{cond}|{constr}|r={r:g}"
                res["bootstrap"]["deltas"][key] = {
                    "binary_iou": S.paired(res["base"][cond]["binary_iou"],
                                           res["envelopes"][cond][constr]["by_radius"][ir]["binary_iou"],
                                           iou_base[:, ic], iou_b[:, ic, ik, ir]),
                    "ssc_miou": S.paired(res["base"][cond]["ssc_miou"],
                                         res["envelopes"][cond][constr]["by_radius"][ir]["ssc_miou"],
                                         miou_base[:, ic], miou_b[:, ic, ik, ir])}

    write_json(os.path.join(ART, f"summary_{ds}.json"), res)

    import csv
    with open(os.path.join(ART, f"per_clip_{ds}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        head = ["clip_id", "group", "unit"]
        for cond in C.CONDITIONS:
            head += [f"{cond}_btp", f"{cond}_bfp", f"{cond}_bfn"]
            for constr in C.CONSTRUCTIONS:
                for r in R:
                    head += [f"{cond}_{constr}_r{r:g}_btp", f"{cond}_{constr}_r{r:g}_bfp",
                             f"{cond}_{constr}_r{r:g}_bfn"]
        w.writerow(head)
        for i in range(n):
            row = [pc["clip_id"][i], pc["group"][i], units[i]]
            for ic in range(len(C.CONDITIONS)):
                row += [int(x) for x in pc["base_binary"][i, ic]]
                for ik in range(len(C.CONSTRUCTIONS)):
                    for ir in range(len(R)):
                        row += [int(x) for x in pc["binary"][i, ic, ik, ir]]
            w.writerow(row)

    b = res["base"]["B-D"]
    o = res["envelopes"]["B-D"]["oracle"]["by_radius"]
    m = res["envelopes"]["B-D"]["morph"]["by_radius"]
    print(f"[{ds}] {n} clips, unit={unit_name} ({U})")
    print(f"  B-D base   IoU {b['binary_iou']:.4f}  mIoU {b['ssc_miou']:.4f}")
    for ir, r in enumerate(R):
        print(f"   r={r:4.1f}  oracle IoU {o[ir]['binary_iou']:.4f} mIoU "
              f"{o[ir]['ssc_miou']:.4f} | morph IoU {m[ir]['binary_iou']:.4f} mIoU "
              f"{m[ir]['ssc_miou']:.4f} P {m[ir]['binary_precision']:.4f} "
              f"vol x{m[ir]['added_volume_ratio']:.2f}")
    md = res["miss_distance"]["B-D"]["all"]
    print(f"  miss p50 {md['quantiles_m']['p50']:.2f} m  p90 {md['quantiles_m']['p90']:.2f} m"
          f"  within 0.4 m {md['miss_fraction_within_radius']['0.4']:.3f}"
          f"  within 4 m {md['miss_fraction_within_radius']['4']:.3f}")
    fr = res["frustum"]["B-D"]["all"]
    print(f"  in-frustum {fr['in_frustum_fraction']:.3f}  classes "
          f"{fr['residual_class_fraction']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
