#!/usr/bin/env python
"""Gate 9 aggregation: condition metrics, pixel depth-error statistics and stratification,
CSV tables, results_table.md, figures, representative-clip selection.

    python tools/gate9_depth_occupancy/analyze.py [--drives 2013_05_28_drive_0006_sync ...]
"""
from __future__ import annotations
import argparse, csv, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import common as G9                                                                        # noqa: E402

LADDER = ("V_all", "V_cam", "LD", "PD", "GF", "Gs")
MATRIX = [k for k in G9.RUNS if k not in ("LD", "PD", "GF", "Gs")]
THRESH = (0.1, 0.2, 0.4, 0.6, 1.0)


def load(drive):
    files = sorted(glob.glob(os.path.join(G9.DATA_ROOT, "clips", drive, "*.npz")))
    clips, pix = [], []
    for f in files:
        z = np.load(f)
        c = {k: z[k] for k in z.files if k.startswith("m_") or k.startswith("ep_support_")}
        c.update({"t": int(z["t"]), "oracle_ratio": float(z["oracle_ratio"]), "n_eval": int(z["n_eval"]), "n_ref_occ": int(z["n_ref_occ"]), "l_scale": float(z["l_scale"])})
        clips.append(c)
        p = z["pixels"]; pix.append(np.concatenate([p, np.full((len(p), 1), c["t"], np.float32)], 1))
        F = [str(x) for x in z["pixel_fields"]] + ["t"]
    return clips, np.concatenate(pix), F


def cond_metrics(clips, name):
    m = np.stack([c[f"m_{name}"] for c in clips]).astype(np.float64); s = m.sum(0)
    tp, fp, fn, po, pof, rf, pt, pout = s
    per = m[:, 0] / np.maximum(m[:, 0] + m[:, 1] + m[:, 2], 1)
    return {"iou": tp / max(tp + fp + fn, 1), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "pred_occ_in_mask_per_clip": po / len(clips), "pred_occ_fraction_of_mask": po / max(sum(c["n_eval"] for c in clips), 1),
            "pred_occ_on_lidar_free": pof / max(po, 1), "false_free_rate": rf / max(tp + fn, 1), "lidar_occ_recovered": tp / max(tp + fn, 1),
            "pred_occ_outside_mask_fraction": pout / max(pt, 1), "iou_per_clip_median": float(np.median(per)), "iou_per_clip_p10_p90": [float(np.percentile(per, 10)), float(np.percentile(per, 90))]}


def depth_stats(P, F, mask=None):
    g = lambda k: P[:, F.index(k)].astype(np.float64)
    m = np.ones(len(P), bool) if mask is None else mask
    zl, zp, rl = g("z_lidar")[m], g("z_pred")[m], g("ray_norm")[m]
    if len(zl) == 0:
        return {"n": 0}
    e = zp - zl; ep = np.abs(e) * rl
    dv = np.stack([g("dvx")[m], g("dvy")[m], g("dvz")[m]], 1)
    out = {"n": int(len(zl)), "signed_err_m_median": float(np.median(e)), "abs_err_m_median": float(np.median(np.abs(e))), "abs_err_m_mean": float(np.mean(np.abs(e))),
           "rel_err_median": float(np.median(e / zl)), "absrel_median": float(np.median(np.abs(e) / zl)), "absrel_mean": float(np.mean(np.abs(e) / zl)),
           "endpoint_err_m_median": float(np.median(ep)), "endpoint_err_m_p90": float(np.percentile(ep, 90)),
           "voxel_disp_median": {ax: float(np.median(np.abs(dv[:, i]))) for i, ax in enumerate("xyz")}, "voxel_disp_total_median": float(np.median(np.abs(dv).sum(1))),
           "within_m": {str(t): float((ep <= t).mean()) for t in THRESH}, "same_voxel": float(g("same_voxel")[m].mean()),
           "in_front": float((e < 0).mean()), "behind": float((e > 0).mean()),
           "pred_ep_on_ref_free": float(g("pred_ep_on_ref_free")[m].mean()), "pred_ep_near_ref_occ": float(g("pred_ep_near_ref_occ")[m].mean()),
           "pred_beyond_surface_by_gt_0p2m": float((e * rl > 0.2).mean()), "carve_through_true_endpoint": float(g("carve_through_true_endpoint")[m].mean()),
           "true_voxel_free_in_PD_map": float(g("true_voxel_free_in_PD_map")[m].mean()), "true_voxel_free_in_GF_map": float(g("true_voxel_free_in_GF_map")[m].mean())}
    close_wrong = (ep <= 0.6) & (g("same_voxel")[m] == 0)
    out["within_0p6_but_wrong_voxel"] = float(close_wrong.mean())
    for t in (0.2, 0.4, 0.6):
        out[f"within_{t}_wrong_voxel"] = float(((ep <= t) & (g("same_voxel")[m] == 0)).mean())
    return out


def scale_decomposition(P, F):
    """How much AbsRel remains after the best per-clip and per-frame scalar (oracle)."""
    g = lambda k: P[:, F.index(k)].astype(np.float64)
    zl, zp, fr, t = g("z_lidar"), g("z_pred"), g("frame"), g("t")
    key_c = t; key_f = t * 10 + fr
    out = {"absrel_raw": float(np.median(np.abs(zp - zl) / zl))}
    for name, key in (("per_clip", key_c), ("per_frame", key_f)):
        order = np.argsort(key, kind="stable"); ks = key[order]; ratio = (zl / np.maximum(zp, 1e-6))[order]
        bounds = np.r_[0, np.flatnonzero(np.diff(ks)) + 1, len(ks)]; sc = np.empty(len(ks))
        for a, b in zip(bounds[:-1], bounds[1:]):
            sc[a:b] = np.median(ratio[a:b])
        zc = np.empty(len(zp)); zc[order] = zp[order] * sc
        out[f"absrel_after_{name}_scale"] = float(np.median(np.abs(zc - zl) / zl))
        ep = np.abs(zc - zl) * g("ray_norm"); out[f"within_0.2m_after_{name}_scale"] = float((ep <= 0.2).mean()); out[f"within_0.6m_after_{name}_scale"] = float((ep <= 0.6).mean())
    return out


def strat(P, F, col, edges=None, labels=None, classes=None):
    x = P[:, F.index(col)].astype(np.float64); res = {}
    if classes is not None:
        for k, lab in classes.items():
            res[lab] = depth_stats(P, F, x == k)
    else:
        for i, lab in enumerate(labels):
            res[lab] = depth_stats(P, F, (x >= edges[i]) & (x < edges[i + 1]))
    return res


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0])
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--drives", nargs="+", default=[G9.PRIMARY]); a = ap.parse_args()
    os.makedirs(os.path.join(G9.ART, "tables"), exist_ok=True); os.makedirs(os.path.join(G9.ART, "figures"), exist_ok=True)
    res = {}
    for d in a.drives:
        clips, P, F = load(d)
        r = {"n_clips": len(clips), "n_pixels_stored": int(len(P)), "conditions": {k: cond_metrics(clips, k) for k in LADDER + tuple(MATRIX)},
             "endpoint_support": {k: {"ref_occ_with_ge1_frame_endpoint": float(sum(c[f"ep_support_{k}"][0] for c in clips) / max(sum(c[f"ep_support_{k}"][2] for c in clips), 1)),
                                      "ref_occ_with_ge2_frame_endpoints": float(sum(c[f"ep_support_{k}"][1] for c in clips) / max(sum(c[f"ep_support_{k}"][2] for c in clips), 1))} for k in ("LD", "PD")},
             "depth": depth_stats(P, F), "scale": scale_decomposition(P, F),
             "by_range": strat(P, F, "z_lidar", G9.RANGE_BINS, G9.RANGE_NAMES),
             "by_edge": strat(P, F, "edge_class", classes={0: "non-edge (<5 %)", 1: "moderate (5-20 %)", 2: "strong (>=20 %)", 3: "isolated"}),
             "by_surface": strat(P, F, "surface_class", classes={0: "planar", 1: "curved/cluttered", 2: "isolated (<6 pts)"}),
             "by_confidence": strat(P, F, "conf", (0, 1.5, 3, 6, 12, 1e9), ("<1.5 (gated)", "1.5-3", "3-6", "6-12", ">=12")),
             "oracle_ratio": {"median": float(np.median([c["oracle_ratio"] for c in clips])), "p10": float(np.percentile([c["oracle_ratio"] for c in clips], 10)), "p90": float(np.percentile([c["oracle_ratio"] for c in clips], 90))}}
        # false-free vs signed error bins
        g = lambda k: P[:, F.index(k)].astype(np.float64); e = g("z_pred") - g("z_lidar"); eb = (-1e9, -2, -1, -0.5, -0.2, 0.2, 0.5, 1, 2, 1e9)
        r["false_free_vs_signed_error"] = {f"[{eb[i]}, {eb[i+1]})": {"n": int(((e >= eb[i]) & (e < eb[i + 1])).sum()), "carve_through": float(g("carve_through_true_endpoint")[(e >= eb[i]) & (e < eb[i + 1])].mean()),
                                                                     "true_voxel_free_PD_map": float(g("true_voxel_free_in_PD_map")[(e >= eb[i]) & (e < eb[i + 1])].mean())} for i in range(len(eb) - 1) if ((e >= eb[i]) & (e < eb[i + 1])).any()}
        # representative clips by PD IoU: 10th / 50th / 90th percentile
        per = np.asarray([c["m_PD"][0] / max(c["m_PD"][:3].sum(), 1) for c in clips]); order = np.argsort(per)
        r["representative_clips"] = {lab: int(clips[order[int(q * (len(order) - 1))]]["t"]) for lab, q in (("bad", 0.1), ("median", 0.5), ("good", 0.9))}
        res[d] = r
        write_csv(os.path.join(G9.ART, "tables", f"conditions_{d[17:21]}.csv"), [{"condition": k, **{kk: (json.dumps(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()}} for k, v in r["conditions"].items()])
        rows = []
        for group in ("by_range", "by_edge", "by_surface", "by_confidence"):
            for lab, s in r[group].items():
                if s.get("n"):
                    rows.append({"group": group, "bin": lab, **{k: v for k, v in s.items() if not isinstance(v, dict)}, **{f"within_{t}m": s["within_m"][str(t)] for t in THRESH}})
        write_csv(os.path.join(G9.ART, "tables", f"depth_strata_{d[17:21]}.csv"), rows)
        write_csv(os.path.join(G9.ART, "tables", f"per_clip_{d[17:21]}.csv"), [{"t": c["t"], "oracle_ratio": c["oracle_ratio"], **{f"iou_{k}": c[f"m_{k}"][0] / max(c[f"m_{k}"][:3].sum(), 1) for k in LADDER + tuple(MATRIX)}} for c in clips])
        if d == G9.PRIMARY:
            figures(P, F, clips, r)
    tag = "" if a.drives == [G9.PRIMARY] else "_all"
    json.dump(res, open(os.path.join(G9.ART, f"analysis{tag}.json"), "w"), indent=1)
    tables(res, tag)
    return 0


def figures(P, F, clips, r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = os.path.join(G9.ART, "figures"); g = lambda k: P[:, F.index(k)].astype(np.float64)
    zl, zp, rl = g("z_lidar"), g("z_pred"), g("ray_norm"); e = zp - zl; ep = np.abs(e) * rl
    # 1 signed error histogram / ECDF
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(np.clip(e, -8, 8), bins=160, color="#2c6fbb"); ax[0].axvline(0, color="k", lw=0.8); ax[0].set_xlabel("signed depth error z_pred - z_lidar (m, clipped at ±8)"); ax[0].set_ylabel("pixels")
    for lab, m in (("all", np.ones(len(e), bool)),) + tuple((n, (zl >= G9.RANGE_BINS[i]) & (zl < G9.RANGE_BINS[i + 1])) for i, n in enumerate(G9.RANGE_NAMES)):
        s = np.sort(e[m]); ax[1].plot(s, np.linspace(0, 1, len(s)), label=lab)
    ax[1].set_xlim(-8, 8); ax[1].axvline(0, color="k", lw=0.8); ax[1].set_xlabel("signed depth error (m)"); ax[1].set_ylabel("ECDF"); ax[1].legend(fontsize=7)
    for a_ in ax: a_.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "1_signed_error.png"), dpi=120); plt.close(fig)
    # 2-4 endpoint error vs range, within thresholds, exact voxel
    rb = np.linspace(0, 60, 25); c_ = 0.5 * (rb[1:] + rb[:-1]); bi = np.digitize(zl, rb) - 1
    fig, ax = plt.subplots(1, 3, figsize=(17, 4))
    med = [np.median(ep[bi == k]) if (bi == k).sum() > 50 else np.nan for k in range(len(c_))]; p90 = [np.percentile(ep[bi == k], 90) if (bi == k).sum() > 50 else np.nan for k in range(len(c_))]
    ax[0].plot(c_, med, "-o", label="median"); ax[0].plot(c_, p90, "--", label="p90"); ax[0].set_xlabel("LiDAR range (m)"); ax[0].set_ylabel("3D endpoint error (m)"); ax[0].legend()
    for t in (0.2, 0.4, 0.6):
        ax[1].plot(c_, [(ep[bi == k] <= t).mean() if (bi == k).sum() > 50 else np.nan for k in range(len(c_))], "-o", ms=3, label=f"within {t} m")
    ax[1].set_xlabel("LiDAR range (m)"); ax[1].set_ylabel("fraction of pixels"); ax[1].legend()
    sv = g("same_voxel"); ax[2].plot(c_, [sv[bi == k].mean() if (bi == k).sum() > 50 else np.nan for k in range(len(c_))], "-o", ms=3, color="#c0392b", label="same 0.2 m voxel")
    ax[2].plot(c_, [((ep[bi == k] <= 0.6) & (sv[bi == k] == 0)).mean() if (bi == k).sum() > 50 else np.nan for k in range(len(c_))], "-s", ms=3, color="#7a5ea8", label="within 0.6 m but wrong voxel")
    ax[2].set_xlabel("LiDAR range (m)"); ax[2].set_ylabel("fraction of pixels"); ax[2].legend()
    for a_ in ax: a_.grid(alpha=0.3)
    fig.suptitle("frozen local-reset LingBot depth (x clip MoGe scale) at LiDAR-supported pixels, drive 0006"); fig.tight_layout(); fig.savefig(os.path.join(fd, "2_3_4_endpoint_vs_range.png"), dpi=120); plt.close(fig)
    # 5 edge vs non-edge
    fig, ax = plt.subplots(figsize=(8, 4)); ec = g("edge_class")
    for k, lab in ((0, "non-edge"), (1, "moderate"), (2, "strong")):
        s = np.sort(np.abs(e[ec == k]) / zl[ec == k]); ax.plot(s, np.linspace(0, 1, len(s)), label=f"{lab} (n={len(s):,})")
    ax.set_xlim(0, 1); ax.set_xlabel("absolute relative depth error"); ax.set_ylabel("ECDF"); ax.legend(); ax.grid(alpha=0.3); ax.set_title("depth error by LiDAR depth discontinuity (5x5 lattice window)")
    fig.tight_layout(); fig.savefig(os.path.join(fd, "5_edge.png"), dpi=120); plt.close(fig)
    # 6 false-occupied-in-free vs range; 7 false-free vs signed error
    fig, ax = plt.subplots(1, 2, figsize=(12, 4)); of = g("pred_ep_on_ref_free")
    ax[0].plot(c_, [of[bi == k].mean() if (bi == k).sum() > 50 else np.nan for k in range(len(c_))], "-o", ms=3); ax[0].set_xlabel("LiDAR range (m)"); ax[0].set_ylabel("predicted endpoint lands in LiDAR-measured free space")
    eb = np.linspace(-4, 4, 33); ce = 0.5 * (eb[1:] + eb[:-1]); bj = np.digitize(e, eb) - 1; ct = g("carve_through_true_endpoint"); tf = g("true_voxel_free_in_PD_map")
    ax[1].plot(ce, [ct[bj == k].mean() if (bj == k).sum() > 50 else np.nan for k in range(len(ce))], "-o", ms=3, label="ray carve passes the true endpoint")
    ax[1].plot(ce, [tf[bj == k].mean() if (bj == k).sum() > 50 else np.nan for k in range(len(ce))], "-s", ms=3, label="true endpoint voxel ends FREE in the PD map")
    ax[1].set_xlabel("signed depth error (m)"); ax[1].set_ylabel("fraction"); ax[1].legend(fontsize=8)
    for a_ in ax: a_.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "6_7_free_space_errors.png"), dpi=120); plt.close(fig)
    # ladder bar chart
    C = r["conditions"]; names = list(LADDER); fig, ax = plt.subplots(figsize=(10, 4)); x = np.arange(len(names))
    ax.bar(x - 0.25, [C[n]["iou"] for n in names], 0.25, label="IoU"); ax.bar(x, [C[n]["precision"] for n in names], 0.25, label="precision"); ax.bar(x + 0.25, [C[n]["recall"] for n in names], 0.25, label="recall")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.legend(); ax.grid(alpha=0.3, axis="y"); ax.set_title("oracle ladder, raw five-frame map vs the same five LiDAR sweeps (GT pose), drive 0006")
    fig.tight_layout(); fig.savefig(os.path.join(fd, "0_ladder.png"), dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 4)); names = ["LD"] + [n for n in MATRIX]; x = np.arange(len(names))
    ax.bar(x, [C[n]["iou"] for n in names], color=["#2c6fbb" if n.startswith("LD") else "#c0392b" if n.startswith("PD") else "#7a5ea8" for n in names])
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8); ax.set_ylabel("IoU"); ax.grid(alpha=0.3, axis="y"); ax.set_title("mapper matrix (blue: LiDAR depth, red: LingBot depth on LiDAR pixels, purple: dense LingBot)")
    fig.tight_layout(); fig.savefig(os.path.join(fd, "0_mapper_matrix.png"), dpi=120); plt.close(fig)


def tables(res, tag):
    L = ["# Gate 9 -- depth-to-occupancy geometry audit: results\n",
         "Raw five-frame observed-map metrics against the LiDAR measurement of the same five sweeps (GT relative poses, anchor SSC grid, camera-frustum mask). "
         "These are NOT completion IoU and are not comparable to benchmark completion numbers.\n"]
    for d, r in res.items():
        L.append(f"## {d[17:21]} ({r['n_clips']} clips)\n")
        L.append("| condition | IoU | precision | recall | predicted occupied / clip (in mask) | occupied on LiDAR-free | false-free rate | IoU per clip median [p10, p90] |"); L.append("|---|---|---|---|---|---|---|---|")
        for k, v in r["conditions"].items():
            L.append(f"| {k} | {v['iou']:.3f} | {v['precision']:.3f} | {v['recall']:.3f} | {v['pred_occ_in_mask_per_clip']:,.0f} | {v['pred_occ_on_lidar_free']:.3f} | {v['false_free_rate']:.4f} | {v['iou_per_clip_median']:.3f} [{v['iou_per_clip_p10_p90'][0]:.3f}, {v['iou_per_clip_p10_p90'][1]:.3f}] |")
        es = r["endpoint_support"]
        L.append(f"\nSame-surface endpoint support (LiDAR-occupied voxels holding an endpoint from >=1 / >=2 of the five frames): LD {es['LD']['ref_occ_with_ge1_frame_endpoint']:.3f} / {es['LD']['ref_occ_with_ge2_frame_endpoints']:.3f}; PD {es['PD']['ref_occ_with_ge1_frame_endpoint']:.3f} / {es['PD']['ref_occ_with_ge2_frame_endpoints']:.3f}\n")
        D = r["depth"]
        L.append("### LingBot depth at LiDAR-supported pixels\n")
        L.append(f"n = {D['n']:,} stored pixels (20 % deterministic subsample). Signed error median {D['signed_err_m_median']:+.2f} m; |error| median {D['abs_err_m_median']:.2f} m (mean {D['abs_err_m_mean']:.2f}); AbsRel median {D['absrel_median']:.3f} (mean {D['absrel_mean']:.3f}); "
                 f"relative error median {D['rel_err_median']:+.3f}; endpoint error median {D['endpoint_err_m_median']:.2f} m (p90 {D['endpoint_err_m_p90']:.2f}); voxel displacement median |dx|,|dy|,|dz| = {D['voxel_disp_median']['x']:.0f}, {D['voxel_disp_median']['y']:.0f}, {D['voxel_disp_median']['z']:.0f}, total {D['voxel_disp_total_median']:.0f} voxels; "
                 f"in front {D['in_front']:.3f}, behind {D['behind']:.3f}.\n")
        L.append("| within 0.1 m | 0.2 m | 0.4 m | 0.6 m | 1.0 m | exact voxel | within 0.2 m wrong voxel | within 0.4 m wrong voxel | within 0.6 m wrong voxel | endpoint on LiDAR-free | endpoint near LiDAR-occupied (±1 voxel) | beyond surface > 0.2 m | carve passes true endpoint | true endpoint voxel FREE in PD map | in GF map |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        w = D["within_m"]
        L.append(f"| {w['0.1']:.3f} | {w['0.2']:.3f} | {w['0.4']:.3f} | {w['0.6']:.3f} | {w['1.0']:.3f} | {D['same_voxel']:.4f} | {D['within_0.2_wrong_voxel']:.3f} | {D['within_0.4_wrong_voxel']:.3f} | {D['within_0.6_wrong_voxel']:.3f} | "
                 f"{D['pred_ep_on_ref_free']:.3f} | {D['pred_ep_near_ref_occ']:.3f} | {D['pred_beyond_surface_by_gt_0p2m']:.3f} | {D['carve_through_true_endpoint']:.3f} | {D['true_voxel_free_in_PD_map']:.4f} | {D['true_voxel_free_in_GF_map']:.4f} |")
        S_ = r["scale"]
        L.append(f"\nScale decomposition (oracle scalars from LiDAR): AbsRel raw {S_['absrel_raw']:.3f} -> after best per-clip scalar {S_['absrel_after_per_clip_scale']:.3f} -> per-frame scalar {S_['absrel_after_per_frame_scale']:.3f}; "
                 f"within 0.2 m {w['0.2']:.3f} -> {S_['within_0.2m_after_per_clip_scale']:.3f} / {S_['within_0.2m_after_per_frame_scale']:.3f}; within 0.6 m {w['0.6']:.3f} -> {S_['within_0.6m_after_per_clip_scale']:.3f} / {S_['within_0.6m_after_per_frame_scale']:.3f}. "
                 f"Per-clip LiDAR/LingBot ratio median {r['oracle_ratio']['median']:.3f} [p10 {r['oracle_ratio']['p10']:.3f}, p90 {r['oracle_ratio']['p90']:.3f}].\n")
        for group, title in (("by_range", "range"), ("by_edge", "LiDAR depth discontinuity"), ("by_surface", "local surface"), ("by_confidence", "LingBot confidence")):
            L.append(f"**by {title}**\n"); L.append("| bin | n | AbsRel median | |err| m median | signed m median | endpoint m median | within 0.2 | within 0.4 | within 0.6 | exact voxel | on LiDAR-free | carve passes true endpoint |"); L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
            for lab, s in r[group].items():
                if s.get("n"):
                    L.append(f"| {lab} | {s['n']:,} | {s['absrel_median']:.3f} | {s['abs_err_m_median']:.2f} | {s['signed_err_m_median']:+.2f} | {s['endpoint_err_m_median']:.2f} | {s['within_m']['0.2']:.3f} | {s['within_m']['0.4']:.3f} | {s['within_m']['0.6']:.3f} | {s['same_voxel']:.4f} | {s['pred_ep_on_ref_free']:.3f} | {s['carve_through_true_endpoint']:.3f} |")
            L.append("")
        L.append("**carving vs signed depth error**\n"); L.append("| signed error bin (m) | n | carve passes true endpoint | true endpoint voxel FREE in PD map |"); L.append("|---|---|---|---|")
        for k, v in r["false_free_vs_signed_error"].items():
            L.append(f"| {k} | {v['n']:,} | {v['carve_through']:.3f} | {v['true_voxel_free_PD_map']:.4f} |")
        L.append(f"\nRepresentative clips (PD IoU 10th / 50th / 90th percentile): {r['representative_clips']}\n")
    open(os.path.join(G9.ART, f"results_table{tag}.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
