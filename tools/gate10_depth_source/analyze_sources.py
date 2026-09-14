#!/usr/bin/env python
"""Gate 10 aggregation: occupancy per condition, gap closed, clip-block bootstrap, depth
metrics on identical matched pixels, strata (range / edge / LingBot confidence), CSV tables,
results_table[_all].md, figures. Representative clips = Gate 9's rule (LB-M IoU on the Gate 9
support at the 10th / 50th / 90th percentile), fixed before any Gate 10 number was seen.

    python tools/gate10_depth_source/analyze_sources.py [--drives 2013_05_28_drive_0006_sync ...]
"""
from __future__ import annotations
import argparse, csv, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g10 as G10                                                                          # noqa: E402
sys.path.insert(0, G10.G9_DIR)
import common as G9                                                                        # noqa: E402

COND = ("LD", "LB", "LBM", "M", "LBL", "ML")
DEPTH_COND = ("LB", "LBM", "M", "LBL", "ML")
NAME = {"LD": "LD", "LB": "LB", "LBM": "LB-M", "M": "M", "LBL": "LB-LiDAR", "ML": "M-LiDAR", "LD_g9": "LD (Gate 9 support)", "LBM_g9": "LB-M (Gate 9 support) = Gate 9 PD"}


def load(drive):
    clips, pix, F = [], [], None
    for f in sorted(glob.glob(os.path.join(G10.DATA_ROOT, "clips", drive, "*.npz"))):
        z = np.load(f)
        c = {k: z[k] for k in z.files if k.startswith("m_")}
        c.update({k: float(z[k]) for k in ("l_scale", "oracle_scale_LBL", "oracle_scale_ML")})
        c.update({"t": int(z["t"]), "n_eval": int(z["n_eval"]), "n_support": int(z["n_support_pixels"].sum()), "n_matched": int(z["n_matched_pixels"].sum()), "fov": z["moge_fov_x_deg"]})
        clips.append(c)
        p = z["pixels"]; pix.append(np.concatenate([p, np.full((len(p), 1), c["t"], np.float32)], 1)); F = [str(x) for x in z["pixel_fields"]] + ["t"]
    return clips, np.concatenate(pix), F


def iou_of(m):
    return m[..., 0] / np.maximum(m[..., 0] + m[..., 1] + m[..., 2], 1)


def cond_metrics(clips, name):
    m = np.stack([c[f"m_{name}"] for c in clips]).astype(np.float64); s = m.sum(0)
    tp, fp, fn, po, pof, rf, pt, pout = s; per = iou_of(m)
    return {"iou": tp / max(tp + fp + fn, 1), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "pred_occ_fraction_of_mask": po / max(sum(c["n_eval"] for c in clips), 1), "pred_occ_in_mask_per_clip": po / len(clips),
            "pred_occ_on_lidar_free": pof / max(po, 1), "lidar_occ_recovered": tp / max(tp + fn, 1), "false_free_rate": rf / max(tp + fn, 1),
            "iou_per_clip_median": float(np.median(per)), "iou_per_clip_p10": float(np.percentile(per, 10)), "iou_per_clip_p90": float(np.percentile(per, 90))}


def bootstrap(clips, fn, seed=0):
    """Contiguous blocks of BOOT_BLOCK clips (stride-3 anchors -> 60 stream frames), pooled-count statistic."""
    n = len(clips); nb = int(np.ceil(n / G10.BOOT_BLOCK)); rng = np.random.default_rng(seed); vals = []
    blocks = [list(range(i * G10.BOOT_BLOCK, min((i + 1) * G10.BOOT_BLOCK, n))) for i in range(nb)]
    for _ in range(G10.N_BOOT):
        sel = [clips[i] for b in rng.integers(0, nb, nb) for i in blocks[b]]; vals.append(fn(sel))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def pooled_iou(clips, k):
    s = np.stack([c[f"m_{k}"] for c in clips]).astype(np.float64).sum(0); return s[0] / max(s[:3].sum(), 1)


def gap(clips, x, ref):
    ld, lb = pooled_iou(clips, "LD"), pooled_iou(clips, ref)
    return (pooled_iou(clips, x) - lb) / (ld - lb)


def depth_stats(P, F, X, mask):
    g = lambda k: P[:, F.index(k)].astype(np.float64)
    zl, zx, rl = g("z_lidar")[mask], g(f"z_{X}")[mask], g("ray_norm")[mask]
    if len(zl) == 0:
        return {"n": 0}
    e = zx - zl; ep = np.abs(e) * rl
    return {"n": int(len(zl)), "absrel_median": float(np.median(np.abs(e) / zl)), "absrel_mean": float(np.mean(np.abs(e) / zl)),
            "signed_rel_median": float(np.median(e / zl)), "signed_err_m_median": float(np.median(e)), "abs_err_m_median": float(np.median(np.abs(e))),
            "endpoint_m_median": float(np.median(ep)), "endpoint_m_p75": float(np.percentile(ep, 75)), "endpoint_m_p90": float(np.percentile(ep, 90)), "endpoint_m_p95": float(np.percentile(ep, 95)),
            "same_voxel": float(g(f"same_voxel_{X}")[mask].mean()), **{f"within_{t}m": float((ep <= t).mean()) for t in G10.THRESH},
            "in_front": float((e < 0).mean()), "behind": float((e > 0).mean()), "ep_on_lidar_free": float(g(f"ep_on_ref_free_{X}")[mask].mean()),
            "ep_in_grid": float(g(f"in_grid_{X}")[mask].mean())}


def scalar_structural(P, F):
    """Per-frame oracle scalar on top of the per-clip one (how much is left once scale drifts within a clip)."""
    g = lambda k: P[:, F.index(k)].astype(np.float64); m = g("matched") > 0
    zl, rl, key = g("z_lidar")[m], g("ray_norm")[m], (g("t") * 10 + g("frame"))[m]
    out = {}
    for X in ("LB", "M"):
        z = g(f"z_{X}")[m]; order = np.argsort(key, kind="stable"); ks = key[order]; ratio = (zl / np.maximum(z, 1e-9))[order]
        b = np.r_[0, np.flatnonzero(np.diff(ks)) + 1, len(ks)]; sc = np.empty(len(ks))
        for a_, b_ in zip(b[:-1], b[1:]):
            sc[a_:b_] = np.median(ratio[a_:b_])
        zc = np.empty(len(z)); zc[order] = z[order] * sc; ep = np.abs(zc - zl) * rl
        out[X] = {"absrel_median_after_per_frame_scalar": float(np.median(np.abs(zc - zl) / zl)), "endpoint_m_median_after_per_frame_scalar": float(np.median(ep)),
                  **{f"within_{t}m_after_per_frame_scalar": float((ep <= t).mean()) for t in (0.2, 0.4, 0.6)}}
    return out


def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(r) for r in rows]


def analyze_drive(d):
    clips, P, F = load(d); g = lambda k: P[:, F.index(k)].astype(np.float64); matched = g("matched") > 0
    r = {"n_clips": len(clips), "n_pixel_rows": int(len(P)), "n_matched_rows": int(matched.sum()),
         "matched_fraction_of_support": sum(c["n_matched"] for c in clips) / max(sum(c["n_support"] for c in clips), 1),
         "moge_fov_x_deg": sorted({round(float(x), 4) for c in clips for x in c["fov"]}),
         "conditions": {k: cond_metrics(clips, k) for k in COND + ("LD_g9", "LBM_g9")}}
    r["gap"] = {}
    for ref in ("LBM", "LB"):
        r["gap"][ref] = {x: {"iou_gain": pooled_iou(clips, x) - pooled_iou(clips, ref), "gap_closed": gap(clips, x, ref),
                             "gap_closed_ci95": bootstrap(clips, lambda s, x=x, ref=ref: gap(s, x, ref))} for x in COND if x != ref}
        r["gap"][ref]["_denominator"] = pooled_iou(clips, "LD") - pooled_iou(clips, ref)
    r["paired_iou_diff_ci95"] = {f"{a}-{b}": {"diff": pooled_iou(clips, a) - pooled_iou(clips, b), "ci95": bootstrap(clips, lambda s, a=a, b=b: pooled_iou(s, a) - pooled_iou(s, b))}
                                 for a, b in (("M", "LBM"), ("LBM", "LB"), ("LBL", "LBM"), ("ML", "M"), ("LD", "M"), ("ML", "LBL"))}
    per = {k: iou_of(np.stack([c[f"m_{k}"] for c in clips])) for k in COND}
    r["clip_win_rate"] = {"M>LBM": float((per["M"] > per["LBM"]).mean()), "LBL>LBM": float((per["LBL"] > per["LBM"]).mean()), "ML>LBL": float((per["ML"] > per["LBL"]).mean())}
    r["scales"] = {k: {"median": float(np.median([c[k] for c in clips])), "p10": float(np.percentile([c[k] for c in clips], 10)), "p90": float(np.percentile([c[k] for c in clips], 90))}
                   for k in ("l_scale", "oracle_scale_LBL", "oracle_scale_ML")}
    r["scales"]["LBL_over_LBM_ratio"] = {q: float(np.percentile([c["oracle_scale_LBL"] / c["l_scale"] for c in clips], p)) for q, p in (("p10", 10), ("median", 50), ("p90", 90))}
    r["depth"] = {X: depth_stats(P, F, X, matched) for X in DEPTH_COND}
    r["scalar_vs_structural"] = scalar_structural(P, F)
    zm, zb, zl_ = g("z_M")[matched], g("z_LBM")[matched], g("z_lidar")[matched]
    em, eb = (zm - zl_) / zl_, (zb - zl_) / zl_
    r["M_vs_LBM_agreement"] = {"median_abs_depth_diff_m": float(np.median(np.abs(zm - zb))), "frac_abs_diff_gt_0p2m": float((np.abs(zm - zb) > 0.2).mean()),
                               "signed_rel_error_correlation": float(np.corrcoef(em, eb)[0, 1]), "both_same_voxel": float(((g("same_voxel_M") > 0) & (g("same_voxel_LBM") > 0))[matched].mean()),
                               "either_same_voxel": float(((g("same_voxel_M") > 0) | (g("same_voxel_LBM") > 0))[matched].mean()),
                               "both_within_0.4m": float(((np.abs(zm - zl_) * g("ray_norm")[matched] <= 0.4) & (np.abs(zb - zl_) * g("ray_norm")[matched] <= 0.4)).mean())}
    zl = g("z_lidar"); ec = g("edge_class"); cf = g("conf")
    r["by_range"] = {lab: {X: depth_stats(P, F, X, matched & (zl >= G9.RANGE_BINS[i]) & (zl < G9.RANGE_BINS[i + 1])) for X in DEPTH_COND} for i, lab in enumerate(G9.RANGE_NAMES)}
    r["by_edge"] = {lab: {X: depth_stats(P, F, X, matched & (ec == k)) for X in DEPTH_COND} for k, lab in G10.EDGE_NAMES.items()}
    r["by_confidence"] = {lab: {X: depth_stats(P, F, X, matched & (cf >= G10.CONF_BINS[i]) & (cf < G10.CONF_BINS[i + 1])) for X in ("LB", "LBM")} for i, lab in enumerate(G10.CONF_NAMES)}
    g9 = iou_of(np.stack([c["m_LBM_g9"] for c in clips])); order = np.argsort(g9, kind="stable")
    r["representative_clips"] = {lab: int(clips[order[int(q * (len(order) - 1))]]["t"]) for lab, q in (("bad", 0.1), ("median", 0.5), ("good", 0.9))}
    tag = d[17:21]; T = os.path.join(G10.ART, "tables")
    write_csv(os.path.join(T, f"occupancy_{tag}.csv"), [{"condition": NAME[k], **r["conditions"][k]} for k in r["conditions"]])
    write_csv(os.path.join(T, f"gap_closed_{tag}.csv"), [{"reference": NAME[ref], "condition": NAME[x], "iou": pooled_iou(clips, x), "iou_gain": v["iou_gain"], "gap_closed": v["gap_closed"], "ci95_lo": v["gap_closed_ci95"][0], "ci95_hi": v["gap_closed_ci95"][1]}
                                                          for ref in r["gap"] for x, v in r["gap"][ref].items() if not x.startswith("_")])
    write_csv(os.path.join(T, f"depth_{tag}.csv"), [{"condition": NAME[X], **r["depth"][X]} for X in DEPTH_COND])
    rows = []
    for group in ("by_range", "by_edge", "by_confidence"):
        for lab, dd in r[group].items():
            for X, s in dd.items():
                if s.get("n"):
                    rows.append({"group": group, "bin": lab, "condition": NAME[X], **s})
    write_csv(os.path.join(T, f"depth_strata_{tag}.csv"), rows)
    write_csv(os.path.join(T, f"per_clip_{tag}.csv"), [{"t": c["t"], "l_scale": c["l_scale"], "oracle_scale_LBL": c["oracle_scale_LBL"], "oracle_scale_ML": c["oracle_scale_ML"],
                                                        **{f"iou_{NAME[k]}": float(iou_of(c[f"m_{k}"])) for k in COND + ("LD_g9", "LBM_g9")}} for c in clips])
    return r, (clips, P, F)


def figures(d, r, clips, P, F):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = os.path.join(G10.ART, "figures"); tag = d[17:21]; g = lambda k: P[:, F.index(k)].astype(np.float64); m = g("matched") > 0
    zl, rl = g("z_lidar")[m], g("ray_norm")[m]
    col = {"LD": "#222222", "LB": "#bbbbbb", "LBM": "#c0392b", "M": "#2c6fbb", "LBL": "#e59866", "ML": "#85c1e9"}
    C = r["conditions"]; G = r["gap"]["LBM"]
    # 1 IoU + gap closed
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2)); x = np.arange(len(COND))
    ax[0].bar(x, [C[k]["iou"] for k in COND], color=[col[k] for k in COND])
    for i, k in enumerate(COND):
        ax[0].text(i, C[k]["iou"] + 0.005, f"{C[k]['iou']:.3f}", ha="center", fontsize=8)
    ax[0].set_xticks(x); ax[0].set_xticklabels([NAME[k] for k in COND]); ax[0].set_ylabel("occupied IoU (matched pixels, GT pose)"); ax[0].grid(alpha=0.3, axis="y")
    ks = [k for k in COND if k not in ("LD", "LBM")]; xx = np.arange(len(ks)); gc = [G[k]["gap_closed"] for k in ks]
    err = np.array([[G[k]["gap_closed"] - G[k]["gap_closed_ci95"][0], G[k]["gap_closed_ci95"][1] - G[k]["gap_closed"]] for k in ks]).T
    ax[1].bar(xx, gc, color=[col[k] for k in ks], yerr=err, capsize=3); ax[1].axhline(0, color="k", lw=0.8); ax[1].axhline(1, color="k", lw=0.8, ls="--")
    for i, v in enumerate(gc):
        ax[1].text(i, v + (0.03 if v >= 0 else -0.08), f"{100*v:+.1f} %", ha="center", fontsize=8)
    ax[1].set_xticks(xx); ax[1].set_xticklabels([NAME[k] for k in ks]); ax[1].set_ylabel("fraction of the LD - LB-M gap closed"); ax[1].grid(alpha=0.3, axis="y")
    fig.suptitle(f"{tag}: occupancy by depth source (95 % CI: clip-block bootstrap)"); fig.tight_layout(); fig.savefig(os.path.join(fd, f"1_iou_gap_{tag}.png"), dpi=120); plt.close(fig)
    if d != G10.PRIMARY:
        return
    # 2 endpoint ECDF, 3 within thresholds
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for X in DEPTH_COND:
        if X == "LB":
            continue
        ep = np.sort(np.abs(g(f"z_{X}")[m] - zl) * rl); ax[0].plot(ep, np.linspace(0, 1, len(ep)), color=col[X], label=NAME[X])
    for t in (0.2, 0.4, 0.6):
        ax[0].axvline(t, color="k", lw=0.5, ls=":")
    ax[0].set_xscale("log"); ax[0].set_xlim(0.01, 30); ax[0].set_xlabel("3D endpoint error (m, log)"); ax[0].set_ylabel("ECDF"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    th = ("0.2", "0.4", "0.6"); ks = [X for X in DEPTH_COND if X != "LB"]; w = 0.8 / len(ks)
    for i, X in enumerate(ks):
        vals = [r["depth"][X][f"within_{t}m"] for t in th] + [r["depth"][X]["same_voxel"]]
        ax[1].bar(np.arange(4) + (i - (len(ks) - 1) / 2) * w, vals, w, color=col[X], label=NAME[X])
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(["within 0.2 m", "within 0.4 m", "within 0.6 m", "exact 0.2 m voxel"]); ax[1].set_ylabel("fraction of matched pixels"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(fd, "2_3_endpoint_ecdf_within.png"), dpi=120); plt.close(fig)
    # 4 endpoint vs range, 5 signed relative vs range
    rb = np.linspace(0, 60, 25); c_ = 0.5 * (rb[1:] + rb[:-1]); bi = np.digitize(zl, rb) - 1; ok = [(bi == k).sum() > 200 for k in range(len(c_))]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for X in ks:
        e = g(f"z_{X}")[m] - zl; ep = np.abs(e) * rl
        ax[0].plot(c_, [np.median(ep[bi == k]) if ok[k] else np.nan for k in range(len(c_))], "-o", ms=3, color=col[X], label=NAME[X])
        ax[1].plot(c_, [np.median((e / zl)[bi == k]) if ok[k] else np.nan for k in range(len(c_))], "-o", ms=3, color=col[X], label=NAME[X])
        ax[1].fill_between(c_, [np.percentile((e / zl)[bi == k], 25) if ok[k] else np.nan for k in range(len(c_))], [np.percentile((e / zl)[bi == k], 75) if ok[k] else np.nan for k in range(len(c_))], color=col[X], alpha=0.12)
    ax[0].axhline(0.2, color="k", lw=0.6, ls=":"); ax[0].set_xlabel("LiDAR range (m)"); ax[0].set_ylabel("median 3D endpoint error (m)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].axhline(0, color="k", lw=0.8); ax[1].set_xlabel("LiDAR range (m)"); ax[1].set_ylabel("signed relative depth error (median, IQR)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "4_5_error_vs_range.png"), dpi=120); plt.close(fig)
    # 6 occupied in LiDAR-free space
    fig, ax = plt.subplots(figsize=(8, 4)); x = np.arange(len(COND))
    ax.bar(x, [C[k]["pred_occ_on_lidar_free"] for k in COND], color=[col[k] for k in COND])
    for i, k in enumerate(COND):
        ax.text(i, C[k]["pred_occ_on_lidar_free"] + 0.01, f"{C[k]['pred_occ_on_lidar_free']:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([NAME[k] for k in COND]); ax.set_ylabel("predicted-occupied voxels in LiDAR-measured free space"); ax.set_ylim(0, 1); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(fd, "6_occupied_in_free.png"), dpi=120); plt.close(fig)


def tables(res, tag):
    L = ["# Gate 10 -- depth-source sufficiency: results\n",
         "Raw five-frame observed maps vs the LiDAR measurement of the same five sweeps (GT relative poses, anchor SSC grid, camera-frustum mask). "
         "Every condition uses the same matched pixels (LiDAR z-buffer support AND MoGe-2 valid) and the frozen Gate 9 mapper. Not completion IoU.\n"]
    for d, r in res.items():
        L.append(f"## {d[17:21]} ({r['n_clips']} clips; matched pixels = {100*r['matched_fraction_of_support']:.2f} % of Gate 9 support)\n")
        L.append("| condition | IoU | precision | recall | predicted occupied / mask | occupied in LiDAR-free | LiDAR occupied recovered | false-free rate | clip IoU median [p10, p90] |"); L.append("|---|---|---|---|---|---|---|---|---|")
        for k, v in r["conditions"].items():
            L.append(f"| {NAME[k]} | {v['iou']:.3f} | {v['precision']:.3f} | {v['recall']:.3f} | {v['pred_occ_fraction_of_mask']:.3f} | {v['pred_occ_on_lidar_free']:.3f} | {v['lidar_occ_recovered']:.3f} | {v['false_free_rate']:.4f} | {v['iou_per_clip_median']:.3f} [{v['iou_per_clip_p10']:.3f}, {v['iou_per_clip_p90']:.3f}] |")
        for ref in ("LBM", "LB"):
            G = r["gap"][ref]
            L.append(f"\n**Gap closed, reference {NAME[ref]}** (denominator IoU_LD - IoU_{NAME[ref]} = {G['_denominator']:.3f})\n")
            L.append("| condition | IoU gain over reference | gap closed | 95 % CI |"); L.append("|---|---|---|---|")
            for x, v in G.items():
                if not x.startswith("_"):
                    L.append(f"| {NAME[x]} | {v['iou_gain']:+.3f} | {100*v['gap_closed']:+.1f} % | [{100*v['gap_closed_ci95'][0]:+.1f}, {100*v['gap_closed_ci95'][1]:+.1f}] |")
        L.append("\n**Paired IoU differences (clip-block bootstrap)**: " + "; ".join(f"{k} {v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]" for k, v in r["paired_iou_diff_ci95"].items())
                 + f". Clip win rates: {r['clip_win_rate']}\n")
        S_ = r["scales"]
        L.append(f"Scales (median [p10, p90]): deployable MoGe clip scale {S_['l_scale']['median']:.2f} [{S_['l_scale']['p10']:.2f}, {S_['l_scale']['p90']:.2f}]; LiDAR oracle for LingBot {S_['oracle_scale_LBL']['median']:.2f} [{S_['oracle_scale_LBL']['p10']:.2f}, {S_['oracle_scale_LBL']['p90']:.2f}]; "
                 f"oracle/deployable {S_['LBL_over_LBM_ratio']['median']:.3f} [{S_['LBL_over_LBM_ratio']['p10']:.3f}, {S_['LBL_over_LBM_ratio']['p90']:.3f}]; LiDAR oracle for MoGe {S_['oracle_scale_ML']['median']:.3f} [{S_['oracle_scale_ML']['p10']:.3f}, {S_['oracle_scale_ML']['p90']:.3f}]. MoGe fov_x supplied: {r['moge_fov_x_deg']} deg\n")
        L.append(f"### Depth on identical matched pixels (n = {r['n_matched_rows']:,}, Gate 9's deterministic 20 % subsample)\n")
        L.append("| source | AbsRel median | AbsRel mean | signed rel median | signed m median | abs m median | endpoint median | p75 | p90 | p95 | exact voxel | within 0.1 | 0.2 | 0.4 | 0.6 | 1.0 m | in front | behind | endpoint in LiDAR-free |")
        L.append("|---|" + "---|" * 18)
        for X, s in r["depth"].items():
            L.append(f"| {NAME[X]} | {s['absrel_median']:.3f} | {s['absrel_mean']:.3f} | {s['signed_rel_median']:+.3f} | {s['signed_err_m_median']:+.2f} | {s['abs_err_m_median']:.2f} | {s['endpoint_m_median']:.2f} | {s['endpoint_m_p75']:.2f} | {s['endpoint_m_p90']:.2f} | {s['endpoint_m_p95']:.2f} | {s['same_voxel']:.4f} | "
                     f"{s['within_0.1m']:.3f} | {s['within_0.2m']:.3f} | {s['within_0.4m']:.3f} | {s['within_0.6m']:.3f} | {s['within_1.0m']:.3f} | {s['in_front']:.3f} | {s['behind']:.3f} | {s['ep_on_lidar_free']:.3f} |")
        ss = r["scalar_vs_structural"]
        L.append("\nPer-frame LiDAR scalar (diagnostic, on top of nothing): " + "; ".join(f"{NAME[X]}: AbsRel {v['absrel_median_after_per_frame_scalar']:.3f}, endpoint median {v['endpoint_m_median_after_per_frame_scalar']:.2f} m, within 0.2/0.4/0.6 m {v['within_0.2m_after_per_frame_scalar']:.3f}/{v['within_0.4m_after_per_frame_scalar']:.3f}/{v['within_0.6m_after_per_frame_scalar']:.3f}" for X, v in ss.items()) + "\n")
        for group, title in (("by_range", "range"), ("by_edge", "LiDAR depth discontinuity (Gate 9 classes)"), ("by_confidence", "LingBot confidence (Gate 9 bins; LingBot sources only -- MoGe-2 exposes no confidence, only a validity mask)")):
            L.append(f"**by {title}**\n"); L.append("| bin | source | n | AbsRel median | signed rel median | endpoint median | p90 | within 0.2 | within 0.4 | within 0.6 | exact voxel | endpoint in LiDAR-free |"); L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
            for lab, dd in r[group].items():
                for X, s in dd.items():
                    if s.get("n") and X != "LB":
                        L.append(f"| {lab} | {NAME[X]} | {s['n']:,} | {s['absrel_median']:.3f} | {s['signed_rel_median']:+.3f} | {s['endpoint_m_median']:.2f} | {s['endpoint_m_p90']:.2f} | {s['within_0.2m']:.3f} | {s['within_0.4m']:.3f} | {s['within_0.6m']:.3f} | {s['same_voxel']:.4f} | {s['ep_on_lidar_free']:.3f} |")
            L.append("")
        L.append(f"Representative clips (Gate 9 rule, LB-M IoU on Gate 9 support at p10 / p50 / p90): {r['representative_clips']}\n")
    open(os.path.join(G10.ART, f"results_table{tag}.md"), "w").write("\n".join(L) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--drives", nargs="+", default=[G10.PRIMARY]); a = ap.parse_args()
    for sub in ("tables", "figures"):
        os.makedirs(os.path.join(G10.ART, sub), exist_ok=True)
    res = {}
    for d in a.drives:
        r, (clips, P, F) = analyze_drive(d); res[d] = r; figures(d, r, clips, P, F)
        print(f"{d[17:21]}: " + ", ".join(f"{NAME[k]} {r['conditions'][k]['iou']:.3f}" for k in COND))
    tag = "" if a.drives == [G10.PRIMARY] else "_all"
    json.dump(res, open(os.path.join(G10.ART, f"analysis{tag}.json"), "w"), indent=1)
    tables(res, tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
