#!/usr/bin/env python
"""Gate 14 Stage 1 aggregation: observed-map metrics for LBM-L / LBM-G / NEW-L / NEW-G (shared and native
support) and LD-G, factorisation quantities, frozen promotion rule, depth / endpoint / registration
diagnostics, range and edge strata, tables, results_table[_all].md, figures 1-6 and promotion_result.json.

    python tools/gate14_depth_pose/analyze14.py [--drives ...]

Written before any NEW occupancy result was inspected; the promotion rule is read from promotion_rule.json
and implemented exactly as stated there (S_shared quantities on 0006).
"""
from __future__ import annotations
import argparse, csv, glob, hashlib, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g14                                                                                   # noqa: E402

MF = ["tp", "fp", "fn", "pred_occ_in_mask", "pred_occ_on_ref_free", "ref_occ_marked_free", "pred_occ_total", "ref_occ"]
KEYS = [f"{c}|{s}" for s in g14.SUPPORTS for c in g14.CONDITIONS] + ["LD-G|lidar"]
RANGES = list(g14.RANGE_NAMES) + ["all"]
THR = (0.2, 0.4, 0.6, 1.0)
EDGES = {0: "non-edge (<5 %)", 1: "moderate (5-20 %)", 2: "strong (>=20 %)"}
BOOT_BLOCK, N_BOOT = 20, 2000


def load(d):
    clips, pix = [], []
    for f in sorted(glob.glob(os.path.join(g14.CLIP_ROOT, d, "*.npz"))):
        z = np.load(f)
        c = {"t": int(z["t"]), "s_clip": float(z["s_clip"]), "s_new": float(z["s_new"]), "registration": z["registration"], "n_pixels": z["n_pixels"], "n_in_crop": int(z["n_in_crop"])}
        for k in KEYS:
            c[k] = z[f"m_{k}"].astype(np.float64)
        for k in z.files:
            if k.startswith("occ_") or k == "ref_occ":
                c[k] = z[k]
        clips.append(c); pix.append(z["pixels"]); F = [str(x) for x in z["pixel_fields"]]
    return clips, np.concatenate(pix), F


def om(v):
    """observed-map metrics from a pooled [len(MF)] vector."""
    tp, fp, fn, po, pof, rf, pt, ro = v
    sd = lambda a, b: float(a / b) if b > 0 else float("nan")
    return {"iou": sd(tp, tp + fp + fn), "precision": sd(tp, tp + fp), "recall": sd(tp, tp + fn), "pred_occ_in_mask": po, "pred_to_ref_ratio": sd(po, ro),
            "occupied_in_lidar_free": sd(pof, po), "false_free_rate": sd(rf, ro), "ref_occ": ro}


def pooled(clips, k, b=5):
    return sum(c[k][b] for c in clips)


def boot(clips, fn):
    n = len(clips); nb = int(np.ceil(n / BOOT_BLOCK)); rng = np.random.default_rng(0)
    blocks = [list(range(i * BOOT_BLOCK, min((i + 1) * BOOT_BLOCK, n))) for i in range(nb)]; v = []
    for _ in range(N_BOOT):
        v.append(fn([clips[i] for bb in rng.integers(0, nb, nb) for i in blocks[bb]]))
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def factors(clips, sup):
    I = lambda cl, cond: om(pooled(cl, f"{cond}|{sup}"))["iou"]
    ld = lambda cl: om(pooled(cl, "LD-G|lidar"))["iou"]
    q = {"pose_cost_LBM": lambda cl: I(cl, "LBM-G") - I(cl, "LBM-L"), "pose_cost_NEW": lambda cl: I(cl, "NEW-G") - I(cl, "NEW-L"),
         "depth_gain_GT": lambda cl: I(cl, "NEW-G") - I(cl, "LBM-G"), "deployable_gain": lambda cl: I(cl, "NEW-L") - I(cl, "LBM-L"),
         "gap_closed": lambda cl: (I(cl, "NEW-L") - I(cl, "LBM-L")) / (ld(cl) - I(cl, "LBM-L")),
         "gap_closed_GT": lambda cl: (I(cl, "NEW-G") - I(cl, "LBM-G")) / (ld(cl) - I(cl, "LBM-G"))}
    return {k: {"value": fn(clips), "ci95": boot(clips, fn)} for k, fn in q.items()}


def depth_stats(P, F, src, m):
    g = lambda k: P[m, F.index(k)].astype(np.float64)
    zl, zx, rl = g("z_lidar"), g(f"z_{src}"), g("ray_norm"); ok = np.isfinite(zx)
    zl, zx, rl = zl[ok], zx[ok], rl[ok]
    if len(zl) == 0:
        return {"n": 0}
    e = zx - zl; ep = np.abs(e) * rl
    return {"n": int(len(zl)), "absrel_median": float(np.median(np.abs(e) / zl)), "signed_rel_median": float(np.median(e / zl)), "abs_err_m_median": float(np.median(np.abs(e))),
            "endpoint_m_median": float(np.median(ep)), "endpoint_m_p75": float(np.percentile(ep, 75)), "endpoint_m_p90": float(np.percentile(ep, 90)), "endpoint_m_p95": float(np.percentile(ep, 95)),
            **{f"within_{t}m": float((ep <= t).mean()) for t in THR}, "exact_voxel": float(g(f"same_voxel_{src}")[ok].mean()), "endpoint_on_lidar_free": float(g(f"ep_on_ref_free_{src}")[ok].mean())}


def analyze(d, rule):
    clips, P, F = load(d); g = lambda k: P[:, F.index(k)].astype(np.float64)
    r = {"n_clips": len(clips)}
    r["observed"] = {k: om(pooled(clips, k)) for k in KEYS}
    r["by_range"] = {k: {rn: om(pooled(clips, k, b)) for b, rn in enumerate(RANGES)} for k in KEYS}
    r["factors"] = {s: factors(clips, s) for s in g14.SUPPORTS}
    npx = sum(c["n_pixels"] for c in clips)
    r["support"] = {"shared_pixels": int(npx[0, 0]), "native_LBM_pixels": int(npx[1, 0]), "native_NEW_pixels": int(npx[1, 1]),
                    "shared_over_native_LBM": float(npx[0, 0] / npx[1, 0]), "native_NEW_over_native_LBM": float(npx[1, 1] / npx[1, 0]),
                    "lattice_in_must3r_crop_fraction": float(np.mean([c["n_in_crop"] for c in clips]) / (140 * 518))}
    sh = (g("lb_valid") > 0) & (g("new_valid") > 0)
    r["depth_shared"] = {s: depth_stats(P, F, s, sh) for s in ("LBM", "NEW")}
    r["depth_native"] = {"LBM": depth_stats(P, F, "LBM", g("lb_valid") > 0), "NEW": depth_stats(P, F, "NEW", g("new_valid") > 0)}
    zl = g("z_lidar"); ec = g("edge_class")
    r["depth_by_range_shared"] = {rn: {s: depth_stats(P, F, s, sh & (zl >= g14.RANGE_BINS[i]) & (zl < g14.RANGE_BINS[i + 1])) for s in ("LBM", "NEW")} for i, rn in enumerate(g14.RANGE_NAMES)}
    r["depth_by_edge_shared"] = {lab: {s: depth_stats(P, F, s, sh & (ec == k)) for s in ("LBM", "NEW")} for k, lab in EDGES.items()}
    reg = np.concatenate([c["registration"] for c in clips])
    r["registration"] = {"rel_translation_err_m_median": float(np.median(reg[:, 0])), "rel_translation_err_m_p90": float(np.percentile(reg[:, 0], 90)),
                         "rel_rotation_err_deg_median": float(np.median(reg[:, 1])), "rel_rotation_err_deg_p90": float(np.percentile(reg[:, 1], 90)),
                         "gt_translation_m_median": float(np.median(reg[:, 2])), "pose_endpoint_displacement_m_median": float(np.median(g("pose_disp_m"))),
                         "pose_endpoint_displacement_m_p90": float(np.percentile(g("pose_disp_m"), 90)), "reprojection_px_median": float(np.nanmedian(g("reproj_px"))),
                         "reprojection_px_p90": float(np.nanpercentile(g("reproj_px"), 90)), "reprojection_valid_fraction": float(np.isfinite(g("reproj_px")).mean()),
                         "pose_displacement_by_range_median_m": {rn: float(np.median(g("pose_disp_m")[(zl >= g14.RANGE_BINS[i]) & (zl < g14.RANGE_BINS[i + 1])])) for i, rn in enumerate(g14.RANGE_NAMES)}}
    r["scales"] = {"s_clip_median": float(np.median([c["s_clip"] for c in clips])), "s_new_median": float(np.median([c["s_new"] for c in clips]))}
    # ---- frozen promotion rule (S_shared), exactly as promotion_rule.json
    O = r["observed"]; iL, iN = O["LBM-L|shared"]["iou"], O["NEW-L|shared"]["iou"]
    ref_by_band = {rn: r["by_range"]["LBM-L|shared"][rn]["ref_occ"] for rn in g14.RANGE_NAMES}; tot = sum(ref_by_band.values())
    major = [rn for rn in g14.RANGE_NAMES if ref_by_band[rn] >= 0.05 * tot]
    C1 = (iN - iL) >= 0.05 or ((iN - iL) / iL) >= 0.50
    C2 = (O["LBM-L|shared"]["occupied_in_lidar_free"] - O["NEW-L|shared"]["occupied_in_lidar_free"]) >= 0.05
    C3 = C1
    C4 = all(r["by_range"]["NEW-L|shared"][rn]["iou"] > r["by_range"]["LBM-L|shared"][rn]["iou"] for rn in major)
    dg = O["NEW-G|shared"]["iou"] - O["LBM-G|shared"]["iou"]; dg_rel = dg / O["LBM-G|shared"]["iou"]
    r["promotion"] = {"C1_iou_gain": bool(C1), "C2_free_space": bool(C2), "C3_not_density": bool(C3), "C4_range_consistency": bool(C4), "promote": bool(C1 and C2 and C3 and C4),
                      "values": {"iou_LBM-L": iL, "iou_NEW-L": iN, "gain": iN - iL, "relative": (iN - iL) / iL,
                                 "occ_in_free_LBM-L": O["LBM-L|shared"]["occupied_in_lidar_free"], "occ_in_free_NEW-L": O["NEW-L|shared"]["occupied_in_lidar_free"],
                                 "major_bands": major, "band_ious": {rn: [r["by_range"]["LBM-L|shared"][rn]["iou"], r["by_range"]["NEW-L|shared"][rn]["iou"]] for rn in g14.RANGE_NAMES},
                                 "native_gain": O["NEW-L|native"]["iou"] - O["LBM-L|native"]["iou"]},
                      "stage1_guide": {"depth_gain_GT": dg, "depth_gain_GT_relative": dg_rel, "depth_substantially_better_GT": bool(dg >= 0.05 or dg_rel >= 0.50),
                                       "iou_LD-G": O["LD-G|lidar"]["iou"], "iou_NEW-G": O["NEW-G|shared"]["iou"], "pose_cost_NEW": r["factors"]["shared"]["pose_cost_NEW"]["value"],
                                       "residual_depth_gap_GT": O["LD-G|lidar"]["iou"] - O["NEW-G|shared"]["iou"]},
                      "rule_sha256": hashlib.sha256(open(g14.PROMOTION_RULE, "rb").read()).hexdigest()}
    reps = {k: c["t"] for c in clips for k in ("x",) if False}
    r["representative_clips"] = sorted({c["t"] for c in clips if "occ_LBM-L" in c})
    T = os.path.join(g14.ART, "tables"); os.makedirs(T, exist_ok=True); tag = d[17:21]
    with open(os.path.join(T, f"observed_{tag}.csv"), "w", newline="") as fh:
        w = csv.writer(fh); keys = list(r["observed"][KEYS[0]]); w.writerow(["condition", "range"] + keys)
        for k in KEYS:
            w.writerow([k, "all"] + [r["observed"][k][x] for x in keys])
            for rn in g14.RANGE_NAMES:
                w.writerow([k, rn] + [r["by_range"][k][rn][x] for x in keys])
    with open(os.path.join(T, f"factors_{tag}.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["support", "quantity", "value", "ci_lo", "ci_hi"])
        for s, fs in r["factors"].items():
            for k, v in fs.items():
                w.writerow([s, k, v["value"], *v["ci95"]])
    with open(os.path.join(T, f"depth_{tag}.csv"), "w", newline="") as fh:
        w = csv.writer(fh); keys = list(r["depth_shared"]["LBM"]); w.writerow(["group", "bin", "source"] + keys)
        for grp, dd in (("shared", {"all": r["depth_shared"]}), ("native", {"all": r["depth_native"]}), ("range_shared", r["depth_by_range_shared"]), ("edge_shared", r["depth_by_edge_shared"])):
            for b, ss in dd.items():
                for s, v in ss.items():
                    w.writerow([grp, b, s] + [v.get(x, "") for x in keys])
    return r, clips, P, F


def figures(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = os.path.join(g14.ART, "figures"); os.makedirs(os.path.join(fd, "maps"), exist_ok=True)
    col = {"LBM-L": "#c0392b", "LBM-G": "#e6907f", "NEW-L": "#2c6fbb", "NEW-G": "#8fb8e6", "LD-G": "#222222"}
    for d, (r, clips, P, F) in res.items():
        tag = d[17:21]; O = r["observed"]
        fig, ax = plt.subplots(1, 3, figsize=(17, 4.3))
        names = ["LBM-L", "LBM-G", "NEW-L", "NEW-G"]; x = np.arange(len(names) + 1)
        for j, s in enumerate(g14.SUPPORTS):
            vals = [O[f"{n}|{s}"]["iou"] for n in names] + ([O["LD-G|lidar"]["iou"]] if j == 0 else [np.nan])
            ax[0].bar(x + (j - 0.5) * 0.38, vals, 0.38, color=[col[n] for n in names] + [col["LD-G"]], alpha=1.0 if s == "shared" else 0.5, label=f"{s} support")
        ax[0].set_xticks(x); ax[0].set_xticklabels(names + ["LD-G"]); ax[0].set_ylabel("observed-map IoU"); ax[0].legend(); ax[0].grid(alpha=0.3, axis="y"); ax[0].set_title(f"1. raw IoU ({tag}; solid shared, light native)")
        Fs = r["factors"]["shared"]; ks = ["deployable_gain", "depth_gain_GT"]
        ax[1].bar(ks, [Fs[k]["value"] for k in ks], color=["#2c6fbb", "#8fb8e6"], yerr=np.array([[Fs[k]["value"] - Fs[k]["ci95"][0], Fs[k]["ci95"][1] - Fs[k]["value"]] for k in ks]).T, capsize=3)
        ax[1].axhline(0.05, color="k", ls=":", lw=0.8); ax[1].axhline(0, color="k", lw=0.8); ax[1].set_ylabel("IoU gain (shared)"); ax[1].set_title("2. deployable gain (NEW-L - LBM-L) and GT-pose depth gain"); ax[1].grid(alpha=0.3, axis="y")
        ks = ["pose_cost_LBM", "pose_cost_NEW"]
        ax[2].bar(ks, [Fs[k]["value"] for k in ks], color=["#c0392b", "#2c6fbb"], yerr=np.array([[Fs[k]["value"] - Fs[k]["ci95"][0], Fs[k]["ci95"][1] - Fs[k]["value"]] for k in ks]).T, capsize=3)
        ax[2].axhline(0, color="k", lw=0.8); ax[2].set_ylabel("IoU(G) - IoU(L)"); ax[2].set_title("3. pose cost before vs after better depth"); ax[2].grid(alpha=0.3, axis="y")
        fig.tight_layout(); fig.savefig(os.path.join(fd, f"1_2_3_iou_gain_posecost_{tag}.png"), dpi=120); plt.close(fig)
        fig, ax = plt.subplots(1, 3, figsize=(17, 4.3)); g = lambda k: P[:, F.index(k)].astype(np.float64); sh = (g("lb_valid") > 0) & (g("new_valid") > 0)
        for s, c in (("LBM", col["LBM-L"]), ("NEW", col["NEW-L"])):
            zx = g(f"z_{s}")[sh]; ok = np.isfinite(zx); ep = np.sort(np.abs(zx[ok] - g("z_lidar")[sh][ok]) * g("ray_norm")[sh][ok])
            ax[0].plot(ep, np.linspace(0, 1, len(ep)), color=c, label=s)
        for t in (0.2, 0.6):
            ax[0].axvline(t, color="k", ls=":", lw=0.6)
        ax[0].set_xscale("log"); ax[0].set_xlim(0.01, 30); ax[0].set_xlabel("3D endpoint error (m), shared LiDAR-supported pixels"); ax[0].set_ylabel("ECDF"); ax[0].legend(); ax[0].grid(alpha=0.3); ax[0].set_title("4. endpoint-error ECDF")
        xr = np.arange(5)
        for k in ["LBM-L", "LBM-G", "NEW-L", "NEW-G"]:
            ax[1].plot(xr, [r["by_range"][f"{k}|shared"][rn]["iou"] for rn in g14.RANGE_NAMES], "-o", color=col[k], label=k)
            ax[2].plot(xr, [r["by_range"][f"{k}|shared"][rn]["occupied_in_lidar_free"] for rn in g14.RANGE_NAMES], "-o", color=col[k], label=k)
        ax[1].plot(xr, [r["by_range"]["LD-G|lidar"][rn]["iou"] for rn in g14.RANGE_NAMES], "-o", color=col["LD-G"], label="LD-G")
        ax[2].plot(xr, [r["by_range"]["LD-G|lidar"][rn]["occupied_in_lidar_free"] for rn in g14.RANGE_NAMES], "-o", color=col["LD-G"], label="LD-G")
        for a_, tt in zip(ax[1:], ("5a. IoU by voxel range (shared)", "5b. occupied-in-LiDAR-free by range (shared)")):
            a_.set_xticks(xr); a_.set_xticklabels(g14.RANGE_NAMES); a_.legend(fontsize=7); a_.grid(alpha=0.3); a_.set_title(tt)
        fig.tight_layout(); fig.savefig(os.path.join(fd, f"4_5_endpoint_range_{tag}.png"), dpi=120); plt.close(fig)
        DIMS = (256, 256, 32); N = int(np.prod(DIMS))
        for c in clips:
            if "occ_LBM-L" not in c:
                continue
            ref = np.zeros(N, bool); ref[c["ref_occ"]] = True; rr = ref.reshape(DIMS).any(2)
            fig, ax = plt.subplots(1, 5, figsize=(22, 4.6))
            for j, k in enumerate(["LBM-L", "LBM-G", "NEW-L", "NEW-G", "LD-G"]):
                o = np.zeros(N, bool); o[c[f"occ_{k}"]] = True; o = o.reshape(DIMS).any(2)
                img = np.ones(DIMS[:2] + (3,)); img[rr & ~o] = (0.8, 0.8, 0.8); img[o & ~rr] = (0.85, 0.2, 0.2); img[o & rr] = (0.1, 0.55, 0.2)
                key = f"{k}|shared" if k != "LD-G" else "LD-G|lidar"; v = c[key][5]; iou = v[0] / max(v[:3].sum(), 1)
                ax[j].imshow(np.transpose(img, (1, 0, 2)), origin="lower"); ax[j].set_xticks([]); ax[j].set_yticks([]); ax[j].set_title(f"{k}: IoU {iou:.3f}", fontsize=10)
            fig.suptitle(f"6. {tag} t={c['t']} (Gate 9 rule representative), shared support; bird's-eye: green hit, red occupied elsewhere, grey LiDAR-occupied missed", fontsize=9)
            fig.tight_layout(); fig.savefig(os.path.join(fd, "maps", f"6_{tag}_t{c['t']:05d}.png"), dpi=80); plt.close(fig)


def table(res, tag):
    L = ["# Gate 14 Stage 1 -- depth x pose observed geometry: results\n",
         "Observed five-frame maps vs the five-sweep LiDAR measurement inside the camera frusta (Gate 9 protocol). LBM = LingBot x causal MoGe-2 clip scale; NEW = MUSt3R-512 (official code) "
         "x causal MoGe-2 clip scale; L = LingBot local-reset pose + pred_K; G = GT pose + calibrated camera. shared = LingBot conf >= 1.5 AND MUSt3R-valid pixels; native = each source's own support. "
         "LD-G = LiDAR camera depth + GT pose (Gate 9 LD). Voxel range = anchor-camera z.\n"]
    for d, (r, *_rest) in res.items():
        L.append(f"## {d[17:21]} ({r['n_clips']} clips)\n")
        S_ = r["support"]
        L.append(f"Support: shared {S_['shared_pixels']:,} px = {100*S_['shared_over_native_LBM']:.1f} % of LBM native; NEW native / LBM native = {S_['native_NEW_over_native_LBM']:.2f}; "
                 f"lattice inside the MUSt3R crop {100*S_['lattice_in_must3r_crop_fraction']:.1f} %. Scales: s_clip median {r['scales']['s_clip_median']:.2f}, s_new median {r['scales']['s_new_median']:.3f}.\n")
        L.append("| condition | IoU | precision | recall | predicted occ / ref occ | occupied in LiDAR-free | false-free |"); L.append("|---|---|---|---|---|---|---|")
        for k in KEYS:
            v = r["observed"][k]; L.append(f"| {k} | {v['iou']:.3f} | {v['precision']:.3f} | {v['recall']:.3f} | {v['pred_to_ref_ratio']:.2f} | {v['occupied_in_lidar_free']:.3f} | {v['false_free_rate']:.4f} |")
        for s in g14.SUPPORTS:
            L.append(f"\nFactors ({s}): " + "; ".join(f"{k} {v['value']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]" for k, v in r["factors"][s].items()))
        L.append("\n| depth (shared LiDAR-supported pixels) | n | AbsRel | signed rel | abs m | endpoint med / p75 / p90 / p95 | within 0.2 / 0.4 / 0.6 / 1.0 m | exact voxel | endpoint on LiDAR-free |"); L.append("|---|---|---|---|---|---|---|---|---|")
        for grp, dd in (("shared", r["depth_shared"]), ("native", r["depth_native"])):
            for s, v in dd.items():
                L.append(f"| {s} {grp} | {v['n']:,} | {v['absrel_median']:.3f} | {v['signed_rel_median']:+.3f} | {v['abs_err_m_median']:.2f} | {v['endpoint_m_median']:.2f} / {v['endpoint_m_p75']:.2f} / {v['endpoint_m_p90']:.2f} / {v['endpoint_m_p95']:.2f} | "
                         f"{v['within_0.2m']:.3f} / {v['within_0.4m']:.3f} / {v['within_0.6m']:.3f} / {v['within_1.0m']:.3f} | {v['exact_voxel']:.4f} | {v['endpoint_on_lidar_free']:.3f} |")
        L.append("\n| range (shared) | LBM-L IoU / occ-in-free | LBM-G | NEW-L | NEW-G | LD-G | endpoint med LBM / NEW | within 0.2 LBM / NEW |"); L.append("|---|---|---|---|---|---|---|---|")
        for rn in g14.RANGE_NAMES:
            cell = lambda k: f"{r['by_range'][k][rn]['iou']:.3f} / {r['by_range'][k][rn]['occupied_in_lidar_free']:.3f}"
            dr = r["depth_by_range_shared"][rn]
            L.append(f"| {rn} | {cell('LBM-L|shared')} | {cell('LBM-G|shared')} | {cell('NEW-L|shared')} | {cell('NEW-G|shared')} | {cell('LD-G|lidar')} | "
                     f"{dr['LBM'].get('endpoint_m_median', float('nan')):.2f} / {dr['NEW'].get('endpoint_m_median', float('nan')):.2f} | {dr['LBM'].get('within_0.2m', float('nan')):.3f} / {dr['NEW'].get('within_0.2m', float('nan')):.3f} |")
        L.append("\n| edge (shared) | n | AbsRel LBM / NEW | endpoint med LBM / NEW | within 0.2 LBM / NEW | exact voxel LBM / NEW |"); L.append("|---|---|---|---|---|---|")
        for lab, dd in r["depth_by_edge_shared"].items():
            L.append(f"| {lab} | {dd['LBM']['n']:,} | {dd['LBM']['absrel_median']:.3f} / {dd['NEW']['absrel_median']:.3f} | {dd['LBM']['endpoint_m_median']:.2f} / {dd['NEW']['endpoint_m_median']:.2f} | "
                     f"{dd['LBM']['within_0.2m']:.3f} / {dd['NEW']['within_0.2m']:.3f} | {dd['LBM']['exact_voxel']:.4f} / {dd['NEW']['exact_voxel']:.4f} |")
        L.append(f"\nRegistration (L vs G): {json.dumps(r['registration'])}\n")
        L.append(f"Promotion (frozen rule, shared): {json.dumps(r['promotion'])}\n")
    open(os.path.join(g14.ART, f"results_table{tag}.md"), "w").write("\n".join(L) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--drives", nargs="+", default=[g14.PRIMARY]); a = ap.parse_args()
    rule = json.load(open(g14.PROMOTION_RULE))
    res = {d: analyze(d, rule) for d in a.drives}
    figures(res); tag = "" if a.drives == [g14.PRIMARY] else "_all"
    json.dump({d: v[0] for d, v in res.items()}, open(os.path.join(g14.ART, f"analysis{tag}.json"), "w"), indent=1); table(res, tag)
    if g14.PRIMARY in res and tag == "":
        p = res[g14.PRIMARY][0]["promotion"]
        json.dump({"drive": g14.PRIMARY, **p}, open(os.path.join(g14.ART, "promotion_result.json"), "w"), indent=1)
    for d, v in res.items():
        print(d[17:21], {k: round(v[0]["observed"][k]["iou"], 4) for k in KEYS}, "promote", v[0]["promotion"]["promote"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
