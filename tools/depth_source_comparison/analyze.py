#!/usr/bin/env python
"""Aggregate the controlled depth-source comparison: support accounting, Gate 9 reproduction,
primary (S1) occupancy + depth metrics, paired per-clip differences, gap closed, secondary (S0)
occupancy, compact range breakdown, CSVs, comparison_table[_all].md, figures and map slices.

Aggregation: IoU / precision / recall pool voxel counts over all clips. Depth metrics pool the
per-pixel values of Gate 9's deterministic 20 % subsample rows inside the mask over all clips;
bias = median of (z - z_LiDAR) / z_LiDAR (and median of z - z_LiDAR in m); AbsRel median and mean of
|z - z_LiDAR| / z_LiDAR; endpoint error = |z - z_LiDAR| x |ray| (same ray, so the 3D endpoint distance).

    python tools/depth_source_comparison/analyze.py [--drives 2013_05_28_drive_0006_sync ...]
"""
from __future__ import annotations
import argparse, csv, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dsc as D                                                                            # noqa: E402
sys.path.insert(0, D.G9_DIR)
import common as G9                                                                        # noqa: E402

SRCS = ("LBM", "MG", "LBO")
PAIRS = (("MG", "LBM"), ("LBO", "LBM"), ("LBO", "MG"), ("LD", "MG"), ("LD", "LBO"))


def load(drive):
    clips, pix, F = [], [], None
    for f in sorted(glob.glob(os.path.join(D.DATA_ROOT, "clips", drive, "*.npz"))):
        z = np.load(f)
        c = {k: z[k] for k in z.files if k.startswith("m_") or k.startswith("n_pixels_") or k.startswith("occ_") or k in ("range_rejected_S1", "ref_occ")}
        c.update({"t": int(z["t"]), "n_eval": int(z["n_eval"]), "l_scale": float(z["l_scale"]), "lbo_S1": float(z["lbo_scale_S1"]), "lbo_S0": float(z["lbo_scale_S0"])})
        clips.append(c); p = z["pixels"]; pix.append(np.concatenate([p, np.full((len(p), 1), c["t"], np.float32)], 1)); F = [str(x) for x in z["pixel_fields"]] + ["t"]
    return clips, np.concatenate(pix), F


def counts(clips, k):
    return np.stack([c[f"m_{k}"] for c in clips]).astype(np.float64)


def iou_s(s):
    return s[0] / max(s[0] + s[1] + s[2], 1)


def occ(clips, k):
    s = counts(clips, k).sum(0); tp, fp, fn, po, pof = s[:5]
    return {"iou": iou_s(s), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "occ_in_lidar_free_fraction": pof / max(po, 1),
            "occ_in_lidar_free_per_clip": pof / len(clips), "pred_occ_in_mask_per_clip": po / len(clips)}


def per_clip_iou(clips, k):
    m = counts(clips, k); return m[:, 0] / np.maximum(m[:, :3].sum(1), 1)


def boot(clips, fn):
    n = len(clips); nb = int(np.ceil(n / D.BOOT_BLOCK)); rng = np.random.default_rng(0)
    blocks = [list(range(i * D.BOOT_BLOCK, min((i + 1) * D.BOOT_BLOCK, n))) for i in range(nb)]; vals = []
    for _ in range(D.N_BOOT):
        vals.append(fn([clips[i] for b in rng.integers(0, nb, nb) for i in blocks[b]]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def gap(clips, x, ref="LBM", ld="LD"):
    a, b, c = iou_s(counts(clips, x).sum(0)), iou_s(counts(clips, ref).sum(0)), iou_s(counts(clips, ld).sum(0))
    return (a - b) / (c - b)


def depth(P, F, X, m):
    g = lambda k: P[m, F.index(k)].astype(np.float64)
    zl, zx, rl = g("z_lidar"), g(f"z_{X}"), g("ray_norm"); e = zx - zl; ep = np.abs(e) * rl
    return {"n": int(m.sum()), "bias_rel_median": float(np.median(e / zl)), "bias_m_median": float(np.median(e)), "absrel_median": float(np.median(np.abs(e) / zl)),
            "absrel_mean": float(np.mean(np.abs(e) / zl)), "endpoint_m_median": float(np.median(ep)), "endpoint_m_p90": float(np.percentile(ep, 90)),
            "within_0.2m": float((ep <= 0.2).mean()), "within_0.6m": float((ep <= 0.6).mean()), "exact_voxel": float(g(f"same_voxel_{X}").mean()),
            "endpoint_in_lidar_free": float(g(f"ep_on_ref_free_{X}").mean())}


def csvw(path, rows):
    if rows:
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(r) for r in rows]


def pct(q, xs):
    return float(np.percentile(xs, q))


def analyze(d):
    clips, P, F = load(d); g = lambda k: P[:, F.index(k)].astype(np.float64); s1 = g("in_S1") > 0; s0 = g("in_S0") > 0; tag = d[17:21]
    px = {k: int(sum(c[f"n_pixels_{k}"].sum() for c in clips)) for k in ("g9", "g9gate", "S0", "S1")}
    rr = np.stack([c["range_rejected_S1"] for c in clips]).sum(0)
    r = {"n_clips": len(clips),
         "support": {"gate9_support_pixels": px["g9"], "removed_by_moge_mask": 1 - px["S0"] / px["g9"], "removed_by_lingbot_conf_gate": 1 - px["g9gate"] / px["g9"],
                     "S1_pixels": px["S1"], "S1_removed_total": 1 - px["S1"] / px["g9"],
                     "mapper_range_rejected_fraction_of_S1": {"LD": rr[0] / px["S1"], "LB-M": rr[1] / px["S1"], "MG": rr[2] / px["S1"], "LB-O": rr[3] / px["S1"]}},
         "gate9_reproduction": {D.NAME[k]: occ(clips, k) for k in D.G9_REPRO},
         "primary_S1": {D.NAME[k]: occ(clips, k) for k in D.PRIMARY_COND},
         "secondary_S0": {D.NAME[k[:-2]]: occ(clips, k) for k in ("LD_u", "LBM_u", "MG_u", "LBO_u")},
         "depth_S1": {D.NAME[X]: depth(P, F, X, s1) for X in SRCS},
         "depth_S0": {D.NAME[X]: depth(P, F, X if X != "LBO" else "LBO_u", s0) for X in SRCS}}
    zm, zb, zlr, rlr = g("z_MG")[s1], g("z_LBM")[s1], g("z_lidar")[s1], g("ray_norm")[s1]
    r["MG_vs_LBM_on_S1"] = {"median_abs_depth_diff_m": float(np.median(np.abs(zm - zb))), "frac_abs_diff_gt_0.2m": float((np.abs(zm - zb) > 0.2).mean()),
                            "signed_rel_error_correlation": float(np.corrcoef((zm - zlr) / zlr, (zb - zlr) / zlr)[0, 1]),
                            "both_within_0.2m": float(((np.abs(zm - zlr) * rlr <= 0.2) & (np.abs(zb - zlr) * rlr <= 0.2)).mean()),
                            "either_within_0.2m": float(((np.abs(zm - zlr) * rlr <= 0.2) | (np.abs(zb - zlr) * rlr <= 0.2)).mean())}
    r["gap_closed_S1"] = {D.NAME[x]:{"value": gap(clips, x), "ci95": boot(clips, lambda s, x=x: gap(s, x))} for x in ("MG", "LBO")}
    r["gap_closed_S1"]["_denominator_LD_minus_LBM"] = iou_s(counts(clips, "LD").sum(0)) - iou_s(counts(clips, "LBM").sum(0))
    r["gap_closed_S0"] = {D.NAME[x]: gap(clips, x + "_u", "LBM_u", "LD_u") for x in ("MG", "LBO")}
    r["paired_S1"] = {}
    for a_, b_ in PAIRS:
        dd = per_clip_iou(clips, a_) - per_clip_iou(clips, b_)
        r["paired_S1"][f"{D.NAME[a_]} - {D.NAME[b_]}"] = {"pooled_iou_diff": iou_s(counts(clips, a_).sum(0)) - iou_s(counts(clips, b_).sum(0)),
                                                         "pooled_ci95": boot(clips, lambda s, a_=a_, b_=b_: iou_s(counts(s, a_).sum(0)) - iou_s(counts(s, b_).sum(0))),
                                                         "per_clip_mean": float(dd.mean()), "per_clip_median": float(np.median(dd)), "per_clip_p10_p90": [pct(10, dd), pct(90, dd)],
                                                         "win_rate": float((dd > 0).mean())}
    zl = g("z_lidar"); r["range_S1"] = {}
    for i, lab in enumerate(G9.RANGE_NAMES):
        m = s1 & (zl >= G9.RANGE_BINS[i]) & (zl < G9.RANGE_BINS[i + 1])
        r["range_S1"][lab] = {D.NAME[X]: depth(P, F, X, m) for X in SRCS}
    ls = [c["l_scale"] for c in clips]; lo = [c["lbo_S1"] for c in clips]; rat = [b / a for a, b in zip(ls, lo)]
    r["scalars"] = {"moge_clip_scale": [pct(50, ls), pct(10, ls), pct(90, ls)], "lbo_scalar_S1": [pct(50, lo), pct(10, lo), pct(90, lo)], "lbo_over_moge_scale": [pct(50, rat), pct(10, rat), pct(90, rat)]}
    g9 = per_clip_iou(clips, "LBM_g9"); order = np.argsort(g9, kind="stable")
    r["representative_clips"] = {lab: int(clips[order[int(q * (len(order) - 1))]]["t"]) for lab, q in (("bad", 0.1), ("median", 0.5), ("good", 0.9))}
    T = os.path.join(D.ART, "tables")
    csvw(os.path.join(T, f"occupancy_{tag}.csv"), [{"mask": grp, "condition": k, **v} for grp in ("gate9_reproduction", "primary_S1", "secondary_S0") for k, v in r[grp].items()])
    csvw(os.path.join(T, f"depth_{tag}.csv"), [{"mask": grp, "source": k, **v} for grp in ("depth_S1", "depth_S0") for k, v in r[grp].items()])
    csvw(os.path.join(T, f"paired_{tag}.csv"), [{"pair": k, **{kk: (json.dumps(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()}} for k, v in r["paired_S1"].items()])
    csvw(os.path.join(T, f"range_{tag}.csv"), [{"range": lab, "source": X, **v} for lab, dd in r["range_S1"].items() for X, v in dd.items()])
    csvw(os.path.join(T, f"per_clip_{tag}.csv"), [{"t": c["t"], "moge_clip_scale": c["l_scale"], "lbo_scalar_S1": c["lbo_S1"], **{f"iou_{k}": float(iou_s(c[f'm_{k}'])) for k in D.RUNS}} for c in clips])
    return r, clips, P, F


def figures(d, r, clips, P, F):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = os.path.join(D.ART, "figures"); os.makedirs(os.path.join(fd, "maps"), exist_ok=True); tag = d[17:21]
    col = {"LD": "#222222", "LB-M": "#c0392b", "MG": "#2c6fbb", "LB-O": "#e59866"}
    C = r["primary_S1"]; names = list(C)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].bar(names, [C[k]["iou"] for k in names], color=[col[k] for k in names])
    for i, k in enumerate(names):
        ax[0].text(i, C[k]["iou"] + 0.005, f"{C[k]['iou']:.3f}", ha="center", fontsize=8)
    ax[0].set_ylabel("occupied IoU (S1 mask, GT pose)"); ax[0].grid(alpha=0.3, axis="y")
    G = r["gap_closed_S1"]; ks = ["MG", "LB-O"]; v = [G[k]["value"] for k in ks]
    ax[1].bar(ks, v, color=[col[k] for k in ks], yerr=np.array([[G[k]["value"] - G[k]["ci95"][0], G[k]["ci95"][1] - G[k]["value"]] for k in ks]).T, capsize=3)
    for i, x in enumerate(v):
        ax[1].text(i, x + 0.02, f"{100*x:+.1f} %", ha="center", fontsize=8)
    ax[1].axhline(0, color="k", lw=0.8); ax[1].set_ylabel("fraction of LD - LB-M gap closed"); ax[1].grid(alpha=0.3, axis="y")
    fig.suptitle(f"{tag}: controlled depth-source comparison (95 % CI: clip-block bootstrap)"); fig.tight_layout(); fig.savefig(os.path.join(fd, f"iou_gap_{tag}.png"), dpi=120); plt.close(fig)
    if d != D.PRIMARY:
        return
    g = lambda k: P[:, F.index(k)].astype(np.float64); m = g("in_S1") > 0; zl, rl = g("z_lidar")[m], g("ray_norm")[m]
    rb = np.linspace(0, 60, 25); c_ = 0.5 * (rb[1:] + rb[:-1]); bi = np.digitize(zl, rb) - 1; ok = [(bi == k).sum() > 200 for k in range(len(c_))]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for X, lab in (("LBM", "LB-M"), ("MG", "MG"), ("LBO", "LB-O")):
        e = g(f"z_{X}")[m] - zl; ep = np.abs(e) * rl; s_ = np.sort(ep)
        ax[0].plot(s_, np.linspace(0, 1, len(s_)), color=col[lab], label=lab)
        ax[1].plot(c_, [np.median(ep[bi == k]) if ok[k] else np.nan for k in range(len(c_))], "-o", ms=3, color=col[lab], label=lab)
        ax[2].plot(c_, [np.median((e / zl)[bi == k]) if ok[k] else np.nan for k in range(len(c_))], "-o", ms=3, color=col[lab], label=lab)
    for t in (0.2, 0.6):
        ax[0].axvline(t, color="k", lw=0.5, ls=":")
    ax[0].set_xscale("log"); ax[0].set_xlim(0.01, 30); ax[0].set_xlabel("3D endpoint error (m)"); ax[0].set_ylabel("ECDF")
    ax[1].axhline(0.2, color="k", lw=0.5, ls=":"); ax[1].set_xlabel("LiDAR range (m)"); ax[1].set_ylabel("median endpoint error (m)")
    ax[2].axhline(0, color="k", lw=0.8); ax[2].set_xlabel("LiDAR range (m)"); ax[2].set_ylabel("median signed relative depth error")
    for a_ in ax:
        a_.grid(alpha=0.3); a_.legend(fontsize=8, loc="upper left")
    fig.suptitle("0006, S1 matched pixels"); fig.tight_layout(); fig.savefig(os.path.join(fd, "depth_error_0006.png"), dpi=120); plt.close(fig)
    DIMS = G9.DIMS; N = int(np.prod(DIMS)); byt = {c["t"]: c for c in clips}
    for lab, t in r["representative_clips"].items():
        c = byt[t]; ref = np.zeros(N, bool); ref[c["ref_occ"]] = True; ref = ref.reshape(DIMS)
        fig, ax = plt.subplots(1, 4, figsize=(17, 4.6))
        for j, k in enumerate(D.PRIMARY_COND):
            o3 = np.zeros(N, bool); o3[c[f"occ_{k}"]] = True; o = o3.reshape(DIMS).any(2); rr = ref.any(2)
            img = np.ones(o.shape + (3,)); img[rr & ~o] = (0.80, 0.80, 0.80); img[o & rr] = (0.10, 0.60, 0.20); img[o & ~rr] = (0.85, 0.20, 0.20)
            ax[j].imshow(np.transpose(img, (1, 0, 2)), origin="lower"); ax[j].set_xticks([]); ax[j].set_yticks([])
            ax[j].set_title(f"{D.NAME[k]}: IoU {iou_s(c[f'm_{k}']):.3f}", fontsize=10)
        fig.suptitle(f"0006 {lab} clip t={t} (Gate 9 rule), bird's-eye max over height; green = occupied & LiDAR-occupied, red = other occupied, grey = missed", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(fd, "maps", f"0006_{lab}_t{t:05d}.png"), dpi=85); plt.close(fig)


def table(res, tag):
    L = ["# Controlled depth-source comparison: results\n",
         "Raw five-frame observed maps vs the LiDAR measurement of the same five sweeps; KITTI-360 GT relative poses, Gate 9 clips, lattice, grid, "
         "evaluation mask and frozen mapper. Not completion IoU; says nothing about Occ3D numbers.\n",
         "Masks: **S1 (primary)** = Gate 9 LiDAR support AND MoGe-2 valid AND LingBot conf >= 1.5; S0 (secondary) = support AND MoGe-2 valid. "
         "IoU pools voxel counts over clips; depth metrics pool the Gate 9 deterministic 20 % pixel subsample inside the mask; bias = median signed error; "
         "endpoint = |z - z_LiDAR| x |ray|.\n"]
    for d, r in res.items():
        S_ = r["support"]; tagd = d[17:21]
        L.append(f"## {tagd} ({r['n_clips']} clips)\n")
        L.append(f"Support: Gate 9 LiDAR support {S_['gate9_support_pixels']:,} px; MoGe mask removes {100*S_['removed_by_moge_mask']:.2f} %, LingBot gate {100*S_['removed_by_lingbot_conf_gate']:.2f} %, "
                 f"S1 keeps {S_['S1_pixels']:,} ({100*(1-S_['S1_removed_total']):.2f} %). Mapper 1-60 m range rejects of S1 rays: " + ", ".join(f"{k} {100*v:.3f} %" for k, v in S_["mapper_range_rejected_fraction_of_S1"].items()) + ".\n")
        L.append("| primary S1 | IoU | precision | recall | occupied in LiDAR-free (fraction / per clip) |"); L.append("|---|---|---|---|---|")
        for k, v in r["primary_S1"].items():
            L.append(f"| {k} | {v['iou']:.3f} | {v['precision']:.3f} | {v['recall']:.3f} | {v['occ_in_lidar_free_fraction']:.3f} / {v['occ_in_lidar_free_per_clip']:,.0f} |")
        G = r["gap_closed_S1"]
        L.append(f"\nGap closed (S1, denominator IoU_LD - IoU_LB-M = {G['_denominator_LD_minus_LBM']:.3f}): MG {100*G['MG']['value']:+.1f} % [{100*G['MG']['ci95'][0]:+.1f}, {100*G['MG']['ci95'][1]:+.1f}]; "
                 f"LB-O {100*G['LB-O']['value']:+.1f} % [{100*G['LB-O']['ci95'][0]:+.1f}, {100*G['LB-O']['ci95'][1]:+.1f}] (95 % clip-block bootstrap).\n")
        L.append("| depth on S1 | n px | bias rel (median) | bias m (median) | AbsRel median / mean | endpoint median | p90 | within 0.2 m | within 0.6 m | exact 0.2 m voxel |"); L.append("|---|---|---|---|---|---|---|---|---|---|")
        for k, v in r["depth_S1"].items():
            L.append(f"| {k} | {v['n']:,} | {v['bias_rel_median']:+.3f} | {v['bias_m_median']:+.2f} | {v['absrel_median']:.3f} / {v['absrel_mean']:.3f} | {v['endpoint_m_median']:.2f} | {v['endpoint_m_p90']:.2f} | {v['within_0.2m']:.3f} | {v['within_0.6m']:.3f} | {v['exact_voxel']:.4f} |")
        L.append("\n| paired per clip (S1) | pooled IoU diff [95 % CI] | per-clip mean | median [p10, p90] | clips improved |"); L.append("|---|---|---|---|---|")
        for k, v in r["paired_S1"].items():
            L.append(f"| {k} | {v['pooled_iou_diff']:+.3f} [{v['pooled_ci95'][0]:+.3f}, {v['pooled_ci95'][1]:+.3f}] | {v['per_clip_mean']:+.3f} | {v['per_clip_median']:+.3f} [{v['per_clip_p10_p90'][0]:+.3f}, {v['per_clip_p10_p90'][1]:+.3f}] | {100*v['win_rate']:.0f} % |")
        L.append("\n| range (S1) | n px | LB-M endpoint med / within 0.2 / within 0.6 / bias | MG | LB-O |"); L.append("|---|---|---|---|---|")
        for lab, dd in r["range_S1"].items():
            L.append(f"| {lab} | {dd['LB-M']['n']:,} | " + " | ".join(f"{dd[X]['endpoint_m_median']:.2f} / {dd[X]['within_0.2m']:.3f} / {dd[X]['within_0.6m']:.3f} / {dd[X]['bias_rel_median']:+.3f}" for X in ("LB-M", "MG", "LB-O")) + " |")
        L.append("\n| secondary S0 (no LingBot gate) | IoU | precision | recall | occupied in LiDAR-free |"); L.append("|---|---|---|---|---|")
        for k, v in r["secondary_S0"].items():
            L.append(f"| {k} | {v['iou']:.3f} | {v['precision']:.3f} | {v['recall']:.3f} | {v['occ_in_lidar_free_fraction']:.3f} |")
        L.append(f"\nS0 gap closed: MG {100*r['gap_closed_S0']['MG']:+.1f} %, LB-O {100*r['gap_closed_S0']['LB-O']:+.1f} %. S0 depth: " + "; ".join(f"{k} endpoint {v['endpoint_m_median']:.2f} m, within 0.2 {v['within_0.2m']:.3f}" for k, v in r["depth_S0"].items()) + "\n")
        L.append("| original Gate 9 conditions (reproduced exactly, Gate 9 support) | IoU | precision | recall |"); L.append("|---|---|---|---|")
        for k, v in r["gate9_reproduction"].items():
            L.append(f"| {k} | {v['iou']:.3f} | {v['precision']:.3f} | {v['recall']:.3f} |")
        sc = r["scalars"]
        L.append(f"\nScalars median [p10, p90]: MoGe clip scale {sc['moge_clip_scale'][0]:.2f} [{sc['moge_clip_scale'][1]:.2f}, {sc['moge_clip_scale'][2]:.2f}]; LB-O scalar {sc['lbo_scalar_S1'][0]:.2f} [{sc['lbo_scalar_S1'][1]:.2f}, {sc['lbo_scalar_S1'][2]:.2f}]; "
                 f"LB-O / MoGe scale {sc['lbo_over_moge_scale'][0]:.3f} [{sc['lbo_over_moge_scale'][1]:.3f}, {sc['lbo_over_moge_scale'][2]:.3f}]. Representative clips: {r['representative_clips']}\n")
    open(os.path.join(D.ART, f"comparison_table{tag}.md"), "w").write("\n".join(L) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--drives", nargs="+", default=[D.PRIMARY]); a = ap.parse_args()
    for sub in ("tables", "figures"):
        os.makedirs(os.path.join(D.ART, sub), exist_ok=True)
    res = {}
    for d in a.drives:
        r, clips, P, F = analyze(d); res[d] = r; figures(d, r, clips, P, F)
        print(d[17:21], {k: round(v["iou"], 4) for k, v in r["primary_S1"].items()}, {k: (round(v["value"], 4) if isinstance(v, dict) else round(v, 4)) for k, v in r["gap_closed_S1"].items()})
    tag = "" if a.drives == [D.PRIMARY] else "_all"
    json.dump(res, open(os.path.join(D.ART, f"analysis{tag}.json"), "w"), indent=1); table(res, tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
