#!/usr/bin/env python
"""Gate 11 aggregation: voxel-state metrics (all / by range), derived metrics, paired clip-block bootstrap
comparisons, ray-level strata (edge / confidence / range), coverage-vs-volume, LD control, CSVs,
results_table[_all].md and figures 3-5.

Metric definitions (voxel level, inside the evaluation mask E; R = LiDAR-occupied, Fr = LiDAR-free):
  surface coverage   |S&R| / |R|          band coverage  |(S|U)&R| / |R|     false-free |F&R| / |R|
  uncovered          1 - (|S&R| + |U&R| + |F&R|) / |R|
  free correct       |F&Fr| / |Fr|        surface in free |S&Fr| / |S&E|     uncertain in free |U&Fr| / |U&E|
  uncertain fraction |U&E| / |E|
  containment efficiency (SCE)  |(S|U)&R| / |(S|U)&E|
  false certainty rate (FCR)    (|S&Fr| + |F&R|) / (|S&E| + |F&E|)
  IoU / precision / recall with occupied := S;  optimistic IoU with occupied := S or U
Counts pool over clips. Voxel range = anchor-camera z of the voxel centre (Gate 9 bins).

    python tools/gate11_uncertainty/analyze_u.py [--drives ...]
"""
from __future__ import annotations
import argparse, csv, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g11 as G                                                                            # noqa: E402

CF = ["R", "Fr", "E", "S_R", "U_R", "F_R", "S_Fr", "U_Fr", "F_Fr", "S_E", "U_E", "F_E", "S_all", "U_all", "F_all"]
FAMILY = ("H", "U1_0.2", "U1_0.4", "U1_0.6", "U2", "U3", "U4")
BOOT_BLOCK, N_BOOT = 20, 1000


def load(drive):
    clips, pix, F = [], [], None
    for f in sorted(glob.glob(os.path.join(G.DATA_ROOT, "clips", drive, "*.npz"))):
        z = np.load(f); names = [str(x) for x in z["condition_names"]]
        c = {"t": int(z["t"]), "names": names, "n_rays_LBM": int(z["n_rays_LBM"]), "n_rays_lidar": int(z["n_rays_lidar"]), "rep": [int(x) for x in z["representative"]]}
        for n in names:
            c[f"c_{n}"] = z[f"c_{n}"].astype(np.float64); c[f"m_{n}"] = z[f"m_{n}"]
        clips.append(c); p = z["pixels"]; pix.append(p); F = [str(x) for x in z["pixel_fields"]]
    return clips, np.concatenate(pix), F


def vm(c):
    """voxel metrics from a 15-vector of pooled counts."""
    R, Fr, E, SR, UR, FR, SFr, UFr, FFr, SE, UE, FE = c[:12]; sd = lambda a, b: float(a / b) if b > 0 else float("nan")
    return {"surface_coverage": sd(SR, R), "band_coverage": sd(SR + UR, R), "uncertain_only_coverage": sd(UR, R), "false_free_rate": sd(FR, R),
            "uncovered": sd(R - SR - UR - FR, R), "free_correct": sd(FFr, Fr), "surface_in_lidar_free": sd(SFr, SE), "uncertain_in_lidar_free": sd(UFr, UE),
            "free_contradicted": sd(FR, FE), "uncertain_fraction_of_mask": sd(UE, E), "claimed_fraction_of_mask": sd(SE + UE, E),
            "SCE": sd(SR + UR, SE + UE), "FCR": sd(SFr + FR, SE + FE), "iou": sd(SR, SE + R - SR), "precision": sd(SR, SE), "recall": sd(SR, R),
            "iou_optimistic": sd(SR + UR, SE + UE + R - SR - UR)}


def pooled(clips, name, b=5):
    return sum(c[f"c_{name}"][b] for c in clips)


def boot(clips, fn):
    n = len(clips); nb = int(np.ceil(n / BOOT_BLOCK)); rng = np.random.default_rng(0)
    blocks = [list(range(i * BOOT_BLOCK, min((i + 1) * BOOT_BLOCK, n))) for i in range(nb)]; v = []
    for _ in range(N_BOOT):
        v.append(fn([clips[i] for bb in rng.integers(0, nb, nb) for i in blocks[bb]]))
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def interp_volume(points, target):
    """points: [(claimed fraction of mask, uncertain fraction, band coverage)] sorted by claimed fraction; first crossing."""
    for (v0, u0, c0), (v1, u1, c1) in zip(points[:-1], points[1:]):
        if (c0 - target) * (c1 - target) <= 0 and c1 != c0:
            w = (target - c0) / (c1 - c0); return {"claimed_fraction": v0 + w * (v1 - v0), "uncertain_fraction": u0 + w * (u1 - u0), "reached": True}
    return {"reached": False, "max_band_coverage": max(p[2] for p in points)}


def ray_strata(P, F, names):
    g = lambda k: P[:, F.index(k)].astype(np.float64); gate = g("gate") > 0
    groups = {"edge": {lab: gate & (g("edge_class") == k) for k, lab in G.EDGE_NAMES.items() if k < 3},
              "confidence": {lab: gate & (g("conf") >= G.CONF_EDGES[i]) & (g("conf") < G.CONF_EDGES[i + 1]) for i, lab in enumerate(("[1.5,3)", "[3,6)", "[6,12)", "[12,inf)"))},
              "range": {lab: gate & (g("z_lidar") >= G.RANGE_BINS[i]) & (g("z_lidar") < G.RANGE_BINS[i + 1]) for i, lab in enumerate(G.RANGE_NAMES)}}
    out = {}
    for grp, bins in groups.items():
        out[grp] = {}
        for lab, m in bins.items():
            row = {"n": int(m.sum()), "lbm_endpoint_on_lidar_free": float(g("lbm_ep_on_ref_free")[m].mean()) if m.any() else float("nan"),
                   "abs_endpoint_err_median_m": float(np.median(np.abs(g("z_lidar")[m] - g("z_lbm")[m]) * g("ray_norm")[m])) if m.any() else float("nan")}
            for n in names:
                st = g(f"state_{n}")[m]
                row[n] = {"contains": float(g(f"contains_{n}")[m].mean()), "true_voxel_S": float((st == 1).mean()), "true_voxel_U": float((st == 2).mean()),
                          "true_voxel_F": float((st == 3).mean()), "true_voxel_unknown": float((st == 0).mean())}
                if f"uf_{n}" in F:
                    row[n]["mean_width_m"] = float((g(f"uf_{n}")[m] + g(f"ub_{n}")[m]).mean())
            out[grp][lab] = row
    return out


def analyze(d):
    clips, P, F = load(d); names = clips[0]["names"]; tag = d[17:21]
    lbm = [n for n in names if not n.endswith("_LD")]; ld = [n for n in names if n.endswith("_LD")]
    r = {"n_clips": len(clips), "conditions": names, "representative_clips": dict(zip(("bad", "median", "good"), clips[0]["rep"]))}
    r["voxel"] = {n: vm(pooled(clips, n)) for n in names}
    nr = {"LBM": sum(c["n_rays_LBM"] for c in clips), "lidar": sum(c["n_rays_lidar"] for c in clips)}
    for n in names:
        cc = pooled(clips, n); rays = nr["lidar" if n.endswith("_LD") else "LBM"]
        r["voxel"][n].update({"surface_voxels_per_ray": cc[12] / rays, "uncertain_voxels_per_ray": cc[13] / rays, "uncertain_m3_per_clip": cc[13] * 0.008 / len(clips),
                              "uncertain_m3_per_clip_in_mask": cc[10] * 0.008 / len(clips)})
    r["by_range"] = {n: {lab: vm(pooled(clips, n, i)) for i, lab in enumerate(G.RANGE_NAMES)} for n in names}
    fam = [n for n in FAMILY if n in names]
    pts = sorted([(r["voxel"][n]["claimed_fraction_of_mask"], r["voxel"][n]["uncertain_fraction_of_mask"], r["voxel"][n]["band_coverage"]) for n in fam])
    r["coverage_volume"] = {str(tg): interp_volume(pts, tg) for tg in (0.5, 0.75, 0.9)}
    u1 = sorted([(r["voxel"][n]["band_coverage"], r["voxel"][n]["SCE"]) for n in ("U1_0.2", "U1_0.4", "U1_0.6")])
    def u1_sce_at(cov):
        xs, ys = zip(*u1)
        return float(np.interp(cov, xs, ys)) if xs[0] <= cov <= xs[-1] else float("nan")
    r["U1_curve_SCE_at_band_coverage"] = {n: {"band_coverage": r["voxel"][n]["band_coverage"], "SCE": r["voxel"][n]["SCE"], "U1_SCE_interpolated": u1_sce_at(r["voxel"][n]["band_coverage"]),
                                              "U1_linear_extrapolated": float(np.polyval(np.polyfit(*zip(*u1), 1), r["voxel"][n]["band_coverage"]))} for n in ("U2", "U3", "U4") if n in names}
    def stat(key, a_, b_=None):
        def fn(s):
            va = vm(pooled(s, a_))[key]
            return va - vm(pooled(s, b_))[key] if b_ else va
        return fn
    pairs = [("band_coverage", x, "H") for x in fam if x != "H"] + [("FCR", x, "H") for x in fam if x != "H"] + [("SCE", "U2", "U1_0.6"), ("SCE", "U3", "U2"), ("band_coverage", "U3", "U2"),
             ("uncertain_fraction_of_mask", "U3", "U2"), ("SCE", "U4", "U3"), ("band_coverage", "U4", "U3")] + \
            [(k, x, x + "h") for x in fam if x + "h" in names for k in ("false_free_rate", "FCR", "band_coverage")]
    r["paired"] = [{"metric": k, "a": a_, "b": b_, "diff": stat(k, a_, b_)(clips), "ci95": boot(clips, stat(k, a_, b_))} for k, a_, b_ in pairs]
    r["ray_strata"] = ray_strata(P, F, [n for n in ("H", "U1_0.4", "U1_0.6", "U2", "U3", "U4") if n in names])
    Hm = r["voxel"]["H"]
    r["guide"] = {n: {"band_gain_vs_H_surface": r["voxel"][n]["band_coverage"] - Hm["surface_coverage"], "substantial_gain": r["voxel"][n]["band_coverage"] >= Hm["surface_coverage"] + 0.15,
                      "useful": bool(r["voxel"][n]["band_coverage"] >= 0.5 and r["voxel"][n]["SCE"] >= Hm["precision"] and r["voxel"][n]["false_free_rate"] <= Hm["false_free_rate"] and r["voxel"][n]["uncertain_fraction_of_mask"] <= 0.25),
                      "useful_range_bins": [lab for lab in G.RANGE_NAMES if r["by_range"][n][lab]["band_coverage"] >= 0.5 and r["by_range"][n][lab]["SCE"] >= r["by_range"]["H"][lab]["precision"]
                                            and r["by_range"][n][lab]["uncertain_fraction_of_mask"] <= 0.25]} for n in fam}
    if "U3" in names:
        r["guide"]["confidence_benefit"] = bool(r["voxel"]["U3"]["uncertain_fraction_of_mask"] <= 0.9 * r["voxel"]["U2"]["uncertain_fraction_of_mask"] and r["voxel"]["U2"]["band_coverage"] - r["voxel"]["U3"]["band_coverage"] <= 0.02)
    r["guide"]["carving_benefit"] = {x: bool(r["voxel"][x]["false_free_rate"] <= 0.8 * r["voxel"][x + "h"]["false_free_rate"]) for x in fam if x + "h" in names}
    r["gate9_reproduction"] = {"H == Gate 9 PD_gate": None}
    T = os.path.join(G.ART, "tables")
    with open(os.path.join(T, f"voxel_metrics_{tag}.csv"), "w", newline="") as fh:
        keys = list(r["voxel"]["H"]); w = csv.writer(fh); w.writerow(["condition", "range"] + keys)
        for n in names:
            w.writerow([n, "all"] + [r["voxel"][n][k] for k in keys])
            for lab in G.RANGE_NAMES:
                w.writerow([n, lab] + [r["by_range"][n][lab].get(k, "") for k in keys])
    with open(os.path.join(T, f"paired_{tag}.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["metric", "a", "b", "diff", "ci_lo", "ci_hi"]); [w.writerow([p["metric"], p["a"], p["b"], p["diff"], *p["ci95"]]) for p in r["paired"]]
    with open(os.path.join(T, f"ray_strata_{tag}.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["group", "bin", "n", "lbm_endpoint_on_lidar_free", "abs_endpoint_err_median_m", "condition", "contains", "true_voxel_S", "true_voxel_U", "true_voxel_F", "true_voxel_unknown", "mean_width_m"])
        for grp, bins in r["ray_strata"].items():
            for lab, row in bins.items():
                for n, v in row.items():
                    if isinstance(v, dict):
                        w.writerow([grp, lab, row["n"], row["lbm_endpoint_on_lidar_free"], row["abs_endpoint_err_median_m"], n, v["contains"], v["true_voxel_S"], v["true_voxel_U"], v["true_voxel_F"], v["true_voxel_unknown"], v.get("mean_width_m", "")])
    with open(os.path.join(T, f"per_clip_{tag}.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["t"] + [f"{k}_{n}" for n in names for k in ("band_coverage", "FCR", "uncertain_fraction", "iou")])
        for c in clips:
            row = [c["t"]]
            for n in names:
                m_ = vm(c[f"c_{n}"][5]); row += [m_["band_coverage"], m_["FCR"], m_["uncertain_fraction_of_mask"], m_["iou"]]
            w.writerow(row)
    return r


def figures(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = os.path.join(G.ART, "figures"); os.makedirs(fd, exist_ok=True)
    col = {"H": "#222222", "U1_0.2": "#c6dbef", "U1_0.4": "#6baed6", "U1_0.6": "#2171b5", "U2": "#e6550d", "U3": "#a63603", "U4": "#31a354"}
    x = np.arange(len(G.RANGE_NAMES))
    for d, r in res.items():
        tag = d[17:21]; fam = [n for n in FAMILY if n in r["conditions"]]
        fig, ax = plt.subplots(1, 3, figsize=(17, 4.3))
        for n in fam:
            br = r["by_range"][n]
            ax[0].plot(x, [br[l]["band_coverage"] for l in G.RANGE_NAMES], "-o", color=col[n], label=f"{n} band (S or U)")
            ax[0].plot(x, [br[l]["surface_coverage"] for l in G.RANGE_NAMES], ":", color=col[n])
            ax[1].plot(x, [br[l]["FCR"] for l in G.RANGE_NAMES], "-o", color=col[n], label=n)
            ax[2].plot(x, [br[l]["false_free_rate"] for l in G.RANGE_NAMES], "-o", color=col[n], label=n)
        for a_, t_ in zip(ax, ("3. LiDAR-occupied voxels covered (solid: S or U; dotted: S only)", "4a. false certainty rate (definite S/F contradicting LiDAR)", "4b. false-free rate (LiDAR-occupied marked free)")):
            a_.set_xticks(x); a_.set_xticklabels(G.RANGE_NAMES); a_.set_xlabel("voxel range (anchor z)"); a_.grid(alpha=0.3); a_.set_title(t_, fontsize=9)
        ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
        fig.suptitle(f"{tag}: coverage and false certainty vs range"); fig.tight_layout(); fig.savefig(os.path.join(fd, f"3_4_range_{tag}.png"), dpi=120); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5)); mk = {"0006": "o", "0003": "s", "0007": "^", "0010": "D"}
    for d, r in res.items():
        tag = d[17:21]
        for n in [n for n in FAMILY if n in r["conditions"]]:
            v = r["voxel"][n]
            ax[0].scatter(v["uncertain_fraction_of_mask"], v["band_coverage"], color=col[n], marker=mk[tag], s=40, label=n if tag == "0006" else None)
            ax[1].scatter(v["claimed_fraction_of_mask"], v["SCE"], color=col[n], marker=mk[tag], s=40)
    ax[0].axhline(0.5, color="k", lw=0.6, ls=":"); ax[0].axvline(0.25, color="k", lw=0.6, ls=":")
    ax[0].set_xlabel("uncertain voxels / evaluation mask"); ax[0].set_ylabel("band coverage of LiDAR-occupied voxels"); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
    ax[1].set_xlabel("claimed (S or U) voxels / evaluation mask"); ax[1].set_ylabel("containment efficiency |(S|U)&R| / |(S|U)&E|"); ax[1].grid(alpha=0.3)
    fig.suptitle("5. uncertainty volume vs surface coverage (markers: o 0006, s 0003, ^ 0007, D 0010; dotted: predeclared 0.5 coverage / 25 % mask)"); fig.tight_layout()
    fig.savefig(os.path.join(fd, "5_volume_vs_coverage.png"), dpi=120); plt.close(fig)


def table(res, tag):
    L = ["# Gate 11 -- uncertainty-aware observed geometry: results\n",
         "Rays: Gate 9 LiDAR support AND LingBot conf >= 1.5; depth LB-M = s_clip x z_LingBot (LD = LiDAR control, 0006 only); GT relative poses; Gate 9 grid, mask, reference. "
         "States: S surface (mapper log-odds > 0), F definite free (mapper free), U uncertain (inside an interval, neither S nor F). Conversions: IoU uses occupied := S; optimistic IoU uses S or U. "
         "Suffix h = hard carving (to z - 0.2 m) with the same interval; no suffix = conservative carving (to z - u_front). Voxel range = anchor-camera z.\n"]
    for d, r in res.items():
        L.append(f"## {d[17:21]} ({r['n_clips']} clips)\n")
        L.append("| condition | surface cov | band cov (S or U) | false-free | uncovered | free correct | S in LiDAR-free | U in LiDAR-free | uncertain / mask | S vox / ray | U vox / ray | SCE | FCR | IoU (S) | precision | recall | IoU (S or U) |")
        L.append("|---|" + "---|" * 16)
        for n in r["conditions"]:
            v = r["voxel"][n]
            L.append(f"| {n} | {v['surface_coverage']:.3f} | {v['band_coverage']:.3f} | {v['false_free_rate']:.4f} | {v['uncovered']:.3f} | {v['free_correct']:.3f} | {v['surface_in_lidar_free']:.3f} | {v['uncertain_in_lidar_free']:.3f} | "
                     f"{v['uncertain_fraction_of_mask']:.3f} | {v['surface_voxels_per_ray']:.3f} | {v['uncertain_voxels_per_ray']:.3f} | {v['SCE']:.3f} | {v['FCR']:.3f} | {v['iou']:.3f} | {v['precision']:.3f} | {v['recall']:.3f} | {v['iou_optimistic']:.3f} |")
        L.append("\n**Paired differences (clip-block bootstrap, 95 % CI)**\n"); L.append("| metric | a - b | diff | CI |"); L.append("|---|---|---|---|")
        for p in r["paired"]:
            L.append(f"| {p['metric']} | {p['a']} - {p['b']} | {p['diff']:+.4f} | [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}] |")
        L.append("\n**By voxel range** (band coverage / false-free / FCR / uncertain fraction / SCE / IoU(S))\n")
        fam = [n for n in FAMILY if n in r["conditions"]] + [n for n in r["conditions"] if n.endswith("_LD")]
        L.append("| condition | " + " | ".join(G.RANGE_NAMES) + " |"); L.append("|---|" + "---|" * len(G.RANGE_NAMES))
        for n in fam:
            L.append(f"| {n} | " + " | ".join(f"{r['by_range'][n][l]['band_coverage']:.3f} / {r['by_range'][n][l]['false_free_rate']:.3f} / {r['by_range'][n][l]['FCR']:.3f} / {r['by_range'][n][l]['uncertain_fraction_of_mask']:.3f} / {r['by_range'][n][l]['SCE']:.3f} / {r['by_range'][n][l]['iou']:.3f}" for l in G.RANGE_NAMES) + " |")
        L.append("\n**Ray level** (contains = the ray's own interval holds the LiDAR endpoint; S/U/F/unk = final state of the LiDAR endpoint voxel)\n")
        L.append("| group | bin | n | LB-M endpoint err median m | condition | contains | S | U | F | unknown | mean width m |"); L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for grp, bins in r["ray_strata"].items():
            for lab, row in bins.items():
                for n, v in row.items():
                    if isinstance(v, dict):
                        L.append(f"| {grp} | {lab} | {row['n']:,} | {row['abs_endpoint_err_median_m']:.2f} | {n} | {v['contains']:.3f} | {v['true_voxel_S']:.3f} | {v['true_voxel_U']:.3f} | {v['true_voxel_F']:.4f} | {v['true_voxel_unknown']:.3f} | {v.get('mean_width_m', float('nan')):.2f} |")
        L.append(f"\nCoverage vs volume (family H, U1, U2, U3, U4; linear between family points): {json.dumps(r['coverage_volume'])}\n")
        L.append(f"U2/U3/U4 against the U1 SCE curve: {json.dumps(r['U1_curve_SCE_at_band_coverage'])}\n")
        L.append(f"Predeclared guide: {json.dumps(r['guide'])}\n\nRepresentative clips: {r['representative_clips']}\n")
    open(os.path.join(G.ART, f"results_table{tag}.md"), "w").write("\n".join(L) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--drives", nargs="+", default=[G.PRIMARY]); a = ap.parse_args()
    os.makedirs(os.path.join(G.ART, "tables"), exist_ok=True)
    res = {d: analyze(d) for d in a.drives}
    for d, r in res.items():
        print(d[17:21], {n: (round(r["voxel"][n]["band_coverage"], 3), round(r["voxel"][n]["uncertain_fraction_of_mask"], 3), round(r["voxel"][n]["FCR"], 3)) for n in r["conditions"]})
    figures(res); tag = "" if a.drives == [G.PRIMARY] else "_all"
    json.dump(res, open(os.path.join(G.ART, f"analysis{tag}.json"), "w"), indent=1); table(res, tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
