#!/usr/bin/env python
"""Aggregate the per-pair rows: same-point vs control similarity, AUCs, viewpoint
stratification, per-class results, perturbation summary (Steps 6-10).

    python tools/direct_correspondence_audit/analyze.py
"""
from __future__ import annotations
import glob, json, os, sys, time
import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.direct_correspondence_audit import common as DC                                # noqa: E402
from core.datasets import kitti360_splits as SRC, vocab                                   # noqa: E402

PREVIOUS = {"drive_0006_mapper_voxel_pairs": {"feature_cos_same": 0.918, "feature_cos_local": 0.915, "feature_auc_local": 0.520,
                                             "prob_cos_same": 0.762, "prob_cos_local": 0.748, "prob_auc_local": 0.514, "top1_same": 0.568, "top1_local": 0.551}}


def auc(t, c):
    t = t[np.isfinite(t)]; c = c[np.isfinite(c)]
    if len(t) < 5 or len(c) < 5:
        return None
    r = rankdata(np.concatenate([t, c])); n1, n2 = len(t), len(c)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def block(R, F, mask):
    """Metrics on a subset of rows."""
    g = lambda k: R[mask, F.index(k)]
    n = int(mask.sum())
    if n < 20:
        return {"n": n}
    out = {"n": n, "same": {"feature_cos": float(np.nanmean(g("cos_same"))), "feature_cos_median": float(np.nanmedian(g("cos_same"))),
                            "prob_cos": float(np.nanmean(g("pcos_same"))), "js": float(np.nanmean(g("js_same"))), "top1": float(np.nanmean(g("top1_same"))),
                            "ref_prob_cos": float(np.nanmean(g("refcos_same"))), "ref_top1": float(np.nanmean(g("reftop1_same")))},
           "controls": {}}
    for c in ("C0", "C1", "C2", "C3"):
        cc = g(f"cos_{c}"); pc = g(f"pcos_{c}"); tc = g(f"top1_{c}"); v = np.isfinite(cc)
        if v.sum() < 20:
            continue
        out["controls"][c] = {"n": int(v.sum()),
                              "feature": {"control_cos": float(np.nanmean(cc)), "margin": float(np.nanmean(g("cos_same")[v]) - np.nanmean(cc)),
                                          "margin_over_control_std": float((np.nanmean(g("cos_same")[v]) - np.nanmean(cc)) / max(np.nanstd(cc), 1e-9)),
                                          "auc": auc(g("cos_same")[v], cc[v])},
                              "prob": {"control_cos": float(np.nanmean(pc)), "margin": float(np.nanmean(g("pcos_same")[v]) - np.nanmean(pc)), "auc": auc(g("pcos_same")[v], pc[v])},
                              "top1": {"control": float(np.nanmean(tc)), "margin": float(np.nanmean(g("top1_same")[v]) - np.nanmean(tc))}}
    if "C2" in out["controls"]:
        out["controls"]["C2"]["mean_distance_m"] = float(np.nanmean(g("d_C2")))
        if "cos_C2_sameview" in F:
            sv = g("cos_C2_sameview"); v = np.isfinite(sv)
            out["controls"]["C2_same_view"] = {"n": int(v.sum()), "feature_cos_same_view_local": float(np.nanmean(sv)),
                                               "prob_cos_same_view_local": float(np.nanmean(g("pcos_C2_sameview"))), "top1_same_view_local": float(np.nanmean(g("top1_C2_sameview"))),
                                               "note": "f_A(X) vs f_A(X'): spatial resolution of the feature without any view change"}
    return out


def strat(R, F, base, edges, names, col):
    x = R[:, F.index(col)]; out = {}
    for k in range(len(names)):
        m = base & (x >= edges[k]) & (x < edges[k + 1])
        out[names[k]] = block(R, F, m)
    return out


def main() -> int:
    DC.require_data_root(); names = list(vocab.load("kitti360").phrases)
    drives = list(SRC.TRAIN_DRIVES) + [SRC.VAL_DRIVE]
    out = {"previous_mapper_based": PREVIOUS, "drives": {}, "tolerances": DC.TOLS}
    t0 = time.time()
    for d in drives:
        files = sorted(glob.glob(os.path.join(DC.ROW_DIR, d, "*.npz")))
        if not files:
            continue
        parts = []; F = None
        for f in files:
            with np.load(f) as z:
                parts.append(z["rows"]); F = [str(x) for x in z["fields"]]
        R = np.concatenate(parts); strict = R[:, F.index("strict")] > 0; allm = np.ones(len(R), bool)
        pc = json.load(open(os.path.join(DC.ART, "pairs", f"{d}.json")))["counts"]
        r = {"n_source_frames": len(files), "n_rows_loose": int(len(R)), "n_rows_strict": int(strict.sum()), "pair_counts": pc,
             "strict": block(R, F, strict), "loose": block(R, F, allm)}
        r["by_baseline"] = strat(R, F, strict, DC.BASELINE_EDGES, DC.BASELINE_NAMES, "baseline_m")
        r["by_angle"] = strat(R, F, strict, DC.ANGLE_EDGES, DC.ANGLE_NAMES, "angle_deg")
        r["by_depth_B"] = strat(R, F, strict, DC.DEPTH_EDGES, DC.DEPTH_NAMES, "z_B")
        r["by_offset"] = {str(int(o)): block(R, F, strict & (R[:, F.index("offset")] == o)) for o in np.unique(R[:, F.index("offset")])}
        res = np.abs(R[:, F.index("resid_B")])
        r["by_visibility_residual"] = {"<0.1 m": block(R, F, strict & (res < 0.1)), "0.1-0.3 m": block(R, F, strict & (res >= 0.1) & (res < 0.3)), "0.3 m-loose": block(R, F, allm & ~strict)}
        cls = R[:, F.index("cls_A")].astype(int); r["per_class"] = {}
        for k in np.unique(cls):
            m = strict & (cls == k)
            if m.sum() >= 300:
                b = block(R, F, m); r["per_class"][names[k]] = {"n": b["n"], "top1": b["same"]["top1"], "feature_cos": b["same"]["feature_cos"], "prob_cos": b["same"]["prob_cos"],
                                                                "C2_feature_margin": b["controls"].get("C2", {}).get("feature", {}).get("margin"), "C2_feature_auc": b["controls"].get("C2", {}).get("feature", {}).get("auc"),
                                                                "C2_prob_auc": b["controls"].get("C2", {}).get("prob", {}).get("auc"), "C2_top1_control": b["controls"].get("C2", {}).get("top1", {}).get("control")}
        # perturbation
        p = {k: R[strict, F.index(f"pert_{k}")] for k in ("1tok", "2px", "5px")}
        cs = R[strict, F.index("cos_same")]
        r["perturbation"] = {"n": int(np.isfinite(p["1tok"]).sum()),
                             **{f"cos_after_{k}": float(np.nanmean(v)) for k, v in p.items()},
                             **{f"one_minus_cos_{k}": float(1 - np.nanmean(v)) for k, v in p.items()},
                             "one_minus_cos_cross_view_same_point": float(1 - np.nanmean(cs)),
                             "one_minus_cos_local_control": float(1 - np.nanmean(R[strict, F.index("cos_C2")])),
                             "fraction_cross_view_diff_explained_by_1tok": float((1 - np.nanmean(p["1tok"])) / max(1 - np.nanmean(cs), 1e-9))}
        # distributions
        bins = np.linspace(0.5, 1.0, 51)
        r["hist"] = {"bins": bins.tolist(), "feature_same": np.histogram(cs, bins)[0].tolist(), "feature_C2": np.histogram(R[strict, F.index("cos_C2")][np.isfinite(R[strict, F.index("cos_C2")])], bins)[0].tolist(),
                     "feature_C0": np.histogram(R[strict, F.index("cos_C0")][np.isfinite(R[strict, F.index("cos_C0")])], bins)[0].tolist(),
                     "prob_same": np.histogram(R[strict, F.index("pcos_same")], np.linspace(0, 1, 51))[0].tolist(), "prob_C2": np.histogram(R[strict, F.index("pcos_C2")][np.isfinite(R[strict, F.index("pcos_C2")])], np.linspace(0, 1, 51))[0].tolist(),
                     "prob_bins": np.linspace(0, 1, 51).tolist()}
        out["drives"][d] = r
        s = r["strict"]; c2 = s["controls"]["C2"]
        print(f"  {d[17:21]}: strict {s['n']:,} | same: F cos {s['same']['feature_cos']:.3f} P cos {s['same']['prob_cos']:.3f} top1 {s['same']['top1']:.3f} ref-top1 {s['same']['ref_top1']:.3f} | "
              f"C2: F ctrl {c2['feature']['control_cos']:.3f} margin {c2['feature']['margin']:+.3f} AUC {c2['feature']['auc']:.3f}; P AUC {c2['prob']['auc']:.3f}; top1 ctrl {c2['top1']['control']:.3f} | "
              f"pert 1tok {r['perturbation']['cos_after_1tok']:.3f} ({(time.time()-t0)/60:.1f} min)", flush=True)
    out["seconds"] = time.time() - t0; out["env"] = DC.environment()
    DC.write_json(os.path.join(DC.ART, "analysis.json"), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
