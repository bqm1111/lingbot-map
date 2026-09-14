#!/usr/bin/env python
"""Derive the U2 / U3 / U4 width rules from 0006 depth residuals ONLY (run before any occupancy run).

Reads only the "pixels" / "pixel_fields" arrays of the Gate 9 0006 clip records (z_lidar, z_pred = LB-M,
conf, ray_norm, edge_class). No occupancy array, metric or map is read. Writes uncertainty_rules.json,
tables/residual_*.csv and figures 1-2.

    python tools/gate11_uncertainty/derive_rules.py
"""
from __future__ import annotations
import csv, datetime, glob, hashlib, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g11 as G                                                                            # noqa: E402

G9_CLIPS = os.path.join("/media/SSD1/MINH_DATASETS/lingbot_gate9_depth_occupancy", "clips", G.PRIMARY)


def pava_nonincreasing(v, w):
    v = [float(x) for x in v]; w = [float(x) for x in w]; blocks = [[v[i], w[i], 1] for i in range(len(v))]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0]:                       # violation of non-increasing
            a, b = blocks[i], blocks[i + 1]; ww = a[1] + b[1]
            blocks[i] = [(a[0] * a[1] + b[0] * b[1]) / ww, ww, a[2] + b[2]]; del blocks[i + 1]; i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for val, _, n in blocks:
        out += [val] * n
    return out


def main() -> int:
    files = sorted(glob.glob(os.path.join(G9_CLIPS, "*.npz"))); rows = []; F = None; h = hashlib.sha256()
    for f in files:
        with np.load(f) as z:
            P = z["pixels"]; F = [str(x) for x in z["pixel_fields"]]
        rows.append(P); h.update(os.path.basename(f).encode()); h.update(P.tobytes())
    P = np.concatenate(rows); g = lambda k: P[:, F.index(k)].astype(np.float64)
    m = g("conf") >= 1.5
    zl, zp, rl, c, edge = g("z_lidar")[m], g("z_pred")[m], g("ray_norm")[m], g("conf")[m], g("edge_class")[m]
    r = (zl - zp) * rl; e = np.abs(r); q = G.DERIVATION["coverage_quantile"]; umin = G.DERIVATION["u_min_m"]
    # U2
    ed = np.asarray(G.ZBIN_EDGES); bins = []
    for i in range(len(ed) - 1):
        s = (zp >= ed[i]) & (zp < ed[i + 1])
        if s.sum() >= 1000:
            bins.append({"z_lo": ed[i], "z_hi": ed[i + 1], "n": int(s.sum()), "z_mean": float(zp[s].mean()), "p50": float(np.quantile(e[s], 0.5)),
                         "p68": float(np.quantile(e[s], q)), "p90": float(np.quantile(e[s], 0.9)), "signed_q16": float(np.quantile(r[s], 0.16)), "signed_q84": float(np.quantile(r[s], 0.84))})
    X = np.array([[1.0, b["z_mean"]] for b in bins]); y = np.array([b["p68"] for b in bins]); w = np.array([b["n"] for b in bins], np.float64)
    a_, b_ = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (w * y))
    u2 = np.maximum(umin, a_ + b_ * zp)
    # U3
    k = np.clip(np.digitize(c, np.asarray(G.CONF_EDGES[1:-1])), 0, 3); ratio = e / u2
    g_raw = [float(np.quantile(ratio[k == j], q)) for j in range(4)]; n_k = [int((k == j).sum()) for j in range(4)]
    g_mono = pava_nonincreasing(g_raw, n_k)
    u3 = np.maximum(umin, u2 * np.asarray(g_mono)[k])
    # U4
    nres = r / u3; q16, q84 = float(np.quantile(nres, 0.16)), float(np.quantile(nres, 0.84))
    lo, hi = sorted([abs(q16), abs(q84)]); asym = hi / max(lo, 1e-9); include = asym >= 1.5
    uf4, ub4 = np.maximum(umin, -q16 * u3), np.maximum(umin, q84 * u3)
    def cov(uf, ub):
        return float(((r >= -uf) & (r <= ub)).mean())
    rules = {"derived_at": datetime.datetime.now().isoformat(timespec="seconds"), "input_rows_sha256": h.hexdigest(), "n_rows": int(m.sum()), "n_clips": len(files),
             "derivation": G.DERIVATION, "u_min_m": umin,
             "U2": {"a": float(a_), "b": float(b_), "bins": bins},
             "U3": {"g_raw": g_raw, "g": g_mono, "bucket_counts": n_k, "buckets": ["[1.5,3)", "[3,6)", "[6,12)", "[12,inf)"]},
             "U4": {"q16": q16, "q84": q84, "asymmetry_ratio": asym, "include": bool(include)},
             "residual_summary": {"median_r": float(np.median(r)), "frac_true_surface_behind_prediction": float((r > 0).mean()), "p68_abs": float(np.quantile(e, q))},
             "interval_containment_on_0006_rows": {**{f"U1_{u}": cov(np.full_like(r, u), np.full_like(r, u)) for u in G.U1_WIDTHS}, "U2": cov(u2, u2), "U3": cov(u3, u3), "U4": cov(uf4, ub4),
                                                   "mean_width_m": {"U2": float(np.mean(2 * u2)), "U3": float(np.mean(2 * u3)), "U4": float(np.mean(uf4 + ub4))}},
             "required_half_width_for_containment": {}}
    for i, lab in enumerate(G.RANGE_NAMES):
        s = (zp >= G.RANGE_BINS[i]) & (zp < G.RANGE_BINS[i + 1])
        rules["required_half_width_for_containment"][lab] = {"n": int(s.sum()), **({f"p{int(100*t)}": float(np.quantile(e[s], t)) for t in (0.5, 0.75, 0.9)} if s.any() else {})}
    rules["required_half_width_for_containment"]["all"] = {"n": int(len(e)), **{f"p{int(100*t)}": float(np.quantile(e, t)) for t in (0.5, 0.75, 0.9)}}
    os.makedirs(os.path.join(G.ART, "tables"), exist_ok=True); os.makedirs(os.path.join(G.ART, "figures"), exist_ok=True)
    json.dump(rules, open(G.RULES, "w"), indent=1)
    with open(os.path.join(G.ART, "tables", "residual_by_zpred_0006.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(bins[0])); wr.writeheader(); [wr.writerow(b) for b in bins]
    with open(os.path.join(G.ART, "tables", "residual_by_confidence_0006.csv"), "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["bucket", "n", "P68(|r|/u2) raw", "g (monotone)", "P50 |r| m", "P68 |r| m", "P90 |r| m"])
        for j in range(4):
            wr.writerow([rules["U3"]["buckets"][j], n_k[j], g_raw[j], g_mono[j], float(np.quantile(e[k == j], 0.5)), float(np.quantile(e[k == j], q)), float(np.quantile(e[k == j], 0.9))])
    # figures 1-2 (width rules against the residual statistics they were fitted to)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    zz = np.linspace(1, 60, 200)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    ax[0].plot([b["z_mean"] for b in bins], [b["p50"] for b in bins], "o", ms=3, color="#999999", label="P50 |r|")
    ax[0].plot([b["z_mean"] for b in bins], [b["p68"] for b in bins], "o", ms=4, color="#222222", label="P68 |r| (fit target)")
    ax[0].plot([b["z_mean"] for b in bins], [b["p90"] for b in bins], "o", ms=3, color="#bbbbbb", label="P90 |r|")
    for u in G.U1_WIDTHS:
        ax[0].axhline(u, lw=0.8, ls=":", color="#2c6fbb")
    ax[0].plot(zz, np.maximum(umin, a_ + b_ * zz), color="#c0392b", lw=2, label=f"U2: max(0.2, {a_:.3f} + {b_:.4f} z)")
    for j, col in zip(range(4), ("#fdae6b", "#e6550d", "#a63603", "#7f2704")):
        ax[0].plot(zz, np.maximum(umin, np.maximum(umin, a_ + b_ * zz) * g_mono[j]), lw=1, color=col, label=f"U3 conf {rules['U3']['buckets'][j]} (g={g_mono[j]:.2f})")
    ax[0].set_xlabel("predicted depth z (m)"); ax[0].set_ylabel("half-width u / |residual| (m along ray)"); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3); ax[0].set_title("1. uncertainty width vs range (dotted: U1)")
    ax[1].bar(range(4), g_raw, 0.4, color="#bbbbbb", label="P68(|r|/u2) per bucket"); ax[1].bar(np.arange(4) + 0.4, g_mono, 0.4, color="#c0392b", label="g (monotone)")
    ax[1].set_xticks(np.arange(4) + 0.2); ax[1].set_xticklabels(rules["U3"]["buckets"]); ax[1].set_xlabel("LingBot confidence"); ax[1].set_ylabel("width multiplier on u2"); ax[1].legend(); ax[1].grid(alpha=0.3, axis="y")
    ax[1].set_title(f"2. width vs confidence; U4 q16/q84 = {q16:.2f}/{q84:.2f} (ratio {asym:.2f}, include={include})")
    fig.tight_layout(); fig.savefig(os.path.join(G.ART, "figures", "1_2_width_rules.png"), dpi=120); plt.close(fig)
    print(json.dumps({k: rules[k] for k in ("U3", "U4", "residual_summary", "interval_containment_on_0006_rows", "required_half_width_for_containment")}, indent=1), "\nU2 a,b", a_, b_)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
