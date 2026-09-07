#!/usr/bin/env python
"""Gate 5.2 step 7 — scale and depth diagnostics against the LiDAR oracle.

Diagnostic only. Reads the finalised deployable scale tables and the oracle produced by
``lidar_oracle.py``; nothing computed here can alter a prediction.

    python tools/gate5_2/diagnostics.py
"""
from __future__ import annotations

import argparse, csv, os, sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                                "depth_gate")))

from decompose_residual import pearson, spearman
from gates.scale_gate.config import REPO_ROOT, load_config, write_json

CONDS = ("C0", "C3", "A", "B", "OR")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/gate5_2/kitti360_transfer.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    art = os.path.join(REPO_ROOT, cfg.experiment.output_dir)

    rows = list(csv.DictReader(open(os.path.join(art, "oracle_scales.csv"))))
    frames = {}
    for v in ("A", "B"):
        p = os.path.join(art, f"scale_frames_{v}.csv")
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                frames.setdefault(v, []).append(r)

    S = {c: np.array([float(r[f"s_{c}"] if c != "OR" else r["s_oracle"]) for r in rows])
         for c in CONDS}
    ok = np.isfinite(S["OR"]) & (S["OR"] > 0)
    for c in CONDS:
        ok &= np.isfinite(S[c]) & (S[c] > 0)
    print(f"{int(ok.sum())}/{len(rows)} clips with a finite oracle and all four scales")

    out = {"n_clips": len(rows), "n_with_oracle": int(ok.sum()), "scale": {}}
    Sor = S["OR"][ok]
    for c in CONDS:
        s = S[c][ok]
        ratio = s / Sor
        d = {"median": float(np.median(s)),
             **{f"p{int(p*100):02d}": float(np.quantile(s, p))
                for p in (0.05, 0.25, 0.75, 0.95)},
             "median_abs_relative_scale_error": float(np.median(np.abs(ratio - 1.0))),
             "median_abs_log_scale_error": float(np.median(np.abs(np.log(ratio)))),
             "median_ratio_to_oracle": float(np.median(ratio)),
             "pearson_with_oracle": pearson(s, Sor),
             "spearman_with_oracle": spearman(s, Sor),
             "log_ratio_mad": float(np.median(np.abs(
                 np.log(ratio) - np.median(np.log(ratio)))))}
        if c in frames:
            disp = np.array([float(x["s_frame"]) for x in frames[c]], float)
            cid = np.array([x["clip_id"] for x in frames[c]])
            per = []
            for u in np.unique(cid):
                v = disp[cid == u]
                v = v[np.isfinite(v) & (v > 0)]
                if v.size > 1:
                    per.append(float(np.std(np.log(v))))
            d["per_frame_log_scale_dispersion_median"] = float(np.median(per))
        out["scale"][c] = d

    # depth diagnostics were accumulated by lidar_oracle.py over the same pixel pairs
    dd = os.path.join(art, "depth_diagnostics.json")
    if os.path.exists(dd):
        import json
        j = json.load(open(dd))
        out["depth"] = j["by_condition"]
        out["oracle_definition"] = j["oracle"]
        out["pinned_deployable_scale_sha256"] = j["pinned_deployable_scale_sha256"]

    # secondary oracle agreement: an honest statement of how quantisation shifts it
    sv = np.array([float(r["s_oracle_voxel_centre"]) for r in rows])
    sa = np.array([float(r["s_oracle_anchor_only"]) for r in rows])
    m = ok & np.isfinite(sv) & (sv > 0)
    out["oracle_variants"] = {
        "voxel_centre_vs_projected": {
            "n": int(m.sum()),
            "median_ratio": float(np.median(sv[m] / S["OR"][m])),
            "pearson": pearson(sv[m], S["OR"][m])},
        "anchor_only_vs_five_frame": {
            "n": int((ok & np.isfinite(sa)).sum()),
            "median_ratio": float(np.median(sa[ok & np.isfinite(sa)] /
                                            S["OR"][ok & np.isfinite(sa)])),
            "pearson": pearson(sa[ok & np.isfinite(sa)], S["OR"][ok & np.isfinite(sa)])}}

    write_json(os.path.join(art, "scale_diagnostics.json"), out)
    hdr = f"{'cond':5s} {'median':>8s} {'|rel err|':>10s} {'|log err|':>10s} " \
          f"{'ratio':>7s} {'pearson':>8s} {'spearman':>9s}"
    print(hdr)
    for c in CONDS:
        d = out["scale"][c]
        print(f"{c:5s} {d['median']:8.3f} {d['median_abs_relative_scale_error']:10.4f} "
              f"{d['median_abs_log_scale_error']:10.4f} {d['median_ratio_to_oracle']:7.4f} "
              f"{d['pearson_with_oracle']:8.4f} {d['spearman_with_oracle']:9.4f}")
    if "depth" in out:
        print(f"\n{'cond':5s} {'AbsRel':>8s} {'medRel':>8s} {'medLog':>8s} {'RMSE':>8s} "
              f"{'d1':>7s} {'d2':>7s} {'d3':>7s}")
        for c in CONDS:
            k = f"{c}|all|all"
            if k in out["depth"]:
                d = out["depth"][k]
                print(f"{c:5s} {d['absrel']:8.4f} {d['med_rel']:8.4f} {d['med_log']:8.4f} "
                      f"{d['rmse']:8.3f} {d['d1']:7.4f} {d['d2']:7.4f} {d['d3']:7.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
