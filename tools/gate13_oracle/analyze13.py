#!/usr/bin/env python
"""Gate 13 aggregation: per-seed / per-region / per-range tables, paired O - M statistics with a paired
clip-block bootstrap, learning curves and plateau summary, M-vs-Gate-12-H reproduction, input quality,
predeclared decision guide, results_table.md and figures 1-5.

    python tools/gate13_oracle/analyze13.py
"""
from __future__ import annotations
import csv, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g13                                                                                   # noqa: E402
from g13 import g12                                                                          # noqa: E402

REGIONS = ("full", "past5", "unobs5_later", "observed20", "unobserved_later20", "H_evidence", "no_H_evidence", "O_evidence", "no_O_evidence")
RANGES = list(g12.RANGE_NAMES) + ["all"]
BOOT_BLOCK, N_BOOT = 20, 2000
M_REPRO_TOL = {"per_seed_abs": 0.003, "mean_abs": 0.0015}                                   # fixed before training
# predeclared guide (brief section 22; applied with judgement, not mechanically)
DECISION_GUIDE = {
    "A_geometry_dominant": "mean full-region O - M >= +0.05 IoU, all seeds positive, CI > 0",
    "B_both": "+0.02 <= mean full-region O - M < +0.05, consistent across seeds",
    "C_completion_limiting": "mean full-region O - M < +0.02 while O's observed-map IoU is clearly higher than M's",
    "D_observed_only": "past5 (observed) O - M >= +0.05 but unobs5_later O - M < +0.02 (or CI including 0)",
    "plateau": "different plateaus if the O - M gap at steps 5000-6000 (stride-5) is >= the gap at step 2000 and both arms change < 0.005 over 5000 -> 6000",
}


def iou_of(c):
    return c[..., 0] / np.maximum(c.sum(-1), 1)


def pr(c):
    return c[..., 0] / np.maximum(c[..., 0] + c[..., 1], 1), c[..., 0] / np.maximum(c[..., 0] + c[..., 2], 1)


def load_counts(art, prefix, cond, seed, step=6000, stride=1):
    p = os.path.join(art, "eval", f"{prefix}_{cond}_seed{seed}_step{step:05d}" + (f"_stride{stride}" if stride > 1 else "") + "_counts.npz")
    if not os.path.exists(p):
        return None, None, None
    z = np.load(p)
    return z["ids"], z["counts"].astype(np.float64), [str(x) for x in z["regions"]]


def main() -> int:
    C, ids = {}, None
    for cond in g13.CONDITIONS:
        for s in g13.SEEDS:
            i_, c_, regs = load_counts(g13.ART, "g13", cond, s); assert c_ is not None and tuple(regs) == REGIONS, f"eval {cond} {s}"
            ids = i_ if ids is None else ids; assert np.array_equal(ids, i_); C[(cond, s)] = c_
    nA = len(ids); T = os.path.join(g13.ART, "tables"); os.makedirs(T, exist_ok=True); fd = os.path.join(g13.ART, "figures"); os.makedirs(fd, exist_ok=True)
    pooled = {k: v.sum(0) for k, v in C.items()}
    with open(os.path.join(T, "per_seed.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["cond", "seed", "iou_full", "precision_full", "recall_full", "iou_past5", "iou_unobs5_later"])
        for (cond, s), c in pooled.items():
            P, R = pr(c); w.writerow([cond, s, iou_of(c[0, 5]), P[0, 5], R[0, 5], iou_of(c[1, 5]), iou_of(c[2, 5])])
    with open(os.path.join(T, "per_region.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["cond", "seed", "region", "iou", "precision", "recall", "tp", "fp", "fn"])
        for (cond, s), c in pooled.items():
            P, R = pr(c)
            for ri, r in enumerate(REGIONS):
                w.writerow([cond, s, r, iou_of(c[ri, 5]), P[ri, 5], R[ri, 5], *c[ri, 5].astype(int)])
    with open(os.path.join(T, "per_range.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["cond", "seed", "region", "range", "iou", "precision", "recall", "tp", "fp", "fn"])
        for (cond, s), c in pooled.items():
            P, R = pr(c)
            for ri, r in enumerate(REGIONS):
                for b, rn in enumerate(RANGES):
                    w.writerow([cond, s, r, rn, iou_of(c[ri, b]), P[ri, b], R[ri, b], *c[ri, b].astype(int)])
    blocks = [np.arange(i, min(i + BOOT_BLOCK, nA)) for i in range(0, nA, BOOT_BLOCK)]; rng = np.random.default_rng(0)
    draws = [np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))]) for _ in range(N_BOOT)]
    res = {"n_anchors": int(nA), "decision_guide": DECISION_GUIDE, "regions": {}}
    for ri, r in enumerate(REGIONS):
        res["regions"][r] = {}
        for b, rn in enumerate(RANGES):
            per = {cond: [float(iou_of(pooled[(cond, s)][ri, b])) for s in g13.SEEDS] for cond in g13.CONDITIONS}
            P = {cond: float(np.mean([pr(pooled[(cond, s)])[0][ri, b] for s in g13.SEEDS])) for cond in g13.CONDITIONS}
            R = {cond: float(np.mean([pr(pooled[(cond, s)])[1][ri, b] for s in g13.SEEDS])) for cond in g13.CONDITIONS}
            diff = np.asarray(per["O"]) - np.asarray(per["M"])
            bs = [np.mean([iou_of(C[("O", s)][dr, ri, b].sum(0)) - iou_of(C[("M", s)][dr, ri, b].sum(0)) for s in g13.SEEDS]) for dr in draws]
            mM = float(np.mean(per["M"]))
            res["regions"][r][rn] = {"M": per["M"], "O": per["O"], "mean_M": mM, "mean_O": float(np.mean(per["O"])), "diff_per_seed": diff.tolist(), "mean_diff": float(diff.mean()),
                                     "sd_diff": float(diff.std(ddof=1)), "relative_gain": float(diff.mean() / mM) if mM > 0 else float("nan"),
                                     "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "seeds_positive": int((diff > 0).sum()),
                                     "precision": P, "recall": R, "gt_occupied_voxels": float(np.mean([pooled[("M", s)][ri, b, 0] + pooled[("M", s)][ri, b, 2] for s in g13.SEEDS]))}
    with open(os.path.join(T, "paired.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["region", "range", "mean_M", "mean_O", "mean_diff", "relative_gain", "sd_diff", "ci_lo", "ci_hi", "seeds_positive", "P_M", "P_O", "R_M", "R_O"])
        for r in REGIONS:
            for rn in RANGES:
                v = res["regions"][r][rn]; w.writerow([r, rn, v["mean_M"], v["mean_O"], v["mean_diff"], v["relative_gain"], v["sd_diff"], *v["ci95"], v["seeds_positive"], v["precision"]["M"], v["precision"]["O"], v["recall"]["M"], v["recall"]["O"]])
    # M vs Gate 12 H reproduction
    g12h = [float(iou_of(load_counts(g13.G12_ART, "g12", "H", s)[1].sum(0)[0, 5])) for s in g13.SEEDS]
    dM = np.asarray(res["regions"]["full"]["all"]["M"]) - np.asarray(g12h)
    res["m_reproduction"] = {"gate12_H": g12h, "gate13_M": res["regions"]["full"]["all"]["M"], "diff": dM.tolist(), "tolerance": M_REPRO_TOL,
                             "pass": bool(np.all(np.abs(dM) <= M_REPRO_TOL["per_seed_abs"]) and abs(dM.mean()) <= M_REPRO_TOL["mean_abs"])}
    # learning curves
    lc = []; curves = {}
    for cond in g13.CONDITIONS:
        for s in g13.SEEDS:
            pts = []
            for st in range(1000, 7000, 1000):
                _, c_, _ = load_counts(g13.ART, "g13", cond, s, st, 5)
                if c_ is not None:
                    q = c_.sum(0); pts.append((st, float(iou_of(q[0, 5])), float(iou_of(q[2, 5])))); lc.append([cond, s, st, pts[-1][1], pts[-1][2]])
            curves[f"{cond}_seed{s}"] = pts
    with open(os.path.join(T, "learning_curves.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["cond", "seed", "step", "iou_full_stride5", "iou_unobs5_later_stride5"]); [w.writerow(x) for x in lc]
    mean_curve = {cond: {st: float(np.mean([p[1] for s in g13.SEEDS for p in curves[f"{cond}_seed{s}"] if p[0] == st])) for st in range(1000, 7000, 1000)} for cond in g13.CONDITIONS}
    gap = {st: mean_curve["O"][st] - mean_curve["M"][st] for st in mean_curve["M"]}
    res["curves"] = curves; res["mean_curve"] = mean_curve
    res["plateau"] = {"gap_by_step": gap, "change_5000_to_6000": {c: mean_curve[c][6000] - mean_curve[c][5000] for c in g13.CONDITIONS},
                      "different_plateaus": bool(np.mean([gap[5000], gap[6000]]) >= gap[2000] and all(abs(mean_curve[c][6000] - mean_curve[c][5000]) < 0.005 for c in g13.CONDITIONS))}
    iq = json.load(open(os.path.join(g13.ART, "input_quality.json"))); res["input_quality"] = iq
    F, P5, U5 = res["regions"]["full"]["all"], res["regions"]["past5"]["all"], res["regions"]["unobs5_later"]["all"]
    res["guide_evaluation"] = {"full_mean_diff": F["mean_diff"], "full_ci": F["ci95"], "seeds_positive": F["seeds_positive"], "past5_diff": P5["mean_diff"], "unobs5_later_diff": U5["mean_diff"],
                               "unobs5_later_ci": U5["ci95"], "band": ("A" if F["mean_diff"] >= 0.05 else "B" if F["mean_diff"] >= 0.02 else "C"),
                               "D_pattern": bool(P5["mean_diff"] >= 0.05 and (U5["mean_diff"] < 0.02 or U5["ci95"][0] <= 0)), "input_iou_M_O": [iq["summary"]["M"]["iou"], iq["summary"]["O"]["iou"]]}
    json.dump(res, open(os.path.join(g13.ART, "analysis.json"), "w"), indent=1)
    figures(res, fd); table(res)
    print(json.dumps({"full": F, "past5": P5, "unobs5_later": U5, "guide": res["guide_evaluation"], "m_repro": res["m_reproduction"], "plateau": res["plateau"]}, indent=1))
    return 0


def figures(res, fd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"M": "#555555", "O": "#1b9e77"}
    F = res["regions"]["full"]["all"]; x = np.arange(3)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(x - 0.18, F["M"], 0.36, color=col["M"], label="M (monocular)"); ax[0].bar(x + 0.18, F["O"], 0.36, color=col["O"], label="O (oracle LiDAR)")
    for i in range(3):
        ax[0].text(i - 0.18, F["M"][i] + 0.003, f"{F['M'][i]:.3f}", ha="center", fontsize=8); ax[0].text(i + 0.18, F["O"][i] + 0.003, f"{F['O'][i]:.3f}", ha="center", fontsize=8)
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"seed {s}" for s in g13.SEEDS]); ax[0].set_ylabel("0006 completion IoU (full)"); ax[0].legend(); ax[0].grid(alpha=0.3, axis="y")
    for k, v in res["curves"].items():
        ax[1].plot([p[0] for p in v], [p[1] for p in v], ["-", "--", ":"][int(k[-1])], color=col[k[0]], marker="o", ms=3, label=k)
    ax[1].set_xlabel("step"); ax[1].set_ylabel("full IoU (every 5th 0006 anchor)"); ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "1_2_per_seed_and_learning_curves.png"), dpi=120); plt.close(fig)
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.2))
    for r, mk in (("full", "o"), ("unobs5_later", "s"), ("past5", "^")):
        mm = [res["regions"][r][rn]["mean_diff"] for rn in g12.RANGE_NAMES]; cc = np.array([res["regions"][r][rn]["ci95"] for rn in g12.RANGE_NAMES])
        ax[0].errorbar(np.arange(5) + {"o": -0.12, "s": 0, "^": 0.12}[mk], mm, yerr=[np.array(mm) - cc[:, 0], cc[:, 1] - np.array(mm)], fmt=mk, capsize=3, label=r)
    ax[0].axhline(0, color="k", lw=0.8); ax[0].set_xticks(range(5)); ax[0].set_xticklabels(g12.RANGE_NAMES); ax[0].set_ylabel("IoU(O) - IoU(M) [95 % CI]"); ax[0].legend(); ax[0].grid(alpha=0.3); ax[0].set_title("3. difference by range")
    regs = ["full", "past5", "unobs5_later", "observed20", "unobserved_later20", "O_evidence", "no_O_evidence", "H_evidence", "no_H_evidence"]
    m = [res["regions"][r]["all"]["mean_diff"] for r in regs]; ci = np.array([res["regions"][r]["all"]["ci95"] for r in regs])
    ax[1].bar(range(len(regs)), m, color=col["O"], yerr=[np.array(m) - ci[:, 0], ci[:, 1] - np.array(m)], capsize=3); ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(range(len(regs))); ax[1].set_xticklabels(regs, rotation=30, ha="right", fontsize=8); ax[1].set_ylabel("IoU(O) - IoU(M)"); ax[1].grid(alpha=0.3, axis="y"); ax[1].set_title("4. observed vs unobserved regions")
    rr = ["full", "past5", "unobs5_later"]; xx = np.arange(len(rr)); w = 0.2
    for j, (cond, key) in enumerate((("M", "precision"), ("O", "precision"), ("M", "recall"), ("O", "recall"))):
        ax[2].bar(xx + (j - 1.5) * w, [res["regions"][r]["all"][key][cond] for r in rr], w, color=col[cond], alpha=1.0 if key == "precision" else 0.5, label=f"{cond} {key}")
    ax[2].set_xticks(xx); ax[2].set_xticklabels(rr); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3, axis="y"); ax[2].set_title("5. precision (solid) / recall (light)")
    fig.tight_layout(); fig.savefig(os.path.join(fd, "3_4_5_range_region_pr.png"), dpi=120); plt.close(fig)


def table(res):
    L = ["# Gate 13 -- oracle observed-geometry completion ceiling: results\n",
         "KITTI-360 0006, 590 anchors, final-step checkpoints, 3 seeds, occupied := final log-odds > 0. Difference = O - M; CI = paired clip-block bootstrap of the mean-over-seeds difference "
         f"(blocks of {BOOT_BLOCK} anchors, {N_BOOT} resamples).\n"]
    iq = res["input_quality"]["summary"]
    L.append(f"Input quality (observed map vs five-sweep LiDAR measurement, 0006): M IoU {iq['M']['iou']:.3f} P {iq['M']['precision']:.3f} R {iq['M']['recall']:.3f} occupied-in-free {iq['M']['occupied_in_lidar_free']:.3f}; "
             f"O IoU {iq['O']['iou']:.3f} P {iq['O']['precision']:.3f} R {iq['O']['recall']:.3f} occupied-in-free {iq['O']['occupied_in_lidar_free']:.3f}. O == Gate 9 LD map on {res['input_quality']['O_equals_gate9_LD_map']}.\n")
    mr = res["m_reproduction"]
    L.append(f"M reproduction of Gate 12 H: Gate 12 {[round(x, 4) for x in mr['gate12_H']]}, Gate 13 M {[round(x, 4) for x in mr['gate13_M']]}, diff {[round(x, 4) for x in mr['diff']]} (tolerance {mr['tolerance']}): {'PASS' if mr['pass'] else 'FAIL'}\n")
    L.append("| region | range | IoU M (seeds) | IoU O (seeds) | mean M | mean O | O - M per seed | mean O - M | relative | 95 % CI | P M / O | R M / O | GT occ voxels |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r, rr in res["regions"].items():
        for rn, v in rr.items():
            if rn != "all" and r not in ("full", "past5", "unobs5_later"):
                continue
            L.append(f"| {r} | {rn} | {' / '.join(f'{x:.4f}' for x in v['M'])} | {' / '.join(f'{x:.4f}' for x in v['O'])} | {v['mean_M']:.4f} | {v['mean_O']:.4f} | {' / '.join(f'{x:+.4f}' for x in v['diff_per_seed'])} | "
                     f"{v['mean_diff']:+.4f} | {100*v['relative_gain']:+.1f} % | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | {v['precision']['M']:.3f} / {v['precision']['O']:.3f} | {v['recall']['M']:.3f} / {v['recall']['O']:.3f} | {v['gt_occupied_voxels']:,.0f} |")
    L.append(f"\nLearning curves, mean over seeds (full IoU, every 5th anchor): M {json.dumps({k: round(v, 4) for k, v in res['mean_curve']['M'].items()})}; O {json.dumps({k: round(v, 4) for k, v in res['mean_curve']['O'].items()})}. Plateau: {json.dumps(res['plateau'])}\n")
    L.append(f"Predeclared guide: {json.dumps(res['decision_guide'], indent=1)}\n\nEvaluation: {json.dumps(res['guide_evaluation'])}\n")
    open(os.path.join(g13.ART, "results_table.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
