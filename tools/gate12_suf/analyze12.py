#!/usr/bin/env python
"""Gate 12 aggregation: per-seed and per-range tables, paired differences (SUF - H), paired
clip-block bootstrap over validation anchors, learning curves, predeclared decision guide,
results_table.md and figures.

    python tools/gate12_suf/analyze12.py [--device cuda:0]
"""
from __future__ import annotations
import argparse, csv, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g12                                                                                  # noqa: E402

REGIONS = ("full", "past5", "unobs5_later", "observed20", "unobserved_later20", "H_evidence", "no_H_evidence")
RANGES = list(g12.RANGE_NAMES) + ["all"]
BOOT_BLOCK, N_BOOT = 20, 2000
# predeclared before any Gate 12 result (thresholds on KITTI-360 0006 only)
DECISION_GUIDE = {
    "meaningful_gain": "mean over seeds of full-region IoU(SUF) - IoU(H) >= +0.005 AND paired bootstrap 95 % CI > 0 AND >= 2 of 3 seeds positive",
    "not_only_observed": "the same CI > 0 in unobs5_later (unobserved by the five causal sweeps, measured later)",
    "no_catastrophic_tradeoff": "neither mean precision nor mean recall drops by more than 0.02",
    "hurts": "mean full-region difference <= -0.005 AND CI < 0 AND >= 2 of 3 seeds negative",
    "short_range_only": "CI > 0 in some <= 20 m range bin(s) of the full region, but CI includes 0 (or < 0) in unobs5_later and in the 20-30 / > 30 m bins",
}


def iou_of(c):
    return c[..., 0] / np.maximum(c.sum(-1), 1)


def pr(c):
    return c[..., 0] / np.maximum(c[..., 0] + c[..., 1], 1), c[..., 0] / np.maximum(c[..., 0] + c[..., 2], 1)


def load_counts(cond, seed, step=6000, stride=1):
    p = os.path.join(g12.ART, "eval", f"g12_{cond}_seed{seed}_step{step:05d}" + (f"_stride{stride}" if stride > 1 else "") + "_counts.npz")
    if not os.path.exists(p):
        return None, None
    z = np.load(p); assert list(z["regions"]) == list(REGIONS)
    return z["ids"], z["counts"].astype(np.float64)                                          # [anchors, regions, 6, 3]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--device", default=None); a = ap.parse_args()
    C, ids = {}, None
    for cond in g12.CONDITIONS:
        for s in g12.SEEDS:
            i_, c_ = load_counts(cond, s)
            assert c_ is not None, f"missing eval for {cond} seed {s}"
            ids = i_ if ids is None else ids; assert np.array_equal(ids, i_); C[(cond, s)] = c_
    nA = len(ids); T = os.path.join(g12.ART, "tables"); os.makedirs(T, exist_ok=True); fd = os.path.join(g12.ART, "figures"); os.makedirs(fd, exist_ok=True)
    pooled = {k: v.sum(0) for k, v in C.items()}                                              # [regions, 6, 3]
    rows = []
    for (cond, s), c in pooled.items():
        P, Rr = pr(c)
        for ri, r in enumerate(REGIONS):
            rows.append({"cond": cond, "seed": s, "region": r, "iou": iou_of(c[ri, 5]), "precision": P[ri, 5], "recall": Rr[ri, 5]})
    with open(os.path.join(T, "per_seed.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(r) for r in rows]
    with open(os.path.join(T, "per_range.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["cond", "seed", "region", "range", "iou", "precision", "recall", "tp", "fp", "fn"])
        for (cond, s), c in pooled.items():
            P, Rr = pr(c)
            for ri, r in enumerate(REGIONS):
                for b, rn in enumerate(RANGES):
                    w.writerow([cond, s, r, rn, iou_of(c[ri, b]), P[ri, b], Rr[ri, b], *c[ri, b].astype(int)])
    # paired statistics
    res = {"n_anchors": int(nA), "decision_guide": DECISION_GUIDE, "regions": {}}
    blocks = [np.arange(i, min(i + BOOT_BLOCK, nA)) for i in range(0, nA, BOOT_BLOCK)]; rng = np.random.default_rng(0)
    draws = [np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))]) for _ in range(N_BOOT)]
    for ri, r in enumerate(REGIONS):
        res["regions"][r] = {}
        for b, rn in enumerate(RANGES):
            per = {cond: [iou_of(pooled[(cond, s)][ri, b]) for s in g12.SEEDS] for cond in g12.CONDITIONS}
            Pm = {cond: [pr(pooled[(cond, s)])[0][ri, b] for s in g12.SEEDS] for cond in g12.CONDITIONS}
            Rm = {cond: [pr(pooled[(cond, s)])[1][ri, b] for s in g12.SEEDS] for cond in g12.CONDITIONS}
            diff = np.asarray(per["SUF"]) - np.asarray(per["H"])
            bs = []
            for dr in draws:
                bs.append(np.mean([iou_of(C[("SUF", s)][dr, ri, b].sum(0)) - iou_of(C[("H", s)][dr, ri, b].sum(0)) for s in g12.SEEDS]))
            gt_vox = float(np.mean([pooled[("H", s)][ri, b, 0] + pooled[("H", s)][ri, b, 2] for s in g12.SEEDS]))
            res["regions"][r][rn] = {"H": per["H"], "SUF": per["SUF"], "mean_H": float(np.mean(per["H"])), "mean_SUF": float(np.mean(per["SUF"])),
                                     "diff_per_seed": diff.tolist(), "mean_diff": float(diff.mean()), "sd_diff": float(diff.std(ddof=1)),
                                     "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "seeds_positive": int((diff > 0).sum()),
                                     "mean_precision": {k: float(np.mean(v)) for k, v in Pm.items()}, "mean_recall": {k: float(np.mean(v)) for k, v in Rm.items()},
                                     "gt_occupied_voxels": gt_vox}
    F = res["regions"]["full"]["all"]; U5 = res["regions"]["unobs5_later"]["all"]
    g = {"meaningful_gain": F["mean_diff"] >= 0.005 and F["ci95"][0] > 0 and F["seeds_positive"] >= 2,
         "not_only_observed": U5["ci95"][0] > 0,
         "no_catastrophic_tradeoff": (F["mean_precision"]["SUF"] - F["mean_precision"]["H"]) > -0.02 and (F["mean_recall"]["SUF"] - F["mean_recall"]["H"]) > -0.02,
         "hurts": F["mean_diff"] <= -0.005 and F["ci95"][1] < 0 and (3 - F["seeds_positive"]) >= 2,
         "short_range_bins_ci_positive": [rn for rn in RANGES[:3] if res["regions"]["full"][rn]["ci95"][0] > 0],
         "far_bins_ci_positive": [rn for rn in RANGES[3:5] if res["regions"]["full"][rn]["ci95"][0] > 0]}
    res["guide_evaluation"] = g
    with open(os.path.join(T, "paired.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["region", "range", "mean_H", "mean_SUF", "mean_diff", "sd_diff", "ci_lo", "ci_hi", "seeds_positive", "diff_seed0", "diff_seed1", "diff_seed2"])
        for r in REGIONS:
            for rn in RANGES:
                v = res["regions"][r][rn]; w.writerow([r, rn, v["mean_H"], v["mean_SUF"], v["mean_diff"], v["sd_diff"], *v["ci95"], v["seeds_positive"], *v["diff_per_seed"]])
    # learning curves (stride-5 checkpoint evals) and training logs
    curves = {}
    for cond in g12.CONDITIONS:
        for s in g12.SEEDS:
            pts = []
            for st in range(1000, 7000, 1000):
                _, c_ = load_counts(cond, s, st, 5)
                if c_ is not None:
                    pts.append((st, float(iou_of(c_.sum(0)[0, 5]))))
            run = json.load(open(os.path.join(g12.ART, "runs", f"train_g12_{cond}_seed{s}.json")))
            vl = [(x["val"]["step"], x["val"]["loss"], x["val"]["iou"]) for x in run["log"] if "val" in x]
            curves[f"{cond}_seed{s}"] = {"eval_full_iou_stride5": pts, "val_loss": [(a_, b_) for a_, b_, _ in vl], "val_crop_iou": [(a_, c2) for a_, _, c2 in vl],
                                         "train_loss_last200": float(np.mean([x["loss"] for x in run["log"] if "loss" in x][-200:]))}
    res["curves"] = curves
    json.dump(res, open(os.path.join(g12.ART, "analysis.json"), "w"), indent=1)
    figures(res, fd)
    table(res)
    print(json.dumps({"full": F, "unobs5_later": U5, "guide": g}, indent=1))
    return 0


def figures(res, fd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"H": "#555555", "SUF": "#d95f02"}
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
    F = res["regions"]["full"]["all"]; x = np.arange(3)
    ax[0].bar(x - 0.18, F["H"], 0.36, color=col["H"], label="H"); ax[0].bar(x + 0.18, F["SUF"], 0.36, color=col["SUF"], label="S/U/F")
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"seed {s}" for s in g12.SEEDS]); ax[0].set_ylabel("completion IoU, 0006 full"); ax[0].legend(); ax[0].grid(alpha=0.3, axis="y")
    lo = min(F["H"] + F["SUF"]); ax[0].set_ylim(max(0, lo - 0.03), max(F["H"] + F["SUF"]) + 0.02)
    regs = ["full", "past5", "unobs5_later", "observed20", "unobserved_later20", "H_evidence", "no_H_evidence"]
    m = [res["regions"][r]["all"]["mean_diff"] for r in regs]; ci = np.array([res["regions"][r]["all"]["ci95"] for r in regs])
    ax[1].errorbar(range(len(regs)), m, yerr=[np.array(m) - ci[:, 0], ci[:, 1] - np.array(m)], fmt="o", color=col["SUF"], capsize=3); ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(range(len(regs))); ax[1].set_xticklabels(regs, rotation=30, ha="right", fontsize=8); ax[1].set_ylabel("IoU(S/U/F) - IoU(H), mean of 3 seeds [95 % CI]"); ax[1].grid(alpha=0.3)
    for r, mk in (("full", "o"), ("unobs5_later", "s")):
        mm = [res["regions"][r][rn]["mean_diff"] for rn in g12.RANGE_NAMES]; cc = np.array([res["regions"][r][rn]["ci95"] for rn in g12.RANGE_NAMES])
        ax[2].errorbar(np.arange(5) + (0.1 if mk == "s" else -0.1), mm, yerr=[np.array(mm) - cc[:, 0], cc[:, 1] - np.array(mm)], fmt=mk, capsize=3, label=r)
    ax[2].axhline(0, color="k", lw=0.8); ax[2].axvline(2.5, color="k", lw=0.6, ls=":"); ax[2].set_xticks(range(5)); ax[2].set_xticklabels(g12.RANGE_NAMES)
    ax[2].set_xlabel("voxel range (anchor z); dotted: S/U/F input cut-off at 20 m"); ax[2].set_ylabel("IoU difference"); ax[2].legend(); ax[2].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "1_per_seed_region_range.png"), dpi=120); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for k, v in res["curves"].items():
        c = col[k.split("_")[0]]; ls = ["-", "--", ":"][int(k[-1])]
        if v["eval_full_iou_stride5"]:
            ax[0].plot(*zip(*v["eval_full_iou_stride5"]), ls, color=c, marker="o", ms=3, label=k)
        ax[1].plot(*zip(*v["val_loss"]), ls, color=c, label=k)
    ax[0].set_xlabel("step"); ax[0].set_ylabel("0006 full IoU (every 5th anchor)"); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
    ax[1].set_xlabel("step"); ax[1].set_ylabel("training-time validation loss (crops)"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "2_learning_curves.png"), dpi=120); plt.close(fig)


def table(res):
    L = ["# Gate 12 -- H vs S/U/F completion: results\n", "KITTI-360 0006, 590 anchors (stride 3), final-step checkpoints, 3 seeds per condition, "
         "occupancy decision final log-odds > 0. IoU pools voxel counts over anchors per seed. Difference = S/U/F - H; CI = paired clip-block bootstrap "
         f"(blocks of {BOOT_BLOCK} anchors, {N_BOOT} resamples) of the mean-over-seeds difference.\n"]
    L.append("| region | range | IoU H (seeds) | IoU S/U/F (seeds) | mean H | mean S/U/F | diff per seed | mean diff | sd | 95 % CI | P H / S/U/F | R H / S/U/F | GT occ voxels |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r, rr in res["regions"].items():
        for rn, v in rr.items():
            if rn != "all" and r not in ("full", "unobs5_later"):
                continue
            L.append(f"| {r} | {rn} | {' / '.join(f'{x:.4f}' for x in v['H'])} | {' / '.join(f'{x:.4f}' for x in v['SUF'])} | {v['mean_H']:.4f} | {v['mean_SUF']:.4f} | "
                     f"{' / '.join(f'{x:+.4f}' for x in v['diff_per_seed'])} | {v['mean_diff']:+.4f} | {v['sd_diff']:.4f} | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | "
                     f"{v['mean_precision']['H']:.3f} / {v['mean_precision']['SUF']:.3f} | {v['mean_recall']['H']:.3f} / {v['mean_recall']['SUF']:.3f} | {v['gt_occupied_voxels']:,.0f} |")
    L.append(f"\nPredeclared guide: {json.dumps(res['decision_guide'], indent=1)}\n\nEvaluation: {json.dumps(res['guide_evaluation'])}\n")
    L.append("Learning curves (full IoU at steps 1000..6000 on every 5th anchor): " + "; ".join(f"{k}: {[round(y, 4) for _, y in v['eval_full_iou_stride5']]}" for k, v in res["curves"].items()) + "\n")
    open(os.path.join(g12.ART, "results_table.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
