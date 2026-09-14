#!/usr/bin/env python
"""Scaling analysis: per-run tables, paired per-seed deltas with temporal-block bootstrap,
learning curves, class-wise changes, observed vs unobserved, qualitative panels.

    python tools/kitti360_data_scaling_7drives/analyze.py
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drives7                                                                            # noqa: E402
from tools.kitti360_data_scaling import common as C0                                      # noqa: E402

A = drives7.ART; REGIONS = ("unobserved_later", "observed", "full"); NC = 19
BLOCK = 20; N_BOOT = 2000


def load_eval(tag, step, stride=1):
    base = os.path.join(A, "eval", f"{tag}_step{step:05d}" + (f"_stride{stride}" if stride > 1 else ""))
    if not os.path.exists(base + ".json"):
        return None, None
    with np.load(base + "_per_anchor.npz") as z:
        pa = {k: z[k] for k in z.files}
    return json.load(open(base + ".json")), pa


def miou(conf):
    cm = conf.sum(0) if conf.ndim == 3 else conf
    tp = np.diag(cm).astype(float); g = cm.sum(1); p = cm.sum(0); den = g + p - tp
    iou = np.where(den > 0, tp / np.maximum(den, 1), np.nan)
    present = g[1:] > 0
    return float(np.nanmean(iou[1:][present])), iou


def occ_iou(o):
    s = o.sum(0); return float(s[0] / max(s.sum(), 1))


def main() -> int:
    budget = json.load(open(os.path.join(A, "budget.json"))); steps = budget["steps"]
    tags = {(r, s): f"kds7_{r}_seed{s}" for r in ("small", "full") for s in (0, 1)}
    finals = {k: load_eval(t, steps) for k, t in tags.items()}
    assert all(v[0] for v in finals.values()), "missing final evaluation"
    runs = {k: json.load(open(os.path.join(A, "runs", f"train_{t}.json"))) for k, t in tags.items()}
    ids = finals[("full", 0)][1]["ids"]
    assert all(np.array_equal(v[1]["ids"], ids) for v in finals.values()), "anchor sets differ"
    out = {"budget": budget, "runs": {}, "paired": {}, "curves": {}, "classwise": {}}
    for (r, s), (ev, pa) in finals.items():
        tr = runs[(r, s)]
        out["runs"][f"{r}_seed{s}"] = {"regions": ev["regions"], "checkpoint": ev["checkpoint"], "checkpoint_sha256": ev["checkpoint_sha256"],
                                       "exposure": tr["exposure"], "n_train_anchors": tr["n_train_anchors"], "gpu_hours": tr["gpu_hours"],
                                       "init_sha256": tr["init_sha256"], "init_state_digest": tr["init_state_digest"]}
    # init identity per seed
    out["init_identical_within_seed"] = {s: runs[("small", s)]["init_state_digest"] == runs[("full", s)]["init_state_digest"] for s in (0, 1)}
    # paired temporal-block bootstrap (same blocks for both conditions)
    nA = len(ids); blocks = [np.arange(i, min(i + BLOCK, nA)) for i in range(0, nA, BLOCK)]
    rng = np.random.default_rng(20260913); picks = [rng.integers(0, len(blocks), len(blocks)) for _ in range(N_BOOT)]
    for s in (0, 1):
        ps, pf = finals[("small", s)][1], finals[("full", s)][1]; row = {}
        for reg in REGIONS:
            ds = miou(pf[f"conf_{reg}"])[0] - miou(ps[f"conf_{reg}"])[0]; do = occ_iou(pf[f"occ_{reg}"]) - occ_iou(ps[f"occ_{reg}"])
            ts = lambda p: p[f"teach_{reg}"].sum(0)[1] / max(p[f"teach_{reg}"].sum(0)[0], 1)
            dt = ts(pf) - ts(ps); bs_s, bs_o, bs_t = [], [], []
            for pk in picks:
                ix = np.concatenate([blocks[k] for k in pk])
                bs_s.append(miou(pf[f"conf_{reg}"][ix])[0] - miou(ps[f"conf_{reg}"][ix])[0])
                bs_o.append(occ_iou(pf[f"occ_{reg}"][ix]) - occ_iou(ps[f"occ_{reg}"][ix]))
                a_ = pf[f"teach_{reg}"][ix].sum(0); b_ = ps[f"teach_{reg}"][ix].sum(0); bs_t.append(a_[1] / max(a_[0], 1) - b_[1] / max(b_[0], 1))
            ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            row[reg] = {"delta_semantic_miou": ds, "ci95_semantic": ci(bs_s), "p_positive_semantic": float((np.asarray(bs_s) > 0).mean()),
                        "delta_occupancy_iou": do, "ci95_occupancy": ci(bs_o), "p_positive_occupancy": float((np.asarray(bs_o) > 0).mean()),
                        "delta_teacher_agreement": float(dt), "ci95_teacher": ci(bs_t)}
        out["paired"][f"seed{s}"] = row
    out["paired"]["mean"] = {reg: {k: float(np.mean([out["paired"][f"seed{s}"][reg][k] for s in (0, 1)])) for k in ("delta_semantic_miou", "delta_occupancy_iou", "delta_teacher_agreement")} for reg in REGIONS}
    out["bootstrap"] = {"unit": f"contiguous blocks of {BLOCK} development anchors (stride-3 stream frames, i.e. 60 stream frames per block, > the 20-frame target horizon)",
                        "n_boot": N_BOOT, "paired": "identical block draws for small and full within a seed", "scope": "evaluation uncertainty only; training variance is shown by the two seeds"}
    # class-wise (primary region)
    names = list(finals[("full", 0)][0]["regions"]["unobserved_later"]["per_class"])
    for s in (0, 1):
        _, i_s = miou(finals[("small", s)][1]["conf_unobserved_later"]); _, i_f = miou(finals[("full", s)][1]["conf_unobserved_later"])
        sup = finals[("full", s)][0]["regions"]["unobserved_later"]["per_class"]
        out["classwise"][f"seed{s}"] = {n: {"support": sup[n]["gt_support"], "small": None if np.isnan(i_s[c + 1]) else float(i_s[c + 1]),
                                            "full": None if np.isnan(i_f[c + 1]) else float(i_f[c + 1])} for c, n in enumerate(names)}
    # learning curves (intermediate checkpoints on every 5th anchor + training logs)
    for (r, s), t in tags.items():
        pts = []
        for f in sorted(glob.glob(os.path.join(A, "eval", f"{t}_step*_stride5.json"))):
            e = json.load(open(f)); st = int(os.path.basename(f).split("_step")[1][:5])
            pts.append({"step": st, **{f"{reg}_{m}": e["regions"][reg][m] for reg in REGIONS for m in ("semantic_miou", "occupancy_iou", "teacher_agreement")}})
        fe = load_eval(t, steps)[1]; sub = np.arange(0, len(ids), 5)
        pts.append({"step": steps, **{f"{reg}_semantic_miou": miou(fe[f"conf_{reg}"][sub])[0] for reg in REGIONS},
                    **{f"{reg}_occupancy_iou": occ_iou(fe[f"occ_{reg}"][sub]) for reg in REGIONS},
                    **{f"{reg}_teacher_agreement": float(fe[f"teach_{reg}"][sub].sum(0)[1] / max(fe[f"teach_{reg}"][sub].sum(0)[0], 1)) for reg in REGIONS}})
        log = runs[(r, s)]["log"]
        tl = [m for m in log if "val" not in m]; vl = [m["val"] for m in log if "val" in m]
        k = 100; tr_s = [int(np.mean([m["step"] for m in tl[i:i + k]])) for i in range(0, len(tl), k)]
        out["curves"][f"{r}_seed{s}"] = {"eval": sorted(pts, key=lambda p: p["step"]),
                                          "train": {"step": tr_s, "loss": [float(np.mean([m["loss"] for m in tl[i:i + k]])) for i in range(0, len(tl), k)],
                                                    "sem_kl": [float(np.mean([m["sem_kl"] for m in tl[i:i + k]])) for i in range(0, len(tl), k)]},
                                          "val": {"step": [v["step"] for v in vl], "loss": [v["loss"] for v in vl], "sem_kl": [v["sem_kl"] for v in vl], "iou": [v["iou"] for v in vl]}}
    # late-training slope of the full runs (optimisation adequacy): last three curve points
    out["late_change_full"] = {}
    for s in (0, 1):
        ev = out["curves"][f"full_seed{s}"]["eval"]; last = [p for p in ev if p["step"] >= steps - 3000]
        out["late_change_full"][f"seed{s}"] = {m: float(last[-1][f"unobserved_later_{m}"] - last[0][f"unobserved_later_{m}"]) if len(last) >= 2 else None
                                               for m in ("semantic_miou", "occupancy_iou")} | {"window_steps": [last[0]["step"], last[-1]["step"]] if last else None}
    C0.write_json(os.path.join(A, "analysis.json"), out)
    figures(out, steps)
    table(out)
    print(json.dumps(out["paired"], indent=1))
    return 0


def figures(out, steps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = os.path.join(A, "figures"); os.makedirs(fd, exist_ok=True)
    col = {"small": "#d98c3a", "full": "#2c6fbb"}; ls = {0: "-", 1: "--"}
    fig, ax = plt.subplots(2, 3, figsize=(17, 8))
    for key, c in out["curves"].items():
        r, s = key.split("_seed"); s = int(s); e = c["eval"]
        for j, (reg, m, ttl) in enumerate((("unobserved_later", "semantic_miou", "semantic mIoU, unobserved->later measured (primary)"),
                                             ("unobserved_later", "occupancy_iou", "occupancy IoU, unobserved->later measured"),
                                             ("observed", "semantic_miou", "semantic mIoU, observed"))):
            ax[0, j].plot([p["step"] for p in e], [p[f"{reg}_{m}"] for p in e], ls[s], marker="o", ms=3, color=col[r], label=f"{r} seed {s}"); ax[0, j].set_title(ttl, fontsize=9)
        ax[1, 0].plot(c["train"]["step"], c["train"]["loss"], ls[s], color=col[r], label=f"{r} seed {s}"); ax[1, 0].set_title("training loss (100-step means)", fontsize=9)
        ax[1, 1].plot(c["val"]["step"], c["val"]["loss"], ls[s], marker=".", color=col[r], label=f"{r} seed {s}"); ax[1, 1].set_title("validation loss (first 120 dev anchors, crops)", fontsize=9)
        ax[1, 2].plot([p["step"] for p in e], [p["unobserved_later_teacher_agreement"] for p in e], ls[s], marker="o", ms=3, color=col[r], label=f"{r} seed {s}"); ax[1, 2].set_title("teacher agreement, unobserved->later measured", fontsize=9)
    for a_ in ax.ravel():
        a_.set_xlabel("optimizer step"); a_.grid(alpha=0.3); a_.legend(fontsize=7)
    fig.suptitle("learning curves (intermediate checkpoints on every 5th development anchor; final point on the same subset)"); fig.tight_layout()
    fig.savefig(os.path.join(fd, "learning_curves.png"), dpi=120); plt.close(fig)
    # paired deltas
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for j, (m, lab) in enumerate((("semantic", "Δ semantic mIoU (full − small)"), ("occupancy", "Δ occupancy IoU (full − small)"))):
        for k_, reg in enumerate(REGIONS):
            for s in (0, 1):
                p = out["paired"][f"seed{s}"][reg]; d = p[f"delta_{'semantic_miou' if m == 'semantic' else 'occupancy_iou'}"]; ci = p[f"ci95_{m}"]
                x = k_ + (s - 0.5) * 0.25
                ax[j].errorbar([x], [d], yerr=[[d - ci[0]], [ci[1] - d]], fmt="o", color=["#2c6fbb", "#7a5ea8"][s], capsize=4, label=f"seed {s}" if k_ == 0 else None)
        ax[j].axhline(0, color="k", lw=0.8); ax[j].set_xticks(range(len(REGIONS))); ax[j].set_xticklabels(REGIONS); ax[j].set_title(lab); ax[j].grid(alpha=0.3); ax[j].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "paired_deltas.png"), dpi=120); plt.close(fig)
    # class-wise
    fig, ax = plt.subplots(figsize=(14, 4.5))
    cw = out["classwise"]; names = [n for n in cw["seed0"] if cw["seed0"][n]["support"] > 0]; names.sort(key=lambda n: -cw["seed0"][n]["support"]); x = np.arange(len(names))
    for s in (0, 1):
        d = [(cw[f"seed{s}"][n]["full"] or 0) - (cw[f"seed{s}"][n]["small"] or 0) for n in names]
        ax.bar(x + (s - 0.5) * 0.4, d, 0.4, color=["#2c6fbb", "#7a5ea8"][s], label=f"seed {s}")
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(x); ax.set_xticklabels([f"{n}\n{cw['seed0'][n]['support']:,}" for n in names], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Δ IoU (full − small)"); ax.set_title("class-wise change, unobserved->later measured (GT support below names)"); ax.grid(alpha=0.3, axis="y"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(fd, "classwise_delta.png"), dpi=120); plt.close(fig)


def table(out):
    L = ["# Seven-drive data scaling -- result tables\n", "“100 %” = all eligible anchors across the seven SSCBench-KITTI-360 training drives 0000, 0002, 0003, 0004, 0005, 0007, 0010.\n",
         "| run | train anchors | steps | epochs | unique anchors seen | sem mIoU unobs→later | occ IoU unobs→later | sem mIoU observed | occ IoU observed | sem mIoU full | occ IoU full | teacher agr. unobs→later |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in out["runs"].items():
        g = v["regions"]
        L.append(f"| {k} | {v['n_train_anchors']:,} | {out['budget']['steps']:,} | {v['exposure']['effective_epochs']:.2f} | {v['exposure']['unique_anchors_seen']:,} | "
                 f"{g['unobserved_later']['semantic_miou']:.4f} | {g['unobserved_later']['occupancy_iou']:.4f} | {g['observed']['semantic_miou']:.4f} | {g['observed']['occupancy_iou']:.4f} | "
                 f"{g['full']['semantic_miou']:.4f} | {g['full']['occupancy_iou']:.4f} | {g['unobserved_later']['teacher_agreement']:.4f} |")
    L += ["", "| seed | region | Δ sem mIoU [95 % block CI] | Δ occ IoU [95 % block CI] | Δ teacher agreement [CI] |", "|---|---|---|---|---|"]
    for s in (0, 1):
        for reg in REGIONS:
            p = out["paired"][f"seed{s}"][reg]
            L.append(f"| {s} | {reg} | {p['delta_semantic_miou']:+.4f} [{p['ci95_semantic'][0]:+.4f}, {p['ci95_semantic'][1]:+.4f}] | {p['delta_occupancy_iou']:+.4f} [{p['ci95_occupancy'][0]:+.4f}, {p['ci95_occupancy'][1]:+.4f}] | "
                     f"{p['delta_teacher_agreement']:+.4f} [{p['ci95_teacher'][0]:+.4f}, {p['ci95_teacher'][1]:+.4f}] |")
    L += ["", "| class | GT support (unobs→later) | small s0 | full s0 | small s1 | full s1 |", "|---|---|---|---|---|---|"]
    cw = out["classwise"]
    for n in sorted(cw["seed0"], key=lambda n: -cw["seed0"][n]["support"]):
        f = lambda x: "-" if x is None else f"{x:.3f}"
        L.append(f"| {n} | {cw['seed0'][n]['support']:,} | {f(cw['seed0'][n]['small'])} | {f(cw['seed0'][n]['full'])} | {f(cw['seed1'][n]['small'])} | {f(cw['seed1'][n]['full'])} |")
    open(os.path.join(A, "results_table.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
