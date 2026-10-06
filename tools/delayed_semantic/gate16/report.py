"""Gate 16 tables, figures, deterministic examples, candidate_config.yaml and code_freeze.json (from analysis.json)."""
from __future__ import annotations

import dataclasses
import glob
import hashlib
import json
import os
import subprocess
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from tools.delayed_semantic import config as PC  # noqa: E402
from tools.delayed_semantic.gate16 import config as G  # noqa: E402
from tools.delayed_semantic.gate16.analysis import NAMES  # noqa: E402
from tools.delayed_semantic.regions import PRIMARY  # noqa: E402
from tools.delayed_semantic.report import PALETTE  # noqa: E402

A = G.ARTIFACT_DIR
COL = {"C0": "#bdbdbd", "B0": "#7f7f7f", "B1": "#1f77b4", "B2": "#d62728", "B3": "#2ca02c", "B3geom": "#98df8a", "oracle": "#000000"}
pct = lambda x: "—" if x is None else f"{100 * x:.2f}"
REG_LABEL = {PRIMARY: "completed_unobserved (primary)", "completed_geo_observed": "geometry observed, semantically unsupported",
             "completed_geo_unobserved": "geometry unobserved", "completed_later_revealed": "later revealed", "completed_revealed_unseen": "later revealed, no H_t camera return",
             "completed_not_revealed": "not revealed", "primary_candidate_hit": "primary ∩ candidate-hit", "primary_candidate_miss": "primary ∩ candidate-miss",
             "ssc_full": "SSC full", "ssc_never_observed": "SSC without evidence", "observed_current": "observed in frame t", "observed_past_only": "observed only earlier",
             "never_observed": "never observed (GT occupancy)"}


def _t(head: List[str], rows: List[List]) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]) + "\n"


def _csv(path: str, head: List[str], rows: List[List]) -> None:
    with open(path, "w") as fh:
        fh.write(",".join(head) + "\n")
        for r in rows:
            fh.write(",".join("" if c is None else (f"{c:.6f}" if isinstance(c, float) else str(c)) for c in r) + "\n")


def ceiling_md(c: Dict) -> str:
    rows = [[REG_LABEL.get(r, r), f"{v['voxels']:,}", pct(v["C0_acc"]), pct(v["C16_hit"]), pct(v["oracle_acc"])] for r, v in c["regions"].items()]
    th = c["selection_threshold"]
    out = ["# Candidate ceiling (0006, reconstructed with the frozen candidate rule; evaluation only; frozen before any B3 score)\n",
           _t(["region", "voxels", "C0 nearest-candidate acc %", "C16 any-of-16 hit %", "oracle selection acc %"], rows),
           f"\nPrimary mIoU: nearest candidate {pct(c['miou']['C0'])} %, oracle selection {pct(c['miou']['oracle'])} %; on candidate-hit voxels nearest {pct(c['miou']['C0_hit'])} %.\n",
           f"Candidate-miss fraction of the primary region: {pct(c['candidate_miss_fraction'])} %. History-wide evidence ceiling (context only): {pct(c['history_wide_context'])} %.\n",
           f"\n**Selection headroom** H = {pct(th['O_oracle_acc'])} − {pct(th['N_nearest_candidate_acc'])} = {pct(th['H'])} points; "
           f"**threshold** T = N + 0.5 H = **{pct(th['T'])} %**.\n",
           "\n## Per class (primary)\n", _t(["class", "GT voxels", "nearest hit %", "any-of-16 hit %"], [[k, f"{v['voxels']:,}", pct(v["C0_acc"]), pct(v["C16_hit"])] for k, v in c["per_class"].items()]),
           "\n## Per distance to nearest evidence\n", _t(["bin", "voxels", "nearest %", "any-of-16 %"], [[k, f"{v['voxels']:,}", pct(v["C0_acc"]), pct(v["C16_hit"])] for k, v in c["per_distance"].items()]),
           "\n## Per reveal delay\n", _t(["bin", "voxels", "nearest %", "any-of-16 %"], [[k, f"{v['voxels']:,}", pct(v["C0_acc"]), pct(v["C16_hit"])] for k, v in c["per_reveal"].items()])]
    return "\n".join(out)


def selection_md(a: Dict) -> str:
    s = a["selection"]; out = ["# Stage 1 — candidate selection (primary region, 0006)\n",
                               f"N (nearest candidate) {pct(s['N'])} %, O (oracle) {pct(s['O'])} %, H {pct(s['H'])} points, T {pct(s['T'])} %.\n"]
    rows = [["C0 nearest candidate", pct(s["C0_acc"]), "", "", "", "0.00"], ["B0 nearest evidence (EDT)", pct(s["B0"]["readout_acc"]), pct(s["B0"]["top_weight_acc"]), pct(s["B0"]["hit_acc"]), pct(s["B0"]["miss_acc"]), f"{s['B0']['R']:.2f}"]]
    for m in ("B1", "B2"):
        rows.append([f"{m} (pilot, seed 0)", pct(s[m]["readout_acc"]), pct(s[m]["top_weight_acc"]), pct(s[m]["hit_acc"]), pct(s[m]["miss_acc"]), f"{s[m]['R']:.2f}"])
    for v in ("B3", "B3geom"):
        for sd, r in s[v]["per_seed"].items():
            rows.append([sd, pct(r["readout_acc"]), pct(r["top_weight_acc"]), pct(r["hit_acc"]), pct(r["miss_acc"]), f"{r['R']:.2f}"])
        mm = s[v]["mean"]; rows.append([f"**{v} mean**", pct(mm["readout_acc"]), pct(mm["top_weight_acc"]), pct(mm["hit_acc"]), pct(mm["miss_acc"]), f"**{mm['R']:.2f}**"])
    rows.append(["oracle selection", pct(s["oracle_acc"]), "", "100.00", "0.00", "1.00"])
    out += [_t(["method", "readout acc %", "top-weight candidate acc %", "candidate-hit acc %", "candidate-miss acc %", "recovered fraction R"], rows)]
    for v in ("B3", "B3geom"):
        g, h = s[v]["acc_gain_vs_B0"], s[v]["hit_acc_gain_vs_B0"]
        out.append(f"\n{v} − B0 accuracy: {100*g['delta']:+.2f} [{100*g['ci95'][0]:+.2f}, {100*g['ci95'][1]:+.2f}]; on candidate-hit {100*h['delta']:+.2f} [{100*h['ci95'][0]:+.2f}, {100*h['ci95'][1]:+.2f}] (seed-pooled, 20-anchor blocks).\n")
    rows = [[m, f"{v['voxels']:,}", pct(v["top1"]), pct(v["top3"]), pct(v["mean_weight_on_best"]), "n/a" if m == "B0" else f"{v['kl_to_candidate_target']:.4f}",
             pct(v["top_weight_gt_acc"]), pct(v["delayed_best_gt_acc"]), pct(v["nearest_gt_acc"])] for m, v in s["label_free"].items()]
    out += ["\n## Label-free agreement with the delayed candidate target (0006 delayed-target voxels, not evidence) and GT diagnostic there\n",
            _t(["method", "voxels", "top-1 = best %", "top-3 ∋ best %", "mean weight on best %", "KL(target‖w)", "top-weight class = GT %", "delayed-best class = GT %", "nearest class = GT %"], rows)]
    c = a["criteria"]
    out += ["\n## Criteria\n", _t(["criterion", "value", "pass"], [
        ["S1 seed-mean acc ≥ T, all seeds > B0, CI lo > 0", f"{pct(c['S1']['seed_mean_acc'])} vs T {pct(c['S1']['T'])}; R {c['S1']['R']:.2f}; seeds {[round(100*x, 2) for x in c['S1']['seed_gains']]}; CI [{100*c['S1']['ci95'][0]:+.2f}, {100*c['S1']['ci95'][1]:+.2f}]", c["S1"]["pass"]],
        ["S2 candidate-hit gain ≥ +2, all seeds, CI lo > 0", f"{100*c['S2']['gain']:+.2f}; seeds {[round(100*x, 2) for x in c['S2']['seed_gains']]}; CI [{100*c['S2']['ci95'][0]:+.2f}, {100*c['S2']['ci95'][1]:+.2f}]", c["S2"]["pass"]],
        ["M1 mIoU gain ≥ +1, all seeds, CI lo > 0, > B2, ≥ 5 classes", f"{100*c['M1']['gain']:+.2f}; seeds {[round(100*x, 2) for x in c['M1']['seed_gains']]}; CI [{100*c['M1']['ci95'][0]:+.2f}, {100*c['M1']['ci95'][1]:+.2f}]; B3−B2 {100*c['M1']['B3_minus_B2']:+.2f}; {c['M1']['classes_up']} classes", c["M1"]["pass"]],
        ["Occ occupancy identical", "", c["Occ"]["pass"]],
        ["D delayed target learned but GT selection worse", f"top-1 B3 {pct(c['D']['B3_top1'])} vs B1 {pct(c['D']['B1_top1'])}; acc-gain CI upper {100*c['D']['acc_gain_ci_upper']:+.2f}", c["D"]["pass"]]]),
        f"\n**Decision {a['decision']}.**\n"]
    return "\n".join(out)


def results_md(a: Dict) -> str:
    sem = a["semantic_miou"]; regs = list(sem)
    rows = [[REG_LABEL.get(r, r), pct(sem[r]["B0"]), pct(sem[r]["B1"]), pct(sem[r]["B2"]), f"{pct(sem[r]['B3'])} ({' / '.join(pct(x) for x in sem[r]['B3_per_seed'])})",
             f"{pct(sem[r]['B3geom'])} ({' / '.join(pct(x) for x in sem[r]['B3geom_per_seed'])})"] for r in regs]
    out = ["# Gate 16 results — 0006 (590 anchors), geometry seed 0; B1/B2 pilot seed 0; B3/B3geom seeds 0/1/2 (mean (per seed))\n",
           "## Semantic mIoU %\n", _t(["region", "B0", "B1", "B2", "B3", "B3geom"], rows), "\n## Paired differences (mIoU points; seed-pooled B3 vs single-seed controls; 95 % block bootstrap)\n"]
    rows = [[k.replace("-", " − "), REG_LABEL.get(r, r), f"{100*d['delta']:+.2f}", f"[{100*d['ci95'][0]:+.2f}, {100*d['ci95'][1]:+.2f}]"] for k, v in a["semantic_deltas"].items() for r, d in v.items()]
    out += [_t(["pair", "region", "Δ", "CI"], rows)]
    rows = [[c, f"{v['gt_voxels']:,}", pct(v["nearest_hit"]), pct(v["any16_hit"]), pct(v["B3_selected_correct"]), pct(v["B0_iou"]), pct(v["B3_iou"]), v["limited_by"]] for c, v in a["per_class"].items()]
    out += ["\n## Per class (primary): coverage vs association\n", _t(["class", "GT voxels", "nearest hit %", "any-of-16 %", "B3 selected correct %", "B0 IoU %", "B3 IoU %", "larger gap"], rows),
            "\n'larger gap' compares candidate absence (1 − any-of-16) with the association gap (any-of-16 − B3 selected correct); descriptive, not a criterion.\n"]
    for name, key in (("distance to nearest evidence", "per_distance"), ("reveal delay", "per_reveal")):
        rows = [[b, f"{v['voxels']:,}", pct(v["candidate_hit_rate"]), pct(v["B0_acc"]), pct(v["B3_top_weight_acc"]), pct(v["B3_readout_acc"]), pct(v["B0_miou"]), pct(v["B3_miou"])] for b, v in a[key].items()]
        out += [f"\n## By {name} (primary)\n", _t(["bin", "voxels", "candidate-hit %", "B0 acc %", "B3 top-weight acc %", "B3 readout acc %", "B0 mIoU %", "B3 mIoU %"], rows)]
    out += [f"\n**Decision {a['decision']}.**\n"]
    return "\n".join(out)


def tables(a: Dict, c: Dict) -> None:
    os.makedirs(G.TABLES, exist_ok=True); s = a["selection"]
    rows = [["C0", s["C0_acc"], None, None, None, 0.0]] + [[m, s[m]["readout_acc"], s[m]["top_weight_acc"], s[m]["hit_acc"], s[m]["miss_acc"], s[m]["R"]] for m in ("B0", "B1", "B2")]
    rows += [[sd, r["readout_acc"], r["top_weight_acc"], r["hit_acc"], r["miss_acc"], r["R"]] for v in ("B3", "B3geom") for sd, r in s[v]["per_seed"].items()]
    rows += [["oracle", s["oracle_acc"], None, 1.0, 0.0, 1.0]]
    _csv(os.path.join(G.TABLES, "selection.csv"), ["method", "readout_acc", "top_weight_acc", "hit_acc", "miss_acc", "recovered_fraction"], rows)
    heads = ["B0", "B1", "B2", "B3", "B3geom"]
    _csv(os.path.join(G.TABLES, "per_region.csv"), ["region"] + heads + [f"B3_s{i}" for i in G.SEEDS],
         [[r] + [v[h] for h in heads] + v["B3_per_seed"] for r, v in a["semantic_miou"].items()])
    _csv(os.path.join(G.TABLES, "per_class.csv"), ["class", "gt_voxels", "B0_iou", "B3_iou", "B3_selected_correct", "limited_by"],
         [[k, v["gt_voxels"], v["B0_iou"], v["B3_iou"], v["B3_selected_correct"], v["limited_by"]] for k, v in a["per_class"].items()])
    _csv(os.path.join(G.TABLES, "per_distance.csv"), ["bin", "voxels", "candidate_hit_rate", "B0_acc", "B3_top_weight_acc", "B3_readout_acc", "B0_miou", "B3_miou"],
         [[b, v["voxels"], v["candidate_hit_rate"], v["B0_acc"], v["B3_top_weight_acc"], v["B3_readout_acc"], v["B0_miou"], v["B3_miou"]] for b, v in a["per_distance"].items()]
         + [[f"reveal {b}", v["voxels"], v["candidate_hit_rate"], v["B0_acc"], v["B3_top_weight_acc"], v["B3_readout_acc"], v["B0_miou"], v["B3_miou"]] for b, v in a["per_reveal"].items()])
    _csv(os.path.join(G.TABLES, "candidate_coverage.csv"), ["class", "gt_voxels", "nearest_hit", "any16_hit", "B3_selected_correct", "absence_gap", "association_gap"],
         [[k, v["gt_voxels"], v["nearest_hit"], v["any16_hit"], v["B3_selected_correct"], v["absence_gap"], v["association_gap"]] for k, v in a["per_class"].items()])


def figures(a: Dict) -> List[str]:
    os.makedirs(G.FIGURES, exist_ok=True); s = a["selection"]; out = []
    def save(fig, name):
        p = os.path.join(G.FIGURES, name); fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig); out.append(p)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    bars = [("C0 nearest", s["C0_acc"], COL["C0"]), ("B0", s["B0"]["readout_acc"], COL["B0"]), ("B1", s["B1"]["readout_acc"], COL["B1"]), ("B2", s["B2"]["readout_acc"], COL["B2"]),
            ("B3", s["B3"]["mean"]["readout_acc"], COL["B3"]), ("B3geom", s["B3geom"]["mean"]["readout_acc"], COL["B3geom"]), ("oracle", s["oracle_acc"], COL["oracle"])]
    ax[0].bar([b[0] for b in bars], [100 * b[1] for b in bars], color=[b[2] for b in bars])
    for i, v in enumerate(("B3", "B3geom")):
        ax[0].scatter([4 + i] * 3, [100 * r["readout_acc"] for r in s[v]["per_seed"].values()], color="k", s=10, zorder=3)
    ax[0].axhline(100 * s["T"], ls="--", color="k", lw=1); ax[0].text(0, 100 * s["T"] + 0.3, f"T = {100*s['T']:.2f} %", fontsize=8)
    ax[0].set_ylim(100 * s["C0_acc"] - 6, 100 * s["oracle_acc"] + 3); ax[0].set_ylabel("GT-class accuracy, primary region %"); ax[0].set_title("1. selection accuracy (dots = seeds)")
    names = ["B0", "B1", "B2"] + [f"B3 s{i}" for i in G.SEEDS] + [f"B3geom s{i}" for i in G.SEEDS]
    vals = [s["B0"]["R"], s["B1"]["R"], s["B2"]["R"]] + [r["R"] for r in s["B3"]["per_seed"].values()] + [r["R"] for r in s["B3geom"]["per_seed"].values()]
    ax[1].bar(names, vals, color=[COL["B0"], COL["B1"], COL["B2"]] + [COL["B3"]] * 3 + [COL["B3geom"]] * 3)
    ax[1].axhline(0.5, ls="--", color="k", lw=1); ax[1].axhline(0, color="k", lw=0.6); ax[1].set_ylabel("recovered fraction R"); ax[1].set_title("2. recovered candidate-selection headroom")
    ax[1].tick_params(axis="x", rotation=45)
    save(fig, "1_2_selection_and_recovered_fraction.png")
    sem = a["semantic_miou"]; regs = [PRIMARY, "completed_geo_observed", "completed_geo_unobserved", "completed_later_revealed", "completed_revealed_unseen", "ssc_full"]
    fig, ax = plt.subplots(figsize=(14, 4.5)); w = 0.16
    for i, m in enumerate(["B0", "B1", "B2", "B3", "B3geom"]):
        ax.bar(np.arange(len(regs)) + (i - 2) * w, [100 * sem[r][m] for r in regs], w, color=COL[m], label=m)
    ax.set_xticks(range(len(regs))); ax.set_xticklabels([REG_LABEL[r] for r in regs], rotation=15, fontsize=8); ax.set_ylabel("mIoU %"); ax.legend(); ax.set_title("3. semantic mIoU")
    save(fig, "3_semantic_miou.png")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for j, (metric, key) in enumerate((("accuracy %", "acc"), ("mIoU %", "miou"))):
        for i, m in enumerate(["B0", "B1", "B2", "B3", "B3geom"]):
            if key == "acc":
                vals = [s[m]["hit_acc"], s[m]["miss_acc"]] if m in ("B0", "B1", "B2") else [s[m]["mean"]["hit_acc"], s[m]["mean"]["miss_acc"]]
            else:
                vals = [sem["primary_candidate_hit"][m], sem["primary_candidate_miss"][m]]
            ax[j].bar(np.arange(2) + (i - 2) * 0.16, [100 * v for v in vals], 0.16, color=COL[m], label=m)
        ax[j].set_xticks([0, 1]); ax[j].set_xticklabels(["candidate-hit", "candidate-miss"]); ax[j].set_ylabel(metric); ax[j].legend(fontsize=7)
    fig.suptitle("4. candidate-hit vs candidate-miss (primary)"); save(fig, "4_hit_vs_miss.png")
    pc = a["per_class"]; cls = [k for k, v in pc.items() if v["gt_voxels"]]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    x = np.arange(len(cls))
    ax[0].bar(x - 0.27, [100 * pc[c]["nearest_hit"] for c in cls], 0.27, color=COL["C0"], label="nearest hit")
    ax[0].bar(x, [100 * pc[c]["B3_selected_correct"] for c in cls], 0.27, color=COL["B3"], label="B3 selected correct")
    ax[0].bar(x + 0.27, [100 * pc[c]["any16_hit"] for c in cls], 0.27, color=COL["oracle"], label="any-of-16")
    ax[0].set_xticks(x); ax[0].set_xticklabels(cls, rotation=60, ha="right", fontsize=8); ax[0].set_ylabel("%"); ax[0].legend(); ax[0].set_title("5a. per-class candidate coverage and selection")
    ax[1].scatter([100 * pc[c]["any16_hit"] for c in cls], [100 * pc[c]["B3_iou"] for c in cls], c=[PALETTE[NAMES.index(c) + 1] / 255 for c in cls], s=40, edgecolors="k")
    for c in cls:
        ax[1].annotate(c, (100 * pc[c]["any16_hit"], 100 * pc[c]["B3_iou"]), fontsize=7)
    ax[1].set_xlabel("any-of-16 candidate hit rate %"); ax[1].set_ylabel("B3 IoU %"); ax[1].set_title("5b. coverage vs B3 IoU")
    save(fig, "5_per_class_coverage.png")
    for n, key, title in ((6, "per_distance", "distance to nearest evidence"), (7, "per_reveal", "reveal delay")):
        d = a[key]; b = list(d)
        fig, ax = plt.subplots(1, 2, figsize=(13, 4))
        ax[0].plot(b, [100 * d[k]["candidate_hit_rate"] for k in b], "o-", color=COL["oracle"], label="candidate-hit (oracle)")
        ax[0].plot(b, [100 * d[k]["B0_acc"] for k in b], "o-", color=COL["B0"], label="B0")
        ax[0].plot(b, [100 * d[k]["B3_top_weight_acc"] for k in b], "o-", color=COL["B3"], label="B3 top-weight")
        ax[0].plot(b, [100 * d[k]["B3_readout_acc"] for k in b], "o--", color=COL["B3"], label="B3 readout")
        ax[0].set_ylabel("accuracy %"); ax[0].legend(fontsize=7)
        ax[1].plot(b, [100 * d[k]["B0_miou"] for k in b], "o-", color=COL["B0"], label="B0"); ax[1].plot(b, [100 * d[k]["B3_miou"] for k in b], "o-", color=COL["B3"], label="B3")
        ax[1].set_ylabel("mIoU %"); ax[1].legend()
        fig.suptitle(f"{n}. primary region by {title}"); save(fig, f"{n}_by_{key.split('_')[1]}.png")
    return out


@torch.no_grad()
def examples(device: str) -> str:
    """Deterministic: anchor = median B3-seed-0 − B0 accuracy gain among anchors with >= 2,000 primary voxels; in it, the lowest flat-index primary voxel
    of each case (a) nearest wrong / B3 right, (b) a correct candidate exists but B3's top-weight candidate is wrong, (c) candidate-miss."""
    from tools.delayed_semantic.gate16 import analysis as AN, context as CX, evaluate as EV
    from tools.delayed_semantic.gate16.candidate_metrics import accuracy
    from tools.delayed_semantic.stream import DriveStream
    from tools.semantic_completion.evidence import trident as TR
    from tools.semantic_completion.frozen_geometry import load_frozen_geometry
    z = AN.load(); dev = torch.device(device)
    ac = lambda cm: np.trace(cm[:, 1:, 1:], axis1=1, axis2=2) / np.maximum(cm[:, 1:].sum((1, 2)), 1)
    b0, b3 = z[f"conf__B0__{PRIMARY}"], z[f"conf__B3_s0__{PRIMARY}"]; ok = b0[:, 1:].sum((1, 2)) >= 2000
    gain = ac(b3) - ac(b0); ids = z["ids"][ok]; t = int(ids[np.argsort(gain[ok], kind="stable")[len(ids) // 2]])
    text, _, _ = TR.load_text_embeddings(); text = text.to(dev)
    ctx = CX.student_context(DriveStream(PC.DEV_DRIVE), t, load_frozen_geometry(PC.GEOMETRY_SEED, dev), text, dev); tr = CX.truth(t, dev)
    gt = tr["sem_gt"]; P = ((gt >= 1) & (gt <= 18)) & ctx["O"] & ~ctx["E"]
    q = P.nonzero(as_tuple=True)[0]
    head = EV.load_head(G.head_path("B3", 0), dev)
    chunks = CX.query_candidates(ctx, q, text); cs = CX.cat_sets([c for _, c in chunks]); sc = EV.score_heads({"B3": head}, ctx, chunks, text)["B3"]
    g = gt[q]; hit = (cs.classes == g[:, None]).any(-1); top = sc["top"]; topcls = cs.classes.gather(1, top[:, None]).squeeze(1)
    cases = {"a_nearest_wrong_B3_right": (cs.classes[:, 0] != g) & (topcls == g), "b_candidate_exists_B3_wrong": hit & (topcls != g), "c_candidate_miss": ~hit}
    evflat = ctx["ev"].flat_index; evlab = ctx["ev_label"][evflat]
    X, Y, Zd = PC.GRID_DIMS
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.5))
    for ax, (name, m) in zip(axs, cases.items()):
        if not m.any():
            ax.set_title(f"{name}: no voxel"); ax.axis("off"); continue
        i = int(m.nonzero(as_tuple=True)[0][0]); v = int(q[i]); vx, vy = v // (Y * Zd), (v // Zd) % Y
        ex, ey = (evflat // (Y * Zd)).cpu().numpy(), ((evflat // Zd) % Y).cpu().numpy()
        near = (np.abs(ex - vx) <= 30) & (np.abs(ey - vy) <= 30)
        ax.scatter(ey[near], ex[near], c=PALETTE[evlab.cpu().numpy()[near]] / 255, s=6, alpha=0.5)
        cx, cy = (cs.voxel_ids[i] // (Y * Zd)).cpu().numpy(), ((cs.voxel_ids[i] // Zd) % Y).cpu().numpy(); w = sc["weights"][i].cpu().numpy()
        ax.scatter(cy, cx, s=40 + 900 * w, facecolors="none", edgecolors="k", linewidths=1.2)
        ax.scatter([cy[0]], [cx[0]], marker="s", s=90, facecolors="none", edgecolors="#7f7f7f", linewidths=2)
        ax.scatter([vy], [vx], marker="*", s=300, c=[PALETTE[int(g[i])] / 255], edgecolors="k")
        ax.set_xlim(vy - 30, vy + 30); ax.set_ylim(vx - 30, vx + 30); ax.set_aspect("equal")
        ax.set_title(f"{name}\nGT {NAMES[int(g[i])-1]} | nearest {NAMES[int(cs.classes[i,0])-1]} | B3 top {NAMES[int(topcls[i])-1]} (w={w.max():.2f}) | classes {sorted({NAMES[int(c)-1] for c in cs.classes[i]})}", fontsize=7)
        ax.set_xlabel("grid y (voxels)"); ax.set_ylabel("grid x (voxels, forward)")
    fig.suptitle(f"8. deterministic examples, 0006 t={t} (median-gain anchor; lowest flat index per case). star = query (GT colour), grey square = nearest, circles = 16 candidates sized by B3-seed-0 weight, dots = evidence by teacher class (±6 m)", fontsize=9)
    p = os.path.join(G.FIGURES, "8_examples.png"); fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    return p


def code_freeze() -> Dict:
    files = sorted(glob.glob(os.path.join(G.REPO_ROOT, "tools/delayed_semantic/gate16/*.py"))) + [os.path.join(G.REPO_ROOT, f) for f in (
        "tools/delayed_semantic/model.py", "tools/delayed_semantic/train.py", "tests/delayed_semantic/test_gate16.py", "artifacts/gate16_candidate_selection/predeclaration.md",
        "artifacts/gate16_candidate_selection/candidate_ceiling.json")] + sorted(glob.glob(os.path.join(G.CHECKPOINTS, "*_step03000.pt"))) + list(G.PILOT_HEADS.values())
    rec = {"git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=G.REPO_ROOT, capture_output=True, text=True).stdout.strip(),
           "sha256": {os.path.relpath(f, G.REPO_ROOT): hashlib.sha256(open(f, "rb").read()).hexdigest() for f in files},
           "frozen_stages": {s: json.load(open(os.path.join(A, f"frozen_{s}.json")))["frozen_at"] for s in ("predeclaration", "ceiling", "code")}}
    json.dump(rec, open(os.path.join(A, "code_freeze.json"), "w"), indent=1)
    return rec


def write_all(device: str = "cuda:0") -> Dict:
    a = json.load(open(os.path.join(A, "analysis.json"))); c = json.load(open(os.path.join(A, "candidate_ceiling.json")))
    open(os.path.join(A, "candidate_ceiling.md"), "w").write(ceiling_md(c))
    open(os.path.join(A, "selection_results.md"), "w").write(selection_md(a))
    open(os.path.join(A, "results_table.md"), "w").write(results_md(a))
    tables(a, c)
    cfg = {"protocol": G.PROTOCOL_VERSION, "k_candidates": G.K, "candidate_rule": "16 nearest observed evidence voxel centres, Euclidean, ascending (model.nearest_candidates)",
           "tau_target": G.TAU_TARGET, "compatibility": "r_i = <p_i, q_v>, p_i = softmax(40 cos(f_i, text)), q_v = delayed target distribution",
           "seeds": list(G.SEEDS), "variants": G.VARIANTS, "train": dataclasses.asdict(PC.TrainConfig()), "model": dataclasses.asdict(PC.ModelConfig()),
           "delayed_targets": "pilot delayed_targets (10 s, tiers A+B), reused unchanged", "selection_threshold": c["selection_threshold"],
           "geometry_seed": PC.GEOMETRY_SEED, "pilot_controls": {k: os.path.relpath(v, G.REPO_ROOT) for k, v in G.PILOT_HEADS.items()}}
    yaml.safe_dump(json.loads(json.dumps(cfg)), open(os.path.join(A, "candidate_config.yaml"), "w"), sort_keys=False, width=120)
    figs = figures(a) + [examples(device)]
    code_freeze()
    return {"figures": figs}


__all__ = ["write_all", "code_freeze", "examples", "figures", "tables"]
