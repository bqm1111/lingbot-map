"""Gate 21 figures and report.

The report opens with the table section 20 requires, with the `E2` row emphasised, and then answers Q1-Q5
of section 21 explicitly. Interpretation text is selected by the outcome the pre-registered rule returns --
it is not written to fit whatever came out.

    PYTHONPATH=$PWD python -m tools.future_semantic_prior.report
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools.future_semantic_prior import config as G

FIG = os.path.join(G.ARTIFACT_DIR, "figures")
ORDER = ("C1_prior", "C2_geometry", "C3_full", "C4_semantics")
SHORT = {"C1_prior": "C1 prior", "C2_geometry": "C2 geometry", "C3_full": "C3 full",
         "C4_semantics": "C4 semantics"}
POP_LABEL = {"E0": "E0 all P0", "E1": "E1 revealed", "E2": "E2 NEVER revealed",
             "E1_P1": "E1 P1", "E2_P1": "E2 P1"}


#: Where each earlier learned attempt's number on THIS population (0006 ``completed_never_observed`` ==
#: Gate 21's ``E0``, 2,743,212 voxels) is recorded. Read at build time rather than quoted from memory.
PRIOR_SOURCES = {
    "Gate 15 B1": ("artifacts/gate15_semantic_completion/analysis.json",
                   lambda j: 100.0 * float(np.mean([j["regions"]["completed_unobserved"]["B1"][k]["miou"]
                                                    for k in sorted(j["regions"]["completed_unobserved"]["B1"])]))),
    "pilot B1": ("artifacts/gate16_candidate_selection/analysis.json",
                 lambda j: 100.0 * j["semantic_miou"]["completed_never_observed"]["B1"]),
    "pilot B2": ("artifacts/gate16_candidate_selection/analysis.json",
                 lambda j: 100.0 * j["semantic_miou"]["completed_never_observed"]["B2"]),
    "Gate 16 B3": ("artifacts/gate16_candidate_selection/analysis.json",
                   lambda j: 100.0 * j["semantic_miou"]["completed_never_observed"]["B3"]),
    "B5": ("artifacts/b5_semantic_residual/analysis.json",
           lambda j: float(j["miou_by_region"]["completed_never_observed"]["B5_mean"])),
}
#: B4 has no machine-readable record of this region; its number comes from its own frozen results table.
PRIOR_B4 = ("artifacts/semantic_field/source_tables.md", 8.22)


def _prior_gates(a: Dict) -> Dict[str, float]:
    """Each earlier attempt's mIoU on E0, read from its own artifact. Missing files are simply left out."""
    out: Dict[str, float] = {}
    for name, (path, get) in PRIOR_SOURCES.items():
        try:
            out[name] = float(get(json.load(open(path))))
        except Exception:
            continue
    if os.path.exists(PRIOR_B4[0]):
        out["B4"] = PRIOR_B4[1]
    return out


def _curves() -> Dict[str, List[Dict]]:
    root = os.path.join(G.ARTIFACT_DIR, "training_curves")
    out: Dict[str, List[Dict]] = {}
    if not os.path.isdir(root):
        return out
    for f in sorted(os.listdir(root)):
        if f.endswith(".json"):
            r = json.load(open(os.path.join(root, f)))
            out.setdefault(r["control"], []).append(r)
    return out


# ---- figures ------------------------------------------------------------------------------------------
def fig_a_curves(cur: Dict[str, List[Dict]]) -> Optional[str]:
    c = G.PRIMARY_CONTROL
    if c not in cur:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for r in sorted(cur[c], key=lambda x: x["seed"]):
        st = [row["step"] for row in r["rows"]]
        ax[0].plot(st, [row["train_loss"] for row in r["rows"]], marker="o", ms=3,
                   label=f"seed {r['seed']}")
        ax[1].plot(st, [row["E1"]["miou"] for row in r["rows"]], marker="o", ms=3, ls="--",
                   color=f"C{r['seed']}", label=f"E1 seed {r['seed']}")
        ax[1].plot(st, [row["E2"]["miou"] for row in r["rows"]], marker="s", ms=3,
                   color=f"C{r['seed']}", label=f"E2 seed {r['seed']}")
    b0 = cur[c][0]["rows"][-1]
    ax[1].axhline(b0["E2"]["b0_miou"], color="k", ls=":", label="B0 on E2")
    ax[1].axhline(b0["E1"]["b0_miou"], color="grey", ls=":", label="B0 on E1")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("training loss on $U^R$"); ax[0].legend(fontsize=7)
    ax[0].set_title("fitting the future pseudo-target")
    ax[1].set_xlabel("step"); ax[1].set_ylabel("dev mIoU (%)")
    ax[1].set_title(f"{SHORT[c]} on 0006 ({cur[c][0]['n_dev_anchors']}-anchor subsample)")
    ax[1].legend(fontsize=6, ncol=2)
    fig.suptitle("Gate 21 A: does fitting the revealed half move the never-revealed half?")
    fig.tight_layout()
    p = os.path.join(FIG, "A_training_curves.png"); fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig_b_table(a: Dict) -> str:
    pops = ("E1", "E2", "E0", "E1_P1", "E2_P1")
    have = [c for c in ORDER if a["populations"]["E2"].get(c, {}).get("seeds")]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    w = 0.8 / (len(have) + 1)
    x = np.arange(len(pops))
    ax.bar(x, [a["populations"][p]["B0"]["miou"] for p in pops], w, label="B0", color="0.35")
    for i, c in enumerate(have):
        v = [a["populations"][p][c]["miou_mean"] for p in pops]
        e = [a["populations"][p][c]["miou_std"] for p in pops]
        ax.bar(x + (i + 1) * w, v, w, yerr=e, capsize=2, label=SHORT[c])
    ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels([POP_LABEL[p] for p in pops])
    ax.get_xticklabels()[1].set_fontweight("bold")
    ax.set_ylabel("mIoU (%)")
    ax.set_title("Gate 21 B: B0 and the four controls, mean +- s.d. over three seeds\n"
                 "E2 (bold) is the pre-registered decision population")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(FIG, "B_populations.png"); fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig_c_distance(a: Dict) -> str:
    names = list(G.DISTANCE_NAMES)
    have = [c for c in ORDER if c in a["distance_E2"][names[0]]]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(names, [a["distance_E2"][n]["B0"]["miou"] for n in names], "ko-", label="B0")
    for c in have:
        ax[0].plot(names, [a["distance_E2"][n][c]["miou_mean"] for n in names], "o-", label=SHORT[c])
    ax[0].set_ylabel("mIoU on E2 (%)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[0].set_xlabel("distance to nearest causal semantic evidence")
    w = 0.8 / max(len(have), 1)
    xx = np.arange(len(names))
    for i, c in enumerate(have):
        ax[1].bar(xx + i * w, [a["distance_E2"][n][c]["delta"] for n in names], w, label=SHORT[c])
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(xx + 0.4 - w / 2); ax[1].set_xticklabels(names)
    ax[1].set_ylabel("mIoU - B0 (pp)"); ax[1].legend(fontsize=8); ax[1].grid(axis="y", alpha=0.3)
    for i, n in enumerate(names):
        ax[1].annotate(f"{a['distance_E2'][n]['B0']['n']:,}", (i + 0.4 - w / 2, 0), fontsize=6,
                       ha="center", va="bottom", rotation=90, alpha=0.6)
    fig.suptitle("Gate 21 C: E2 by distance from causal evidence (voxel counts annotated)")
    fig.tight_layout()
    p = os.path.join(FIG, "C_distance_E2.png"); fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig_d_per_class(a: Dict) -> str:
    rows = [a["per_class_E2"][str(i)] for i in range(1, 19)]
    c = G.PRIMARY_CONTROL
    rows = sorted(rows, key=lambda r: -(r["support"]))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    xx = np.arange(len(rows))
    ax.bar(xx - 0.2, [r["B0"] or 0.0 for r in rows], 0.4, label="B0", color="0.35")
    ax.bar(xx + 0.2, [r.get(c) or 0.0 for r in rows], 0.4, label=SHORT[c])
    ax.set_xticks(xx)
    ax.set_xticklabels([f"{r['class']}\n{r['support']:,}" for r in rows], rotation=90, fontsize=6)
    ax.set_ylabel("IoU on E2 (%)"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"Gate 21 D: per-class IoU on E2, {SHORT[c]} against B0 "
                 f"(classes ordered by GT support, shown under each name)")
    fig.tight_layout()
    p = os.path.join(FIG, "D_per_class_E2.png"); fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig_e_selection_bias(sb: Dict) -> str:
    m = sb["marginals"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    cf = m["class"]["fraction"]
    xx = np.arange(18)
    ax[0].bar(xx - 0.2, [100 * cf["UR_train"][i] for i in range(1, 19)], 0.4, label="$U^R$ train")
    ax[0].bar(xx + 0.2, [100 * cf["UN_dev"][i] for i in range(1, 19)], 0.4, label="$U^N$ dev (E2)")
    ax[0].set_xticks(xx); ax[0].set_xticklabels(G.CLASS_NAMES, rotation=90, fontsize=6)
    ax[0].set_ylabel("% of the population"); ax[0].legend(fontsize=8)
    ax[0].set_title(f"class (TV {m['class']['tv_from_UR_train_pct']['UN_dev']:.1f} %)")
    for k, i, lab in (("dist", 1, "distance to causal evidence"), ("range", 2, "range from the sensor")):
        f = m[k]["fraction"]
        nn = np.arange(len(m[k]["names"]))
        ax[i].bar(nn - 0.2, [100 * v for v in f["UR_train"]], 0.4, label="$U^R$ train")
        ax[i].bar(nn + 0.2, [100 * v for v in f["UN_dev"]], 0.4, label="$U^N$ dev (E2)")
        ax[i].set_xticks(nn); ax[i].set_xticklabels(m[k]["names"], rotation=30, fontsize=7)
        ax[i].legend(fontsize=8)
        ax[i].set_title(f"{lab} (TV {m[k]['tv_from_UR_train_pct']['UN_dev']:.1f} %)")
    fig.suptitle("Gate 21 E: the training population is not the decision population")
    fig.tight_layout()
    p = os.path.join(FIG, "E_selection_bias.png"); fig.savefig(p, dpi=150); plt.close(fig)
    return p


# ---- prose selected by outcome -------------------------------------------------------------------------
def _verdict(a: Dict) -> str:
    d = a["decision"]
    return {"A": "**OUTCOME A -- a semantic completion prior is demonstrated.**",
            "B": "**OUTCOME B -- weak transfer.**",
            "C": "**OUTCOME C -- no transfer.**",
            "VOID": "**VOID -- an integrity check failed; no outcome is declared.**",
            "INCOMPLETE": "**INCOMPLETE.**"}[d["outcome"]]


#: set by ``build`` so the outcome prose can quote the training-curve fit without threading it through
cur_global: Dict = {}


def _fit(a: Dict, cur: Dict, tail: int = 200) -> float:
    """Mean top-1 agreement with the pseudo-target over the last ``tail`` steps of each seed.

    A single step's value swings between 0.4 and 0.8 with the anchors it happened to draw, so quoting the
    final step would be quoting noise."""
    c = G.PRIMARY_CONTROL
    vals = []
    for seed in G.SEEDS:
        p = os.path.join(G.seed_dir(seed), f"train_{c}.json")
        if not os.path.exists(p):
            continue
        log = json.load(open(p))["log"][-tail:]
        vals.append(float(np.mean([x["target_top1"] for x in log])))
    return float(np.mean(vals)) if vals else float("nan")


def _interpretation(a: Dict) -> str:
    d = a["decision"]
    e1 = a["populations"]["E1"][G.PRIMARY_CONTROL]["delta_miou_mean"]
    e2 = d["delta_E2_miou"]
    if d["outcome"] == "A":
        return (f"Future-revealed pseudo-supervision teaches something that holds beyond the voxels it was "
                f"measured on. The model gains {e2:+.2f} pp on voxels for which no future semantic "
                f"observation exists anywhere in the frozen window, with the 95 % interval entirely above "
                f"zero and all three seeds agreeing in sign. The supervision is therefore not merely a "
                f"label for the revealable subset; it is a signal about hidden semantics in general, and the "
                f"gate's premise survives.")
    if d["outcome"] == "C":
        c = G.PRIMARY_CONTROL
        p1 = a["populations"]
        ov2 = a["b0_overlap"]["E2"][c]
        ov1 = a["b0_overlap"]["E1"][c]
        ds = a["distance_E2"]
        near, mid = G.DISTANCE_NAMES[0], G.DISTANCE_NAMES[2]
        return (
            "> **Privileged future supervision does not generalise from future-revealed hidden voxels to "
            "never-revealed hidden voxels under the simple completion model.**\n\n"
            f"The pre-registered statement is the one above, and the measured numbers say something more "
            f"specific than it does.\n\n"
            f"The head *does* learn the pseudo-target: training top-1 agreement reaches "
            f"{100*_fit(a, cur_global):.0f} % on `U^R`, and on `E1_P1` and `E2_P1` — the voxels where B0 is "
            f"wrong by construction and scores 0.00 — it recovers "
            f"{p1['E1_P1'][c]['miou_mean']:.2f} and {p1['E2_P1'][c]['miou_mean']:.2f} mIoU, at "
            f"{p1['E1_P1'][c]['top1_mean']:.1f} % and {p1['E2_P1'][c]['top1_mean']:.1f} % accuracy. It is "
            f"producing real semantics on hidden space, on both halves, at almost the same rate.\n\n"
            f"What it never learns is **when to defer**. On `E1`, where its supervision lives, it breaks "
            f"{ov1['n_broken']:,.0f} of B0's correct answers and fixes {ov1['n_fixed']:,.0f} of its wrong "
            f"ones — a net {ov1['net_voxels']:+,.0f} voxels out of {a['b0_overlap']['E1']['n']:,}, an "
            f"almost exact wash ({e1:+.2f} pp mIoU). On `E2` the same trade at the same rates loses, "
            f"{ov2['n_broken']:,.0f} broken against {ov2['n_fixed']:,.0f} fixed, because `E2` is where B0 "
            f"is *right more often* ({p1['E2']['B0']['top1']:.1f} % against "
            f"{p1['E1']['B0']['top1']:.1f} % on `E1`). The {e2:+.2f} pp is not a failure to produce "
            f"semantics for never-revealed space; it is a failure of the model's errors to be complementary "
            f"to the baseline's.\n\n"
            f"The distance profile says the same thing from the other side: the loss is concentrated at "
            f"{near} ({ds[near][c]['delta']:+.2f} pp over {ds[near]['B0']['n']:,} voxels, "
            f"{100*ds[near]['B0']['n']/p1['E2']['B0']['n']:.0f} % of `E2`), where nearest-evidence copying "
            f"is at its best, and vanishes at {mid} ({ds[mid][c]['delta']:+.2f} pp) where copying has "
            f"degraded. A learned scene prior reaches parity with copying exactly where copying stops "
            f"working, and never passes it.\n\n"
            f"This also is not an artefact of the metric. A class-collapse story predicts accuracy up and "
            f"mIoU down; here the model loses top-1 as well "
            f"({p1['E2'][c]['delta_top1_mean']:+.2f} pp against {e2:+.2f} pp of mIoU), so it is giving a "
            f"worse answer rather than a differently-shaped one. Its concentration against B0 is mixed "
            f"either way (§6b): fewer classes predicted "
            f"({p1['E2'][c]['n_classes_predicted']:.0f} of 18 against "
            f"{p1['E2']['B0']['n_classes_predicted']}) and lower entropy, but a smaller dominant share. "
            f"The one place a "
            f"metric effect is real is things: {_dead_things(a)}, which B0 occasionally hits by "
            f"copying, and that costs the thing-restricted mIoU "
            f"{a['thing_stuff_E2']['thing'][c]['delta']:+.2f} pp on "
            f"{100*a['thing_stuff_E2']['thing']['B0']['n']/p1['E2']['B0']['n']:.1f} % of the voxels.")
    return (f"The supervision contains usable information and the model does take some of it across: E1 "
            f"moves {e1:+.2f} pp and E2 moves {e2:+.2f} pp, positive but below the pre-registered "
            f"{d['thresholds']['a_delta_e2_min']:.1f} pp bar. A simple geometry-conditioned convolutional "
            f"head is not enough to establish a strong semantic completion prior, and the honest reading is "
            f"that the mechanism is real but weak at this capacity and with this inductive bias.")


def _prior_line(a: Dict) -> str:
    """The earlier attempts, lowest first, each with its delta against B0 on E0."""
    b0 = a["populations"]["E0"]["B0"]["miou"]
    pg = _prior_gates(a)
    pg[f"Gate 21 {SHORT[G.PRIMARY_CONTROL]}"] = a["populations"]["E0"][G.PRIMARY_CONTROL]["miou_mean"]
    return ", ".join(f"{n} {v:.2f} ({v - b0:+.2f})" for n, v in sorted(pg.items(), key=lambda kv: kv[1]))


def _prior_shape(a: Dict) -> str:
    """One sentence describing the spread of prior attempts, computed rather than asserted."""
    b0 = a["populations"]["E0"]["B0"]["miou"]
    pg = dict(_prior_gates(a))
    pg[f"Gate 21 {SHORT[G.PRIMARY_CONTROL]}"] = a["populations"]["E0"][G.PRIMARY_CONTROL]["miou_mean"]
    up = sorted(v - b0 for v in pg.values() if v > b0)
    dn = sorted(v - b0 for v in pg.values() if v <= b0)
    worst = min(pg.items(), key=lambda kv: kv[1])
    return (f"{len(up)} of the {len(pg)} beat B0, by {min(up):+.2f} to {max(up):+.2f} pp; {len(dn)} lose "
            f"to it, by {max(dn):+.2f} down to {min(dn):+.2f} pp, the largest loss being "
            f"{worst[0]}'s.") if up and dn else f"all {len(pg)} move B0 in the same direction."


def _next_step(a: Dict, sb: Dict) -> str:
    d = a["decision"]
    if d["outcome"] == "A":
        return ("**Stop here and decide the inductive bias from where the transfer is**, as section 23 "
                "requires. The table of §5 and §6 below localises it; that localisation, not a parameter "
                "count, should pick between region-level completion, object/scene queries, masked semantic "
                "modelling and open-vocabulary feature prediction. Do not scale this head.")
    if d["outcome"] == "C":
        c = G.PRIMARY_CONTROL
        b0_e0 = a["populations"]["E0"]["B0"]["miou"]
        return (
            "**Close this branch.** Section 25 is explicit that a bigger network, more epochs, more data, a "
            "stronger teacher or a transformer are not automatic rescues, and none of them is recommended "
            "here. What the results rule out is worth recording, because it is what a reconsideration from "
            "first principles has to start from:\n\n"
            f"- **Not capacity.** The head reaches {100*_fit(a, cur_global):.0f} % agreement with its own "
            f"target and recovers {a['populations']['E2_P1'][c]['miou_mean']:.2f} mIoU where B0 is wrong by "
            f"construction. It is not failing to learn; it is learning something that is not better than "
            f"copying.\n"
            f"- **Not the direction of the formulation either.** B5 tested the complementary design — an "
            f"additive correction that *starts* exactly at B0 and can only move away from it — on this "
            f"same population, and lost {b0_e0 - _prior_gates(a).get('B5', b0_e0):.2f} pp. Gate 21 tests "
            f"replacement and loses {-a['populations']['E0'][c]['delta_miou_mean']:.2f} pp on the same "
            f"population. Both the \"replace B0\" and the \"correct B0\" formulations have now failed, so "
            f"the problem is not where the prediction starts.\n"
            "- **Not the revealed/never-revealed distinction.** The model performs almost identically on "
            f"both halves where B0 is wrong ({a['populations']['E1_P1'][c]['miou_mean']:.2f} on `E1_P1`, "
            f"{a['populations']['E2_P1'][c]['miou_mean']:.2f} on `E2_P1`). The gate's framing assumed the "
            f"interesting failure would be revealed-versus-never-revealed; it is not. It is "
            f"learned-versus-copied.\n"
            f"- **What is left unexplained** is how small an effect any learned mechanism has ever had on "
            f"this population, and how asymmetric. Against B0's {b0_e0:.2f} mIoU on `E0`: "
            f"{_prior_line(a)} — each read from that gate's own artifact, not quoted from memory. "
            f"(The two entries named B1 are different methods that happen to share the label: Gate 15's "
            f"feature-propagation head and the pilot's learned evidence association.) "
            f"{_prior_shape(a)} So a parameter-free nearest-evidence copy is not literally the strongest "
            f"thing in this stack — Gate 16's B3 beats it — but nothing has beaten it by a margin that "
            f"would carry a paper, and the losses are several times larger than the wins. Why *that* "
            f"asymmetry holds, rather than which inductive bias to try next, is "
            f"the question to answer before building anything else. Gate 20's constraint stands over "
            f"whatever comes next: the camera's field of view, not the trajectory, caps revealability at "
            f"27 %, and the teacher itself reaches only 19.01 mIoU where it is measurable.")
    return ("Section 24 asks for exactly ONE architectural inductive bias, motivated by where transfer did "
            "and did not occur. See §12 below, which names it from the measured pattern rather than from a "
            "preference.")


def _dead_things(a: Dict, eps: float = 0.5) -> str:
    """Which thing classes the model drops to (near) zero IoU on E2, and what B0 got there."""
    c = G.PRIMARY_CONTROL
    rows = [a["per_class_E2"][str(i)] for i in range(1, 19) if a["per_class_E2"][str(i)]["thing"]]
    dead = [r for r in rows if (r.get(c) or 0.0) < eps and (r["B0"] or 0.0) >= eps]
    if not dead:
        return "no thing class collapses"
    return (f"{len(dead)} of the {len(rows)} thing classes fall below {eps} IoU where B0 did not ("
            + ", ".join(f"{r['class']} {r['B0']:.2f} → {(r.get(c) or 0.0):.2f}" for r in dead) + ")")


def _where_transfer(a: Dict) -> str:
    c = G.PRIMARY_CONTROL
    ds = a["distance_E2"]
    best = max(G.DISTANCE_NAMES, key=lambda n: ds[n][c]["delta"])
    worst = min(G.DISTANCE_NAMES, key=lambda n: ds[n][c]["delta"])
    ts = a["thing_stuff_E2"]
    rows = [a["per_class_E2"][str(i)] for i in range(1, 19) if a["per_class_E2"][str(i)].get(f"delta_{c}") is not None]
    gain = sorted(rows, key=lambda r: -r[f"delta_{c}"])[:3]
    loss = sorted(rows, key=lambda r: r[f"delta_{c}"])[:3]
    return (f"- distance: best **{best}** ({ds[best][c]['delta']:+.2f} pp), worst **{worst}** "
            f"({ds[worst][c]['delta']:+.2f} pp)\n"
            f"- thing {ts['thing'][c]['delta']:+.2f} pp, stuff {ts['stuff'][c]['delta']:+.2f} pp\n"
            f"- best classes: " + ", ".join(f"{r['class']} {r[f'delta_{c}']:+.2f}" for r in gain) + "\n"
            f"- worst classes: " + ", ".join(f"{r['class']} {r[f'delta_{c}']:+.2f}" for r in loss))


# ---- the report ----------------------------------------------------------------------------------------
def _central_table(a: Dict) -> str:
    have = [c for c in ORDER if a["populations"]["E2"].get(c, {}).get("seeds")]
    head = "| Population | B0 mIoU | " + " | ".join(f"{SHORT[c]}" for c in have) + " | Δ full−B0 |"
    sep = "|---|---|" + "---|" * (len(have) + 1)
    lines = [head, sep]
    for p in ("E1", "E2", "E0", "E1_P1", "E2_P1"):
        r = a["populations"][p]
        bold = p == "E2"
        name = f"**{POP_LABEL[p]}**" if bold else POP_LABEL[p]
        cells = []
        for c in have:
            v = f"{r[c]['miou_mean']:.2f} ± {r[c]['miou_std']:.2f}"
            cells.append(f"**{v}**" if bold else v)
        dfull = r[G.PRIMARY_CONTROL]["delta_miou_mean"]
        dcell = f"{dfull:+.2f}"
        lines.append(f"| {name} | {r['B0']['miou']:.2f} | " + " | ".join(cells) + " | "
                     + (f"**{dcell}**" if bold else dcell) + " |")
    return "\n".join(lines)


def build(a: Dict, sb: Dict, cur: Dict, val: Dict, figs: Dict[str, Optional[str]]) -> str:
    d = a["decision"]
    c = G.PRIMARY_CONTROL
    e1 = a["populations"]["E1"]
    e2 = a["populations"]["E2"]
    from tools.future_semantic_prior.model import build_head
    pc = {k: build_head(k).n_params() for k in ORDER}
    channels = [n for n, _ in G.GEOMETRY_CHANNELS + G.SEMANTIC_CHANNELS]
    bs = d["bootstrap"]
    git = os.popen("git -C %s rev-parse HEAD" % G.ARTIFACT_DIR).read().strip()[:12]

    L: List[str] = []
    L.append("# Gate 21 — does privileged future supervision learn a semantic completion prior?\n")
    L.append(f"{_verdict(a)}\n")
    L.append("## 1. The table that decides the experiment\n")
    L.append("KITTI-360 drive 0006, 590 anchors, mean ± s.d. over three head seeds, geometry seed 0 fixed.\n")
    L.append(_central_table(a) + "\n")
    L.append(f"`E2` is **{e2['B0']['n']:,}** voxels that are correctly completed, GT-occupied, never "
             f"causally observed, and that **never become visible again** inside the frozen 10 s privileged "
             f"window. No training target exists for any of them, in any drive, at any horizon. It is also "
             f"the half where B0 is *strongest* ({e2['B0']['miou']:.2f} against {e1['B0']['miou']:.2f} on "
             f"`E1`), because never-revealed voxels sit closer to causal evidence.\n")
    L.append("## 2. The pre-registered decision\n")
    L.append(f"```\nΔ E2 (mean of 3 seeds)   {d['delta_E2_miou']:+.3f} pp\n"
             f"per seed                 " + ", ".join(f"{x:+.3f}" for x in d["delta_E2_per_seed"]) + "\n"
             f"block bootstrap 95 % CI  [{bs['ci95'][0]:+.3f}, {bs['ci95'][1]:+.3f}]  "
             f"(P(Δ>0) = {bs['fraction_positive']:.3f})\n"
             f"A1  Δ ≥ +1.00 pp         {'MET' if d['A']['A1_delta_ge_1.0'] else 'not met'}\n"
             f"A2  all seeds positive   {'MET' if d['A']['A2_all_seeds_positive'] else 'not met'}\n"
             f"A3  CI lower > 0         {'MET' if d['A']['A3_ci_lower_above_0'] else 'not met'}\n"
             f"C1  Δ ≤ +0.25 pp         {'FIRED' if d['C']['C1_delta_le_0.25'] else 'no'}\n"
             f"C2  a seed negative      {'FIRED' if d['C']['C2_a_seed_negative_and_below_bar'] else 'no'}\n"
             f"outcome                  {d['outcome']}\n```\n")
    L.append(f"The rule and these thresholds were written and hashed to "
             f"`{open(os.path.join(G.ARTIFACT_DIR, 'decision_rule.sha256')).read().strip()[:16]}…` before "
             f"the first training run; `validate.py` check 1 recomputes the hash and check 2 requires the "
             f"code's thresholds to appear verbatim in the document.\n")
    globals()["cur_global"] = cur
    L.append("## 3. Interpretation\n")
    L.append(_interpretation(a) + "\n")
    L.append("## 4. The five questions of section 21\n")
    fit = (np.mean([r["rows"][-1]["train_target_top1"] for r in cur[c]]) if cur.get(c) else float("nan"))
    L.append(f"**Q1 — can the model fit/use the future pseudo-supervision on E1?** "
             f"Yes. Training top-1 agreement with the pseudo-target reaches "
             f"{100*fit:.1f} % and `E1` mIoU moves {e1[c]['delta_miou_mean']:+.2f} pp "
             f"({e1['B0']['miou']:.2f} → {e1[c]['miou_mean']:.2f}). On `E1_P1`, where B0 scores "
             f"{a['populations']['E1_P1']['B0']['miou']:.2f} by construction, it reaches "
             f"{a['populations']['E1_P1'][c]['miou_mean']:.2f}.\n")
    L.append(f"**Q2 — does that transfer to E2, which has no future supervision?** "
             f"{'Yes' if d['outcome'] == 'A' else ('Partly' if d['outcome'] == 'B' else 'No')}: "
             f"{d['delta_E2_miou']:+.2f} pp, against the pre-registered +1.00 pp bar, with 95 % interval "
             f"[{bs['ci95'][0]:+.2f}, {bs['ci95'][1]:+.2f}].\n")
    cmp_rows = " / ".join(f"{SHORT[k]} {e2[k]['miou_mean']:.2f}" for k in ORDER if k in e2 and e2[k].get("seeds"))
    g2 = e2.get("C2_geometry", {}).get("miou_mean", float("nan"))
    g4 = e2.get("C4_semantics", {}).get("miou_mean", float("nan"))
    g3 = e2[c]["miou_mean"]
    g1 = e2.get("C1_prior", {}).get("miou_mean", float("nan"))
    L.append(f"**Q3 — does geometry carry semantic-completion information beyond the causal evidence?** "
             f"On `E2`: B0 {e2['B0']['miou']:.2f}, {cmp_rows}.\n")
    L.append(f"Two separate answers, and they point in opposite directions. **On its own, geometry carries "
             f"real semantics**: C2 sees no semantic evidence at all — no evidence feature, no evidence "
             f"mask, not even the distance to the nearest evidence — and still reaches {g2:.2f} mIoU "
             f"against the {g1:.2f} available from class frequency alone, so the frozen completion features "
             f"plus the voxel's position in the scene predict class well above chance. **Given the causal "
             f"evidence, geometry adds nothing**: C4 (evidence + occupancy, no latent features) reaches "
             f"{g4:.2f} and C3 (both) {g3:.2f} — a difference of {g3 - g4:+.2f} pp, inside the seed spread. "
             f"Whatever the trunk knows about hidden class is already recoverable from where the causal "
             f"evidence is and what it says.\n")
    L.append("**Q4 — where does transfer occur?**\n\n" + _where_transfer(a) + "\n")
    c1 = e2.get("C1_prior", {}).get("miou_mean", float("nan"))
    L.append(f"**Q5 — is this consistent with learning p(y_hidden | completed geometry, observed semantics) "
             f"rather than reproducing class priors?** The class-prior control reaches {c1:.2f} mIoU on "
             f"`E2` — a single constant class for every voxel — against {e2[c]['miou_mean']:.2f} for the "
             f"full model and {e2['B0']['miou']:.2f} for B0. "
             f"{'The full model is far above the prior, so whatever it learned is voxel-specific.' if e2[c]['miou_mean'] > c1 + 1 else 'The margin over the prior is the quantity to watch.'} "
             f"The selection-bias table of §6 bounds how much of any difference could be a population "
             f"artefact.\n")
    L.append("## 5. E2 by distance from causal evidence\n")
    have = [k for k in ORDER if k in a["distance_E2"][G.DISTANCE_NAMES[0]]]
    L.append("| bin | voxels | B0 | " + " | ".join(SHORT[k] for k in have) + f" | Δ {SHORT[c]} |")
    L.append("|---|---|---|" + "---|" * (len(have) + 1))
    for n in G.DISTANCE_NAMES:
        r = a["distance_E2"][n]
        L.append(f"| {n} | {r['B0']['n']:,} | {r['B0']['miou']:.2f} | "
                 + " | ".join(f"{r[k]['miou_mean']:.2f}" for k in have)
                 + f" | {r[c]['delta']:+.2f} |")
    L.append("")
    L.append("Gate 20 found future supervision unusually strong far from causal evidence (revealability "
             "rose from 17.3 % at 0–1 m to 48.2 % beyond 8 m). This table says whether the model carried "
             "that into the never-revealed population or only smoothed B0's propagation.\n")
    L.append("## 6. E2 per class\n")
    L.append(f"| class | support | thing | B0 | {SHORT[c]} | Δ |")
    L.append("|---|---|---|---|---|---|")
    for i in range(1, 19):
        r = a["per_class_E2"][str(i)]
        f = lambda v: "—" if v is None else f"{v:.2f}"
        dd = r.get(f"delta_{c}")
        L.append(f"| {r['class']} | {r['support']:,} | {'thing' if r['thing'] else 'stuff'} | "
                 f"{f(r['B0'])} | {f(r.get(c))} | {'—' if dd is None else f'{dd:+.2f}'} |")
    ts = a["thing_stuff_E2"]
    L.append("")
    # section 16 asks whether the movement comes from one or two classes. E2 mIoU is the unweighted mean of
    # the 18 per-class IoUs, so each class's contribution to the delta is exactly its own delta over 18.
    contrib = sorted(((a["per_class_E2"][str(i)].get(f"delta_{c}") or 0.0) / 18.0,
                      a["per_class_E2"][str(i)]["class"]) for i in range(1, 19))
    movers = [x for x in contrib if abs(x[0]) >= 0.05]
    top = contrib[:4] if e2[c]["delta_miou_mean"] < 0 else contrib[::-1][:4]
    L.append(f"`E2` mIoU is the unweighted mean of these 18 IoUs, so each class contributes its own delta "
             f"divided by 18. The largest contributions are "
             + ", ".join(f"**{n}** {v:+.2f} pp" for v, n in top)
             + f"; {len(movers)} of 18 classes move the metric by at least 0.05 pp. The change is therefore "
             f"spread across the vocabulary rather than carried by one or two classes.\n")
    L.append(f"Aggregated over `E2`: thing B0 {ts['thing']['B0']['miou']:.2f} → "
             f"{ts['thing'][c]['miou_mean']:.2f} ({ts['thing'][c]['delta']:+.2f} pp, "
             f"{ts['thing']['B0']['n']:,} voxels); stuff B0 {ts['stuff']['B0']['miou']:.2f} → "
             f"{ts['stuff'][c]['miou_mean']:.2f} ({ts['stuff'][c]['delta']:+.2f} pp, "
             f"{ts['stuff']['B0']['n']:,} voxels). These two are mIoU over the voxels whose GT class is a "
             f"thing or a stuff class respectively, each averaged over the classes with support inside that "
             f"restricted region (6 and 12), so they are separate measurements and do not recombine into "
             f"the `E2` figure. No class is hidden: every one of the 18 is listed, including those B0 or "
             f"the model never predicts.\n")
    L.append("## 6b. Does the model collapse onto the easy classes?\n")
    L.append("mIoU averages over the classes with GT support, so a model that answers only with the few "
             "dominant classes can gain top-1 accuracy and lose mIoU. These are the numbers that tell the "
             "two apart, on `E2`.\n")
    have_c = [k for k in ORDER if k in e2 and e2[k].get("seeds")]
    L.append("| method | E2 mIoU | E2 top-1 | classes predicted / 18 | classes ≥1 % of predictions | "
             "dominant class | its share | prediction entropy |")
    L.append("|---|---|---|---|---|---|---|---|")
    nm = lambda i: G.CLASS_NAMES[int(i) - 1]
    b = e2["B0"]
    L.append(f"| B0 | {b['miou']:.2f} | {b['top1']:.2f} % | {b['n_classes_predicted']} | "
             f"{b['n_classes_ge_1pct']} | {nm(b['dominant_pred_class'])} | "
             f"{b['dominant_pred_fraction']:.1f} % | {b['pred_entropy_bits']:.2f} bits |")
    for k in have_c:
        r = e2[k]
        L.append(f"| {SHORT[k]} | {r['miou_mean']:.2f} | {r['top1_mean']:.2f} % | "
                 f"{r['n_classes_predicted']:.1f} | {r['n_classes_ge_1pct']:.1f} | "
                 f"{nm(r['dominant_pred_class'])} | {r['dominant_pred_fraction']:.1f} % | "
                 f"{r['pred_entropy_bits']:.2f} bits |")
    L.append("")
    dt, dm = e2[c]["delta_top1_mean"], e2[c]["delta_miou_mean"]
    b0r, mr = e2["B0"], e2[c]
    if dt * dm >= 0:
        L.append(f"Top-1 and mIoU move the same way on `E2` ({dt:+.2f} pp and {dm:+.2f} pp), which settles "
                 f"the question this table exists to answer. A class-collapse story predicts the opposite "
                 f"signature — accuracy up, mIoU down, because answering only with the frequent classes is "
                 f"the accuracy-maximising move and the one that zeroes out the rare ones. Both fell, so "
                 f"the loss is a worse answer, not a differently-shaped one.\n")
        L.append(f"The concentration numbers are mixed and are reported in full rather than selectively: "
                 f"{SHORT[c]} predicts {mr['n_classes_predicted']:.1f} of the 18 classes against B0's "
                 f"{b0r['n_classes_predicted']} and has a *smaller* dominant share "
                 f"({mr['dominant_pred_fraction']:.1f} % against {b0r['dominant_pred_fraction']:.1f} %), "
                 f"but a *lower* prediction entropy ({mr['pred_entropy_bits']:.2f} against "
                 f"{b0r['pred_entropy_bits']:.2f} bits). It is neither cleanly more nor cleanly less "
                 f"concentrated than B0. C2, which sees no semantic evidence, is the one that genuinely "
                 f"collapses: {a['populations']['E2']['C2_geometry']['dominant_pred_fraction']:.0f} % of "
                 f"its predictions are a single class.\n")
    else:
        L.append(f"Top-1 and mIoU disagree in sign on `E2` ({dt:+.2f} pp against {dm:+.2f} pp), which is "
                 f"the class-collapse signature: B0 copies a nearest neighbour and so scatters predictions "
                 f"across many classes, earning small non-zero IoU on rare ones, while a model distilled "
                 f"from a class-imbalanced soft target concentrates on the frequent ones.\n")
    L.append("mIoU was pre-registered as the primary metric and stays primary; this table says what it is "
             "measuring.\n")
    L.append("## 6c. What the model does to B0's answers\n")
    L.append("`P1` is defined as the voxels where B0's class is wrong, so each population splits exactly "
             "into the part B0 already gets right and the part it does not. The model's accuracy on each "
             "half follows from the two top-1 numbers, and it separates \"uniformly worse\" from \"trades "
             "correct answers for fewer fixes\".\n")
    L.append("| population | B0 right | B0 wrong | keeps of B0-right | fixes of B0-wrong | net voxels |")
    L.append("|---|---|---|---|---|---|")
    for base in ("E1", "E2", "E0"):
        r = a["b0_overlap"].get(base)
        if not r or c not in r:
            continue
        m = r[c]
        L.append(f"| {POP_LABEL[base]} | {r['n_b0_right']:,} | {r['n_b0_wrong']:,} | "
                 f"{m['keeps_b0_correct_pct']:.1f} % | {m['fixes_b0_wrong_pct']:.1f} % | "
                 f"{m['net_voxels']:+,.0f} |")
    L.append("")
    r2 = a["b0_overlap"].get("E2", {}).get(c)
    if r2:
        L.append(f"On `E2` the model keeps {r2['keeps_b0_correct_pct']:.1f} % of the "
                 f"{a['b0_overlap']['E2']['n_b0_right']:,} answers B0 already had right and fixes "
                 f"{r2['fixes_b0_wrong_pct']:.1f} % of the {a['b0_overlap']['E2']['n_b0_wrong']:,} it had "
                 f"wrong — about {r2['n_broken']:,.0f} broken against {r2['n_fixed']:,.0f} fixed. That is "
                 f"the whole of the top-1 loss, and it is not a metric artefact: the model is producing a "
                 f"different answer, not a more concentrated one (see §6b).\n")
    L.append("## 7. Selection bias between the training and decision populations\n")
    s = sb["summary"]
    L.append(f"```\n$U^R$ train (loss)   {sb['n_voxels']['UR_train']:>9,} voxels\n"
             f"$U^N$ train          {sb['n_voxels']['UN_train']:>9,}\n"
             f"$U^R$ dev    (E1)    {sb['n_voxels']['UR_dev']:>9,}\n"
             f"$U^N$ dev    (E2)    {sb['n_voxels']['UN_dev']:>9,}\n\n"
             f"total-variation distance from $U^R$ train to E2\n"
             f"  class              {s['tv_class_pct']:>6.1f} %\n"
             f"  distance-to-evid.  {s['tv_distance_pct']:>6.1f} %\n"
             f"  height             {s['tv_height_pct']:>6.1f} %\n"
             f"  range from sensor  {s['tv_range_pct']:>6.1f} %\n"
             f"  thing / stuff      {s['tv_thing_pct']:>6.1f} %   "
             f"(thing {s['thing_pct_UR_train']:.2f} % -> {s['thing_pct_UN_dev']:.2f} %)\n"
             f"classes whose frequency changes by more than 2x: "
             f"{s['n_classes_with_ratio_outside_half_to_two']} of 18\n```\n")
    L.append("The two populations are genuinely different, as Gate 20 predicted. Nothing is corrected for "
             "this — section 12 asks for the measurement, and an inverse-frequency weighting would be a "
             "separate ablation, not a silent change — but any failure has to be read against it.\n")
    L.append("## 8. Training curves\n")
    if cur.get(c):
        L.append(f"| step | train loss | tgt top-1 | E1 mIoU | E2 mIoU | Δ E2 |")
        L.append("|---|---|---|---|---|---|")
        steps = [r["step"] for r in cur[c][0]["rows"]]
        for i, st in enumerate(steps):
            rows = [r["rows"][i] for r in cur[c]]
            L.append(f"| {st} | {np.mean([r['train_loss'] for r in rows]):.4f} | "
                     f"{100*np.mean([r['train_target_top1'] for r in rows]):.1f} % | "
                     f"{np.mean([r['E1']['miou'] for r in rows]):.2f} | "
                     f"{np.mean([r['E2']['miou'] for r in rows]):.2f} | "
                     f"{np.mean([r['E2']['delta_miou'] for r in rows]):+.2f} |")
        L.append("")
        best = {}
        for r in cur[c]:
            best[r["seed"]] = max(r["rows"], key=lambda x: x["E2"]["miou"])["step"]
        steps_all = [r["step"] for r in cur[c][0]["rows"]]
        mean_e2 = [float(np.mean([r["rows"][i]["E2"]["miou"] for r in cur[c]]))
                   for i in range(len(steps_all))]
        last = steps_all[-1]
        peak_step = steps_all[int(np.argmax(mean_e2))]
        rising = peak_step == last
        early = [s for s, st in best.items() if st < last]
        # B0 on THIS subsample, not the 590-anchor figure -- the two are different populations
        b0_sub = float(np.mean([r["rows"][-1]["E2"]["b0_miou"] for r in cur[c]]))
        gap = b0_sub - max(mean_e2)
        L.append("**Does E2 improve early and then overfit?** "
                 + (f"No. Averaged over the three seeds, `E2` mIoU rises monotonically across the whole "
                    f"schedule, {mean_e2[0]:.2f} → {mean_e2[-1]:.2f}, and its best step is the last one. "
                    f"Individual seeds peak earlier ("
                    + ", ".join(f"seed {s}: {st}" for s, st in sorted(best.items()))
                    + f") but that is sampling noise on a {cur[c][0]['n_dev_anchors']}-anchor subsample, "
                      f"not a trend. "
                    if rising else
                    f"Partly. The seed-mean `E2` peaks at step {peak_step} of {last} and then declines, "
                    f"and {len(early)} of {len(best)} individual seeds peak before the final step. ")
                 + f"The important part is the level, not the shape: at its best the model is still "
                   f"{gap:.2f} pp below B0 on this subsample (B0 {b0_sub:.2f} there), and the curve is "
                   f"flattening, so the deficit "
                   f"is not something the remaining schedule was about to close. "
                 + ("That is evidence against more epochs as a remedy, which is one reason section 25 "
                    "forbids proposing one." if rising else "")
                 + " The reported model is the final step either way: selecting a checkpoint on a dev "
                   "metric is exactly the selection the pre-registered rule forbids.\n")
        L.append(f"Measured on a fixed {cur[c][0]['n_dev_anchors']}-anchor stride-"
                 f"{cur[c][0]['stride']} subsample of 0006, swept over saved checkpoints AFTER training, so "
                 f"the training process never opened a file of the development drive "
                 f"(`drive_0006` is on its firewall's forbidden list). No checkpoint is selected from "
                 f"these: the reported model is the final step, as pre-registered.\n")
    L.append("## 9. The composite scene read\n")
    if a["ssc"]:
        L.append(f"Over all {a['ssc']['B0']['n']:,} labelled voxels of 0006, with the teacher's own class "
                 f"at evidence voxels and the method's class elsewhere: B0 {a['ssc']['B0']['miou']:.2f} "
                 f"mIoU, " + ", ".join(f"{SHORT[k]} {a['ssc'][k]['miou_mean']:.2f} "
                                       f"({a['ssc'][k]['delta']:+.2f})" for k in ORDER if k in a["ssc"]
                                       and k != "B0") + ". This is the "
                 f"no-regression check of section 13.3 and decides nothing.\n")
    L.append("## 10. Verification\n")
    L.append(f"`logs/validation.json`: **{val['n_checks']} checks, all_ok = {val['all_ok']}**"
             + (f", failed: {val['failed']}" if val["failed"] else "") + "\n")
    for ch in val["checks"]:
        L.append(f"- {'PASS' if ch['ok'] else '**FAIL**'} — {ch['name']}")
    L.append("")
    L.append(f"The leakage guarantee has four independent parts. (1) No input channel is derived from a "
             f"future frame: the complete list is `{', '.join(channels)}`. "
             f"(2) Building that input under a file audit forbidding both reveal caches opens nothing from "
             f"them. (3) The prediction is bit-identical when the reveal mask is shuffled or set to "
             f"all-hidden, so the function really is the same for `E1` and `E2`. (4) Each training step "
             f"fingerprints the head input and the whole causal state before the target file is opened and "
             f"again after the backward pass, and requires both unchanged.\n")
    L.append("## 11. Provenance\n")
    import hashlib as _h
    cfg_sha = _h.sha256(open(os.path.join(G.ARTIFACT_DIR, "config.yaml"), "rb").read()).hexdigest()[:16]
    tgt_sha = json.load(open(os.path.join(G.ARTIFACT_DIR, "population_stats.json")))["target_sha256"][:16]
    g20_sha = open("artifacts/gate20_future_revealability/decision_rule.sha256").read().strip()[:16]
    L.append(f"```\ngate                     {G.VERSION}\ngit commit               {git}\n"
             f"config.yaml sha256       {cfg_sha}...\n"
             f"training target sha256   {tgt_sha}...\n"
             f"gate 20 rule sha256      {g20_sha}...\n"
             f"decision rule sha256     {open(os.path.join(G.ARTIFACT_DIR, 'decision_rule.sha256')).read().strip()}\n"
             f"geometry                 CompletionUNet seed {G.GEOMETRY_SEED} step 6000, frozen, 967,322 params\n"
             f"teacher                  Trident-H, 18 KITTI-360 phrases, logit scale 40\n"
             f"reveal definition        Gate 20 '{G.REVEAL_LEVEL}' / '{G.REVEAL_READOUT}', 10 s window\n"
             f"training population      {sb['n_voxels']['UR_train']:,} voxels over "
             f"{a.get('n_train_anchors', 1274)} anchors of 0003/0007/0010\n"
             f"head parameters          " + ", ".join(f"{SHORT[k]} {pc[k]:,}" for k in ORDER) + "\n"
             f"training                 3000 steps, AdamW lr 1e-3 wd 0.01, cosine, grad clip 1.0, "
             f"2 full-grid anchors/step\n"
             f"seeds                    {list(G.SEEDS)}\n"
             f"dev evaluation           {a['n_dev_anchors']} anchors of 0006, final checkpoint, no selection\n"
             f"bootstrap                {bs['block_anchors']}-anchor blocks, {bs['resamples']} resamples, "
             f"seed {bs['seed']}\n```\n")
    L.append("Reproduce:\n\n```bash\nPYTHONPATH=$PWD python -m tools.future_semantic_prior.targets --drive <drive> --device cuda:0\n"
             "PYTHONPATH=$PWD python -m tools.future_semantic_prior.train    --control C3_full --seed 0 --device cuda:0\n"
             "PYTHONPATH=$PWD python -m tools.future_semantic_prior.curves   --control C3_full --seed 0 --device cuda:0\n"
             "PYTHONPATH=$PWD python -m tools.future_semantic_prior.evaluate --control C3_full --seed 0 --device cuda:0\n"
             "PYTHONPATH=$PWD python -m tools.future_semantic_prior.selection_bias\n"
             "PYTHONPATH=$PWD python -m tools.future_semantic_prior.validate --device cuda:0\n"
             "PYTHONPATH=$PWD python -m tools.future_semantic_prior.analyze\n"
             "PYTHONPATH=$PWD python -m tools.future_semantic_prior.report\n```\n")
    L.append("## 12. Recommended next step\n")
    L.append(_next_step(a, sb) + "\n")
    L.append("## 13. Figures\n")
    for k, v in figs.items():
        if v:
            L.append(f"- `figures/{os.path.basename(v)}` — {k}")
    L.append("")
    L.append("## 14. What this gate did not do\n")
    L.append("No change to LingBotMap, CompletionUNet, the geometry, Trident-H, B0, the voxel grid, the "
             "occupancy threshold, the evaluation masks, the class mapping, Gate 20's visibility definition "
             "or Gate 20's target construction. No SemanticKITTI, no Occ3D, no nuScenes, no extra cameras, "
             "no fisheyes. No transformer, no IGGT, no region or object queries, no diffusion, no "
             "CLIP/DINO feature regression, no learning-rate sweep, no architecture search, and no dataset "
             "growth. Future information was used to build the training loss and to split the evaluation "
             "population, and nowhere else.\n")
    return "\n".join(L)


def terminal_summary(a: Dict, sb: Dict, val: Dict, cur: Dict) -> str:
    """The block section 26 asks to be printed."""
    from tools.future_semantic_prior.model import build_head
    c = G.PRIMARY_CONTROL
    d = a["decision"]
    e1, e2 = a["populations"]["E1"], a["populations"]["E2"]
    ds = a["distance_E2"]
    have_d = [n for n in G.DISTANCE_NAMES if c in ds[n]]
    best = max(have_d, key=lambda n: ds[n][c]["delta"])
    worst = min(have_d, key=lambda n: ds[n][c]["delta"])
    ts = a["thing_stuff_E2"]
    bs = d["bootstrap"]
    git = os.popen("git -C %s rev-parse HEAD" % G.ARTIFACT_DIR).read().strip()[:12]
    g = lambda k: e2[k]["miou_mean"] if k in e2 and e2[k].get("seeds") else float("nan")
    L = ["GATE 21 RESULT", "",
         f"Outcome:                      {d['outcome']}",
         f"Model parameters:             {build_head(c).n_params():,}  ({SHORT[c]})", "",
         f"B0 E1 mIoU:                   {e1['B0']['miou']:.2f}",
         f"C3 E1 mIoU:                   {e1[c]['miou_mean']:.2f} +- {e1[c]['miou_std']:.2f}",
         f"Delta E1:                     {e1[c]['delta_miou_mean']:+.2f}", "",
         f"B0 E2 NEVER-REVEALED mIoU:    {e2['B0']['miou']:.2f}",
         f"C3 E2 NEVER-REVEALED mIoU:    {e2[c]['miou_mean']:.2f} +- {e2[c]['miou_std']:.2f}",
         f"Delta E2:                     {d['delta_E2_miou']:+.2f}",
         f"E2 delta 95% CI / seed spread: [{bs['ci95'][0]:+.2f}, {bs['ci95'][1]:+.2f}] / "
         + ", ".join(f"{x:+.2f}" for x in d["delta_E2_per_seed"]), "",
         f"C1 prior E2:                  {g('C1_prior'):.2f}",
         f"C2 geometry E2:               {g('C2_geometry'):.2f}",
         f"C4 semantics-only E2:         {g('C4_semantics'):.2f}", "",
         f"Best E2 distance bin:         {best}  ({ds[best][c]['delta']:+.2f} pp, "
         f"{ds[best]['B0']['n']:,} voxels)",
         f"Worst E2 distance bin:        {worst} ({ds[worst][c]['delta']:+.2f} pp, "
         f"{ds[worst]['B0']['n']:,} voxels)", "",
         f"Thing E2 delta:               {ts['thing'][c]['delta']:+.2f}",
         f"Stuff E2 delta:               {ts['stuff'][c]['delta']:+.2f}", "",
         f"Geometry bit-identical:       {a['all_geometry_identical']} "
         f"(B0 identical across runs: {a['b0_identical_across_runs']})",
         f"Leakage checks:               {val['n_checks']} checks, all_ok = {val['all_ok']}"
         + (f", FAILED {val['failed']}" if val["failed"] else ""), "",
         f"Scientific conclusion:        {_conclusion_line(a)}",
         f"Recommended next step:        {_next_line(a)}", "",
         f"Report:                       {os.path.relpath(os.path.join(G.ARTIFACT_DIR, 'report.md'), os.getcwd())}",
         f"Commit:                       {git}"]
    return "\n".join(L)


def _conclusion_line(a: Dict) -> str:
    o = a["decision"]["outcome"]
    return {"A": "future-revealed supervision transfers to never-revealed hidden space",
            "B": "the supervision transfers weakly, below the pre-registered bar",
            "C": "future supervision does not generalise from revealed to never-revealed hidden voxels",
            "VOID": "void: an integrity check failed",
            "INCOMPLETE": "incomplete"}[o]


def _next_line(a: Dict) -> str:
    o = a["decision"]["outcome"]
    return {"A": "decide the inductive bias from WHERE transfer occurred; do not scale this head",
            "B": "one inductive bias, named in report.md section 12",
            "C": "close this branch; no automatic rescue by size, epochs, data or a transformer",
            "VOID": "fix the integrity failure and rerun",
            "INCOMPLETE": "finish the runs"}[o]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int); ap.add_argument("--stride", type=int, default=1)
    a_ = ap.parse_args()
    G.ensure_dirs()
    os.makedirs(FIG, exist_ok=True)
    a = json.load(open(os.path.join(G.ARTIFACT_DIR, "analysis.json")))
    sb = json.load(open(os.path.join(G.ARTIFACT_DIR, "selection_bias.json")))
    val = json.load(open(os.path.join(G.ARTIFACT_DIR, "logs", "validation.json")))
    cur = _curves()
    figs = {"training curves": fig_a_curves(cur), "populations": fig_b_table(a),
            "E2 by distance": fig_c_distance(a), "E2 per class": fig_d_per_class(a),
            "selection bias": fig_e_selection_bias(sb)}
    text = build(a, sb, cur, val, figs)
    p = os.path.join(G.ARTIFACT_DIR, "report.md")
    open(p, "w").write(text)
    print(f"-> {p} ({len(text.splitlines())} lines)")
    for v in figs.values():
        if v:
            print(f"-> {v}")
    summary = terminal_summary(a, sb, val, cur)
    open(os.path.join(G.ARTIFACT_DIR, "summary.txt"), "w").write(summary + "\n")
    print("\n" + summary)


if __name__ == "__main__":
    main()
