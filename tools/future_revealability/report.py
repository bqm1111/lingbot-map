"""Gate 20 figures and report.

    PYTHONPATH=$PWD python -m tools.future_revealability.report
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from typing import Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tools.future_revealability import config as G

NH = len(G.HORIZONS_S)
FIG = os.path.join(G.ARTIFACT_DIR, "figures")
LEVEL_STYLE = {"geometric": ("#9aa0a6", "-", "any future LiDAR return (ceiling)"),
               "in_frustum": ("#f9ab00", "-", "... and inside a future image (no occlusion test)"),
               "visible": ("#1a73e8", "-", "camera-visible + Trident-H token (DECISION)"),
               "deployable": ("#d93025", "-", "+ frozen label-free target selector")}


def _ax(title: str, xl: str, yl: str, ax=None):
    ax = ax or plt.gca()
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xl, fontsize=9); ax.set_ylabel(yl, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


def plot_a(A: Dict) -> str:
    t = A["train"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, pop, name in ((axes[0], "P0", "P0 — correctly completed, never observed"),
                          (axes[1], "P1", "P1 — P0 that B0 currently gets wrong")):
        for lv in G.DIAG_LEVELS:
            c, ls, lab = LEVEL_STYLE[lv]
            y = t["curves"][f"{lv}|{pop}"]["micro"]
            ax.plot(G.HORIZONS_S, y, ls, color=c, marker="o", ms=4, lw=1.8, label=lab)
            lo = [ci[0] for ci in t["curves"][f"{lv}|{pop}"]["ci95"]]
            hi = [ci[1] for ci in t["curves"][f"{lv}|{pop}"]["ci95"]]
            ax.fill_between(G.HORIZONS_S, lo, hi, color=c, alpha=0.12, linewidth=0)
        _ax(f"{name}\n{t['n_pop'][pop]:,} voxels, training drives", "future horizon Δ (s)",
            "revealed by Δ (% of population)", ax)
        ax.set_ylim(0, 100); ax.set_xlim(0, 10.2)
    axes[0].legend(fontsize=7.5, loc="upper left", frameon=False)
    fig.suptitle("A — future horizon vs fraction of hidden completed geometry that becomes observable", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = os.path.join(FIG, "A_horizon_curve.png"); fig.savefig(p, dpi=170); plt.close(fig)
    return p


def plot_b(A: Dict) -> str:
    t = A["train"]
    x = np.arange(len(G.DISTANCE_NAMES)); w = 0.21
    fig, ax = plt.subplots(figsize=(8.2, 4.1))
    for i, lv in enumerate(G.DIAG_LEVELS):
        c, _, lab = LEVEL_STYLE[lv]
        y = [t["distance"][f"{lv}|P0"]["revealability"][b][NH - 1] for b in range(len(G.DISTANCE_NAMES))]
        ax.bar(x + (i - 1.5) * w, y, w, color=c, label=lab)
    for b, n in enumerate(t["distance"]["visible|P0"]["n_target"]):
        ax.text(x[b], 2, f"n={n:,}", ha="center", fontsize=7, color="#444", rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(G.DISTANCE_NAMES)
    _ax("B — revealability by distance to the nearest causal semantic evidence (Δ = 10 s, P0)",
        "distance to nearest evidence", "revealed by 10 s (%)", ax)
    ax.set_ylim(0, 100); ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    p = os.path.join(FIG, "B_distance.png"); fig.savefig(p, dpi=170); plt.close(fig)
    return p


def plot_c(A: Dict) -> str:
    pc = A["train"]["per_class"]["P0"]
    names = sorted(pc, key=lambda n: -pc[n]["n_target"])          # sorted by target support (documented)
    y = [pc[n]["revealability"] or 0.0 for n in names]
    col = ["#d93025" if pc[n]["thing"] else "#1a73e8" for n in names]
    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    ax.bar(np.arange(len(names)), y, color=col)
    for i, n in enumerate(names):
        ax.text(i, (y[i] or 0) + 1.5, f"{pc[n]['n_target']:,}", ha="center", fontsize=6.5, rotation=90, color="#444")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ev = A["train"]["curves"]["visible|P0"]["micro"][NH - 1]
    ax.axhline(ev, color="#5f6368", ls="--", lw=1, label=f"all classes ({ev:.1f}%)")
    _ax("C — per-class future revealability of P0 (Δ = 10 s, decision level), sorted by target support",
        "", "revealed by 10 s (%)", ax)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, frameon=False,
              handles=[plt.Line2D([], [], color="#d93025", lw=6, label="thing"),
                       plt.Line2D([], [], color="#1a73e8", lw=6, label="stuff"),
                       plt.Line2D([], [], color="#5f6368", ls="--", lw=1, label=f"all classes ({ev:.1f}%)")])
    fig.tight_layout()
    p = os.path.join(FIG, "C_per_class.png"); fig.savefig(p, dpi=170); plt.close(fig)
    return p


def plot_d(A: Dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))
    for pop, c in (("P0", "#1a73e8"), ("P1", "#d93025")):
        rt = A["train"]["reveal_time"][pop]
        cdf = np.asarray(rt["cdf"])
        tot = cdf[-1] if cdf[-1] > 0 else 1.0
        edges = np.arange(0.05, 10.05, 0.05)
        axes[0].plot(edges, cdf, color=c, lw=1.8, label=f"{pop} (of all {pop})")
        axes[1].plot(edges, 100.0 * cdf / tot, color=c, lw=1.8, label=f"{pop} (of revealed {pop})")
    _ax("time to first reveal — share of the whole population", "Δ (s)", "revealed by Δ (%)", axes[0])
    _ax("time to first reveal — share of the EVENTUALLY revealed", "Δ (s)", "cumulative (%)", axes[1])
    for ax in axes:
        ax.legend(fontsize=8, frameon=False); ax.set_xlim(0, 10)
    axes[0].set_ylim(0, 100); axes[1].set_ylim(0, 102)
    fig.suptitle("D — when does a hidden completed voxel become observable again?", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = os.path.join(FIG, "D_reveal_time.png"); fig.savefig(p, dpi=170); plt.close(fig)
    return p


def plot_e(A: Dict) -> str:
    r = A["train"]["region"]
    fr = list(G.REGION_FRACTIONS)
    x = np.arange(len(fr)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for i, (key, c, lab) in enumerate((("", "#5f6368", "all regions"), ("_thing", "#d93025", "thing"),
                                       ("_stuff", "#1a73e8", "stuff"))):
        den = r[f"n_eligible{key}"] or 1
        y = [100.0 * r[f"ge_{int(f*100)}{key}"] / den for f in fr]
        ax.bar(x + (i - 1) * w, y, w, color=c, label=f"{lab} (n={r[f'n_eligible{key}']:,})")
    ax.set_xticks(x); ax.set_xticklabels([f"≥{int(f*100)}%" for f in fr])
    _ax("E — GT connected components: how much of their HIDDEN support gets revealed (Δ = 10 s)",
        "fraction of the component's hidden voxels revealed", "% of components", ax)
    ax.set_ylim(0, 100); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    p = os.path.join(FIG, "E_region_coverage.png"); fig.savefig(p, dpi=170); plt.close(fig)
    return p


def figures(A: Dict) -> List[str]:
    os.makedirs(FIG, exist_ok=True)
    return [plot_a(A), plot_b(A), plot_c(A), plot_d(A), plot_e(A)]


def _row(vals, fmt="{:.2f}") -> str:
    return " | ".join("—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt.format(v) for v in vals)


def write_report(A: Dict, commit: str) -> str:
    t, dv, D = A["train"], A["dev"], A["decision"]
    lv, ro = G.DECISION_LEVEL, G.DECISION_READOUT
    hs = G.HORIZONS_S
    L: List[str] = []
    w = L.append
    oc = D["outcome"]
    title = {"A": "OUTCOME A — future revealability supports the paper hypothesis",
             "B": "OUTCOME B — future revealability is useful but incomplete",
             "C": "OUTCOME C — future revealability is insufficient"}[oc]
    w(f"# {title}\n")
    p0, p1 = D["values"]["p0_eventual"], D["values"]["p1_eventual"]
    far = D["values"]["far_eventual"]
    q = t["future_target_quality"][ro]["eventual"]
    rt = t["reveal_time"]["P0"]
    w(f"Training drives 0003/0007/0010, {t['n_anchors']} anchors, **{t['n_pop']['P0']:,}** P0 voxels. "
      f"Of the geometry the frozen causal system correctly completes but has never observed, "
      f"**{p0:.1f}%** becomes genuinely observable again within the 10 s privileged window, "
      f"**{p1:.1f}%** of the subset B0 currently gets wrong, and **{far:.1f}%** of the population more than "
      f"4 m from any causal semantic evidence. The future teacher on the voxels it does reach scores "
      f"**{q['top1']:.2f}%** top-1 / **{q['miou']:.2f}** mIoU, against the frozen current-evidence teacher's "
      f"{G.GATE18_TEACHER['top1']}% / {G.GATE18_TEACHER['miou']}. Median time to first reveal "
      f"**{rt['median_s']:.2f} s**, 90th percentile **{rt['p90_s']:.2f} s**; "
      f"**{rt['never_revealed_frac']:.1f}%** is never revealed.\n")

    w("## 1. Headline\n")
    w("| | P0 | P1 | P0 that B0 already gets right |")
    w("|---|---:|---:|---:|")
    for lvl in G.DIAG_LEVELS:
        w(f"| {lvl} — revealed by 10 s | " + _row([t['curves'][f'{lvl}|{p}']['micro'][NH-1] for p in ('P0','P1','P0_correct')], "{:.2f}") + " |")
    w(f"| population (voxels) | {t['n_pop']['P0']:,} | {t['n_pop']['P1']:,} | {t['n_pop']['P0_correct']:,} |")
    w("")
    w(f"Decision level is **`{lv}`** and decision readout **`{ro}`** (`decision_rule.md` §2, §4), on the "
      f"micro aggregate over the training drives.\n")

    w("## 2. Pre-registered decision rule\n")
    w(f"Hashed before the run: `{open(os.path.join(G.ARTIFACT_DIR,'decision_rule.sha256')).read().split()[0]}`\n")
    w("| id | condition | value | met |")
    w("|---|---|---:|---|")
    T = D["thresholds"]
    rowsA = [("A1", f"P0 eventual ≥ {T['a_p0_eventual']:.0f}%", p0), ("A2", f"P1 eventual ≥ {T['a_p1_eventual']:.0f}%", p1),
             ("A3", f">4 m eventual ≥ {T['a_far_eventual']:.0f}%", far),
             ("A4", f"top-1 ≥ {T['a_teacher_top1']:.0f}% and mIoU ≥ {T['a_teacher_miou']:.1f}", None),
             ("A5", f"≥ {T['a_min_classes_20pct']}/18 classes ≥ 20%", float(D['values']['n_classes_ge_20pct']))]
    for k, cond, v in rowsA:
        vs = "—" if v is None else (f"{v:.0f}" if k == "A5" else f"{v:.2f}")
        if k == "A4":
            vs = f"{q['top1']:.2f} / {q['miou']:.2f}"
        w(f"| {k} | {cond} | {vs} | {'yes' if D['A'][k] else 'no'} |")
    rowsC = [("C1", f"P0 eventual < {T['c_p0_eventual']:.0f}%", f"{p0:.2f}"), ("C2", f"P1 eventual < {T['c_p1_eventual']:.0f}%", f"{p1:.2f}"),
             ("C3", f">4 m eventual < {T['c_far_eventual']:.0f}%", f"{far:.2f}"),
             ("C4", f"top-1 < {T['c_teacher_top1']:.0f}% or mIoU < {T['c_teacher_miou']:.1f}", f"{q['top1']:.2f} / {q['miou']:.2f}")]
    for k, cond, vs in rowsC:
        w(f"| {k} | {cond} | {vs} | {'**FIRES**' if D['C'][k] else 'no'} |")
    w("")
    w(f"**Outcome {oc}.** No threshold was moved after the results were seen.\n")

    w("## 3. Population\n")
    w("```")
    w("P0 = sem_gt in 1..18  &  frozen CompletionUNet occupied  &  no Trident-H evidence in t-4..t")
    w("P1 = P0 & (frozen B0 nearest-evidence class != sem_gt)")
    w("```")
    w("P0 is bit-identical to the frozen region `completed_never_observed` — the primary region of the "
      "pilot, Gate 16, Gate 17 and Gate 18 — asserted on every anchor.\n")
    w("| | anchors | P0 | P1 | P1 share of P0 | B0 mIoU on P0 |")
    w("|---|---:|---:|---:|---:|---:|")
    for name, s in (("training drives", t), ("0006 (held out)", dv)):
        w(f"| {name} | {s['n_anchors']} | {s['n_pop']['P0']:,} | {s['n_pop']['P1']:,} | "
          f"{100.0*s['n_pop']['P1']/s['n_pop']['P0']:.1f}% | {s['b0_on_P0']['miou']:.2f} |")
    w("")

    w("## 4. Horizon curve\n")
    w("Micro = total revealed / total target voxels; macro = mean of the per-anchor ratios. 95 % intervals "
      "are a 20-anchor block bootstrap, 2,000 resamples, seed 0.\n")
    for pop in ("P0", "P1"):
        w(f"**{pop}** (N_target = {t['n_pop'][pop]:,})\n")
        w("| Δ (s) | " + " | ".join(f"{lvl} micro" for lvl in G.DIAG_LEVELS) + " | visible macro | visible 95 % CI | visible N_revealed |")
        w("|---:|" + "---:|" * (len(G.DIAG_LEVELS) + 3))
        for h in range(NH):
            c = t["curves"]
            ci = c[f"{lv}|{pop}"]["ci95"][h]
            w(f"| {hs[h]:.0f} | " + _row([c[f'{lvl}|{pop}']['micro'][h] for lvl in G.DIAG_LEVELS], "{:.2f}")
              + f" | {c[f'{lv}|{pop}']['macro'][h]:.2f} | [{ci[0]:.2f}, {ci[1]:.2f}] | {c[f'{lv}|{pop}']['n_revealed'][h]:,} |")
        w("")
    w("![A](figures/A_horizon_curve.png)\n")
    w("### Which filter costs the coverage\n")
    c = t["curves"]
    g, fr, vi, dp = (c[f"{l}|P0"]["micro"][NH - 1] for l in ("geometric", "in_frustum", "visible", "deployable"))
    w("| level | revealed by 10 s | lost at this step |")
    w("|---|---:|---:|")
    w(f"| a future LiDAR sweep puts a return in the voxel | {g:.2f}% | — |")
    w(f"| ... and that return lands inside a future image | {fr:.2f}% | **−{g-fr:.2f} pp** (camera field of view) |")
    w(f"| ... and it is the nearest surface on its ray (z-buffer) | {vi:.2f}% | −{fr-vi:.2f} pp (occlusion) |")
    w(f"| ... and the label-free target selector keeps it | {dp:.2f}% | −{vi-dp:.2f} pp (occupancy/persistence rules) |")
    w("")
    w(f"The drive re-measures almost all of this geometry — **{g:.1f}%** of P0 gets a future LiDAR return. "
      f"The supervision is lost at the **camera**: {g-fr:.1f} percentage points disappear because the return "
      f"falls outside the 1408x376 perspective frustum, which is {fr-vi:.1f} pp more than occlusion costs. "
      f"This is a property of the sensor rig, not of the trajectory or of the scene.\n")

    w("## 5. P0 vs P1\n")
    w("| | P0 | P1 | P0 B0-correct |")
    w("|---|---:|---:|---:|")
    for lvl in G.DIAG_LEVELS:
        w(f"| {lvl} eventual | " + _row([t['curves'][f'{lvl}|{p}']['micro'][NH-1] for p in ('P0','P1','P0_correct')]) + " |")
    w(f"| median reveal time (s) | " + _row([t['reveal_time'][p]['median_s'] for p in ('P0','P1','P0_correct')]) + " |")
    w(f"| p90 reveal time (s) | " + _row([t['reveal_time'][p]['p90_s'] for p in ('P0','P1','P0_correct')]) + " |")
    w(f"| never revealed (%) | " + _row([t['reveal_time'][p]['never_revealed_frac'] for p in ('P0','P1','P0_correct')]) + " |")
    w("")

    w("## 6. Distance to causal semantic evidence\n")
    w("Same distance definition as Gate 18 and the pilot (Euclidean distance transform to the nearest "
      "evidence voxel centre).\n")
    w("| distance | P0 voxels | geometric | in frustum | **visible** | deployable | P1 visible | median reveal (s) | p90 (s) |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for b, nm in enumerate(G.DISTANCE_NAMES):
        d = t["distance"]; rtb = t["distance_reveal_time"][nm]
        w(f"| {nm} | {d['visible|P0']['n_target'][b]:,} | " +
          _row([d[f'{lvl}|P0']['revealability'][b][NH-1] for lvl in G.DIAG_LEVELS]) +
          f" | {d['visible|P1']['revealability'][b][NH-1]:.2f} | {rtb['median_s']:.2f} | {rtb['p90_s']:.2f} |")
    fci = t["distance"][f"{lv}|P0"]["far_ci95"]; vci = t["distance"][f"{lv}|P0"]["very_far_ci95"]
    w("")
    w(f"- pooled **> 4 m** (4–8 m and > 8 m, {t['distance'][f'{lv}|P0']['far_n_target']:,} voxels): "
      f"**{far:.2f}%** [{fci[0]:.2f}, {fci[1]:.2f}]")
    w(f"- pooled **> 8 m** ({t['distance'][f'{lv}|P0']['very_far_n_target']:,} voxels): "
      f"**{t['distance'][f'{lv}|P0']['very_far_micro'][NH-1]:.2f}%** [{vci[0]:.2f}, {vci[1]:.2f}]")
    w("")
    w("![B](figures/B_distance.png)\n")

    w("## 7. Class, thing and stuff\n")
    w("GT classes are used here for **auditing only**; no supervision in this gate is built from them.\n")
    w("| class | | P0 voxels | revealed | revealability | median reveal (s) | P1 revealability |")
    w("|---|---|---:|---:|---:|---:|---:|")
    pc, pc1 = t["per_class"]["P0"], t["per_class"]["P1"]
    for n in sorted(pc, key=lambda n: -pc[n]["n_target"]):
        r = pc[n]
        w(f"| {n} | {'thing' if r['thing'] else 'stuff'} | {r['n_target']:,} | {r['n_revealed']:,} | "
          + ("—" if r["revealability"] is None else f"{r['revealability']:.2f}")
          + f" | {r['median_reveal_s']:.2f} | "
          + ("—" if pc1[n]["revealability"] is None else f"{pc1[n]['revealability']:.2f}") + " |")
    ts = t["thing_stuff"]
    w("")
    w(f"- **thing** ({ts['thing']['n_target']:,} voxels): **{ts['thing']['revealability']:.2f}%**")
    w(f"- **stuff** ({ts['stuff']['n_target']:,} voxels): **{ts['stuff']['revealability']:.2f}%**")
    w(f"- classes with ≥ 20 % revealability: **{t['n_classes_ge_20pct']} of {t['n_classes_with_support']}** with GT support")
    w("")
    w("![C](figures/C_per_class.png)\n")

    w("## 8. Quality of the future semantic target\n")
    w("On revealed P0 voxels with a GT label, using only the views inside the horizon. mIoU is the frozen "
      "definition (classes 1..18 with GT support; class 0 excluded).\n")
    w("| readout | voxels | top-1 | mIoU | thing mIoU | stuff mIoU |")
    w("|---|---:|---:|---:|---:|---:|")
    for r in ("first", "latest", "mean_probability"):
        qq = t["future_target_quality"][r]
        w(f"| {r}{' **(decision)**' if r == ro else ''} | {qq['eventual']['n']:,} | {qq['eventual']['top1']:.2f} | "
          f"{qq['eventual']['miou']:.2f} | " + _row([qq['thing']['miou'], qq['stuff']['miou']]) + " |")
    g18 = G.GATE18_TEACHER
    w(f"| *Gate 18 current-evidence teacher* | — | *{g18['top1']}* | *{g18['miou']}* | *{g18['thing_miou']}* | *{g18['stuff_miou']}* |")
    w(f"| *frozen B0 on all of P0* | {t['b0_on_P0']['n']:,} | {t['b0_on_P0']['top1']:.2f} | {t['b0_on_P0']['miou']:.2f} | — | — |")
    w("")
    h2 = t["head_to_head_on_revealed"]
    w("**The comparison that matters: B0 and the future target on exactly the same voxels.**\n")
    w(f"| on the {h2['future_target']['n']:,} revealed P0 voxels | top-1 | mIoU | thing mIoU | stuff mIoU |")
    w("|---|---:|---:|---:|---:|")
    for k, nm in (("b0", "frozen B0 (what we produce today)"), ("future_target", "future pseudo-target")):
        x = h2[k]
        w(f"| {nm} | {x['top1']:.2f} | {x['miou']:.2f} | " + _row([x['thing_miou'], x['stuff_miou']]) + " |")
    w(f"| difference | +{h2['future_target']['top1']-h2['b0']['top1']:.2f} | "
      f"**+{h2['future_target']['miou']-h2['b0']['miou']:.2f}** | "
      f"+{h2['future_target']['thing_miou']-h2['b0']['thing_miou']:.2f} | "
      f"+{h2['future_target']['stuff_miou']-h2['b0']['stuff_miou']:.2f} |")
    w("")
    w(f"B0 scores **{h2['b0_on_unrevealed']['miou']:.2f}** mIoU on the part that is *never* revealed and only "
      f"**{h2['b0']['miou']:.2f}** on the revealed part — the supervision is concentrated exactly where the "
      f"current method is weakest.\n")
    dq = t["deployable_level_quality"]
    w("**Moving surfaces.** The `visible` level deliberately drops the frozen selector's static-persistence "
      "test, so a voxel can be \"revealed\" by a surface that moved into it after `t`. Restricting the same "
      "target to the voxels that selector keeps is the check:\n")
    w("| on the deployable-selected voxels | voxels | top-1 | mIoU | thing mIoU | stuff mIoU |")
    w("|---|---:|---:|---:|---:|---:|")
    for k, nm in (("b0", "frozen B0"), ("future_target", "future pseudo-target")):
        x = dq[k]
        w(f"| {nm} | {x['n']:,} | {x['top1']:.2f} | {x['miou']:.2f} | " + _row([x['thing_miou'], x['stuff_miou']]) + " |")
    w("")
    cl = t["ceilings_on_all_P0"]
    w("**What that is worth over all of P0**, with no learning at all:\n")
    w("| prediction on all of P0 | top-1 | mIoU | Δ mIoU vs B0 |")
    w("|---|---:|---:|---:|")
    base = cl["b0"]["miou"]
    for k, nm in (("b0", "frozen B0 everywhere"),
                  ("future_where_revealed_else_b0", "future target where revealed, B0 elsewhere"),
                  ("gt_where_revealed_else_b0", "GT where revealed, B0 elsewhere (perfect fit, no generalisation)")):
        x = cl[k]
        w(f"| {nm} | {x['top1']:.2f} | {x['miou']:.2f} | " + ("—" if k == "b0" else f"+{x['miou']-base:.2f}") + " |")
    w("")
    w("More future views of the same voxel (`mean_probability`, Δ = 10 s):\n")
    w("| future views | voxels | top-1 | mIoU |")
    w("|---|---:|---:|---:|")
    for k, v in t["quality_by_n_views"].items():
        w(f"| {k} | {v['n']:,} | " + _row([v['top1'], v['miou']]) + " |")
    w("")
    w(f"On **P1** only — the voxels B0 gets wrong — the future target scores "
      f"**{t['quality_on_P1']['top1']:.2f}%** top-1 / **{t['quality_on_P1']['miou']:.2f}** mIoU "
      f"({t['quality_on_P1']['n']:,} voxels).\n")
    w("Across horizons (decision readout):\n")
    w("| Δ (s) | " + " | ".join(f"{h:.0f}" for h in hs) + " |")
    w("|---|" + "---:|" * NH)
    bh = t["future_target_quality"][ro]["by_horizon"]
    w("| top-1 | " + _row([x["top1"] for x in bh]) + " |")
    w("| mIoU | " + _row([x["miou"] for x in bh]) + " |")
    w("| voxels | " + " | ".join(f"{x['n']:,}" for x in bh) + " |")
    w("")

    w("## 9. Region analysis\n")
    r = t["region"]
    w(f"GT connected components (per-class, 6-connectivity — Gate 17's definition, reused unchanged) with at "
      f"least {G.MIN_REGION_HIDDEN_VOX} hidden P0 voxels: **{r['n_eligible']:,}** "
      f"({r['n_eligible_thing']:,} thing, {r['n_eligible_stuff']:,} stuff).\n")
    w("| share of the component's hidden support revealed by 10 s | all | thing | stuff |")
    w("|---|---:|---:|---:|")
    for f in G.REGION_FRACTIONS:
        k = int(f * 100)
        w(f"| ≥ {k}% | {100.0*r[f'ge_{k}']/max(r['n_eligible'],1):.2f} | "
          f"{100.0*r[f'ge_{k}_thing']/max(r['n_eligible_thing'],1):.2f} | "
          f"{100.0*r[f'ge_{k}_stuff']/max(r['n_eligible_stuff'],1):.2f} |")
    w("")
    w("This is the weakest-assumption version of the region question: a component whose hidden support is "
      "mostly revealed could be supervised as an object even if individual voxels are missed. It is a "
      "connected-component analysis, not instance annotation — KITTI-360 instance labels are not available "
      "on this machine (see Gate 17's `plan.md`), so adjacent same-class objects merge, which makes these "
      "numbers an upper bound.\n")
    w("![E](figures/E_region_coverage.png)\n")

    w("## 10. Time to reveal\n")
    w("| | median (s) | p90 (s) | never revealed (%) |")
    w("|---|---:|---:|---:|")
    for p in ("P0", "P1", "P0_correct"):
        x = t["reveal_time"][p]
        w(f"| {p} | {x['median_s']:.2f} | {x['p90_s']:.2f} | {x['never_revealed_frac']:.2f} |")
    w("")
    w("![D](figures/D_reveal_time.png)\n")

    w("## 11. 0006, held out from every threshold\n")
    w("| | P0 | geometric | in frustum | **visible** | deployable | future top-1 | future mIoU |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, s in (("training drives", t), ("0006", dv)):
        qq = s["future_target_quality"][ro]["eventual"]
        w(f"| {name} | {s['n_pop']['P0']:,} | " + _row([s['curves'][f'{lvl}|P0']['micro'][NH-1] for lvl in G.DIAG_LEVELS])
          + f" | {qq['top1']:.2f} | {qq['miou']:.2f} |")
    w("")
    w("Per drive:\n")
    w("| drive | anchors | P0 | visible eventual | deployable eventual | future mIoU |")
    w("|---|---:|---:|---:|---:|---:|")
    for d, s in A["per_drive"].items():
        qq = s["future_target_quality"][ro]["eventual"]
        w(f"| {d[17:21]}{' (dev)' if d == G.DEV_DRIVE else ''} | {s['n_anchors']} | {s['n_pop']['P0']:,} | "
          f"{s['curves'][f'visible|P0']['micro'][NH-1]:.2f} | {s['curves'][f'deployable|P0']['micro'][NH-1]:.2f} | "
          + ("—" if qq['miou'] is None else f"{qq['miou']:.2f}") + " |")
    w("")

    w("## 12. Leakage and validation\n")
    v = json.load(open(os.path.join(G.ARTIFACT_DIR, "logs", "validation.json")))
    w("| # | check | result |")
    w("|---|---|---|")
    labels = {"1_zero_horizon_is_zero": "horizon 0 revealability is exactly zero, every anchor, every level",
              "2_P0_is_frozen_primary_region": "P0 count on 0006 equals the frozen `completed_never_observed` (2,743,212)",
              "2b_B0_reproduces_pilot_miou": "B0 on P0 reproduces the pilot's 8.80 mIoU on 0006",
              "3_P0_has_no_observed_voxel": "P0 ∩ evidence = ∅ (asserted per anchor)",
              "4_future_transforms_correct": "the `visible` level reproduces the frozen delayed-target cache's first-view times, view counts and readout",
              "5_no_feedback_into_causal_state": "the causal-state digest is unchanged after the future is opened",
              "6_counts_stable_across_reruns": "a rebuild is bit-identical",
              "7_coordinates_match_frozen_pipeline": "grid dims/origin/voxel size, floor binning round-trip, pose identity",
              "8_frozen_cache_counts_reproduced": "frozen `completed_later_revealed` size (559,021) and bit-exact 0006 GT masks",
              "levels_nested": "geometric ⊇ in_frustum ⊇ visible ⊇ deployable on every voxel"}
    for k, ok in v["checks"].items():
        w(f"| {k.split('_')[0]} | {labels.get(k, k)} | {'pass' if ok else '**FAIL**'} |")
    w("")
    w(f"All checks: **{'pass' if v['all_pass'] else 'FAIL'}**. Full record: `logs/validation.json`.\n")
    w("The future is opened strictly after the causal state is computed and digested, and the digest is "
      "re-checked afterwards; the causal state itself is computed inside a file audit that forbids LiDAR, "
      "GT poses, delayed targets and human labels. Future information can therefore only answer *when* a "
      "hidden voxel becomes observable and *what* it would then say.\n")

    w("## 13. Interpretation\n")
    w(INTERPRETATION[oc](A))
    w("\n## 14. Recommended next experiment\n")
    w(NEXT_STEP[oc](A))
    w("")
    w("---\n")
    w(f"Version `{G.VERSION}` · commit `{commit}` · `decision_rule.md` hashed before the run · "
      f"{t['n_anchors'] + dv['n_anchors']} anchors · no training, no model change.\n")
    p = os.path.join(G.ARTIFACT_DIR, "report.md")
    open(p, "w").write("\n".join(L))
    return p


def _facts(A: Dict) -> Dict:
    t = A["train"]
    c = t["curves"]
    f = {l: c[f"{l}|P0"]["micro"][NH - 1] for l in G.DIAG_LEVELS}
    f["p1"] = c["visible|P1"]["micro"][NH - 1]
    f["near"] = t["distance"]["visible|P0"]["revealability"][0][NH - 1]
    f["far"] = t["distance"]["visible|P0"]["far_micro"][NH - 1]
    f["vfar"] = t["distance"]["visible|P0"]["very_far_micro"][NH - 1]
    f["median"] = t["reveal_time"]["P0"]["median_s"]
    f["p90"] = t["reveal_time"]["P0"]["p90_s"]
    f["never"] = t["reveal_time"]["P0"]["never_revealed_frac"]
    f["thing"] = t["thing_stuff"]["thing"]["revealability"]
    f["stuff"] = t["thing_stuff"]["stuff"]["revealability"]
    f["ncls"] = t["n_classes_ge_20pct"]
    f["nsup"] = t["n_classes_with_support"]
    h2 = t["head_to_head_on_revealed"]
    f["b0_rev"] = h2["b0"]["miou"]; f["fut_rev"] = h2["future_target"]["miou"]
    f["b0_rev_top1"] = h2["b0"]["top1"]; f["fut_rev_top1"] = h2["future_target"]["top1"]
    f["b0_unrev"] = h2["b0_on_unrevealed"]["miou"]
    f["p1_top1"] = t["quality_on_P1"]["top1"]
    cl = t["ceilings_on_all_P0"]
    f["b0_all"] = cl["b0"]["miou"]
    f["hyb"] = cl["future_where_revealed_else_b0"]["miou"]
    f["ceil"] = cl["gt_where_revealed_else_b0"]["miou"]
    r = t["region"]
    f["reg50"] = r["frac_ge_50"]; f["reg75"] = r["frac_ge_75"]; f["reg25"] = r["frac_ge_25"]
    f["n_reg"] = r["n_eligible"]
    f["n_p0"] = t["n_pop"]["P0"]; f["n_p1"] = t["n_pop"]["P1"]
    f["n_rev"] = c["visible|P0"]["n_revealed"][NH - 1]
    f["dev_p0"] = A["dev"]["curves"]["visible|P0"]["micro"][NH - 1]
    f["fut_thing"] = t["future_target_quality"]["mean_probability"]["thing"]["miou"]
    f["b0_thing"] = h2["b0"]["thing_miou"]
    f["dep_thing"] = t["deployable_level_quality"]["future_target"]["thing_miou"]
    f["thing_share"] = 100.0 * t["thing_stuff"]["thing"]["n_target"] / t["n_pop"]["P0"]
    return f


def _interp_b(A: Dict) -> str:
    f = _facts(A)
    return f"""**The supervision exists, it is good, and it covers about a quarter of the problem.** Five findings, in
the order that matters.

**1. The drive re-measures this geometry; the camera does not see it.** {f['geometric']:.1f}% of P0 gets a
future LiDAR return — the trajectory really does carry the vehicle past almost all of the space it
completed blind. But only {f['in_frustum']:.1f}% of those returns land inside a future image, and
{f['visible']:.1f}% survive the occlusion test with a usable Trident-H token. The field of view costs
{f['geometric']-f['in_frustum']:.1f} percentage points — far more than occlusion's
{f['in_frustum']-f['visible']:.1f} pp. The ceiling on this idea is set by the sensor rig, not by the scene
or the route.

**2. Reveals are immediate or never.** Median time to first reveal is **{f['median']:.2f} s** — the very
next stream frame — and the 90th percentile is **{f['p90']:.2f} s**. Horizons beyond 2 s add almost
nothing, and **{f['never']:.1f}%** of P0 is never revealed inside the 10 s window. This is the mirror image
of Gate 18's finding that the past leaves the forward volume: the future is in front of you, but the window
in which a hidden voxel is simultaneously ahead of the vehicle and inside the frustum is about a second and
a half wide. Extending the horizon is not the lever; widening the field of view would be.

**3. The coverage is best exactly where the current method is worst.** Revealability *rises* with distance
from causal semantic evidence — {f['near']:.1f}% at 0–1 m to {f['vfar']:.1f}% beyond 8 m — and P1, the
subset B0 gets wrong, is better covered ({f['p1']:.1f}%) than P0 as a whole ({f['visible']:.1f}%). This is
the opposite of the failure mode the gate was designed to catch. Gate 18 showed propagation collapses with
distance; future observation is densest precisely there, because voxels far from any evidence are mostly
voxels *ahead* of the vehicle, beyond the reconstruction's depth range but inside the grid.

**4. No class collapse.** Thing {f['thing']:.1f}% against stuff {f['stuff']:.1f}%, and
**{f['ncls']} of {f['nsup']}** classes with GT support clear 20% revealability. The expected bias — plentiful
road, building and vegetation, no cars, poles or signs — does not appear.

**5. Where the target exists it is much better than what we produce today.** On the same
{f['n_rev']:,} revealed voxels the future pseudo-target scores **{f['fut_rev']:.2f} mIoU /
{f['fut_rev_top1']:.2f}% top-1** against B0's **{f['b0_rev']:.2f} / {f['b0_rev_top1']:.2f}%** — a gap of
**+{f['fut_rev']-f['b0_rev']:.2f} mIoU**. On revealed P1, where B0 is wrong by construction, the future
target is right {f['p1_top1']:.1f}% of the time. Substituting it wherever it exists, with no learning at
all, lifts mIoU over *all* of P0 from {f['b0_all']:.2f} to **{f['hyb']:.2f}** (+{f['hyb']-f['b0_all']:.2f});
a learner that fit this supervision perfectly and generalised nothing would reach {f['ceil']:.2f}
(+{f['ceil']-f['b0_all']:.2f}). For scale, every learned semantic mechanism tried so far moved the same
metric by less than half a point (B1 +0.33, B2 +0.41, B4 −0.58, B5 −2.04).

**One number that looks bad and is not.** The future target's thing mIoU is {f['fut_thing']:.2f}, well under
the current-evidence teacher's {G.GATE18_TEACHER['thing_miou']}. Two things rule out the obvious worry. The
static-persistence filter — the frozen selector's defence against a surface that moved into the voxel after
`t` — barely changes it ({f['dep_thing']:.2f}), so moving objects are not the explanation. And B0 on the same
voxels scores {f['b0_thing']:.2f}, so the future target is still several times better on objects than what we
produce today. The gap to the teacher is a population effect: Gate 18 measured the teacher on surfaces it was
looking straight at, while these are object voxels that were *hidden* at `t` — boundaries, far cars, partly
occluded shapes — and things are only {f['thing_share']:.1f}% of P0 to begin with.

**Why this is B and not A.** The pre-registered coverage bars were 50% of P0 and 40% of P1; the measured
{f['visible']:.1f}% and {f['p1']:.1f}% miss them. {f['never']:.1f}% of the population this mechanism is
supposed to solve receives no future view at all, so future observation cannot be the *sole* source of
semantic supervision — a model trained on it is supervised on a minority of its own target population, and
that minority is systematically the far, ahead-of-vehicle part rather than the occluded part. The decision
rule therefore licenses proceeding only if partial reveals can be turned into useful supervision, which is
what the region analysis speaks to: **{f['reg25']:.0f}%** of the {f['n_reg']:,} hidden GT components get at
least a quarter of their hidden support revealed and **{f['reg50']:.0f}%** get at least half, so
region-level consistency is available as an auxiliary signal rather than a replacement.

**One caveat stated plainly.** Future evidence here is lifted from raw LiDAR returns registered with GT
poses, while causal evidence is lifted from the model's own predicted geometry. The future side therefore
has better geometry than the deployed system ever would, which makes every number in this report an upper
bound on what a real pipeline could harvest. The direction of that bias is the safe one for a falsification
test — the hypothesis survived a test tilted in its favour, and the honest coverage figure is somewhere at
or below {f['visible']:.1f}%."""


def _next_b(A: Dict) -> str:
    f = _facts(A)
    return f"""**Train the smallest possible semantic-completion head on future-revealed pseudo-labels only, and test
whether it generalises to the never-revealed part of the same population.**

That last clause is the experiment. Coverage of {f['visible']:.1f}% is enough to train on and not enough to
deploy on, so the only question that decides the paper is whether supervision drawn from the revealed
{f['visible']:.1f}% transfers to the {f['never']:.1f}% that never becomes observable. Nothing measured in
this gate answers it, and no further audit can.

Concretely, and with no new architecture: reuse the frozen `CompletionUNet` trunk and a small semantic head
(the Gate 15 head is the obvious candidate), supervise it **only** on training-drive P0 voxels that this
gate marks as future-revealed, using the `mean_probability` target, and evaluate on 0006's P0 split into its
revealed and never-revealed halves — reporting B0 on each half as the control. Pre-register the
never-revealed half as the gate before training, because the revealed half is where the supervision lives
and a gain there proves only that the model can fit its own targets. The headroom is bounded above by the
{f['ceil']:.2f} mIoU figure in section 8 and below by {f['hyb']:.2f} (pure substitution, no learning), so a
result inside that band is a real mechanism and a result at or under B0's {f['b0_all']:.2f} on the
never-revealed half closes the branch.

Two things worth recording but **not** worth doing now. The field-of-view finding in section 4 implies a
large, cheap coverage lever — KITTI-360 carries a second perspective camera and two fisheyes, and the
{f['geometric']-f['in_frustum']:.1f} pp lost to the frustum is partly recoverable from them. That is data
engineering, not the paper's mechanism, and it should only be spent after the generalisation question is
answered. Likewise, the region-level signal ({f['reg50']:.0f}% of hidden components at least half revealed)
is a plausible second loss term, but adding it before the voxel-level question is settled would confound
the one test that matters."""


def _interp_a(A: Dict) -> str:
    f = _facts(A)
    return (f"Future observation reveals {f['visible']:.1f}% of P0 and {f['p1']:.1f}% of P1, including "
            f"{f['far']:.1f}% of the population more than 4 m from any causal evidence, and the resulting "
            f"pseudo-target scores {f['fut_rev']:.2f} mIoU against B0's {f['b0_rev']:.2f} on the same voxels. "
            f"Ordinary unlabelled driving sequences contain substantial free supervision specifically for "
            f"semantic completion.")


def _next_a(A: Dict) -> str:
    return ("Build the smallest possible semantic-completion learner on this supervision: the frozen "
            "`CompletionUNet` trunk plus a small semantic head, trained only on future-revealed voxels of the "
            "training drives, evaluated on 0006's P0 with B0 as the control.")


def _interp_c(A: Dict) -> str:
    f = _facts(A)
    return (f"Only {f['visible']:.1f}% of P0 and {f['p1']:.1f}% of P1 become semantically observable later, "
            f"and {f['far']:.1f}% of the population beyond 4 m from causal evidence. Future observation cannot "
            f"provide enough supervision for the proposed mechanism.")


def _next_c(A: Dict) -> str:
    return ("Stop this branch. The semantic prior has to come from somewhere other than later observation of "
            "the same drive -- the standing candidate is the teacher comparison on the frozen 0006 evidence "
            "voxels, which both Gate 18 and Gate 19 recommended and which needs no training.")


INTERPRETATION: Dict[str, object] = {"A": _interp_a, "B": _interp_b, "C": _interp_c}
NEXT_STEP: Dict[str, object] = {"A": _next_a, "B": _next_b, "C": _next_c}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--analysis", default=os.path.join(G.ARTIFACT_DIR, "analysis.json"))
    a = ap.parse_args()
    A = json.load(open(a.analysis))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    figs = figures(A)
    p = write_report(A, commit)
    t, D = A["train"], A["decision"]
    q = t["future_target_quality"][G.DECISION_READOUT]["eventual"]
    print("\n".join([
        f"Outcome:                 {D['outcome']}",
        f"P0 size:                 {t['n_pop']['P0']:,} voxels ({t['n_anchors']} anchors, training drives)",
        f"P1 size:                 {t['n_pop']['P1']:,} voxels",
        f"Eventual P0 revealability: {D['values']['p0_eventual']:.2f}%",
        f"Eventual P1 revealability: {D['values']['p1_eventual']:.2f}%",
        f">4m revealability:       {D['values']['far_eventual']:.2f}%",
        f">8m revealability:       {t['distance']['visible|P0']['very_far_micro'][NH-1]:.2f}%",
        f"Future teacher top-1:    {q['top1']:.2f}%",
        f"Future teacher mIoU:     {q['miou']:.2f}",
        f"Median reveal time:      {t['reveal_time']['P0']['median_s']:.2f} s",
        f"Report path:             {p}",
        f"Git commit:              {commit}",
        f"Figures:                 {', '.join(os.path.basename(f) for f in figs)}"]))


if __name__ == "__main__":
    main()
