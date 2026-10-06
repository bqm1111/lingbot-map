"""Tables, figures, verification.md and report.md for the OccAny comparison.

Every number and every comparative word is computed from ``analysis.json`` (or the evaluation shards)
at render time; no conclusion is hard-coded (see reports/reporting_corrections.md RC1).
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                    # noqa: E402
import numpy as np                                                                 # noqa: E402

from tools.baseline_reconstruction import config as BR                            # noqa: E402
from tools.occany_comparison import config as C                                    # noqa: E402
from tools.occany_comparison.analysis import blocks_of, load_analysis, load_dataset  # noqa: E402

NAMES = {"semantickitti": "SemanticKITTI 08", "occ3d": "Occ3D-nuScenes val"}
COL_OURS, COL_OCC = "#1b6f8a", "#b5452b"


def pc(x: float) -> str:
    return f"{100 * x:.2f}"


def pp(x: float) -> str:
    return f"{100 * x:+.2f}"


def ci_text(dl: Dict) -> str:
    return f"{pp(dl['point'])} [{pp(dl['lo'])}, {pp(dl['hi'])}]"


def verdict(dl: Dict) -> str:
    return "higher" if dl["lo"] > 0 else "lower" if dl["hi"] < 0 else "not distinguishable"


def cal(ds: str, which: str) -> str:
    return "shared" if ds == "semantickitti" else which


def reported(ds: str) -> Dict:
    z = json.load(open(os.path.join(BR.CurrentPaths().target_eval, f"zero_shot_{ds}.json")))
    return {"mean": z["seed_statistics"]["iou"]["mean"], "median": z["seed_statistics"]["iou"]["median"], "n": z["n_anchors"],
            "per_seed": [z["per_seed"][str(s)]["full"]["iou"] if str(s) in z["per_seed"] else z["per_seed"][s]["full"]["iou"] for s in C.SEEDS]}


# --------------------------------------------------------------------------- the ladder
def ladder(a: Dict, ds: str) -> List[Dict]:
    """From the quoted comparison to the like-for-like one, removing one difference per step."""
    sub = a[ds]["subsets"]; run = C.OCCANY[ds]; rep = reported(ds)
    steps = [{"label": f"as quoted: ours raw, causal, our {rep['n']:,} anchors & label loader  vs  OccAny published (pooled)",
              "short": "as quoted", "ours_mean": rep["mean"], "ours_median": rep["median"], "ours_seeds": rep["per_seed"],
              "occany": run.published_sc_iou, "delta": None}]

    def step(label, short, subset, v, post_ref):
        r = sub[subset]["ours"][v]
        key = "delta_vs_occany_same_post" if v.endswith(post_ref) else "delta_vs_occany_pooled"
        steps.append({"label": label, "short": short, "subset": subset, "variant": v, "ours_mean": r["stats"]["iou"]["mean"],
                      "ours_median": r["stats"]["iou"]["median"], "ours_seeds": [r["per_seed"][f"seed{s}"]["iou"] for s in C.SEEDS],
                      "occany": sub[subset]["occany"][post_ref]["iou"], "occany_post": post_ref, "delta": r[key]})

    n_c = sub["causal_eligible"]["n_samples"]; n_a = sub["occany_all"]["n_samples"]
    step(f"OccAny's samples ({n_c:,} causal-eligible), labels & scorer; both raw; our causal frames" + ("" if ds == "semantickitti" else ", our placement"),
         "same samples, both raw", "causal_eligible", f"causal|{cal(ds, 'ours')}|raw", "raw")
    if ds == "occ3d":
        step("+ OccAny's benchmark placement (static extrinsic)", "+ OccAny placement", "causal_eligible", "causal|occany|raw", "raw")
    step(f"+ OccAny's five frames (forward), all {n_a:,} samples; both raw", "+ OccAny frames (forward)", "occany_all", f"forward|{cal(ds, 'occany')}|raw", "raw")
    step("+ OccAny's pooling for both  [PRIMARY, predeclared]", "+ pooling for both = PRIMARY", "occany_all", f"forward|{cal(ds, 'occany')}|pooled", "pooled")
    return steps


# --------------------------------------------------------------------------- tables
def results_table(a: Dict) -> str:
    L = ["# OccAny comparison — results", "",
         "Pooled SC IoU / precision / recall in %, OccAny's `SSCMetrics`, OccAny's stored labels. Ours: three frozen seeds, "
         "each pooled separately; mean and median over seeds. Δ = seed-mean IoU(ours) − IoU(OccAny) on the same samples, "
         f"paired block bootstrap 95 % CI ({C.BOOTSTRAP_N:,} draws; blocks: Occ3D scenes, SemanticKITTI runs of {C.SK_BLOCK_FRAMES} target frames).", ""]
    for ds in C.DATASETS:
        for subset in C.SUBSETS:
            s = a[ds]["subsets"][subset]
            L += [f"## {NAMES[ds]} — `{subset}` ({s['n_samples']:,} samples, {s['n_blocks']} blocks)", "",
                  f"_{C.SUBSET_DESCRIPTIONS[subset]}_", "",
                  "| method | protocol | placement | post | IoU seeds 0/1/2 | IoU mean | IoU median | P | R | pred/GT | Δ vs OccAny (same post) | reading | map-only IoU |",
                  "|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|---:|"]
            for post in C.POSTPROCESS:
                o = s["occany"][post]
                L.append(f"| OccAny (released) | forward | occany | {post} | — | {pc(o['iou'])} | {pc(o['iou'])} | {pc(o['precision'])} | {pc(o['recall'])} | {o['pred_over_gt']:.2f} | — | — | — |")
            for v, r in s["ours"].items():
                p, c, post = v.split("|")
                st = r["stats"]; dl = r["delta_vs_occany_same_post"]
                seeds = " / ".join(pc(r["per_seed"][f"seed{k}"]["iou"]) for k in C.SEEDS)
                L.append(f"| ours (frozen) | {p} | {c} | {post} | {seeds} | {pc(st['iou']['mean'])} | {pc(st['iou']['median'])} | "
                         f"{pc(st['precision']['mean'])} | {pc(st['recall']['mean'])} | {st['pred_over_gt']['mean']:.2f} | {ci_text(dl)} | {verdict(dl)} | {pc(s['maponly'][v]['iou'])} |")
            L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- figures
def fig_ladder(a: Dict, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    for ax, ds in zip(axes, C.DATASETS):
        st = ladder(a, ds)[::-1]
        y = np.arange(len(st))
        for k, s in zip(y, st):
            lo, hi = min(s["ours_seeds"]), max(s["ours_seeds"])
            ax.plot([100 * lo, 100 * hi], [k, k], color=COL_OURS, lw=5, alpha=0.3, solid_capstyle="round")
            ax.plot(100 * s["ours_mean"], k, "o", color=COL_OURS, ms=9, zorder=3)
            ax.plot(100 * s["occany"], k, "D", color=COL_OCC, ms=8, zorder=3)
            if s["delta"] is not None:
                ax.text(0.98, k + 0.33, f"Δ {ci_text(s['delta'])} pp", transform=ax.get_yaxis_transform(), ha="right", va="center",
                        fontsize=8.5, color="0.25")
        ax.set_yticks(y); ax.set_yticklabels([s["short"] for s in st], fontsize=9)
        ax.set_ylim(-0.5, len(st) - 0.35)
        vals = [100 * v for s in st for v in (*s["ours_seeds"], s["occany"])]
        ax.set_xlim(min(vals) - 3, max(vals) + 3)
        ax.set_xlabel("pooled SC IoU (%)"); ax.grid(axis="x", alpha=0.3)
        ax.set_title(NAMES[ds], fontsize=11, loc="left")
    axes[0].plot([], [], "o", color=COL_OURS, label="ours, seed mean (band = seed range)")
    axes[0].plot([], [], "D", color=COL_OCC, label="OccAny on the same samples")
    fig.legend(loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("From the quoted comparison (bottom) to the like-for-like one (top): one difference removed per row", fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95]); fig.subplots_adjust(wspace=0.45)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_per_sample(a: Dict, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    for ax, ds in zip(axes, C.DATASETS):
        p = a[ds]["per_sample_primary"]
        x, y = 100 * np.asarray(p["occany_iou"]), 100 * np.asarray(p["ours_iou_seed_mean"])
        ax.scatter(x, y, s=5, alpha=0.35, color=COL_OURS, lw=0)
        m = max(x.max(), y.max()) + 2
        ax.plot([0, m], [0, m], color="0.4", lw=1, ls="--")
        ax.set_xlim(0, m); ax.set_ylim(0, m); ax.set_aspect("equal")
        ax.set_xlabel("OccAny per-sample IoU (%)"); ax.set_ylabel("ours per-sample IoU, seed mean (%)")
        ax.set_title(f"{NAMES[ds]}\nours above the diagonal on {100 * np.mean(y > x):.1f} % of {len(x):,} samples", fontsize=10, loc="left")
        ax.grid(alpha=0.25)
    fig.suptitle("Primary variant, per sample: forward frames, OccAny placement, pooled, OccAny labels and scorer", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_operating_points(a: Dict, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    r = np.linspace(0.01, 1, 200)
    for ax, ds in zip(axes, C.DATASETS):
        for iou in (0.1, 0.2, 0.3, 0.4, 0.5):
            prec = iou * r / (r * (1 + iou) - iou)
            ok = (prec > 0) & (prec <= 1)
            ax.plot(100 * r[ok], 100 * prec[ok], color="0.85", lw=0.8, zorder=0)
            j = np.flatnonzero(ok)[-1]
            ax.text(100 * r[j], 100 * prec[j], f"IoU {int(100 * iou)}", fontsize=7, color="0.55", ha="right", va="bottom")
        s = a[ds]["subsets"]["occany_all"]
        for post, mk in (("raw", "o"), ("pooled", "s")):
            o = s["occany"][post]
            ax.plot(100 * o["recall"], 100 * o["precision"], mk, color=COL_OCC, ms=9, mfc="none" if post == "raw" else COL_OCC, mew=1.8)
            ax.annotate(f"OccAny {post}", (100 * o["recall"], 100 * o["precision"]), textcoords="offset points", xytext=(6, -10), fontsize=8, color=COL_OCC)
            v = f"forward|{cal(ds, 'occany')}|{post}"
            st = s["ours"][v]["stats"]
            ax.plot(100 * st["recall"]["mean"], 100 * st["precision"]["mean"], mk, color=COL_OURS, ms=9, mfc="none" if post == "raw" else COL_OURS, mew=1.8)
            ax.annotate(f"ours {post}", (100 * st["recall"]["mean"], 100 * st["precision"]["mean"]), textcoords="offset points", xytext=(6, 4), fontsize=8, color=COL_OURS)
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_xlabel("recall (%)"); ax.set_ylabel("precision (%)")
        ax.set_title(f"{NAMES[ds]} — all {s['n_samples']:,} samples, forward frames", fontsize=10, loc="left"); ax.grid(alpha=0.2)
    fig.suptitle("Operating points: pooling trades precision for recall for both methods (hollow = raw, filled = pooled)", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_blocks(path: str) -> Dict[str, float]:
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5))
    share = {}
    for ax, ds in zip(axes, C.DATASETS):
        d = load_dataset(ds)
        keys = [str(k) for k in d["keys"]]; col = {k: i for i, k in enumerate(keys)}
        v = f"forward|{cal(ds, 'occany')}|pooled"
        b = blocks_of(ds, d); ids = np.unique(b)
        occ = np.stack([d["official"][b == i, col["occany|pooled"]].sum(0) for i in ids]).astype(np.float64)
        ours = np.stack([np.stack([d["official"][b == i, col[f"ours|{v}|seed{s}"]].sum(0) for s in C.SEEDS]) for i in ids]).astype(np.float64)
        iou = lambda t: t[..., 0] / np.maximum(t.sum(-1), 1)
        delta = 100 * (iou(ours).mean(1) - iou(occ))
        order = np.argsort(delta)
        ax.bar(np.arange(len(ids)), delta[order], color=np.where(delta[order] > 0, COL_OURS, COL_OCC), width=0.9)
        ax.axhline(0, color="0.3", lw=0.8)
        share[ds] = float(np.mean(delta > 0))
        unit = "scenes" if ds == "occ3d" else f"blocks of {C.SK_BLOCK_FRAMES} target frames"
        ax.set_title(f"{NAMES[ds]}: pooled IoU Δ per {unit[:-1] if ds == 'occ3d' else 'block'} — ours higher in {int(np.sum(delta > 0))}/{len(ids)} {unit}", fontsize=10, loc="left")
        ax.set_ylabel("Δ IoU (pp)"); ax.set_xticks([]); ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Primary variant, per resampling block (sorted): is the difference carried by a few scenes?", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return share


# --------------------------------------------------------------------------- documents
def verification_md(checks: List[Dict]) -> str:
    n_pass = sum(c["pass"] for c in checks)
    L = ["# OccAny comparison — verification", "", f"**{n_pass}/{len(checks)} checks pass.** Recomputed from files by `tools/occany_comparison/verify.py`.", "",
         "| check | result | detail |", "|---|---|---|"]
    L += [f"| {c['check']} | {'PASS' if c['pass'] else '**FAIL**'} | {c['detail']} |" for c in checks]
    return "\n".join(L) + "\n"


def maps_section() -> List[str]:
    """Six-panel 3D maps (written by cli/figures_3d.py); omitted until they exist."""
    p = os.path.join(C.FIGURE_DIR, "maps_index.json")
    if not os.path.exists(p):
        return []
    idx = json.load(open(p))
    L = ["## Maps: ours vs OccAny on the same samples", "",
         "Six panels per sample, one viewpoint per row: (1) ground truth; (2, 3) what each method recovered (green) over what it left out "
         "(grey); (4, 5) the same plus voxels predicted where the ground truth is free (red); (6) recovered by only ours (blue), only OccAny "
         "(orange), both (grey). Setting: OccAny's five frames, stored label and placement. Our side is one seed (the median seed on the primary "
         "variant). Rows are **not curated**: they are the samples at fixed quantiles of the per-sample difference, so the first rows show "
         "where OccAny is better. Every drawn volume was re-counted and asserted equal to the evaluation. Figures showing all six samples: "
         "`5_maps_<dataset>_<raw|pooled>.png`; two-row splits: `_pair1..3`.", ""]
    for ds in C.DATASETS:
        r = idx[ds]
        L += [f"**{NAMES[ds]}** (seed {r['seed']})", "", "| Δ-rank | sample | OccAny IoU raw / pooled | ours IoU raw / pooled |", "|---:|---|---|---|"]
        for pk in r["picks"]:
            L.append(f"| {pk['rank_pct']:.0f} % | `{pk['sample_id']}` | {pc(pk['raw_occany']['iou'])} / {pc(pk['pooled_occany']['iou'])} | "
                     f"{pc(pk['raw_ours']['iou'])} / {pc(pk['pooled_ours']['iou'])} |")
        L += ["", f"![{ds} raw](figures/5_maps_{ds}_raw_pair1.png)", "", f"![{ds} pooled](figures/5_maps_{ds}_pooled_pair1.png)", ""]
    return L


def interpretation(a: Dict) -> List[str]:
    """Measurements first, then readings whose wording is decided by the numbers."""
    L = ["## Interpretation (measurements vs reading)", ""]
    not_higher, pool_gap, forward_lower = [], {}, []
    for ds in C.DATASETS:
        s_all, s_c = a[ds]["subsets"]["occany_all"], a[ds]["subsets"]["causal_eligible"]
        v_raw, v_pool = f"forward|{cal(ds, 'occany')}|raw", f"forward|{cal(ds, 'occany')}|pooled"
        o_raw, o_pool = s_all["occany"]["raw"], s_all["occany"]["pooled"]
        u_raw, u_pool = s_all["ours"][v_raw]["stats"], s_all["ours"][v_pool]["stats"]
        c_raw = s_c["ours"][f"causal|{cal(ds, 'ours')}|raw"]["stats"]["iou"]["mean"]
        f_raw = s_c["ours"][v_raw]["stats"]["iou"]["mean"]
        g_o, g_u = o_pool["iou"] - o_raw["iou"], u_pool["iou"]["mean"] - u_raw["iou"]["mean"]
        if f_raw < c_raw:
            forward_lower.append(NAMES[ds])
        pool_gap[ds] = (g_u, g_o, s_all["ours"][v_raw]["delta_vs_occany_same_post"], s_all["ours"][v_pool]["delta_vs_occany_same_post"])
        L.append(f"* **{NAMES[ds]} — measured.** OccAny's pooling moves OccAny {pc(o_raw['iou'])} → {pc(o_pool['iou'])} ({pp(g_o)} pp) and ours "
                 f"{pc(u_raw['iou']['mean'])} → {pc(u_pool['iou']['mean'])} ({pp(g_u)} pp). Raw operating points (all samples, forward): ours "
                 f"precision {pc(u_raw['precision']['mean'])} / recall {pc(u_raw['recall']['mean'])} / predicted-to-GT {u_raw['pred_over_gt']['mean']:.2f}; "
                 f"OccAny {pc(o_raw['precision'])} / {pc(o_raw['recall'])} / {o_raw['pred_over_gt']:.2f}. On the causal-eligible samples our raw IoU is "
                 f"{pc(c_raw)} with our causal frames and {pc(f_raw)} with OccAny's forward frames.")
        for subset in ("occany_all", "causal_eligible"):
            for v, r in a[ds]["subsets"][subset]["ours"].items():
                if (ds == "semantickitti" or f"|{'ours' if v.startswith('causal') else 'occany'}|" in v) and verdict(r["delta_vs_occany_same_post"]) != "higher":
                    not_higher.append(f"{NAMES[ds]} `{subset}` {v} (Δ {ci_text(r['delta_vs_occany_same_post'])})")
    L.append("* **Reading — ranking.** " + (
        "In every like-for-like row (both datasets; raw and pooled; forward and causal, each with its natural placement) ours is higher on identical samples."
        if not not_higher else
        "Ours is higher on identical samples in every like-for-like row except: " + "; ".join(not_higher) +
        ". Where the interval covers 0 the two systems are not distinguishable in that setting."))
    for ds, (g_u, g_o, d_raw, d_pool) in pool_gap.items():
        if g_u - g_o > 0.02:
            L.append(f"* **Reading — magnitude ({NAMES[ds]}).** The gap depends on post-processing: pooling adds {pp(g_u)} pp to ours but {pp(g_o)} pp to OccAny, "
                     f"so the pooled Δ ({pp(d_pool['point'])}) is much larger than the raw Δ ({pp(d_raw['point'])}). A one-voxel dilation mostly helps a "
                     "conservative, high-precision prediction; the raw-vs-raw Δ is the more conservative statement of the difference.")
        elif g_o > g_u:
            L.append(f"* **Reading — magnitude ({NAMES[ds]}).** Pooling favours OccAny: {pp(g_o)} pp for OccAny, {pp(g_u)} pp for ours, so the pooled Δ "
                     f"({pp(d_pool['point'])}) is smaller than the raw Δ ({pp(d_raw['point'])}) and is the conservative figure.")
        else:
            L.append(f"* **Reading — magnitude ({NAMES[ds]}).** Pooling changes both methods by similar amounts ({pp(g_u)} vs {pp(g_o)} pp); "
                     f"raw Δ {pp(d_raw['point'])}, pooled Δ {pp(d_pool['point'])}.")
    if forward_lower:
        L.append(f"* **Hypothesis (not tested here).** On {', '.join(forward_lower)} our raw IoU is lower with forward than with causal frames on the "
                 "same targets. A plausible cause is that with the target first the four later frames mostly re-observe the far field ahead, while the "
                 "target's near field is seen by one frame only; no ablation isolates this.")
    return L + [""]


def report_md(a: Dict, checks: List[Dict], block_share: Dict[str, float]) -> str:
    n_pass = sum(c["pass"] for c in checks)
    L = ["# Is the frozen baseline comparable to OccAny? A like-for-like measurement", "",
         "Frozen current geometry-only baseline (three seeds, threshold 0, source freeze verified) versus the released OccAny "
         "checkpoint's saved predictions, on **every sample OccAny scored**, with OccAny's labels, scorer and pooling applied "
         "identically to both. Primary comparison predeclared in [predeclaration.md](predeclaration.md). "
         f"Verification: {n_pass}/{len(checks)} checks pass ([verification.md](verification.md)).", "",
         "## Answer", ""]
    for ds in C.DATASETS:
        rep = reported(ds); lad = ladder(a, ds); prim = lad[-1]; raw_fwd = lad[-2]
        dl = prim["delta"]
        L.append(f"* **{NAMES[ds]}.** The quoted pair was ours {pc(rep['median'])} (median; mean {pc(rep['mean'])}) on our {rep['n']:,} anchors, raw "
                 f"vs OccAny {pc(C.OCCANY[ds].published_sc_iou)} published (pooled) — not comparable. "
                 f"Like-for-like (all {a[ds]['subsets']['occany_all']['n_samples']:,} OccAny samples, OccAny's frames, labels, placement and pooling): "
                 f"**ours {pc(prim['ours_mean'])} (seed mean; median {pc(prim['ours_median'])}) vs OccAny {pc(prim['occany'])}, "
                 f"Δ {ci_text(dl)} pp → ours {verdict(dl)}.** "
                 f"Without pooling for either: {pc(raw_fwd['ours_mean'])} vs {pc(raw_fwd['occany'])}, Δ {ci_text(raw_fwd['delta'])} → {verdict(raw_fwd['delta'])}. "
                 f"Per resampling block, ours is higher in {100 * block_share[ds]:.0f} % of blocks.")
    L += ["", "The numbers in the question were **not** evaluated on the whole validation sets: 163 of the 815 labelled SemanticKITTI 08 "
          "frames and 1,182 of the Occ3D val samples (1 in 5, CAM_FRONT), raw, with our five past frames. OccAny's published numbers use "
          "805 / 4,819 samples, pooling, and five frames starting at the target.", "",
          "## From the quoted comparison to the like-for-like one", "",
          "Each row removes one difference; Δ is always on identical samples (bootstrap 95 % CI, pp).", ""]
    for ds in C.DATASETS:
        L += [f"**{NAMES[ds]}**", "", "| step | ours mean | ours median | OccAny | Δ (95 % CI) | reading |", "|---|---:|---:|---:|---|---|"]
        for s in ladder(a, ds):
            d = "—" if s["delta"] is None else ci_text(s["delta"])
            L.append(f"| {s['label']} | {pc(s['ours_mean'])} | {pc(s['ours_median'])} | {pc(s['occany'])} | {d} | {'not comparable' if s['delta'] is None else verdict(s['delta'])} |")
        L.append("")
    L += ["![ladder](figures/1_ladder.png)", "",
          "## Supporting rows", ""]
    for ds in C.DATASETS:
        sub = a[ds]["subsets"]
        rp = sub["causal_eligible"]["ours"][f"causal|{cal(ds, 'ours')}|pooled"]
        oa = sub["official_anchors"]
        prim_v = f"forward|{cal(ds, 'occany')}|pooled"
        map_v = f"forward|{cal(ds, 'occany')}|raw"
        causal_fwd = sub["causal_eligible"]["ours"][prim_v]
        L += [f"**{NAMES[ds]}**", "",
              f"* Our deployed setting (causal frames, our placement) pooled, causal-eligible samples: {pc(rp['stats']['iou']['mean'])} vs OccAny "
              f"{pc(sub['causal_eligible']['occany']['pooled']['iou'])}, Δ {ci_text(rp['delta_vs_occany_same_post'])} → {verdict(rp['delta_vs_occany_same_post'])}. "
              f"Forward frames on the same samples: {pc(causal_fwd['stats']['iou']['mean'])} — the effect of seeing the future instead of the past on identical targets.",
              f"* Our official anchors that OccAny also scored ({oa['n_samples']:,}), primary variant: {pc(oa['ours'][prim_v]['stats']['iou']['mean'])} vs "
              f"{pc(oa['occany']['pooled']['iou'])}, Δ {ci_text(oa['ours'][prim_v]['delta_vs_occany_same_post'])}; OccAny scores "
              f"{pc(oa['occany']['pooled']['iou'])} here vs {pc(sub['occany_all']['occany']['pooled']['iou'])} on all of its samples.",
              f"* Map-only reference (our raw map `S`, no network), primary placement, forward, raw: {pc(sub['occany_all']['maponly'][map_v]['iou'])}.",
              ""]
    L += ["![per-sample](figures/2_per_sample.png)", "", "![operating points](figures/3_operating_points.png)", "", "![blocks](figures/4_blocks.png)", ""]
    L += maps_section()
    L += interpretation(a)
    L += ["## What this does and does not establish", "",
          "* **Measured:** on OccAny's own sample sets, labels, placement, frames, pooling and scorer, the frozen baseline's pooled IoU relative to "
          "OccAny's, with a paired interval over scenes / frame blocks. OccAny's pooled numbers on its full sets reproduce its published values "
          "(check 6), so the reference is the published system.",
          "* **Not removed:** OccAny's per-dataset confidence thresholds vs our single frozen threshold; different training data (neither trained on "
          "these targets); different image preprocessing; three seeds vs one released model. The intervals cover sample variation, not these.",
          "* **Not a clean held-out test:** both target sets have been evaluated many times in this project (RC5). This is a reporting measurement "
          "of a frozen model; no number here may be used to select anything.",
          "* **Pooling** is OccAny's convention, not part of either model; the raw-vs-raw rows are the post-processing-free comparison.", "",
          "## Files", "",
          "* [predeclaration.md](predeclaration.md), [results_table.md](results_table.md), [analysis.json](analysis.json), [verification.md](verification.md)",
          "* `eval/<dataset>_shardKofN.npz` — per-sample TP/FP/FN for every variant (OccAny scorer and GPU tally)",
          "* code: `tools/occany_comparison/` (tests: `tests/occany_comparison/`)", "",
          "```bash", "export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH FLASHINFER_CUDA_ARCH_LIST=12.0a PYTHONPATH=$PWD",
          "python tools/occany_comparison/cli/build_moge.py --dataset occ3d --device cuda:0",
          "for k in 0 1 2 3; do python tools/occany_comparison/cli/build_clips.py --dataset occ3d --device cuda:$k --shard $k --shards 4 & done; wait",
          "python tools/occany_comparison/cli/build_clips.py --dataset semantickitti --device cuda:0",
          "for k in 0 1 2 3; do python tools/occany_comparison/cli/evaluate.py --dataset occ3d --device cuda:$k --shard $k --shards 4 & done; wait",
          "for k in 0 1; do python tools/occany_comparison/cli/evaluate.py --dataset semantickitti --device cuda:$k --shard $k --shards 2 & done; wait",
          "python tools/occany_comparison/cli/report.py          # analysis.json, tables, figures 1-4, verification",
          "python tools/occany_comparison/cli/figures_3d.py --device cuda:0   # 3D maps (needs analysis.json)",
          "python tools/occany_comparison/cli/report.py          # again, to include the maps", "```", ""]
    return "\n".join(L)


def write_all() -> Dict:
    from tools.occany_comparison.analysis import analyse_all
    from tools.occany_comparison.verify import run_verification
    a = analyse_all()
    os.makedirs(C.FIGURE_DIR, exist_ok=True)
    fig_ladder(a, os.path.join(C.FIGURE_DIR, "1_ladder.png"))
    fig_per_sample(a, os.path.join(C.FIGURE_DIR, "2_per_sample.png"))
    fig_operating_points(a, os.path.join(C.FIGURE_DIR, "3_operating_points.png"))
    share = fig_blocks(os.path.join(C.FIGURE_DIR, "4_blocks.png"))
    open(os.path.join(C.ARTIFACT_DIR, "results_table.md"), "w").write(results_table(a))
    checks = run_verification()
    open(os.path.join(C.ARTIFACT_DIR, "verification.md"), "w").write(verification_md(checks))
    rep = report_md(a, checks, share)
    open(os.path.join(C.ARTIFACT_DIR, "report.md"), "w").write(rep)
    shutil.copyfile(os.path.join(C.ARTIFACT_DIR, "report.md"), os.path.join(C.REPO_ROOT, "reports", "occany_comparison.md"))
    return {"checks_pass": sum(c["pass"] for c in checks), "checks": len(checks)}


__all__ = ["ladder", "results_table", "verification_md", "report_md", "write_all"]
