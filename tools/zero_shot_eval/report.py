"""Render tables, figures and the numbers document from ``analysis.json`` (and canonical predictions for the qualitative figure).

Nothing here computes a new metric: every number comes from ``analysis.py``.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tools.zero_shot_eval import config as C  # noqa: E402
from tools.zero_shot_eval.practical.protocol import DATASET_SPECS, EVIDENCE_DISTANCE_NAMES, PAPER, REGION  # noqa: E402

POPS = ("ssc", "ssc_no_evidence", "gt_occupied", "observed_evidence", "no_evidence", "completed_no_evidence", "geometrically_unobserved")
POP_LABEL = {"ssc": "SSC (predicted occupancy)", "ssc_no_evidence": "SSC, no-evidence voxels", "gt_occupied": "GT occupied (diagnostic)",
             "observed_evidence": "observed with evidence", "no_evidence": "no evidence (GT occupancy)",
             "completed_no_evidence": "completed, no evidence (CENTRAL)", "geometrically_unobserved": "completed, geometrically unobserved"}
DS_LABEL = {"semantickitti": "SemanticKITTI 08", "occ3d": "Occ3D-nuScenes val"}
COL = {"B0": "#8c8c8c", "B3": "#2ca02c", "OccAny": "#ff7f0e"}
pct = lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.2f}"


def _csv(path: str, head: List[str], rows: List[List]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(",".join(head) + "\n")
        for r in rows:
            fh.write(",".join("" if v is None else (f"{v:.6f}" if isinstance(v, float) else str(v)) for v in r) + "\n")


def _md(head: List[str], rows: List[List]) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]) + "\n"


def tables(a: Dict) -> None:
    T = C.TABLE_DIR; b0b3 = []; per_class = []; per_dist = []; occany = []
    for ds, d in a["datasets"].items():
        spec = DATASET_SPECS[ds]; rows = []
        for region in ("P", "full"):
            for pop in POPS:
                s = d["semantic"][f"{region}__{pop}"]; dl = s["delta_B3_minus_B0"]
                rows.append([region, pop, s["B0"]["miou"], s["B3"]["miou"]] + [s["B3_per_seed"][m] for m in sorted(s["B3_per_seed"])] +
                            [dl["delta"] / 100, dl["ci95"][0] / 100, dl["ci95"][1] / 100, s["B0"]["overall_accuracy"], s["B3"]["overall_accuracy"], s["B0"]["gt_voxels"]])
                b0b3.append([ds, region, pop, s["B0"]["miou"], s["B3"]["miou"], dl["delta"] / 100, dl["ci95"][0] / 100, dl["ci95"][1] / 100,
                             min(s["delta_per_seed"].values()) / 100, dl["block_wins"], dl["blocks"]])
        o = d["occupancy"]
        rows += [["P", f"occupancy_{p}", o[f"ours__P__{p}"]["iou"], None, None, None, None, None, None, None, o[f"ours__P__{p}"]["precision"], o[f"ours__P__{p}"]["recall"], None] for p in ("all", "observed", "unobserved")]
        _csv(os.path.join(T, f"practical_{ds}.csv"), ["region", "population", "B0_miou", "B3_miou", "B3_s0", "B3_s1", "B3_s2", "delta", "ci_lo", "ci_hi", "B0_acc", "B3_acc", "gt_voxels"], rows)
        off = d["official"]
        _csv(os.path.join(T, f"official_{ds}.csv"), ["method", "sc_iou", "sc_precision", "sc_recall", "ssc_miou", "superclass_miou"],
             [["ours (B0/B3 share occupancy)", o["ours__official"]["iou"], o["ours__official"]["precision"], o["ours__official"]["recall"], None, None],
              ["B0", None, None, None, off["B0"]["miou"], off["B0"]["superclass_miou"]], ["B3 (seed mean)", None, None, None, off["B3"]["miou"], off["B3"]["superclass_miou"]]]
             + [[m, None, None, None, off[m]["miou"], off[m]["superclass_miou"]] for m in ("B3_s0", "B3_s1", "B3_s2")]
             + [[f"OccAny released geometry ({t})", o[f"occany_{t}__official"]["iou"], o[f"occany_{t}__official"]["precision"], o[f"occany_{t}__official"]["recall"], None, None] for t in ("raw", "pooled")])
        cen = d["semantic"]["P__completed_no_evidence"]; ssc = d["semantic"]["P__ssc"]
        for c in spec.semantic_classes:
            per_class.append([ds, spec.class_names[c], int(c in spec.dynamic_classes), cen["B0"]["per_class_iou"].get(str(c), cen["B0"]["per_class_iou"].get(c)),
                              cen["B3"]["per_class_iou"].get(str(c), cen["B3"]["per_class_iou"].get(c)),
                              ssc["B0"]["per_class_iou"].get(str(c), ssc["B0"]["per_class_iou"].get(c)), ssc["B3"]["per_class_iou"].get(str(c), ssc["B3"]["per_class_iou"].get(c)),
                              cen["B0"]["gt_frequency"].get(str(c), cen["B0"]["gt_frequency"].get(c)) if "gt_frequency" in cen["B0"] else None])
        for kind, key in (("evidence_distance", "distance"), ("range", "range")):
            for name, v in d[key].items():
                per_dist.append([ds, kind, name, v["voxels"], v["B0_miou"], v["B3_miou"], v["B0_accuracy"], v["B3_accuracy"]])
        pub = dict(PAPER.occany_published_miou)[ds]
        cams = "1 (image_2)" if ds == "semantickitti" else "1 (CAM_FRONT)"
        occany += [[ds, "B0", cams, "5 causal (t-4..t)", "Track P", o["ours__P__all"]["iou"], ssc["B0"]["miou"], cen["B0"]["miou"], o["ours__official"]["iou"], off["B0"]["miou"]],
                   [ds, "B3", cams, "5 causal (t-4..t)", "Track P", o["ours__P__all"]["iou"], ssc["B3"]["miou"], cen["B3"]["miou"], o["ours__official"]["iou"], off["B3"]["miou"]],
                   [ds, "OccAny geometry (released, raw)", "1 front", "5 forward (t..t+8)", "Level A (same evaluator)", o["occany_raw__P__all"]["iou"], None, None, o["occany_raw__official"]["iou"], None],
                   [ds, "OccAny geometry (released, pooled)", "1 front", "5 forward (t..t+8)", "Level A (same evaluator)", o["occany_pooled__P__all"]["iou"], None, None, o["occany_pooled__official"]["iou"], None],
                   [ds, "OccAny + GSAM2 semantics (published)", "1 (5 frames)" if ds == "semantickitti" else "6 surround (1 timestep)", "5 forward" if ds == "semantickitti" else "1 timestep",
                    "Level B (paper Tab. 4)", None, None, None, None, pub / 100]]
    _csv(os.path.join(T, "b0_vs_b3.csv"), ["dataset", "region", "population", "B0_miou", "B3_miou", "delta", "ci_lo", "ci_hi", "min_seed_delta", "block_wins", "blocks"], b0b3)
    _csv(os.path.join(T, "per_class.csv"), ["dataset", "class", "dynamic", "B0_iou_central", "B3_iou_central", "B0_iou_ssc_P", "B3_iou_ssc_P", "gt_share_central"], per_class)
    _csv(os.path.join(T, "per_distance.csv"), ["dataset", "binning", "bin", "voxels", "B0_miou", "B3_miou", "B0_accuracy", "B3_accuracy"], per_dist)
    _csv(os.path.join(T, "occany_comparison.csv"), ["dataset", "method", "cameras", "temporal_context", "evaluation", "track_p_sc_iou", "track_p_ssc_miou",
                                                    "track_p_central_miou", "official_sc_iou", "official_ssc_miou"], occany)


def figures(a: Dict) -> List[str]:
    F = C.FIGURE_DIR; os.makedirs(F, exist_ok=True); out = []
    dss = list(a["datasets"])
    def save(fig, name):
        p = os.path.join(F, name); fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); out.append(p)
    # 2: B3 - B0 gains
    fig, axs = plt.subplots(1, len(dss), figsize=(7 * len(dss), 4.2), squeeze=False)
    for ax, ds in zip(axs[0], dss):
        pops = list(POPS); d = a["datasets"][ds]["semantic"]
        vals = [d[f"P__{p}"]["delta_B3_minus_B0"] for p in pops]
        y = np.arange(len(pops))
        ax.errorbar([v["delta"] for v in vals], y, xerr=[[v["delta"] - v["ci95"][0] for v in vals], [v["ci95"][1] - v["delta"] for v in vals]], fmt="o", color=COL["B3"], capsize=3)
        for i, p in enumerate(pops):
            ax.scatter(list(d[f"P__{p}"]["delta_per_seed"].values()), [i] * 3, s=10, color="k", zorder=3)
        ax.axvline(0, color="k", lw=0.8); ax.axvline(PAPER.meaningful_gain_points, color="k", ls="--", lw=0.6)
        ax.set_yticks(y); ax.set_yticklabels([POP_LABEL[p] for p in pops], fontsize=8); ax.set_xlabel("B3 − B0 mIoU (points), 95 % block bootstrap; dots = seeds")
        ax.set_title(f"{DS_LABEL[ds]} — Track P")
    save(fig, "2_b0_vs_b3_gains.png")
    # 3: observed vs completed vs no-evidence
    fig, axs = plt.subplots(1, len(dss), figsize=(7 * len(dss), 4), squeeze=False)
    pops3 = ("observed_evidence", "no_evidence", "completed_no_evidence", "geometrically_unobserved", "ssc")
    for ax, ds in zip(axs[0], dss):
        d = a["datasets"][ds]["semantic"]; x = np.arange(len(pops3))
        ax.bar(x - 0.2, [100 * d[f"P__{p}"]["B0"]["miou"] for p in pops3], 0.4, color=COL["B0"], label="B0")
        ax.bar(x + 0.2, [100 * d[f"P__{p}"]["B3"]["miou"] for p in pops3], 0.4, color=COL["B3"], label="B3 (seed mean)")
        ax.set_xticks(x); ax.set_xticklabels([POP_LABEL[p] for p in pops3], rotation=20, fontsize=7); ax.set_ylabel("mIoU %"); ax.legend(); ax.set_title(DS_LABEL[ds])
    save(fig, "3_observed_completed_no_evidence.png")
    # 4: distance and range
    fig, axs = plt.subplots(2, len(dss), figsize=(7 * len(dss), 7), squeeze=False)
    for j, ds in enumerate(dss):
        for i, (key, names, title) in enumerate((("distance", EVIDENCE_DISTANCE_NAMES, "distance to nearest evidence"), ("range", REGION.range_names, "range from the camera"))):
            v = a["datasets"][ds][key]; ax = axs[i][j]
            ax.plot(names, [100 * (v[n]["B0_miou"] or 0) for n in names], "o-", color=COL["B0"], label="B0 mIoU")
            ax.plot(names, [100 * (v[n]["B3_miou"] or 0) for n in names], "o-", color=COL["B3"], label="B3 mIoU")
            ax2 = ax.twinx(); ax2.plot(names, [100 * v[n]["B0_accuracy"] for n in names], "s:", color=COL["B0"]); ax2.plot(names, [100 * v[n]["B3_accuracy"] for n in names], "s:", color=COL["B3"])
            ax2.set_ylabel("accuracy % (dotted)"); ax.set_ylabel("mIoU %"); ax.legend(fontsize=7); ax.set_title(f"{DS_LABEL[ds]}: central population by {title}")
    save(fig, "4_distance_and_range.png")
    # 5: per class
    fig, axs = plt.subplots(len(dss), 1, figsize=(15, 4 * len(dss)), squeeze=False)
    for ax, ds in zip(axs[:, 0], dss):
        spec = DATASET_SPECS[ds]; cen = a["datasets"][ds]["semantic"]["P__completed_no_evidence"]
        cls = list(spec.semantic_classes); x = np.arange(len(cls))
        g = lambda s, c: s["per_class_iou"].get(str(c), s["per_class_iou"].get(c))
        ax.bar(x - 0.2, [100 * (g(cen["B0"], c) or 0) for c in cls], 0.4, color=COL["B0"], label="B0")
        ax.bar(x + 0.2, [100 * (g(cen["B3"], c) or 0) for c in cls], 0.4, color=COL["B3"], label="B3")
        ax.set_xticks(x); ax.set_xticklabels([spec.class_names[c] + (" *" if c in spec.dynamic_classes else "") for c in cls], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("IoU %"); ax.legend(); ax.set_title(f"{DS_LABEL[ds]}: per-class IoU, central population (* dynamic class)")
    save(fig, "5_per_class.png")
    # 6: OccAny comparison
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.6))
    x = np.arange(len(dss)); w = 0.2
    for k, (lab, key, col) in enumerate((("ours (5 causal frames, 1 cam)", "ours", COL["B3"]), ("OccAny raw (5 forward frames, 1 cam)", "occany_raw", COL["OccAny"]),
                                         ("OccAny pooled", "occany_pooled", "#ffbb78"))):
        axs[0].bar(x + (k - 1) * w, [100 * a["datasets"][ds]["occupancy"][f"{key}__P__all"]["iou"] for ds in dss], w, color=col, label=lab)
        axs[0].bar(x + (k - 1) * w, [100 * a["datasets"][ds]["occupancy"][f"{key}__official"]["iou"] for ds in dss], w, fill=False, edgecolor="k", hatch="//")
    axs[0].set_xticks(x); axs[0].set_xticklabels([DS_LABEL[d] for d in dss]); axs[0].set_ylabel("SC IoU % (solid: Track P, hatched: official)"); axs[0].legend(fontsize=7)
    axs[0].set_title("Geometry, Level A: same samples, labels and evaluator")
    pub = dict(PAPER.occany_published_miou)
    axs[1].bar(x - 0.3, [100 * a["datasets"][ds]["official"]["B0"]["miou"] for ds in dss], 0.3, color=COL["B0"], label="B0 (ours, Track O)")
    axs[1].bar(x, [100 * a["datasets"][ds]["official"]["B3"]["miou"] for ds in dss], 0.3, color=COL["B3"], label="B3 (ours, Track O)")
    axs[1].bar(x + 0.3, [pub[ds] for ds in dss], 0.3, color=COL["OccAny"], label="OccAny + GSAM2 (published, Level B)")
    axs[1].set_xticks(x); axs[1].set_xticklabels([f"{DS_LABEL['semantickitti']}\nOccAny: 5 forward frames" if d == "semantickitti" else f"{DS_LABEL['occ3d']}\nOccAny: 6 surround cams, full camera mask" for d in dss], fontsize=8)
    axs[1].set_ylabel("SSC mIoU %"); axs[1].set_ylim(0, 9.5); axs[1].legend(fontsize=7, loc="upper center", ncol=3); axs[1].set_title("Semantics: different inputs / protocols — context only")
    save(fig, "6_occany_comparison.png")
    return out


def protocol_and_qualitative(device_free: bool = True) -> List[str]:
    """Figures 1 and 7 from the first evaluated sample of each dataset (deterministic): the Track P region and populations, and B0/B3/GT."""
    from tools.occany_comparison import scoring as SC
    from tools.zero_shot_eval import samples as ZS
    from tools.zero_shot_eval.canonical.prediction import CanonicalPrediction
    from tools.zero_shot_eval.practical.masks import populations, track_p_masks
    out = []
    fig1, ax1 = plt.subplots(1, 2, figsize=(12, 6)); fig7, ax7 = plt.subplots(2, 4, figsize=(18, 9))
    for j, ds in enumerate(C.DATASETS):
        spec = DATASET_SPECS[ds]; ss = ZS.evaluated_samples(ds); s = ss[len(ss) // 2]
        pred = CanonicalPrediction.load(C.prediction_path(ds, s.sample_id)); label = SC.load_occany(ds, s.occany_pkl, pred.dims).label.reshape(-1)
        m = track_p_masks(pred, label, spec); pops = populations(pred, m, m.region)
        X, Y, Z = pred.dims
        top = lambda mask: mask.reshape(X, Y, Z).any(2)
        img = np.full((X, Y, 3), 1.0)
        img[top(m.valid)] = (0.9, 0.9, 0.9); img[top(m.region)] = (0.75, 0.85, 1.0)
        img[top(pops["completed_no_evidence"])] = (0.2, 0.6, 0.2); img[top(pops["observed_evidence"])] = (0.85, 0.3, 0.3)
        ax1[j].imshow(np.rot90(img), interpolation="nearest"); ax1[j].set_title(f"{DS_LABEL[ds]} {s.sample_id}\nlight grey valid, blue Track P region, red observed-with-evidence GT, green completed no-evidence GT", fontsize=8); ax1[j].axis("off")
        rng = np.random.default_rng(0); pal = rng.random((256, 3)); pal[spec.empty_class] = 1.0
        def bev(lab_dense, mask):
            vol = np.where(mask, lab_dense, spec.empty_class).reshape(X, Y, Z); zi = np.where(vol != spec.empty_class, np.arange(Z)[None, None], -1).argmax(2)
            return np.rot90(pal[np.take_along_axis(vol, zi[..., None], 2)[..., 0]])
        R = m.region & pred.occupied
        for k, (title, lab) in enumerate((("GT (region, GT occupied)", label), ("B0", pred.dense_labels("B0", spec.empty_class)),
                                          ("B3 seed 0", pred.dense_labels("B3_s0", spec.empty_class)), ("evidence voxels (region)", np.where(pred.evidence, 1, spec.empty_class)))):
            mask = (m.region & m.gt_occupied) if k == 0 else (m.region & pred.evidence if k == 3 else R)
            ax7[j][k].imshow(bev(lab, mask), interpolation="nearest"); ax7[j][k].set_title(f"{DS_LABEL[ds]}: {title}", fontsize=8); ax7[j][k].axis("off")
    for fig, name in ((fig1, "1_practical_region.png"), (fig7, "7_qualitative.png")):
        p = os.path.join(C.FIGURE_DIR, name); fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig); out.append(p)
    return out


def results_md(a: Dict) -> str:
    out = ["# Final zero-shot results (numbers; interpretation in `final_zero_shot_results.md` / `paper_decision.md`)\n"]
    for ds, d in a["datasets"].items():
        out.append(f"\n## {DS_LABEL[ds]} — {d['n_samples']} samples, {d['blocks']} bootstrap blocks\n")
        rows = []
        for pop in POPS:
            s = d["semantic"][f"P__{pop}"]; dl = s["delta_B3_minus_B0"]
            rows.append([POP_LABEL[pop], f"{s['B0']['gt_voxels']:,}", pct(s["B0"]["miou"]), f"{pct(s['B3']['miou'])} ({' / '.join(pct(v) for v in s['B3_per_seed'].values())})",
                         f"{dl['delta']:+.2f} [{dl['ci95'][0]:+.2f}, {dl['ci95'][1]:+.2f}]", f"{pct(s['B0']['overall_accuracy'])} / {pct(s['B3']['overall_accuracy'])}"])
        out.append("### Track P semantic (mIoU %)\n" + _md(["population", "GT voxels", "B0", "B3 mean (seeds)", "Δ B3−B0 [CI]", "accuracy B0 / B3"], rows))
        o = d["occupancy"]
        rows = [[f"{src} {part}", pct(o[f"{src}__P__{part}"]["iou"]), pct(o[f"{src}__P__{part}"]["precision"]), pct(o[f"{src}__P__{part}"]["recall"])]
                for src in ("ours", "occany_raw", "occany_pooled") for part in ("all", "observed", "unobserved")]
        out.append("\n### Track P occupancy (%)\n" + _md(["source / partition", "IoU", "precision", "recall"], rows))
        rows = [[b, pct(o["ours__range"][b]["iou"]), pct(o["occany_raw__range"][b]["iou"]), pct(d["range"][b]["B0_miou"]), pct(d["range"][b]["B3_miou"])] for b in REGION.range_names]
        out.append("\n### By range (Track P)\n" + _md(["range", "ours SC IoU", "OccAny raw SC IoU", "B0 central mIoU", "B3 central mIoU"], rows))
        rows = [[b, f"{d['distance'][b]['voxels']:,}", pct(d["distance"][b]["B0_miou"]), pct(d["distance"][b]["B3_miou"]), pct(d["distance"][b]["B0_accuracy"]), pct(d["distance"][b]["B3_accuracy"])] for b in EVIDENCE_DISTANCE_NAMES]
        out.append("\n### Central population by distance to nearest evidence\n" + _md(["distance", "voxels", "B0 mIoU", "B3 mIoU", "B0 acc", "B3 acc"], rows))
        off = d["official"]
        out.append(f"\n### Track O (official)\n\nSC IoU / P / R ours {pct(o['ours__official']['iou'])} / {pct(o['ours__official']['precision'])} / {pct(o['ours__official']['recall'])}; "
                   f"OccAny released geometry raw {pct(o['occany_raw__official']['iou'])}, pooled {pct(o['occany_pooled__official']['iou'])}. "
                   f"SSC mIoU B0 {pct(off['B0']['miou'])}, B3 {pct(off['B3']['miou'])} (seeds {' / '.join(pct(off[m]['miou']) for m in ('B3_s0', 'B3_s1', 'B3_s2'))}); "
                   f"superclass mIoU B0 {pct(off['B0']['superclass_miou'])}, B3 {pct(off['B3']['superclass_miou'])}. OccAny published (GSAM2): {dict(PAPER.occany_published_miou)[ds]:.2f}.\n")
        for tag in ("raw", "pooled"):
            dd = o[f"delta_ours_minus_occany_{tag}__P"]; do = o[f"delta_ours_minus_occany_{tag}__official"]
            out.append(f"* ours − OccAny {tag} SC IoU: Track P {dd['delta']:+.2f} [{dd['ci95'][0]:+.2f}, {dd['ci95'][1]:+.2f}]; official {do['delta']:+.2f} [{do['ci95'][0]:+.2f}, {do['ci95'][1]:+.2f}]\n")
        out.append(f"* Track O confusions identical to Track P full-region confusions: {d['track_o_equals_track_p_full']}\n")
    out.append(f"\n## Criteria\n\n```\n{json.dumps(a['criteria'], indent=1)}\n```\n\n**{a['decision']}**\n")
    return "\n".join(out)


def write_all() -> Dict:
    a = json.load(open(os.path.join(C.ARTIFACT_DIR, "analysis.json")))
    tables(a)
    open(os.path.join(C.ARTIFACT_DIR, "results_numbers.md"), "w").write(results_md(a))
    return {"figures": figures(a) + protocol_and_qualitative()}


__all__ = ["tables", "figures", "results_md", "write_all"]
