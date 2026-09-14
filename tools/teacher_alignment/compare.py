#!/usr/bin/env python
"""Aggregate R/S into the result table with paired R - S differences and CIs.

Resampling units as before: contiguous 20-anchor blocks on the source drive, whole
sequences/scenes (or blocks where too few) on the targets. Historical A and B are carried as
labelled reference rows, read from the feature study; they were trained under nominal loss
weights, not the matched rule, and are not the control for R.

    python tools/teacher_alignment/compare.py
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.teacher_alignment import study as ST                                   # noqa: E402
from core.evaluation import scores as SC                                          # noqa: E402
from tools.feature_study.compare import (block_indices, group_indices, iou,       # noqa: E402
                                         paired_ci, N_BOOT, BLOCK)

REGIONS = ("later", "full", "observed")


def load_source(art, tagf):
    rows, counts = {}, {}
    for c in (ST.CONDITIONS if art == ST.ART else ("A", "B")):
        for s in ST.SEEDS:
            tag = tagf(c, s)
            p = os.path.join(art, "source", f"source_{tag}.json")
            if not os.path.exists(p):
                continue
            r = json.load(open(p))
            rows[(c, s)] = {"threshold": r["occupancy_threshold"],
                            **{f"{reg}_{k}": r["regions"][reg]["at_threshold"][k]
                               for reg in REGIONS for k in ("iou", "precision", "recall", "tp",
                                                             "fp", "fn", "tn", "pred_over_gt")},
                            "later_ap": r["regions"]["later"]["average_precision"],
                            "later_auroc": r["regions"]["later"]["auroc"],
                            "later_prevalence": r["regions"]["later"]["prevalence"],
                            "full_ap": r["regions"]["full"]["average_precision"],
                            "full_auroc": r["regions"]["full"]["auroc"],
                            "semantic_threshold": r.get("semantic_threshold"),
                            "teacher_agreement": r.get("teacher_agreement")}
            for reg in REGIONS:
                q = os.path.join(art, "source", f"clipcounts_{tag}_{reg}.npz")
                if os.path.exists(q):
                    with np.load(q) as z:
                        counts[(c, s, reg)] = z["counts"]
    return rows, counts


def load_transfer(art, tagf, conds):
    rows, counts = {}, {}
    for c in conds:
        for s in ST.SEEDS:
            for ds in ("semantickitti", "occ3d"):
                p = os.path.join(art, "transfer", c, f"eval_{ds}_past5_seed{s}.json")
                if not os.path.exists(p):
                    continue
                r = json.load(open(p))
                cb = r["semantic"]["completion"]
                rows[(c, s, ds)] = {"n_clips": r["n_clips"], "threshold": r["threshold"],
                                    "iou": cb["binary_iou"], "precision": cb["binary_precision"],
                                    "recall": cb["binary_recall"], "ssc_miou": cb["ssc_miou"],
                                    "auroc": r["methods"]["completion"]["full"]["auroc"],
                                    "ap_over_prevalence": r["methods"]["completion"]["full"]["ap_over_prevalence"]}
                q = os.path.join(art, "transfer", c, f"scores_{ds}_past5_seed{s}_completion.npz")
                if os.path.exists(q):
                    with np.load(q) as z:
                        blk = {k[len("full_"):]: z[k] for k in z.files if k.startswith("full_")}
                    counts[(c, s, ds)] = (SC.per_clip_counts(blk, r["threshold"]), blk["group"])
    return rows, counts


def training_rows():
    out = {}
    for c in ST.CONDITIONS:
        for s in ST.SEEDS:
            p = os.path.join(ST.ART, f"train_{ST.run_tag(c, s)}.json")
            if not os.path.exists(p):
                continue
            j = json.load(open(p)); st = [x for x in j["log"] if "val" not in x]
            nd = [x for x in st if not x["degenerate"]]
            out[f"{c}_seed{s}"] = {
                "rho": j["rho"], "n_degenerate_steps": j["n_degenerate_steps"],
                "achieved_ratio_median": float(np.median([x["achieved_ratio"] for x in nd])),
                "achieved_ratio_p05_p95": [float(np.percentile([x["achieved_ratio"] for x in nd], q)) for q in (5, 95)],
                "raw_ratio_median": float(np.median([x["raw_ratio"] for x in nd])),
                "norm_factor_median": float(np.median([x["norm_factor"] for x in nd])),
                "grad_cosine_median": float(np.median([x["grad_cosine"] for x in nd])),
                "grad_cosine_mean": float(np.mean([x["grad_cosine"] for x in nd])),
                "final_occ_loss": float(np.mean([x["occ"] for x in st[-100:]])),
                "final_kl_loss": float(np.mean([x["kl"] for x in st[-100:]])),
                "final_train_agree": float(np.mean([x["aux_top1_agree"] for x in st[-100:]])),
                "final_val_kl_aligned": [x["val"]["kl"] for x in j["log"] if "val" in x][-1],
                "final_val_agree_aligned": [x["val"]["aux_top1_agree"] for x in j["log"] if "val" in x][-1],
                "manipulation": j["manipulation"], "minutes": j["seconds"] / 60}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    ST.require_data_root()
    rng = np.random.default_rng(0)
    src, sc = load_source(ST.ART, ST.run_tag)
    hsrc, _ = load_source(ST.FEATURE_STUDY_ART, lambda c, s: f"fcs_{c}_seed{s}")
    tr, tc = load_transfer(ST.ART, ST.run_tag, ST.CONDITIONS)
    htr, _ = load_transfer(ST.FEATURE_STUDY_ART, None, ("A", "B"))
    out = {"n_bootstrap": a.n_boot, "block_anchors": BLOCK, "training": training_rows(),
           "source": {f"{c}_seed{s}": v for (c, s), v in src.items()},
           "source_reference_AB": {f"{c}_seed{s}": v for (c, s), v in hsrc.items()},
           "transfer": {f"{c}_seed{s}_{ds}": v for (c, s, ds), v in tr.items()},
           "transfer_reference_AB": {f"{c}_seed{s}_{ds}": v for (c, s, ds), v in htr.items()},
           "paired": {"source": {}, "transfer": {}}, "env": ST.environment()}
    for reg in REGIONS:
        per = []
        for s in ST.SEEDS:
            if ("R", s, reg) in sc and ("S", s, reg) in sc:
                n = sc[("R", s, reg)].shape[0]
                per.append(paired_ci(sc[("R", s, reg)], sc[("S", s, reg)],
                                     lambda r, n=n: block_indices(n, r), rng, a.n_boot))
        if per:
            out["paired"]["source"][f"R-S_{reg}"] = {
                "per_seed": per, "mean_delta": float(np.mean([x["delta"] for x in per])),
                "seed_spread": [min(x["delta"] for x in per), max(x["delta"] for x in per)]}
    for ds in ("semantickitti", "occ3d"):
        per = []
        for s in ST.SEEDS:
            if ("R", s, ds) in tc and ("S", s, ds) in tc:
                cr, g = tc[("R", s, ds)]; cs, _ = tc[("S", s, ds)]
                per.append(paired_ci(cr, cs, lambda r, g=g: group_indices(g, r), rng, a.n_boot))
        if per:
            out["paired"]["transfer"][f"R-S_{ds}"] = {
                "per_seed": per, "mean_delta": float(np.mean([x["delta"] for x in per])),
                "seed_spread": [min(x["delta"] for x in per), max(x["delta"] for x in per)]}
    ST.write_json(os.path.join(ST.ART, "comparison.json"), out)
    _print(out)
    return 0


def _print(out):
    print("\n  training (gradient budget)")
    for k, v in sorted(out["training"].items()):
        m = v["manipulation"]
        print(f"  {k}: degenerate {v['n_degenerate_steps']}, achieved {v['achieved_ratio_median']:.3f} "
              f"[{v['achieved_ratio_p05_p95'][0]:.3f},{v['achieved_ratio_p05_p95'][1]:.3f}] "
              f"raw {v['raw_ratio_median']:.2f} c {v['norm_factor_median']:.2f} cos "
              f"{v['grad_cosine_median']:+.3f} | occ {v['final_occ_loss']:.4f} kl {v['final_kl_loss']:.4f} "
              f"val-agree(aligned) {v['final_val_agree_aligned']:.3f}"
              + (f" | shuffled: vec {100*m['fraction_vectors_changed']:.1f}% argmax "
                 f"{100*m['fraction_argmax_changed']:.1f}% L1 {m['mean_l1_before_after']:.3f} "
                 f"homog {100*m['fraction_in_homogeneous_strata']:.1f}%" if k.startswith("S") else "")
              + f" tiny-dropped {100*m['fraction_supervision_dropped_tiny_strata']:.2f}%")
    print("\n  source drive 0006 -- LATER IoU per seed")
    for c in ST.CONDITIONS:
        v = [out["source"][f"{c}_seed{s}"]["later_iou"] for s in ST.SEEDS if f"{c}_seed{s}" in out["source"]]
        ag = [out["source"][f"{c}_seed{s}"]["teacher_agreement"]["all"]["top1_agreement"]
              for s in ST.SEEDS if f"{c}_seed{s}" in out["source"]]
        print(f"  {c}: " + " ".join(f"{x:.4f}" for x in v) + f"  median {np.median(v):.4f}"
              + f"  | aligned-teacher agreement " + " ".join(f"{x:.3f}" for x in ag))
    for c in ("A", "B"):
        v = [out["source_reference_AB"][f"{c}_seed{s}"]["later_iou"] for s in ST.SEEDS]
        print(f"  ref {c}: " + " ".join(f"{x:.4f}" for x in v) + f"  median {np.median(v):.4f}")
    for k, v in out["paired"]["source"].items():
        print(f"  paired {k}: " + "  ".join(f"{x['delta']:+.4f}[{x['ci95'][0]:+.4f},{x['ci95'][1]:+.4f}]"
                                            for x in v["per_seed"]) + f"  mean {v['mean_delta']:+.4f}")
    if out["transfer"]:
        print("\n  transfer past5 -- IoU per seed")
        for ds in ("semantickitti", "occ3d"):
            for c in ST.CONDITIONS:
                v = [out["transfer"][f"{c}_seed{s}_{ds}"]["iou"] for s in ST.SEEDS if f"{c}_seed{s}_{ds}" in out["transfer"]]
                m = [out["transfer"][f"{c}_seed{s}_{ds}"]["ssc_miou"] for s in ST.SEEDS if f"{c}_seed{s}_{ds}" in out["transfer"]]
                if v:
                    print(f"  {ds:14s} {c}: " + " ".join(f"{x:.4f}" for x in v)
                          + f"  median {np.median(v):.4f}  mIoU {np.median(m):.4f}")
            for c in ("A", "B"):
                v = [out["transfer_reference_AB"][f"{c}_seed{s}_{ds}"]["iou"] for s in ST.SEEDS]
                print(f"  {ds:14s} ref {c}: " + " ".join(f"{x:.4f}" for x in v) + f"  median {np.median(v):.4f}")
        for k, v in out["paired"]["transfer"].items():
            print(f"  paired {k}: " + "  ".join(f"{x['delta']:+.4f}[{x['ci95'][0]:+.4f},{x['ci95'][1]:+.4f}]"
                                                for x in v["per_seed"]) + f"  mean {v['mean_delta']:+.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
