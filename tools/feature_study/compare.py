#!/usr/bin/env python
"""Aggregate the nine runs into the comparison table, with paired differences and CIs.

Two things this does carefully.

**Pairing.** A, B and C for a given seed saw the same files at the same crops, so their
per-clip counts are paired observations, not independent samples. Differences are therefore
formed *within* a seed and *within* a clip before anything is pooled.

**Resampling unit.** Voxels are not independent -- neighbouring voxels belong to the same
surface -- and neither are consecutive anchors, which share most of their map. The bootstrap
resamples **contiguous blocks of anchors** on the source drive and **whole sequences /
scenes** on the target datasets, never voxels and never overlapping clips.

    python tools/feature_study/compare.py
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# study first: it pins G8C1_DATA_ROOT, which core.datasets reads at import time
from tools.feature_study import study as ST                                       # noqa: E402
from core.evaluation import scores as SC                                          # noqa: E402

N_BOOT = 2000
BLOCK = 20                      #: anchors per contiguous source block


def iou(c):
    tp, fp, fn = c[..., 0].sum(-1), c[..., 1].sum(-1), c[..., 2].sum(-1)
    return tp / np.maximum(tp + fp + fn, 1)


def block_indices(n: int, rng, block: int = BLOCK) -> np.ndarray:
    """Resample contiguous blocks until at least ``n`` anchors are drawn."""
    starts = rng.integers(0, max(n - block, 1), size=int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]


#: below this many distinct sequences/scenes, resampling them is degenerate and the
#: bootstrap falls back to contiguous blocks of anchors within the run
MIN_GROUPS = 5


def group_indices(groups: np.ndarray, rng) -> np.ndarray:
    """Resample whole sequences / scenes; fall back to contiguous anchor blocks if the
    dataset has too few of them for that to be a resampling at all (SemanticKITTI's
    validation split is one sequence)."""
    uq = np.unique(groups)
    if len(uq) < MIN_GROUPS:
        return block_indices(len(groups), rng)
    pick = rng.choice(uq, size=len(uq), replace=True)
    return np.concatenate([np.flatnonzero(groups == g) for g in pick])


def paired_ci(a: np.ndarray, b: np.ndarray, sampler, rng, n_boot: int = N_BOOT):
    """Bootstrap CI of ``IoU(a) - IoU(b)`` under a shared resampling of clips."""
    d = []
    for _ in range(n_boot):
        ix = sampler(rng)
        d.append(iou(a[ix]) - iou(b[ix]))
    d = np.asarray(d)
    return {"delta": float(iou(a) - iou(b)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_positive": float((d > 0).mean())}


def training_table():
    """Loss levels and the auxiliary-vs-occupancy gradient ratio at the shared trunk.

    The ratio is the only honest way to read the two auxiliary objectives against each
    other: both enter at nominal weight 0.5, but a KL over 25 classes and a cosine distance
    over 64 dimensions do not thereby push on the trunk equally hard, and any difference in
    completion could be that rather than the representation.
    """
    rows = {}
    for c in ST.CONDITIONS:
        for s_ in ST.SEEDS:
            p_ = os.path.join(ST.ART, f"train_{ST.run_tag(c, s_)}.json")
            if not os.path.exists(p_):
                continue
            j = json.load(open(p_))
            steps = [x for x in j["log"] if "val" not in x]
            probes = [x for x in steps if "aux_grad_norm" in x]
            vals = [x["val"] for x in j["log"] if "val" in x]
            rows[f"{c}_seed{s_}"] = {
                "n_params_total": j["n_params_total"],
                "n_params_active": j["n_params_active"],
                "final_occ_loss": float(np.mean([x["occ"] for x in steps[-100:]])),
                "final_aux_loss": float(np.mean([x["aux"] for x in steps[-100:]])),
                "final_val_occ_loss": vals[-1]["occ"] if vals else None,
                "final_val_aux_loss": vals[-1].get("aux") if vals else None,
                "mean_supervised_voxels_kl": float(np.mean([x["n_kl_voxels"] for x in steps])),
                "mean_supervised_voxels_feature": float(
                    np.mean([x["n_feature_voxels"] for x in steps])),
                "aux_grad_norm_median": (float(np.median([x["aux_grad_norm"] for x in probes]))
                                         if probes else None),
                "occ_grad_norm_median": (float(np.median([x["occ_grad_norm"] for x in probes]))
                                         if probes else None),
                "aux_over_occ_grad": (float(np.median([x["aux_grad_norm"] / max(x["occ_grad_norm"], 1e-12)
                                                       for x in probes])) if probes else None),
                "minutes": j["seconds"] / 60.0,
                "trace_sha256": j["trace_sha256"][:16],
            }
    return rows


def source_table():
    rows = {}
    for c in ST.CONDITIONS:
        for s in ST.SEEDS:
            p = os.path.join(ST.ART, "source", f"source_{ST.run_tag(c, s)}.json")
            if os.path.exists(p):
                rows[(c, s)] = json.load(open(p))
    return rows


def transfer_table():
    rows = {}
    for c in ST.CONDITIONS:
        for s in ST.SEEDS:
            for ds in ("semantickitti", "occ3d"):
                p = os.path.join(ST.ART, "transfer", c, f"eval_{ds}_past5_seed{s}.json")
                if os.path.exists(p):
                    rows[(c, s, ds)] = json.load(open(p))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    ST.require_data_root()
    rng = np.random.default_rng(0)
    src, tr = source_table(), transfer_table()
    out_training = training_table()
    out = {"n_bootstrap": a.n_boot, "block_anchors": BLOCK,
           "resampling_unit": {"source": "contiguous blocks of 20 drive-0006 anchors",
                               "transfer": f"whole sequences/scenes, or contiguous anchor "
                                           f"blocks where fewer than {MIN_GROUPS} exist"}, "source": {}, "transfer": {},
           "paired": {"source": {}, "transfer": {}}, "training": out_training,
           "env": ST.environment()}

    # ---- source ------------------------------------------------------------------- #
    for (c, s), r in sorted(src.items()):
        out["source"][f"{c}_seed{s}"] = {
            "threshold": r["occupancy_threshold"],
            **{f"{reg}_{k}": r["regions"][reg]["at_threshold"][k]
               for reg in ("full", "observed", "later")
               for k in ("iou", "precision", "recall", "tp", "fp", "fn", "tn",
                         "pred_over_gt")},
            "later_ap": r["regions"]["later"]["average_precision"],
            "later_auroc": r["regions"]["later"]["auroc"],
            "later_prevalence": r["regions"]["later"]["prevalence"],
            "full_ap": r["regions"]["full"]["average_precision"],
            "full_auroc": r["regions"]["full"]["auroc"],
            "feature_diagnostics": r.get("feature_diagnostics"),
        }
    counts = {}
    for (c, s) in src:
        for reg in ("full", "observed", "later"):
            p = os.path.join(ST.ART, "source", f"clipcounts_{ST.run_tag(c, s)}_{reg}.npz")
            if os.path.exists(p):
                with np.load(p) as z:
                    counts[(c, s, reg)] = z["counts"]
    for reg in ("later", "full", "observed"):
        for hi, lo in (("C", "B"), ("B", "A"), ("C", "A")):
            per_seed = []
            for s in ST.SEEDS:
                if (hi, s, reg) in counts and (lo, s, reg) in counts:
                    n = counts[(hi, s, reg)].shape[0]
                    per_seed.append(paired_ci(counts[(hi, s, reg)], counts[(lo, s, reg)],
                                              lambda r, n=n: block_indices(n, r), rng,
                                              a.n_boot))
            if per_seed:
                out["paired"]["source"][f"{hi}-{lo}_{reg}"] = {
                    "per_seed": per_seed,
                    "mean_delta": float(np.mean([x["delta"] for x in per_seed])),
                    "seed_spread": [float(min(x["delta"] for x in per_seed)),
                                    float(max(x["delta"] for x in per_seed))]}

    # ---- transfer ----------------------------------------------------------------- #
    for (c, s, ds), r in sorted(tr.items()):
        cb = r["semantic"]["completion"]
        out["transfer"][f"{c}_seed{s}_{ds}"] = {
            "n_clips": r["n_clips"], "threshold": r["threshold"],
            "iou": cb["binary_iou"], "precision": cb["binary_precision"],
            "recall": cb["binary_recall"], "ssc_miou": cb["ssc_miou"],
            "ap_over_prevalence": r["methods"]["completion"]["full"]["ap_over_prevalence"],
            "auroc": r["methods"]["completion"]["full"]["auroc"],
            "pred_over_gt": r["baselines"].get("mapper_native", {}).get("pred_over_gt"),
            "mapper_native_iou": r["baselines"]["mapper_native"]["binary_iou"],
            "mapper_dilate_iou": r["baselines"]["mapper_dilate"]["binary_iou"]}
    tc = {}
    for (c, s, ds) in tr:
        p = os.path.join(ST.ART, "transfer", c, f"scores_{ds}_past5_seed{s}_completion.npz")
        if not os.path.exists(p):
            continue
        with np.load(p) as z:
            blk = {k[len("full_"):]: z[k] for k in z.files if k.startswith("full_")}
        tau = tr[(c, s, ds)]["threshold"]
        tc[(c, s, ds)] = (SC.per_clip_counts(blk, tau), blk["group"])
    for ds in ("semantickitti", "occ3d"):
        for hi, lo in (("C", "B"), ("B", "A"), ("C", "A")):
            per_seed = []
            for s in ST.SEEDS:
                if (hi, s, ds) in tc and (lo, s, ds) in tc:
                    ch, g = tc[(hi, s, ds)]
                    cl, _ = tc[(lo, s, ds)]
                    per_seed.append(paired_ci(ch, cl,
                                              lambda r, g=g: group_indices(g, r), rng,
                                              a.n_boot))
            if per_seed:
                out["paired"]["transfer"][f"{hi}-{lo}_{ds}"] = {
                    "per_seed": per_seed,
                    "mean_delta": float(np.mean([x["delta"] for x in per_seed])),
                    "seed_spread": [float(min(x["delta"] for x in per_seed)),
                                    float(max(x["delta"] for x in per_seed))]}

    ST.write_json(os.path.join(ST.ART, "comparison.json"), out)
    _print(out)
    return 0


def _print(out):
    def med(pref, key):
        v = [r[key] for k, r in out["source"].items() if k.startswith(pref) and key in r]
        return float(np.median(v)) if v else float("nan")
    if out.get("training"):
        print("\n  training")
        print(f"  {'run':12s} {'occ':>8s} {'aux':>8s} {'|g_aux|/|g_occ|':>16s} "
              f"{'sup voxels':>11s} {'active params':>14s}")
        for k, v in sorted(out["training"].items()):
            r = v["aux_over_occ_grad"]
            n = (v["mean_supervised_voxels_feature"] if k.startswith("C")
                 else v["mean_supervised_voxels_kl"])
            print(f"  {k:12s} {v['final_occ_loss']:8.4f} {v['final_aux_loss']:8.4f} "
                  f"{(f'{r:.4f}' if r else '-'):>16s} {n:11.0f} {v['n_params_active']:14d}")
    print("\n  source (KITTI-360 drive 0006), median over seeds")
    print(f"  {'cond':4s} {'tau':>8s} {'full IoU':>9s} {'LATER IoU':>10s} {'LATER P':>8s} "
          f"{'LATER R':>8s} {'obs IoU':>8s} {'LATER AP/prev':>13s}")
    for c in ST.CONDITIONS:
        pv = med(c, "later_prevalence")
        print(f"  {c:4s} {med(c,'threshold'):+8.4f} {med(c,'full_iou'):9.4f} "
              f"{med(c,'later_iou'):10.4f} {med(c,'later_precision'):8.4f} "
              f"{med(c,'later_recall'):8.4f} {med(c,'observed_iou'):8.4f} "
              f"{med(c,'later_ap')/max(pv,1e-9):13.2f}")
    for k, v in out["paired"]["source"].items():
        print(f"  paired {k:16s} delta {v['mean_delta']:+.4f}  seeds "
              f"[{v['seed_spread'][0]:+.4f}, {v['seed_spread'][1]:+.4f}]")
    if out["transfer"]:
        print("\n  transfer (past5), median over seeds")
        for ds in ("semantickitti", "occ3d"):
            for c in ST.CONDITIONS:
                v = [r["iou"] for k, r in out["transfer"].items()
                     if k.startswith(c) and k.endswith(ds)]
                m = [r["ssc_miou"] for k, r in out["transfer"].items()
                     if k.startswith(c) and k.endswith(ds)]
                if v:
                    print(f"  {ds:14s} {c}  IoU {np.median(v):.4f}  "
                          f"(seeds {min(v):.4f}-{max(v):.4f})  SSC mIoU {np.median(m):.4f}")
        for k, v in out["paired"]["transfer"].items():
            print(f"  paired {k:22s} delta {v['mean_delta']:+.4f}  seeds "
                  f"[{v['seed_spread'][0]:+.4f}, {v['seed_spread'][1]:+.4f}]")


if __name__ == "__main__":
    raise SystemExit(main())
