#!/usr/bin/env python
"""Gate 8C-1: collect three seeds x two targets x three protocols into one result file.

Adds the three-seed mean and standard deviation, scene-level paired bootstrap intervals
over Gate 6's own resampling units, OccAny's official scene-completion counts on our
predictions, and the brief's decision rules applied mechanically.

    python tools/gate8c1/aggregate.py
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, SEEDS                                        # noqa: E402
from gates.scale_gate.config import write_json                                         # noqa: E402
from gates.gate8a import boot as B, scores as SC                                       # noqa: E402
from gates.gate8c1 import occany_eval as OE                                            # noqa: E402

G6 = os.path.join(REPO_ROOT, "artifacts", "gate6")
TARGETS = ("semantickitti", "occ3d")
MODES = ("past5", "stream", "occany_fwd")
CAUSAL = {"past5": True, "stream": True, "occany_fwd": False}


def load(p):
    z = np.load(p, allow_pickle=False); return {k: z[k] for k in z.files}


def blk(ds, tag, name, region):
    p = os.path.join(ART, f"scores_{ds}_{tag}_{name}.npz")
    if not os.path.exists(p):
        return None
    z = load(p)
    return {k[len(region) + 1:]: v for k, v in z.items() if k.startswith(region + "_")}


def one(ds, mode, seed):
    tag = f"{mode}_seed{seed}"
    p = os.path.join(ART, f"eval_{ds}_{tag}.json")
    if not os.path.exists(p):
        return None
    ev = json.load(open(p))
    tau = ev["threshold"]
    fb = blk(ds, tag, "completion", "full"); eb = blk(ds, tag, "completion", "edit")
    mb = blk(ds, tag, "mapper", "full")
    out = {"eval": {k: ev[k] for k in ("dataset", "mode", "seed", "checkpoint", "threshold",
                                       "semantic_threshold", "causal", "n_clips", "n_skipped",
                                       "seconds", "peak_gpu_gib", "latency_ms")},
           "threshold_free": {"completion_full": SC.summarize(fb),
                              "completion_edit": SC.summarize(eb),
                              "mapper_full": SC.summarize(mb)},
           "methods": {}, "semantic": ev["semantic"],
           "newly_completed": ev["newly_completed"], "comparisons": {}}
    clips = [str(c) for c in fb["clip_id"]]
    counts = {"completion": SC.per_clip_counts(fb, tau),
              "mapper_calibrated": SC.per_clip_counts(mb, tau)}
    clip_of = {k: clips for k in counts}
    bc = load(os.path.join(ART, f"bincounts_{ds}_{tag}.npz"))
    bclips = [str(c) for c in bc["clip_id"]]
    assert bclips == clips
    for k in [x for x in bc if x.startswith("pred_")]:
        counts[k[5:]] = bc[k]; clip_of[k[5:]] = bclips
    nval = float(bc["n_valid"].sum()); ngt = float(bc["n_gt"].sum())
    for k, c in counts.items():
        b = np.asarray(c, np.int64).sum(0)
        out["methods"][k] = {"binary_iou": float(b[0] / max(b.sum(), 1)),
                             "precision": float(b[0] / max(b[0] + b[1], 1)),
                             "recall": float(b[0] / max(b[0] + b[2], 1)),
                             "density": float((b[0] + b[1]) / nval),
                             "pred_over_gt_volume": float((b[0] + b[1]) / ngt),
                             "tp": int(b[0]), "fp": int(b[1]), "fn": int(b[2])}
    rs = [k for k in counts if k.startswith("random_editable_s")]
    ious = [out["methods"][k]["binary_iou"] for k in rs]
    out["methods"]["random_editable_mean"] = {
        "binary_iou": float(np.mean(ious)), "binary_iou_sd": float(np.std(ious, ddof=1)),
        "n_seeds": len(rs)}
    for cond, nm in (("B-R", "frozen_5frame_raw"), ("B-D", "frozen_5frame_dil")):
        z = load(os.path.join(G6, f"counts_{ds}_{cond}.npz"))
        cid, bb, pc = B.counts_from_gate6_block(z)
        counts[nm] = bb; clip_of[nm] = cid
        s = bb.sum(0)
        out["methods"][nm] = {"binary_iou": float(s[0] / max(bb.sum(), 1)),
                              "precision": float(s[0] / max(s[0] + s[1], 1)),
                              "recall": float(s[0] / max(s[0] + s[2], 1)),
                              "pred_over_gt_volume": float(s[:2].sum() / max(s[[0, 2]].sum(), 1))}
    for other in [k for k in counts if k != "completion"]:
        if other.startswith("random_editable_s") and other != "random_editable_s0":
            continue
        out["comparisons"][f"completion vs {other}"] = B.paired_binary(
            ds, REPO_ROOT, clip_of[other], counts[other], clips, counts["completion"])
    sc = {}
    for nm in list(ev["semantic"]):
        p2 = os.path.join(ART, f"counts_{ds}_{tag}_{nm}.npz")
        if os.path.exists(p2):
            sc[nm] = B.counts_from_gate6_block(load(p2))
    for cond, nm in (("B-R", "frozen_5frame_raw"), ("B-D", "frozen_5frame_dil")):
        sc[nm] = B.counts_from_gate6_block(load(os.path.join(G6, f"counts_{ds}_{cond}.npz")))
        g6 = json.load(open(os.path.join(G6, f"summary_{ds}.json")))["conditions"][cond]
        out["semantic"][nm] = {"binary_iou": g6["binary_iou_pooled"], "ssc_miou": g6["ssc_miou"],
                               "tp_accuracy": g6["tp_conditioned"]["top1_accuracy"],
                               "coverage_miss": g6["decomposition"]["coverage_miss_fraction"],
                               "naming_error": g6["decomposition"]["naming_error_fraction"],
                               "per_class_iou": {k: v["iou"] for k, v in g6["per_class"].items()}}
    if "completion" in sc:
        ca, ba, pa = sc["completion"]
        for nm, (cb, bb, pb) in sc.items():
            if nm != "completion":
                out["comparisons"][f"SSC completion vs {nm}"] = B.paired_binary(
                    ds, REPO_ROOT, cb, bb, ca, ba, pb, pa)
    # ---- OccAny's official SC counts on our prediction ------------------------------
    if OE.available():
        from gates.gate6 import vocab
        v = vocab.load(ds)
        z = load(os.path.join(ART, f"counts_{ds}_{tag}_completion.npz"))
        b = z["binary"].sum(0)
        out["occany_official_metric"] = {
            "note": "OccAny's SSCMetrics.get_score_completion, verified in tests to agree "
                    "with our counting on the same arrays",
            "ours_tp_fp_fn": [int(b[0]), int(b[1]), int(b[2])],
            **OE.rates(int(b[0]), int(b[1]), int(b[2]))}
    return out


def main() -> int:
    man = json.load(open(os.path.join(ART, "frozen_manifest.json")))
    res = {"manifest": {k: man[k] for k in ("gate", "claim", "code_commit", "frozen_at",
                                            "training_data", "target_datasets_excluded_until_now",
                                            "target_validation_all_pass", "firewall_proof")},
           "seeds": man["seeds"], "published_occany": OE.PUBLISHED_5FRAME,
           "runs": {}, "per_target": {}, "decision": {}}
    for ds in TARGETS:
        res["runs"][ds] = {}
        for mode in MODES:
            rows = {s: one(ds, mode, s) for s in SEEDS}
            rows = {s: r for s, r in rows.items() if r}
            if not rows:
                continue
            res["runs"][ds][mode] = rows
            agg = {}
            for key in ("binary_iou", "precision", "recall", "pred_over_gt_volume"):
                vals = [r["methods"]["completion"][key] for r in rows.values()]
                agg[key] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1))
                            if len(vals) > 1 else 0.0, "per_seed": vals,
                            "median": float(np.median(vals))}
            for key in ("average_precision", "ap_over_prevalence", "auroc", "prevalence"):
                vals = [r["threshold_free"]["completion_full"][key] for r in rows.values()]
                agg[key] = {"mean": float(np.mean(vals)),
                            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                            "per_seed": vals, "median": float(np.median(vals))}
            vals = [r["semantic"]["completion"]["ssc_miou"] for r in rows.values()]
            agg["ssc_miou"] = {"mean": float(np.mean(vals)),
                               "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                               "per_seed": vals, "median": float(np.median(vals))}
            base = {}
            for nm in ("mapper_native", "mapper_dilate", "all_valid_occupied", "editable_fill",
                       "random_editable_mean", "frozen_5frame_raw", "frozen_5frame_dil",
                       "completion_occany_pool", "completion_occany_vote"):
                vv = [r["methods"][nm]["binary_iou"] for r in rows.values() if nm in r["methods"]]
                if vv:
                    base[nm] = {"mean": float(np.mean(vv)), "median": float(np.median(vv))}
            res["per_target"].setdefault(ds, {})[mode] = {
                "n_seeds": len(rows), "causal": CAUSAL[mode], "completion": agg,
                "baselines": base,
                "n_clips": rows[list(rows)[0]]["eval"]["n_clips"]}
    # ---- decision ---------------------------------------------------------------------
    for ds in TARGETS:
        pt = res["per_target"].get(ds, {}).get("past5")
        if not pt:
            continue
        pub = OE.PUBLISHED_5FRAME[ds]["sc_iou"]
        med = pt["completion"]["binary_iou"]["median"]
        strongest = max((k for k in pt["baselines"] if not k.startswith("completion")),
                        key=lambda k: pt["baselines"][k]["median"])
        cis = []
        for s, r in res["runs"][ds]["past5"].items():
            c = r["comparisons"].get(f"completion vs {strongest}")
            if c:
                cis.append(c["binary_iou"])
        beats = bool(cis and all(x["difference"] > 0 and x["ci"][0] > 0 for x in cis))
        sem_med = pt["completion"]["ssc_miou"]["median"]
        sem_base = max((v["ssc_miou"] for k, v in res["runs"][ds]["past5"][list(res["runs"][ds]["past5"])[0]]["semantic"].items()
                        if k != "completion" and v.get("ssc_miou") is not None), default=None)
        res["decision"][ds] = {
            "published_occany_sc_iou": pub, "our_median_sc_iou": med,
            "fraction_of_occany": med / pub,
            "strongest_non_completion_baseline": strongest,
            "strongest_baseline_sc_iou": pt["baselines"][strongest]["median"],
            "beats_strongest_baseline_all_seeds_ci_excludes_zero": beats,
            "ap_over_prevalence_median": pt["completion"]["ap_over_prevalence"]["median"],
            "auroc_median": pt["completion"]["auroc"]["median"],
            "pred_over_gt_volume_median": pt["completion"]["pred_over_gt_volume"]["median"],
            "ssc_miou_median": sem_med, "strongest_non_completion_ssc_miou": sem_base,
            "semantic_beats_baseline": bool(sem_base is not None and sem_med > sem_base)}
    d = res["decision"]
    if len(d) == 2:
        a, b = (d[t] for t in TARGETS)
        match_one = any(x["our_median_sc_iou"] >= x["published_occany_sc_iou"] for x in (a, b))
        ninety_other = all(x["fraction_of_occany"] >= 0.9 for x in (a, b))
        beats_both = all(x["beats_strongest_baseline_all_seeds_ci_excludes_zero"] for x in (a, b))
        sem_both = all(x["semantic_beats_baseline"] for x in (a, b))
        rank_ok = all(x["ap_over_prevalence_median"] >= 1.5 and x["auroc_median"] >= 0.65
                      for x in (a, b))
        no_inflate = all(x["pred_over_gt_volume_median"] <= 2.0 for x in (a, b))
        near_chance = any(x["ap_over_prevalence_median"] < 1.2 or x["auroc_median"] < 0.55
                          for x in (a, b))
        far_below = all(x["fraction_of_occany"] < 0.6 for x in (a, b))
        if match_one and ninety_other and beats_both and sem_both:
            verdict = "STRONG PASS"
        elif ninety_other and rank_ok and no_inflate:
            verdict = "MINIMUM CONTINUE"
        elif near_chance or far_below:
            verdict = "FAIL"
        else:
            verdict = "FAIL (criteria for MINIMUM CONTINUE not met)"
        res["decision"]["overall"] = {
            "verdict": verdict, "match_or_beat_occany_on_at_least_one": match_one,
            "at_least_90pct_of_occany_on_both": ninety_other,
            "beats_strongest_baseline_on_both_ci_excludes_zero": beats_both,
            "semantic_beats_baseline_on_both": sem_both,
            "ranking_ok_on_both (AP/prev>=1.5, AUROC>=0.65)": rank_ok,
            "no_volume_inflation (pred/GT<=2)": no_inflate,
            "near_chance_on_either": near_chance, "far_below_occany_on_both": far_below}
    write_json(os.path.join(ART, "gate8c1_results.json"), res)
    print("== Gate 8C-1")
    for ds in TARGETS:
        for mode in MODES:
            pt = res["per_target"].get(ds, {}).get(mode)
            if not pt:
                continue
            c = pt["completion"]
            print(f"  {ds:14s} {mode:11s} ({'causal' if pt['causal'] else 'NON-causal'}, "
                  f"{pt['n_seeds']} seeds, {pt['n_clips']} clips) SC IoU "
                  f"{c['binary_iou']['median']:.4f}±{c['binary_iou']['sd']:.4f} "
                  f"P {c['precision']['median']:.4f} R {c['recall']['median']:.4f} "
                  f"AP/prev {c['ap_over_prevalence']['median']:.2f} "
                  f"AUROC {c['auroc']['median']:.4f} pred/GT {c['pred_over_gt_volume']['median']:.2f} "
                  f"SSC {c['ssc_miou']['median']:.4f}")
    if "overall" in res["decision"]:
        print(f"\n  VERDICT: {res['decision']['overall']['verdict']}")
        for k, v in res["decision"]["overall"].items():
            if k != "verdict":
                print(f"     {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
