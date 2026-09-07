#!/usr/bin/env python
"""Gate 8A results: the 2x2 ablation table, the source-only selection table, the single
locked KITTI-360 evaluation, and the decision rules applied to them mechanically.

Every interval is a paired, scene-aware bootstrap over Gate 6's own resampling units
(10 000 draws, seed 0), and every threshold-dependent count is recomputed from the stored
score histograms rather than from a binarized artifact.

    python tools/gate8a/aggregate.py
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT, write_json                              # noqa: E402
from gates.gate8a import boot as B, scores as SC                                       # noqa: E402

ART = os.path.join(REPO_ROOT, "artifacts", "gate8a")
G6 = os.path.join(REPO_ROOT, "artifacts", "gate6")
G8 = os.path.join(REPO_ROOT, "artifacts", "gate8")
SRC = ("semantickitti", "occ3d")
CELLS = {"cellA": ("occ_centred", "focal_dice"), "cellB": ("uniform", "focal_dice"),
         "cellC": ("occ_centred", "bce"), "cellD": ("uniform", "bce")}


def load(p):
    z = np.load(p, allow_pickle=False); return {k: z[k] for k in z.files}


def blk(source, tag, name, region):
    z = load(os.path.join(ART, f"scores_{source}_{tag}_{name}.npz"))
    return {k[len(region) + 1:]: v for k, v in z.items() if k.startswith(region + "_")}


def train_meta(cell):
    for p in (os.path.join(ART, f"train_{cell}_*.json"),):
        g = glob.glob(p)
        if g:
            return json.load(open(g[0]))
    if cell == "cellA":
        return json.load(open(os.path.join(G8, "train_completion.json")))
    return None


def _val_kl(meta):
    if not meta:
        return None
    vs = [e["val"] for e in meta["log"] if isinstance(e, dict) and "val" in e]
    best = meta.get("best_step")
    for v in vs:
        if v.get("step") == best:
            return v.get("sem_kl")
    return vs[-1].get("sem_kl") if vs else None


def ablation_rows(tag, candidates):
    rows = []
    for name in candidates:
        cell = name.split("_")[0]
        meta = train_meta(cell)
        r = {"cell": name, "sampler": CELLS[cell][0], "loss": CELLS[cell][1],
             "checkpoint": candidates[name], "ap": {}, "prevalence": {},
             "ap_over_prevalence": {}, "auroc": {}, "edit_ap": {}, "edit_prevalence": {},
             "full_iou_at_zero": {}, "full_iou_at_tau": {}, "edit_at_tau": {},
             "teacher_kl_val": _val_kl(meta),
             "train_seconds": meta and meta.get("seconds"),
             "train_peak_gpu_gib": meta and meta.get("peak_gpu_gib"),
             "best_step": meta and meta.get("best_step")}
        sws = {}
        for s in SRC:
            f = blk(s, tag, name, "full"); e = blk(s, tag, name, "edit")
            sw = SC.sweep(f["pos"], f["neg"]); sws[s] = sw
            swe = SC.sweep(e["pos"], e["neg"])
            r["ap"][s] = SC.average_precision(sw)
            r["prevalence"][s] = float(sw["n_pos"] / sw["n_tot"])
            r["ap_over_prevalence"][s] = r["ap"][s] / r["prevalence"][s]
            r["auroc"][s] = SC.auroc(sw)
            r["edit_ap"][s] = SC.average_precision(swe)
            r["edit_prevalence"][s] = float(swe["n_pos"] / swe["n_tot"])
            r["full_iou_at_zero"][s] = SC.at_threshold(sw, 0.0)
            r["brier_" + s] = float(f["sse"].sum() / f["n_scored"].sum())
            r["ece_" + s] = SC.ece(f["cal_n"], f["cal_p"], f["cal_y"])
        r["macro_ap"] = float(np.mean([r["ap"][s] for s in SRC]))
        iou = np.mean([sws[s]["iou"] for s in SRC], axis=0)
        j = int(np.argmax(iou))
        r["tau"] = float(sws[SRC[0]]["tau"][j])
        r["full_iou_at_tau"] = {s: SC.at_threshold(sws[s], r["tau"]) for s in SRC}
        r["full_iou_at_tau"]["macro"] = float(iou[j])
        r["full_iou_at_zero"]["macro"] = float(np.mean(
            [r["full_iou_at_zero"][s]["iou"] for s in SRC]))
        r["edit_at_tau"] = {s: SC.at_threshold(SC.sweep(blk(s, tag, name, "edit")["pos"],
                                                        blk(s, tag, name, "edit")["neg"]),
                                               r["tau"]) for s in SRC}
        r["ap_exceeds_prevalence_on_both"] = all(r["ap"][s] > r["prevalence"][s] for s in SRC)
        rows.append(r)
    return rows


def heldout(sel):
    """The one locked KITTI-360 read-out and every declared comparison."""
    ds, tag = "kitti360", "locked"
    p = os.path.join(ART, f"eval_{ds}_{tag}.json")
    if not os.path.exists(p):
        return None
    ev = json.load(open(p))
    name = ev["locked"]; tau = ev["threshold"]; mtau = ev["mapper_threshold"]
    out = {"eval": ev, "threshold": tau, "mapper_threshold": mtau,
           "unadapted_transfer": True, "methods": {}, "comparisons": {}}
    fb = blk(ds, tag, name, "full"); mb = blk(ds, tag, "mapper", "full")
    eb = blk(ds, tag, name, "edit"); meb = blk(ds, tag, "mapper", "edit")
    swf, swm = SC.sweep(fb["pos"], fb["neg"]), SC.sweep(mb["pos"], mb["neg"])
    out["threshold_free"] = {
        "completion_full": SC.summarize(fb), "completion_edit": SC.summarize(eb),
        "mapper_full": SC.summarize(mb), "mapper_edit": SC.summarize(meb)}
    clips = [str(c) for c in fb["clip_id"]]
    counts = {"completion": SC.per_clip_counts(fb, tau),
              "mapper_calibrated": SC.per_clip_counts(mb, mtau),
              "mapper_score_ge_zero": SC.per_clip_counts(mb, 0.0)}
    clip_of = {k: clips for k in counts}
    bc = load(os.path.join(ART, f"bincounts_{ds}_{tag}.npz"))
    bclips = [str(c) for c in bc["clip_id"]]
    assert bclips == clips, "score blocks and baseline counts disagree on the anchor order"
    for k in [x for x in bc if x.startswith("pred_")]:
        counts[k[len("pred_"):]] = bc[k]
        clip_of[k[len("pred_"):]] = bclips
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
        "binary_iou_range": [float(min(ious)), float(max(ious))], "n_seeds": len(rs),
        "matched_density": float(np.mean(bc["random_q"])) if "random_q" in bc else None}
    for cond, nm in (("B-R", "frozen_5frame_raw"), ("B-D", "frozen_5frame_dil")):
        z = load(os.path.join(G6, f"counts_{ds}_{cond}.npz"))
        cid, bb, pc = B.counts_from_gate6_block(z)
        counts[nm] = bb; out["methods"][nm] = {
            "binary_iou": float(bb.sum(0)[0] / max(bb.sum(), 1)),
            "precision": float(bb.sum(0)[0] / max(bb.sum(0)[:2].sum(), 1)),
            "recall": float(bb.sum(0)[0] / max(bb.sum(0)[[0, 2]].sum(), 1)),
            "pred_over_gt_volume": float(bb.sum(0)[:2].sum() / max(bb.sum(0)[[0, 2]].sum(), 1))}
        clip_of[nm] = cid
    for other in ("mapper_native", "mapper_calibrated", "mapper_score_ge_zero",
                  "mapper_dilate", "editable_fill",
                  "all_valid_occupied", "random_editable_s0", "frozen_5frame_raw",
                  "frozen_5frame_dil"):
        if other in counts:
            out["comparisons"][f"completion vs {other}"] = B.paired_binary(
                ds, REPO_ROOT, clip_of[other], counts[other], clips, counts["completion"])
    # ---- semantics -------------------------------------------------------------
    sem = ev.get("semantic", {})
    out["semantic"] = sem
    out["newly_completed"] = ev.get("newly_completed")
    sc = {}
    for nm in ("completion", "mapper_native", "mapper_dilate", "mapper_calibrated"):
        p2 = os.path.join(ART, f"counts_{ds}_{tag}_{nm}.npz")
        if os.path.exists(p2):
            sc[nm] = B.counts_from_gate6_block(load(p2))
    for cond, nm in (("B-R", "frozen_5frame_raw"), ("B-D", "frozen_5frame_dil")):
        sc[nm] = B.counts_from_gate6_block(load(os.path.join(G6, f"counts_{ds}_{cond}.npz")))
        g6 = json.load(open(os.path.join(G6, f"summary_{ds}.json")))["conditions"][cond]
        sem[nm] = {"binary_iou": g6["binary_iou_pooled"], "ssc_miou": g6["ssc_miou"],
                   "tp_accuracy": g6["tp_conditioned"]["top1_accuracy"],
                   "coverage_miss": g6["decomposition"]["coverage_miss_fraction"],
                   "naming_error": g6["decomposition"]["naming_error_fraction"],
                   "per_class_iou": {k: v["iou"] for k, v in g6["per_class"].items()}}
    if "completion" in sc:
        # The semantic pass compares `final >= tau` on float32 scores; the histogram
        # compares the bin index, and `final - (-16.0)` loses ~1.9e-6 of float32
        # precision, so scores within one ULP below tau fall in the tau bin. That is a
        # storage-precision limit, not a logic difference: it must stay far below any
        # difference the report leans on, so it is bounded rather than ignored.
        want = counts["completion"].sum(0).astype(np.float64)
        got = sc["completion"][1].sum(0).astype(np.float64)
        rel = float(np.max(np.abs(want - got) / np.maximum(want, 1)))
        assert rel < 1e-5, (
            f"semantic counts {tuple(got.astype(int))} disagree with the score histogram "
            f"{tuple(want.astype(int))} at tau={tau} by {rel:.2e} relative: the two "
            f"binarizations have drifted apart")
        out["binarization_agreement"] = {
            "max_relative_disagreement": rel,
            "voxels": int(np.max(np.abs(want - got))),
            "cause": "float32 bin-edge precision in the score histogram (~1.9e-6 near "
                     "the +16 offset); scores within one ULP below tau bin as >= tau"}
        ca, ba, pa = sc["completion"]
        for nm, (cb, bb, pb) in sc.items():
            if nm == "completion":
                continue
            out["comparisons"][f"SSC completion vs {nm}"] = B.paired_binary(
                ds, REPO_ROOT, cb, bb, ca, ba, pb, pa)
    return out


def decide(res):
    h = res.get("heldout")
    if not h:
        return None
    m = h["methods"]; tf = h["threshold_free"]["completion_full"]
    ci = h["comparisons"]
    def beats(k):
        c = ci.get(f"completion vs {k}")
        return None if not c else (c["binary_iou"]["difference"] > 0
                                   and c["binary_iou"]["ci"][0] > 0)
    ap, prev = tf["average_precision"], tf["prevalence"]
    geo = {
        "ap": ap, "prevalence": prev, "ap_over_prevalence": ap / prev,
        "ap_meaningfully_above_prevalence": ap > 1.5 * prev,
        "beats_incremental_mapper": beats("mapper_native"),
        "beats_calibrated_mapper": beats("mapper_calibrated"),
        "beats_strongest_dilation": beats("mapper_dilate"),
        "beats_editable_fill": beats("editable_fill"),
        "beats_matched_density_random": beats("random_editable_s0"),
        "beats_all_valid_occupied": beats("all_valid_occupied"),
        "pred_over_gt_volume": m["completion"]["pred_over_gt_volume"],
        "volume_ratio_is_reasonable": m["completion"]["pred_over_gt_volume"] < 2.0}
    geo["passed"] = bool(geo["ap_meaningfully_above_prevalence"]
                         and geo["beats_incremental_mapper"]
                         and geo["beats_calibrated_mapper"]
                         and geo["beats_strongest_dilation"]
                         and geo["beats_editable_fill"]
                         and geo["beats_matched_density_random"]
                         and geo["volume_ratio_is_reasonable"])
    sem = h.get("semantic", {})
    best_non_completion = max(
        ((k, v.get("ssc_miou")) for k, v in sem.items()
         if k != "completion" and v.get("ssc_miou") is not None),
        key=lambda kv: kv[1], default=(None, None))
    sc = ci.get(f"SSC completion vs {best_non_completion[0]}") if best_non_completion[0] else None
    nc = h.get("newly_completed") or {}
    tpacc = sem.get("completion", {}).get("tp_accuracy")
    semd = {"completion_ssc_miou": sem.get("completion", {}).get("ssc_miou"),
            "strongest_non_completion": best_non_completion[0],
            "strongest_non_completion_ssc_miou": best_non_completion[1],
            "beats_it": None if not sc else (sc["ssc_miou"]["difference"] > 0
                                             and sc["ssc_miou"]["ci"][0] > 0),
            "newly_completed_semantic_accuracy": nc.get("semantic_accuracy"),
            "all_tp_semantic_accuracy": tpacc,
            "new_tp_carry_information": (
                None if nc.get("semantic_accuracy") is None else
                nc["semantic_accuracy"] > 1.0 / 19)}
    semd["passed"] = bool(semd["beats_it"] and semd["new_tp_carry_information"])
    return {"geometry": geo, "semantics": semd}


def main() -> int:
    res = {"bootstrap": {"n_boot": B.N_BOOT, "seed": B.SEED},
           "sampler_stats": json.load(open(os.path.join(ART, "sampler_stats.json")))
           if os.path.exists(os.path.join(ART, "sampler_stats.json")) else None}
    selp = os.path.join(ART, "selection.json")
    if os.path.exists(selp):
        sel = json.load(open(selp)); res["selection"] = sel
        cands = {r["candidate"]: r["checkpoint"] for r in sel["candidates"]
                 if r["candidate"] != "mapper"}
        res["ablation"] = ablation_rows(sel["tag"], cands)
        # the mapper, scored the same way, as the reference row
        res["mapper_source"] = {s: {"ap": SC.average_precision(
            SC.sweep(blk(s, sel["tag"], "mapper", "full")["pos"],
                     blk(s, sel["tag"], "mapper", "full")["neg"]))} for s in SRC}
        res["heldout"] = heldout(sel)
        res["decision"] = decide(res)
    if os.path.exists(os.path.join(ART, "frozen_manifest.json")):
        res["frozen_manifest"] = json.load(open(os.path.join(ART, "frozen_manifest.json")))
    write_json(os.path.join(ART, "gate8a_results.json"), res)
    for r in res.get("ablation", []):
        print(f"{r['cell']:22s} {r['sampler']:11s} {r['loss']:10s} macroAP {r['macro_ap']:.4f} "
              f"tau {r['tau']:+.3f} macroIoU@tau {r['full_iou_at_tau']['macro']:.4f} "
              f"macroIoU@0 {r['full_iou_at_zero']['macro']:.4f}")
    h = res.get("heldout")
    if h:
        print("\n== KITTI-360 (unadapted transfer, locked)")
        for k, m in sorted(h["methods"].items(), key=lambda kv: -kv[1]["binary_iou"]):
            print(f"   {k:26s} IoU {m['binary_iou']:.4f} "
                  + (f"P {m['precision']:.4f} R {m['recall']:.4f} "
                     f"pred/gt {m['pred_over_gt_volume']:.2f}" if "precision" in m else ""))
        for k, c in h["comparisons"].items():
            if c:
                key = "ssc_miou" if k.startswith("SSC") else "binary_iou"
                print(f"   D {k}: {c[key]['difference']:+.4f} {c[key]['ci']}")
        d = res["decision"]
        print(f"\n   geometry passed: {d['geometry']['passed']}   "
              f"semantics passed: {d['semantics']['passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
