#!/usr/bin/env python
"""Gate 8B results: three leave-one-dataset-out folds, two evaluation settings each, every
declared baseline, paired scene-aware bootstrap intervals over Gate 6's own units, the
OccAny reference comparison, and the outcome classification applied mechanically.

    python tools/gate8b/aggregate.py
"""
from __future__ import annotations
import glob, hashlib, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT                                               # noqa: E402
from gates.scale_gate.config import write_json                                         # noqa: E402
from gates.gate8a import boot as B, scores as SC                                       # noqa: E402
from gates.gate8b.sources import FOLDS, K360_TRAIN_DRIVES, K360_VAL_DRIVE              # noqa: E402

G6 = os.path.join(REPO_ROOT, "artifacts", "gate6")
G8A = os.path.join(REPO_ROOT, "artifacts", "gate8a")
TARGETS = ("semantickitti", "occ3d", "kitti360")
OCCANY_5FRAME = {"semantickitti": {"precision": 0.3679, "recall": 0.4670, "sc_iou": 0.2591},
                 "occ3d": {"precision": 0.3609, "recall": 0.4039, "sc_iou": 0.2355}}
PROTOCOL = {
    "semantickitti": [
        ("target split", "SemanticKITTI seq 08 (val)", "SemanticKITTI seq 08 (val)", True),
        ("camera", "image_2 (left colour)", "image_2 (left colour)", True),
        ("five-frame temporal sampling",
         "target frame FIRST; views at native offsets +0,+10,+20,+30,+40 (10-frame video at "
         "stride 5, recon_view_idx [0,2,4,6,8]); anchors every 5 frames while begin+50 <= end",
         "target frame LAST; frames at offsets -20,-15,-10,-5,0 (stride 5); anchors every 5 "
         "frames from frame 20", False),
        ("voxel grid / resolution", "256x256x32 @ 0.2 m, origin (0,-25.6,-2)",
         "256x256x32 @ 0.2 m, origin (0,-25.6,-2)", True),
        ("coordinate frame", "velodyne frame of the target frame", "velodyne frame of the anchor", True),
        ("valid / unknown mask", "label 255 excluded (predict[target==255] := empty)",
         "keep = target != 255", True),
        ("occupied / free", "any non-empty class = occupied", "any non-empty class = occupied", True),
        ("majority pooling", "apply_majority_pooling, separate mode (official code)",
         "gate8b.pooling, checked bit-for-bit against the official function", True),
        ("metric aggregation", "pooled TP/FP/FN over all evaluated samples",
         "pooled TP/FP/FN over all official clips (Gate 6 metric code)", True),
        ("metric-scale source", "OccAny's own (reconstruction network)",
         "frozen G51-B: calibrated-FOV MoGe-2 median gauge from the five frames", False)],
    "occ3d": [
        ("target split", "nuScenes val (Occ3D-nuScenes), CAM_FRONT",
         "nuScenes val (Occ3D-nuScenes), CAM_FRONT, 150 scenes / 1182 clips", True),
        ("camera", "CAM_FRONT", "CAM_FRONT", True),
        ("five-frame temporal sampling",
         "target sample FIRST; 5 samples at frame_interval 2 (~1.0 s apart, ~4 s forward)",
         "target sample LAST; 5 samples at 0.5 s spacing (2 s backward)", False),
        ("voxel grid / resolution", "200x200x16 @ 0.4 m, origin (-40,-40,-1)",
         "200x200x16 @ 0.4 m (prediction at 0.2 m, any-sub-voxel reduction)", True),
        ("coordinate frame", "ego frame of the reference sample", "ego frame of the anchor", True),
        ("valid / unknown mask", "mask_camera applied, mask_lidar not; x < 100 := 255 (rear half)",
         "mask_camera applied, mask_lidar not; x < 100 cut (Gate 4 frozen setting)", True),
        ("occupied / free", "class 17 = free; all else occupied", "class 17 = free; all else occupied", True),
        ("majority pooling", "apply_majority_pooling, separate mode (official code)",
         "gate8b.pooling, checked bit-for-bit against the official function", True),
        ("metric aggregation", "pooled TP/FP/FN over all evaluated samples",
         "pooled TP/FP/FN over all official clips (Gate 6 metric code)", True),
        ("metric-scale source", "OccAny's own (reconstruction network)",
         "frozen G51-B: calibrated-FOV MoGe-2 median gauge from the five frames", False)],
}


def load(p):
    z = np.load(p, allow_pickle=False); return {k: z[k] for k in z.files}


def blk(art, ds, tag, name, region):
    z = load(os.path.join(art, f"scores_{ds}_{tag}_{name}.npz"))
    return {k[len(region) + 1:]: v for k, v in z.items() if k.startswith(region + "_")}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def setting_block(ds, art, tag, comp_name="selected"):
    """One (target, setting): threshold-free metrics, every baseline, paired intervals."""
    p = os.path.join(art, f"eval_{ds}_{tag}.json")
    if not os.path.exists(p):
        return None
    ev = json.load(open(p))
    tau, mtau = ev["threshold"], ev["mapper_threshold"]
    out = {"eval_file": os.path.relpath(p, REPO_ROOT), "threshold": tau, "mapper_threshold": mtau,
           "n": ev.get("n_anchors", ev.get("n_clips")), "seconds": ev["seconds"],
           "peak_gpu_gib": ev["peak_gpu_gib"], "methods": {}, "comparisons": {}}
    fb = blk(art, ds, tag, comp_name, "full"); eb = blk(art, ds, tag, comp_name, "edit")
    mb = blk(art, ds, tag, "mapper", "full"); meb = blk(art, ds, tag, "mapper", "edit")
    out["threshold_free"] = {"completion_full": SC.summarize(fb), "completion_edit": SC.summarize(eb),
                             "mapper_full": SC.summarize(mb), "mapper_edit": SC.summarize(meb)}
    clips = [str(c) for c in fb["clip_id"]]
    counts = {"completion": SC.per_clip_counts(fb, tau),
              "completion_edit_region": SC.per_clip_counts(eb, tau),
              "mapper_calibrated": SC.per_clip_counts(mb, mtau),
              "mapper_score_ge_zero": SC.per_clip_counts(mb, 0.0)}
    clip_of = {k: clips for k in counts}
    bc = load(os.path.join(art, f"bincounts_{ds}_{tag}.npz"))
    bclips = [str(c) for c in bc["clip_id"]]
    assert bclips == clips
    for k in [x for x in bc if x.startswith("pred_")]:
        counts[k[5:]] = bc[k]; clip_of[k[5:]] = bclips
    nval = float(bc["n_valid"].sum()); ngt = float(bc["n_gt"].sum())
    nedit = float(bc["n_editable"].sum()); ngte = float(bc["n_gt_editable"].sum())
    for k, c in counts.items():
        b = np.asarray(c, np.int64).sum(0)
        den_v, den_g = (nedit, ngte) if k == "completion_edit_region" else (nval, ngt)
        out["methods"][k] = {"binary_iou": float(b[0] / max(b.sum(), 1)),
                             "precision": float(b[0] / max(b[0] + b[1], 1)),
                             "recall": float(b[0] / max(b[0] + b[2], 1)),
                             "density": float((b[0] + b[1]) / max(den_v, 1)),
                             "pred_over_gt_volume": float((b[0] + b[1]) / max(den_g, 1)),
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
        counts[nm] = bb; clip_of[nm] = cid
        s = bb.sum(0); nv6 = float(np.asarray(z["binary"])[:, 3].sum())
        out["methods"][nm] = {"binary_iou": float(s[0] / max(s.sum(), 1)),
                              "precision": float(s[0] / max(s[0] + s[1], 1)),
                              "recall": float(s[0] / max(s[0] + s[2], 1)),
                              "density": float((s[0] + s[1]) / max(nv6, 1)),
                              "pred_over_gt_volume": float((s[0] + s[1]) / max(s[0] + s[2], 1))}
    for other in [k for k in counts if k not in ("completion", "completion_edit_region")]:
        if other.startswith("random_editable_s") and other != "random_editable_s0":
            continue
        out["comparisons"][f"completion vs {other}"] = B.paired_binary(
            ds, REPO_ROOT, clip_of[other], counts[other], clips, counts["completion"])
    # ---- semantics ---------------------------------------------------------------
    sem = dict(ev.get("semantic", {}))
    sc = {}
    for nm in list(sem):
        p2 = os.path.join(art, f"counts_{ds}_{tag}_{nm}.npz")
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
        ca, ba, pa = sc["completion"]
        want = counts["completion"].sum(0).astype(np.float64); got = ba.sum(0).astype(np.float64)
        rel = float(np.max(np.abs(want - got) / np.maximum(want, 1)))
        assert rel < 1e-5, f"binarization drift {rel:.2e} on {ds}/{tag}"
        out["binarization_agreement"] = {"max_relative_disagreement": rel}
        for nm, (cb, bb, pb) in sc.items():
            if nm != "completion":
                out["comparisons"][f"SSC completion vs {nm}"] = B.paired_binary(
                    ds, REPO_ROOT, cb, bb, ca, ba, pb, pa)
    out["semantic"] = sem
    nc = ev.get("newly_completed") or {}
    if nc and "n_observed_true_positives" not in nc and "completion" in sc:
        # the streaming evaluator stores only the newly-completed split; derive the observed one
        decomp = load(os.path.join(art, f"counts_{ds}_{tag}_completion.npz"))["decomp"].sum(0)
        n_corr, n_name = int(decomp[:, 2].sum()), int(decomp[:, 1].sum())
        n_tp = n_corr + n_name
        nc = dict(nc, n_observed_true_positives=n_tp - nc["n_new_true_positives"],
                  observed_semantic_accuracy=(n_corr - nc["n_named_correctly"])
                  / max(n_tp - nc["n_new_true_positives"], 1))
    out["newly_completed"] = nc
    return out


def decide(sb, n_cls):
    if not sb:
        return None
    tf = sb["threshold_free"]["completion_full"]; m = sb["methods"]; ci = sb["comparisons"]
    def beats(k):
        c = ci.get(f"completion vs {k}")
        return None if not c else bool(c["binary_iou"]["difference"] > 0 and c["binary_iou"]["ci"][0] > 0)
    ap, prev = tf["average_precision"], tf["prevalence"]
    geo = {"ap": ap, "prevalence": prev, "ap_over_prevalence": ap / prev, "auroc": tf["auroc"],
           "ap_meaningfully_above_prevalence": ap > 1.5 * prev,
           "beats_incremental_mapper": beats("mapper_native"),
           "beats_calibrated_mapper": beats("mapper_calibrated"),
           "beats_strongest_dilation": beats("mapper_dilate") and beats("frozen_5frame_dil"),
           "beats_editable_fill": beats("editable_fill"),
           "beats_matched_density_random": beats("random_editable_s0"),
           "beats_all_valid_occupied": beats("all_valid_occupied"),
           "pred_over_gt_volume": m["completion"]["pred_over_gt_volume"],
           "volume_ratio_is_reasonable": m["completion"]["pred_over_gt_volume"] < 2.0}
    geo["passed"] = bool(all(geo[k] for k in ("ap_meaningfully_above_prevalence", "beats_incremental_mapper",
                                              "beats_calibrated_mapper", "beats_strongest_dilation",
                                              "beats_editable_fill", "beats_matched_density_random",
                                              "volume_ratio_is_reasonable")))
    sem = sb["semantic"]
    best = max(((k, v.get("ssc_miou")) for k, v in sem.items()
                if not k.startswith("completion") and v.get("ssc_miou") is not None),
               key=lambda kv: kv[1], default=(None, None))
    c = ci.get(f"SSC completion vs {best[0]}") if best[0] else None
    nc = sb.get("newly_completed") or {}
    semd = {"completion_ssc_miou": sem.get("completion", {}).get("ssc_miou"),
            "strongest_non_completion": best[0], "strongest_non_completion_ssc_miou": best[1],
            "beats_it": None if not c else bool(c["ssc_miou"]["difference"] > 0 and c["ssc_miou"]["ci"][0] > 0),
            "newly_completed_semantic_accuracy": nc.get("semantic_accuracy"),
            "observed_semantic_accuracy": nc.get("observed_semantic_accuracy"),
            "all_tp_semantic_accuracy": sem.get("completion", {}).get("tp_accuracy"),
            "new_tp_carry_information": (None if nc.get("semantic_accuracy") is None
                                         else nc["semantic_accuracy"] > 1.0 / n_cls)}
    semd["passed"] = bool(semd["beats_it"] and semd["new_tp_carry_information"])
    return {"geometry": geo, "semantics": semd}


def classify(dec):
    """Outcome A-D over the three folds, from the *primary* setting."""
    g = {t: dec[t]["stream"]["geometry"] for t in TARGETS if dec.get(t, {}).get("stream")}
    s = {t: dec[t]["stream"]["semantics"] for t in TARGETS if dec.get(t, {}).get("stream")}
    if len(g) < 3:
        return {"outcome": "incomplete", "have": list(g)}
    near = {t: (g[t]["ap_over_prevalence"] < 1.5 or abs(g[t]["auroc"] - 0.5) < 0.1) for t in TARGETS}
    lift = {t: g[t]["ap_over_prevalence"] for t in TARGETS}
    if near["kitti360"] and not near["semantickitti"] and not near["occ3d"]:
        outcome = "A: only KITTI-360 fails"
    elif all(near.values()):
        outcome = "B: all held-out folds fail (source-specific representation)"
    elif all(g[t]["passed"] for t in TARGETS) and all(s[t]["passed"] for t in TARGETS):
        outcome = "D: geometry and semantics transfer"
    elif all(g[t]["passed"] for t in TARGETS):
        outcome = "C: geometry transfers but semantics fail"
    else:
        outcome = ("mixed: " + ", ".join(f"{t} geo {'pass' if g[t]['passed'] else 'fail'}"
                                         f"/near-chance={near[t]}" for t in TARGETS))
    return {"outcome": outcome, "near_chance": near, "ap_over_prevalence": lift,
            "auroc": {t: g[t]["auroc"] for t in TARGETS},
            "geometry_passed": {t: g[t]["passed"] for t in TARGETS},
            "semantics_passed": {t: s[t]["passed"] for t in TARGETS}}


def main() -> int:
    from gates.gate6 import vocab
    res = {"bootstrap": {"n_boot": B.N_BOOT, "seed": B.SEED}, "folds": {}, "settings": {},
           "decision": {}, "teacher": {}, "occany": {"published_5frame_single_camera": OCCANY_5FRAME,
                                                     "protocol": {}}}
    # ---- provenance and frozen configs per fold --------------------------------------
    for t in TARGETS:
        f = dict(FOLDS[t])
        if t == "kitti360":
            fz = json.load(open(os.path.join(G8A, "frozen_manifest.json")))
            tr = json.load(open(os.path.join(G8A, "train_cellB_uniform_focal.json")))
            sel = json.load(open(os.path.join(G8A, "selection.json")))
            f.update({"frozen": fz, "training": tr and {k: tr[k] for k in ("seconds", "peak_gpu_gib", "best_step",
                                                                           "n_train", "n_val")},
                      "selection": {"selected": sel["selected"], "mapper_calibrated": sel["mapper_calibrated"],
                                    "candidates": [{k: r[k] for k in ("candidate", "macro_ap",
                                                                      "ap_exceeds_prevalence_on_both", "sources")}
                                                   for r in sel["candidates"]
                                                   if r["candidate"] in ("cellB_best", "cellB_last", "mapper")]},
                      "kitti360_partition": "target only (drive 0006); never trained on"})
        else:
            pf = os.path.join(ART, f"frozen_manifest_{t}.json")
            if os.path.exists(pf):
                fz = json.load(open(pf)); tr = json.load(open(os.path.join(ART, f"train_fold_{t}.json")))
                sel = json.load(open(os.path.join(ART, f"selection_{t}.json")))
                f.update({"frozen": fz, "training": {k: tr[k] for k in ("seconds", "peak_gpu_gib", "best_step",
                                                                        "n_train", "n_val", "samples_drawn")},
                          "selection": {"selected": sel["selected"], "mapper_calibrated": sel["mapper_calibrated"],
                                        "candidates": [{k: r[k] for k in ("candidate", "macro_ap",
                                                                          "ap_exceeds_prevalence_on_both", "sources")}
                                                       for r in sel["candidates"]]},
                          "kitti360_partition": f"train drives {list(K360_TRAIN_DRIVES)}; "
                                                f"source-validation drive {K360_VAL_DRIVE}"})
        res["folds"][t] = f
    # ---- both settings per target ------------------------------------------------------
    for t in TARGETS:
        n_cls = len(vocab.load(t))
        if t == "kitti360":
            st = setting_block(t, G8A, "locked")
        else:
            st = setting_block(t, ART, f"stream_{t}")
        cl = setting_block(t, ART, f"clips_{t}")
        res["settings"][t] = {"stream": st, "clips": cl}
        res["decision"][t] = {"stream": decide(st, n_cls), "clips": decide(cl, n_cls)}
        tp = os.path.join(ART, f"teacher_accuracy_{t}.json")
        if os.path.exists(tp):
            res["teacher"][t] = json.load(open(tp))
    res["outcome"] = classify(res["decision"])
    # ---- OccAny reference comparison (matched setting only) ----------------------------
    for t in ("semantickitti", "occ3d"):
        rows = PROTOCOL[t]
        res["occany"]["protocol"][t] = {"items": [{"item": a, "occany": b, "ours": c, "match": d}
                                                  for a, b, c, d in rows],
                                        "exact": all(d for *_, d in rows),
                                        "label": "reference comparison (protocol differs)"
                                        if not all(d for *_, d in rows) else "exact comparison"}
    # ---- manifest ----------------------------------------------------------------------
    man = {"gate": "8B", "selected_checkpoints": {}, "configs": {}, "frozen_files": {}}
    for t in TARGETS:
        fz = res["folds"][t].get("frozen")
        if fz:
            man["selected_checkpoints"][t] = {"path": fz["checkpoint"], "sha256": fz["checkpoint_sha256"],
                                              "recorded_sha256_matches": sha256(os.path.join(REPO_ROOT, fz["checkpoint"])) == fz["checkpoint_sha256"]}
    for p in sorted(glob.glob(os.path.join(REPO_ROOT, "configs", "gate8b", "*.yaml"))):
        man["configs"][os.path.relpath(p, REPO_ROOT)] = sha256(p)
    for p in ("configs/gate8a/frozen_selection.yaml", "gates/gate8/net.py", "gates/gate8/mapper.py", "gates/gate8/targets.py",
              "gates/gate8/losses.py", "gates/gate8a/regions.py", "gates/gate8a/scores.py", "gates/gate8b/pooling.py", "gates/gate8b/clips.py"):
        man["frozen_files"][p] = sha256(os.path.join(REPO_ROOT, p))
    res["manifest"] = man
    write_json(os.path.join(ART, "gate8b_manifest.json"), man)
    write_json(os.path.join(ART, "gate8b_results.json"), res)
    print("== outcome:", res["outcome"].get("outcome"))
    for t in TARGETS:
        for setting in ("stream", "clips"):
            sb = res["settings"][t][setting]
            if not sb:
                print(f"   {t:14s} {setting:6s} (missing)"); continue
            tf = sb["threshold_free"]["completion_full"]; m = sb["methods"]
            d = res["decision"][t][setting]
            print(f"   {t:14s} {setting:6s} AP {tf['average_precision']:.4f} prev {tf['prevalence']:.4f} "
                  f"x{tf['ap_over_prevalence']:.2f} AUROC {tf['auroc']:.4f} IoU@tau {m['completion']['binary_iou']:.4f} "
                  f"(all-occ {m['all_valid_occupied']['binary_iou']:.4f}, dil {m['mapper_dilate']['binary_iou']:.4f}, "
                  f"5f-dil {m['frozen_5frame_dil']['binary_iou']:.4f}) SSC {sb['semantic'].get('completion', {}).get('ssc_miou', float('nan')):.4f} "
                  f"geo {'PASS' if d['geometry']['passed'] else 'fail'} sem {'PASS' if d['semantics']['passed'] else 'fail'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
