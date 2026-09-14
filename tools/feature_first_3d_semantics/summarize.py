#!/usr/bin/env python
"""results_table.md + config.json for the feature-first study.

    python tools/feature_first_3d_semantics/summarize.py
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.feature_first_3d_semantics import common as FF                                 # noqa: E402
from core.datasets import kitti360_splits as SRC                                          # noqa: E402

SHORT = {d: d[17:21] for d in list(SRC.TRAIN_DRIVES) + [SRC.VAL_DRIVE]}


def main() -> int:
    val = SRC.VAL_DRIVE; drives = list(SRC.TRAIN_DRIVES) + [val]
    ext = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(FF.ART, "extraction", "2013*.json")))]
    lifts = {(l["drive"], l["population"]): l for l in (json.load(open(p)) for p in sorted(glob.glob(os.path.join(FF.ART, "lifting", "*.json"))))}
    ver = json.load(open(os.path.join(FF.ART, "verification.json")))
    L = ["# Feature-first 3D semantics -- results\n"]
    L.append("Representation: Trident-H's correlation-refined CLIP ViT-H-14 token feature (1024-d, unit norm, 34 x 128 tokens per frame), the tensor the "
             "official code projects onto its text embeddings. P_ref = existing SAM-refined per-view probabilities lifted and averaged; P_coarse = per-view "
             "softmax(40 f . text) lifted and averaged (classify-then-fuse); F = feature lifted and averaged, normalised, classified once (fuse-then-classify). "
             "Interleaved frame-disjoint groups A/B/C; identical rays, weights and voxels for all three.\n")
    n_fr = sum(e["n_written"] for e in ext)
    L.append(f"Extraction: {n_fr} frames, {ext[0]['bytes_per_frame']/1e6:.1f} MB/frame ({n_fr*ext[0]['bytes_per_frame']/1e9:.1f} GB); coarse-probability reproduction max |dp| "
             f"{max(e['coarse_reproduction']['max_abs_dp_max'] for e in ext):.4f}, argmax agreement mean {np.mean([e['coarse_reproduction']['argmax_agree_mean'] for e in ext]):.4f}; "
             f"SAM refinement changes {100*(1-np.mean([e['coarse_reproduction']['refined_vs_coarse_agree_mean'] for e in ext])):.1f} % of pixel labels.\n")
    for pop in ("eligible", "all_future_observed"):
        p = os.path.join(FF.ART, f"analysis_{pop}.json")
        if not os.path.exists(p):
            continue
        an = json.load(open(p))
        L.append(f"## Population: {pop}\n")
        L.append("### Support\n")
        L.append("| drive | voxels in population | with evidence | units with >= 2 groups | A&B | A&C | B&C | A&B&C |")
        L.append("|---|---|---|---|---|---|---|---|")
        for d in drives:
            r = an["drives"][d]; s = r["support"]
            L.append(f"| {SHORT[d]} | {s['n_eligible']:,} | {s['n_any_evidence']:,} | {s['n_units_ge2_groups']:,} | {r['pairs'].get('AB',{}).get('n',0):,} | {r['pairs'].get('AC',{}).get('n',0):,} | {r['pairs'].get('BC',{}).get('n',0):,} | {r['fusion'].get('A+B->C',{}).get('n',0):,} |")
        L.append("\n### Step 5 -- same-surface repeatability on matched A/B voxels\n")
        L.append("| drive | n | P_ref top-1 | P_coarse top-1 | F classify-each top-1 | P_ref cos | P_coarse cos | F cos | P_coarse JS | F unit-euclid |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for d in drives:
            ab = an["drives"][d]["pairs"]["AB"]
            L.append(f"| {SHORT[d]} | {ab['n']:,} | {ab['P_ref']['top1']:.3f} | {ab['P_coarse']['top1']:.3f} | {ab['F']['top1_classify_each']:.3f} | {ab['P_ref']['cosine']:.3f} | {ab['P_coarse']['cosine']:.3f} | {ab['F']['cosine']:.3f} | {ab['P_coarse']['js']:.3f} | {ab['F']['euclid_unit']:.3f} |")
        L.append("\n### Step 6 -- controls (A/B pairs): true cosine, control cosine, margin, margin / control std, AUC(true > control)\n")
        for ctl in ("C0_drive", "C1_anchor", "C2_local", "C3_same_class_other_location"):
            L.append(f"**{ctl}**\n")
            L.append("| drive | n | P_coarse true / ctrl / margin / z / AUC | F true / ctrl / margin / z / AUC | P_ref true / ctrl / margin | P_coarse top-1 true / ctrl | P_ref top-1 true / ctrl |")
            L.append("|---|---|---|---|---|---|---|")
            for d in drives:
                c = an["drives"][d]["pairs"]["AB"]["controls"].get(ctl)
                if not c:
                    continue
                f_ = lambda k: f"{c[k]['true_mean']:.3f} / {c[k]['control_mean']:.3f} / {c[k]['margin']:+.3f}" + (f" / {c[k]['margin_over_control_std']:.2f} / {c[k]['auc_true_gt_control']:.3f}" if c[k]['auc_true_gt_control'] is not None else "")
                L.append(f"| {SHORT[d]} | {c['n']:,} | {f_('P_coarse')} | {f_('F')} | {f_('P_ref')} | {c['P_coarse_top1']['true_mean']:.3f} / {c['P_coarse_top1']['control_mean']:.3f} | {c['P_ref_top1']['true_mean']:.3f} / {c['P_ref_top1']['control_mean']:.3f} |")
            L.append("")
        L.append("### Steps 7-8 -- fuse A+B, validate on the held-out group (all three rotations)\n")
        L.append("| drive | rotation | n | P_ref fuse vs held-out (refined) | classify-then-fuse vs held-out view | feature-fuse-then-classify vs held-out view | F vs held-out feature-classified | classify-then-fuse vs held-out feature-classified | cos(f_AB, f_C) | single view vs held-out (P_coarse / F) | P_coarse & F same decision |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for d in drives:
            for rot, fu in an["drives"][d]["fusion"].items():
                L.append(f"| {SHORT[d]} | {rot} | {fu['n']:,} | {fu['P_ref_fuse_vs_C_ref']['top1']:.3f} | {fu['P_coarse_fuse_vs_C_coarse']['top1']:.3f} (cos {fu['P_coarse_fuse_vs_C_coarse']['cosine']:.3f}, JS {fu['P_coarse_fuse_vs_C_coarse']['js']:.3f}) | "
                         f"{fu['F_fuse_classify_vs_C_coarse']['top1']:.3f} (cos {fu['F_fuse_classify_vs_C_coarse']['cosine']:.3f}, JS {fu['F_fuse_classify_vs_C_coarse']['js']:.3f}) | {fu['F_fuse_classify_vs_C_feature_classified']['top1']:.3f} | "
                         f"{fu['P_coarse_fuse_vs_C_feature_classified']['top1']:.3f} | {fu['feature_cosine_fAB_fC']:.3f} | {fu['single_view_A_vs_C']['P_coarse_top1']:.3f} / {fu['single_view_A_vs_C']['F_classify_top1']:.3f} | {fu['P_coarse_vs_F_same_decision']:.3f} |")
        L.append("\n### Step 9 -- frames fused in A+B vs held-out C\n")
        L.append("| drive | frames fused | n | classify-then-fuse top-1 | feature-fuse top-1 | cos(f_AB, f_C) | cos(p_AB, p_C) |")
        L.append("|---|---|---|---|---|---|---|")
        for d in drives:
            for k, v in an["drives"][d]["fusion"]["A+B->C"]["by_frames_fused"].items():
                L.append(f"| {SHORT[d]} | {k} | {v['n']:,} | {v['P_coarse_top1']:.3f} | {v['F_top1']:.3f} | {v['feature_cosine']:.3f} | {v['P_coarse_cosine']:.3f} |")
        L.append("\n### Per class, drive 0006 (A/B pairs; classes with >= 200 units)\n")
        L.append("| class | n | P_coarse top-1 | P_ref top-1 | P_coarse cos | F cos | P_coarse margin C2 | F margin C2 | P_coarse AUC C2 | F AUC C2 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        pc = an["drives"][val]["per_class_AB"]
        for n_, v in sorted(pc.items(), key=lambda x: -x[1]["n"]):
            g = lambda x: "-" if x is None else f"{x:.3f}"
            L.append(f"| {n_} | {v['n']:,} | {v['P_coarse_top1']:.3f} | {v['P_ref_top1']:.3f} | {v['P_coarse_cosine']:.3f} | {v['F_cosine']:.3f} | {g(v['P_coarse_margin_C2'])} | {g(v['F_margin_C2'])} | {g(v['P_coarse_auc_C2'])} | {g(v['F_auc_C2'])} |")
        L.append("\n### Per class, drive 0006 (A+B -> C held-out; classes with >= 100 units)\n")
        L.append("| class | n | P_ref | classify-then-fuse | feature-fuse-then-classify | cos(f_AB, f_C) |")
        L.append("|---|---|---|---|---|---|")
        for n_, v in sorted(an["drives"][val]["fusion"]["A+B->C"]["per_class"].items(), key=lambda x: -x[1]["n"]):
            L.append(f"| {n_} | {v['n']:,} | {v['P_ref_top1']:.3f} | {v['P_coarse_top1']:.3f} | {v['F_top1']:.3f} | {v['feature_cosine']:.3f} |")
        ov = os.path.join(FF.ART, f"openvocab_{pop}.json")
        if os.path.exists(ov):
            o = json.load(open(ov))
            L.append(f"\n### Step 11 -- open-vocabulary query of the fused feature (drive 0006, {o['n_units']:,} A&B units)\n")
            L.append(f"Best of 16 free-text queries, mapped to its vocabulary class: agrees with the fused feature's vocabulary decision on {100*o['query_mapped_class_equals_vocab_decision']:.1f} % of units and with the "
                     f"independent view C on {100*o['query_mapped_class_equals_C_view_decision']:.1f} % (vocabulary decision vs C: {100*o['vocab_decision_equals_C_view_decision']:.1f} %).\n")
            for k, v in o["per_vocab_class"].items():
                L.append(f"- {k}: n={v['n']}, top query '{v['top_query']}' ({100*v['top_query_share']:.0f} %), mapped agreement {100*v['query_mapped_agree']:.0f} %")
            L.append("")
    L.append("### Step 10 -- observed-surface semantic quality against human labels\n")
    L.append("Not measured: the protocol never reads a KITTI-360 human semantic label on the source drives and the target benchmarks are untouched in this step.\n")
    L.append(f"Verification: {ver['n_pass']}/{ver['n_total']} checks pass.\n")
    open(os.path.join(FF.ART, "results_table.md"), "w").write("\n".join(L) + "\n")
    cfg = {"study": "feature_first_3d_semantics", "representation": {"tensor": "Trident.get_trident_seg: clip_features = model.clip.encode_image(...) (recorded by hook), L2-normalised",
           "model": "open_clip ViT-H-14 laion2b_s32b_b79k with Trident DINO/SAM correlation refinement (official Cityscapes config)", "dim": FF.FEAT_DIM, "token_grid": [34, 128],
           "language_aligned": True, "classifier": "logits = f . query_features^T (mean-templated CLIP text embeddings, unit norm); official coarse probs = softmax(40 * bilinear(logits))",
           "relation_to_cached_probs": "cached 25-class vectors = SAM-refined per-view probabilities (refinement rewrites region scores after this classifier)"},
           "teacher": ext[0]["teacher"], "extraction": {"n_frames": n_fr, "dtype": "float16", "bytes": n_fr * ext[0]["bytes_per_frame"]},
           "lifting": {"lattice_hw": [140, 518], "resampling": lifts[(val, "eligible")]["resampling"], "channels": lifts[(val, "eligible")]["channels"],
                       "correspondence": "identical (voxel, frame) pairs and weights as the probability-path evidence cache (checked per anchor)"},
           "grouping": json.load(open(os.path.join(FF.ART, "analysis_eligible.json")))["grouping"],
           "fusion": {"P": "p_AB = (w_A p_A + w_B p_B)/(w_A + w_B)", "F": "f_AB = normalize(w_A f_A + w_B f_B); p'_AB = softmax(40 f_AB . text)"},
           "controls": {"C0": "random within drive", "C1": "within anchor", "C2": "within (anchor, z band 4 vox, xy cell 20 vox)", "C3": "within Trident class across the drive"},
           "text_queries": os.path.join(FF.DATA_ROOT, "text_queries.npz"), "env": FF.environment()}
    FF.write_json(os.path.join(FF.ART, "config.json"), cfg)
    print("\n".join(L[:30]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
