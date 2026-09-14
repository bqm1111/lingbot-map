#!/usr/bin/env python
"""results_table.md + config.json for the direct-correspondence audit.

    python tools/direct_correspondence_audit/summarize.py
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.direct_correspondence_audit import common as DC                                # noqa: E402
from core.datasets import kitti360_splits as SRC                                          # noqa: E402

SHORT = {d: d[17:21] for d in list(SRC.TRAIN_DRIVES) + [SRC.VAL_DRIVE]}


def main() -> int:
    an = json.load(open(os.path.join(DC.ART, "analysis.json"))); mc = json.load(open(os.path.join(DC.ART, "mapper_comparison.json")))
    ver = json.load(open(os.path.join(DC.ART, "verification.json"))); val = SRC.VAL_DRIVE
    pairs = {d: json.load(open(os.path.join(DC.ART, "pairs", f"{d}.json"))) for d in an["drives"]}
    drives = [d for d in list(SRC.TRAIN_DRIVES) + [val] if d in an["drives"]]
    L = ["# Direct semantic-correspondence audit -- results\n"]
    L.append("## Correspondence construction and rejection\n")
    L.append("| drive | source frames | landmarks | candidate (X, B) | behind / out of range | outside image | no LiDAR reference | occluded | in front of reference | accepted loose | accepted strict |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for d in drives:
        c = pairs[d]["counts"]
        L.append(f"| {SHORT[d]} | {c['n_A']} | {c['n_landmarks']:,} | {c['n_candidates']:,} | {c['rej_behind_or_range']:,} | {c['rej_outside']:,} | {c['rej_no_reference']:,} | {c['rej_occluded']:,} | {c['rej_in_front']:,} | {c['accepted_loose']:,} | {c['accepted_strict']:,} |")
    L.append("\n## Same-point similarity and controls (strict visibility)\n")
    L.append("| drive | pairs | feature cos same | C0 random | C1 same frame | C3 same class | C2 local (0.3-1.5 m) | C2 same-view local | Δ vs C2 | AUC vs C2 | AUC vs C3 | AUC vs C0 | prob cos same / C2 | prob AUC vs C2 | top-1 same / C2 / same-view local | refined-label top-1 same |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in drives:
        r = an["drives"][d]["strict"]; c = r["controls"]; s = r["same"]
        L.append(f"| {SHORT[d]} | {r['n']:,} | {s['feature_cos']:.3f} | {c['C0']['feature']['control_cos']:.3f} | {c['C1']['feature']['control_cos']:.3f} | {c['C3']['feature']['control_cos']:.3f} | {c['C2']['feature']['control_cos']:.3f} | "
                 f"{c['C2_same_view']['feature_cos_same_view_local']:.3f} | {c['C2']['feature']['margin']:+.3f} | {c['C2']['feature']['auc']:.3f} | {c['C3']['feature']['auc']:.3f} | {c['C0']['feature']['auc']:.3f} | "
                 f"{s['prob_cos']:.3f} / {c['C2']['prob']['control_cos']:.3f} | {c['C2']['prob']['auc']:.3f} | {s['top1']:.3f} / {c['C2']['top1']['control']:.3f} / {c['C2_same_view']['top1_same_view_local']:.3f} | {s['ref_top1']:.3f} |")
    pv = an["previous_mapper_based"]["drive_0006_mapper_voxel_pairs"]
    L.append(f"\nPrevious mapper-defined same-voxel pairs (0006): feature cos {pv['feature_cos_same']} vs local {pv['feature_cos_local']} (AUC {pv['feature_auc_local']}); "
             f"prob cos {pv['prob_cos_same']} vs {pv['prob_cos_local']} (AUC {pv['prob_auc_local']}); top-1 {pv['top1_same']} vs {pv['top1_local']}.\n")
    r = an["drives"][val]
    L.append("## Viewpoint dependence, drive 0006 (strict)\n")
    for key, ttl in (("by_baseline", "baseline"), ("by_angle", "viewing-angle change"), ("by_depth_B", "depth in B"), ("by_visibility_residual", "visibility residual"), ("by_offset", "stream offset B-A")):
        L.append(f"**{ttl}**\n"); L.append("| bin | n | feature cos same | C2 local | AUC vs C2 | prob cos same | top-1 same | top-1 C2 |"); L.append("|---|---|---|---|---|---|---|---|")
        for n_, b in r[key].items():
            if "same" not in b:
                continue
            c2 = b["controls"].get("C2", {})
            L.append(f"| {n_} | {b['n']:,} | {b['same']['feature_cos']:.3f} | {c2.get('feature', {}).get('control_cos', float('nan')):.3f} | {c2.get('feature', {}).get('auc', float('nan')):.3f} | {b['same']['prob_cos']:.3f} | {b['same']['top1']:.3f} | {c2.get('top1', {}).get('control', float('nan')):.3f} |")
        L.append("")
    L.append("## Per class, drive 0006 (class = A's per-view top-1; >= 300 pairs)\n")
    L.append("| class | n | top-1 same | top-1 C2 | feature cos | feature AUC vs C2 | prob AUC vs C2 |"); L.append("|---|---|---|---|---|---|---|")
    for n_, v in sorted(r["per_class"].items(), key=lambda x: -x[1]["n"]):
        L.append(f"| {n_} | {v['n']:,} | {v['top1']:.3f} | {v['C2_top1_control']:.3f} | {v['feature_cos']:.3f} | {v['C2_feature_auc']:.3f} | {v['C2_prob_auc']:.3f} |")
    p = r["perturbation"]
    L.append(f"\n## Pixel-neighbourhood sensitivity, drive 0006 ({p['n']:,} pairs)\n")
    L.append("| perturbation of the B sample | 1 - cosine |"); L.append("|---|---|")
    for k in ("2px", "5px", "1tok"):
        L.append(f"| {k} | {p['one_minus_cos_' + k]:.4f} |")
    L.append(f"| cross-view, verified same point | {p['one_minus_cos_cross_view_same_point']:.4f} |"); L.append(f"| cross-view, local control point | {p['one_minus_cos_local_control']:.4f} |")
    L.append(f"\nA one-token (11 px) shift explains {100*p['fraction_cross_view_diff_explained_by_1tok']:.0f} % of the cross-view feature change.\n")
    L.append("## Mapper association vs direct correspondence, drive 0006, |offset| <= 3 stream frames\n")
    s = mc["similarity"]
    L.append(f"{mc['n_pairs']:,} verified pairs replayed through the mapper's frozen inputs (LingBot depth/poses, frozen scale {mc['scale']:.2f}).\n")
    L.append(f"- mapper has any frame-B observation in the 0.2 m voxel of X: **{100*mc['mapper_has_B_observation_in_voxel_of_X']:.1f} %** (frame A: {100*mc['mapper_has_A_observation_in_voxel_of_X']:.1f} %; both: {100*mc['both']:.1f} %)")
    o = mc["mapper_pixel_offset_B_px"]; L.append(f"- when it has one, its B pixels lie {o['median']:.0f} px (median; p25 {o['p25']:.0f}, p90 {o['p90']:.0f}) from the true projection u_B; within 1 token (11 px) {100*o['frac_lt_11px']:.1f} %, beyond 30 px {100*o['frac_gt_30px']:.1f} %")
    de = mc["lingbot_depth_error_at_uB_m"]; L.append(f"- LingBot depth at u_B vs LiDAR depth: median |error| {de['median_abs']:.2f} m ({100*de['median_rel']:.1f} % relative); relative error > 10 % on {100*de['frac_rel_gt_10pct']:.0f} % of pairs")
    pe = mc["pose_reprojection_error_px"]; L.append(f"- LingBot relative pose alone (GT camera-frame point at A carried to B): reprojection error median {pe['median']:.0f} px, p90 {pe['p90']:.0f} px")
    ee = mc["endpoint_3d_error_m"]; L.append(f"- LingBot endpoint of the true pixel vs X in the mapper's world: median {ee['median']:.1f} m, p90 {ee['p90']:.1f} m; > 0.2 m (one voxel) on {100*ee['frac_gt_0p2m']:.1f} %")
    L.append(f"- class agreement on the {s['n_both']:,} pairs where the mapper has observations in both frames: direct same point {s['direct_top1_on_both']:.3f}, mapper same voxel {s['mapper_top1_on_both']:.3f}; feature cos {s['direct_cos_on_both']:.3f} vs {s['mapper_cos_on_both']:.3f}\n")
    L.append("| stream position of A | pairs | mapper has B obs | pose reproj err (px) | endpoint 3D err (m) | depth rel err | mapper pixel offset (px) | direct top-1 | direct top-1 (both) | mapper top-1 (both) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for k, v in mc["by_stream_position"].items():
        L.append(f"| {k} | {v['n']:,} | {100*v['mapper_has_B_obs']:.1f} % | {v['pose_reproj_err_px_median']:.0f} | {v['endpoint_3d_err_m_median']:.1f} | {100*v['depth_rel_err_median']:.0f} % | {v['mapper_pix_offset_px_median']:.0f} | {v['direct_top1']:.3f} | {v['direct_top1_on_both']:.3f} | {v['mapper_top1_on_both']:.3f} |")
    L.append(f"\nVerification: {ver['n_pass']}/{ver['n_total']} checks pass.\n")
    open(os.path.join(DC.ART, "results_table.md"), "w").write("\n".join(L) + "\n")
    cfg = {"study": "direct_semantic_correspondence_audit", "geometry_source": "KITTI-360 raw Velodyne sweeps + cam0_to_world.txt + calibration (P_rect_00, R_rect_00, calib_cam_to_velo)",
           "correspondence_rule": pairs[val]["rule"], "calibration": pairs[val]["calibration"], "tolerances": DC.TOLS, "controls": json.load(open(os.path.join(DC.ART, "sampling", f"{val}.json")))["controls"],
           "feature": {"source": "feature-first token cache (Trident-H correlation-refined CLIP ViT-H-14 tokens, 1024-d, unit norm, 34x128)", "sampling": "bilinear at native pixel centres, align_corners=False (== official logit interpolation)",
                       "classification": "softmax(40 f . text) with Trident's own text embeddings"},
           "mapper_replay": {"inputs": "stream cache pred_depth / pred_pose_c2w / pred_K, scale = exp(median(log_s[:5]))", "association": "LingBot endpoint band (-1,0,+1 voxel) into 0.2 m world voxels; X placed via GT camera-frame coordinates at A and LingBot pose of A"},
           "env": DC.environment()}
    DC.write_json(os.path.join(DC.ART, "config.json"), cfg)
    print("\n".join(L[:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
