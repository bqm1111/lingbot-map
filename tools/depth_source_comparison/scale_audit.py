#!/usr/bin/env python
"""Scale-path audit of Gate 9: numerical checks on deterministic 0006 clips.

    python tools/depth_source_comparison/scale_audit.py --device cuda:0

Writes artifacts/depth_source_comparison/scale_audit.{json,md}. Checks, per clip:
 (1) the cached local-reset depth is the raw LingBot output (fresh re-inference, canonical units);
 (2) the MoGe-2 clip scale is exp(median_q frame_candidate) over the five causal frames t-4..t only;
 (3) Gate 9 depth = cached depth x clip scale, applied once (stored pixel rows; mapper scale forced 1);
 (4) Gate 9 convention (scaled depth, metric GT translation, mapper scale 1) == deployable mapper
     convention (canonical depth, translation / s, mapper scale s); a double application differs;
 (5) the deployable local-reset L pipeline multiplies LingBot pose translations by the same scalar;
 (6) Gate 9 GT translations are metric and unchanged;
 (7) Gs applies a residual LiDAR ratio on top of the already-scaled depth.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dsc as D                                                                            # noqa: E402
sys.path.insert(0, D.G9_DIR); sys.path.insert(0, D.REPO_ROOT)
import common as G9                                                                        # noqa: E402
from audit import lattice_K, zbuffer                                                       # noqa: E402  (Gate 9)
from variant_mapper import VariantMapper                                                   # noqa: E402  (Gate 9)
from tools.local_registration_audit import common as LR                                    # noqa: E402
from gates.gate7b import replay, scale as SC                                               # noqa: E402
from core.datasets import kitti360_transforms as TF, kitti360_splits as SRC, sources as S  # noqa: E402
from core.mapping.mapper import FrameInput                                                 # noqa: E402
from core.mapping import depth as D7                                                       # noqa: E402
from core.run._common import REPO_ROOT                                                     # noqa: E402


def loc(rel, needle):
    for i, ln in enumerate(open(os.path.join(REPO_ROOT, rel)).read().splitlines(), 1):
        if needle in ln:
            return f"{rel}:{i}"
    return f"{rel}:NOT FOUND ({needle})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--device", default="cuda:0"); a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev); d = D.PRIMARY
    seg = [s for s in S.segments("kitti360", REPO_ROOT) if s.name == d][0]; nat = [int(f.order) for f in seg.frames]
    geo = TF.DriveGeometry(d, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT); Kl = lattice_K(np.asarray(geo.calib.K, np.float64)); v2c = np.asarray(geo.calib.velo_to_rect_cam, np.float64)
    Tgw = np.linalg.inv(np.asarray(geo.rect_cam_to_velo, np.float64)); grid = type("G", (), {"dims": G9.DIMS, "voxel_size": G9.VS, "origin": tuple(G9.ORIGIN)})()
    moge = D.moge_loader("kitti360"); model, _ = replay.build_model(dev); rows = []
    for t in D.SCALE_AUDIT_CLIPS:
        z = np.load(LR.clip_path(d, t)); g = np.load(os.path.join(G9.DATA_ROOT, "clips", d, f"{t:05d}.npz"))
        idx = [int(x) for x in z["stream_indices"]]; s = float(z["scale"]); canon = z["pred_depth"].astype(np.float64); r = {"t": t, "stream_indices": idx}
        # (1)
        imgs = replay.load_images([seg.frames[j].path for j in idx]); out = replay.replay_segment(model, imgs, 1, dev)
        r["1_rerun_depth_bit_identical"] = bool(np.array_equal(out["pred_depth"], z["pred_depth"]))
        r["1_rerun_depth_max_abs_diff"] = float(np.abs(out["pred_depth"].astype(np.float32) - z["pred_depth"].astype(np.float32)).max())
        pc4 = geo.read_velodyne(nat[idx[4]]) @ v2c[:3, :3].T + v2c[:3, 3]; zb4, _ = zbuffer(pc4, Kl); sup4 = (zb4 > D7.MIN_DEPTH_M) & (zb4 < D7.MAX_DEPTH_M)
        r["1_median_cached_depth_at_lidar_px"] = float(np.median(canon[4][sup4])); r["1_median_lidar_depth_m"] = float(np.median(zb4[sup4]))
        r["1_median_lidar_over_cached"] = float(np.median(zb4[sup4] / canon[4][sup4]))
        # (2)
        logs = []
        for q in range(5):
            md, mm, _ = moge(str(z["keys"][q])); logs.append(SC.frame_candidate(md.astype(np.float32), mm, z["pred_depth"][q].astype(np.float32), z["pred_depth_conf"][q].astype(np.float32))["log_s"])
        r["2_causal_frames_t-4..t"] = idx == [t - 4, t - 3, t - 2, t - 1, t]; r["2_keys_are_those_frames"] = [str(k) for k in z["keys"]] == [seg.frames[j].key for j in idx]
        r["2_log_s_frames"] = [round(x, 6) for x in logs]; r["2_scale_stored"] = s; r["2_scale_recomputed"] = float(np.exp(np.nanmedian(logs))); r["2_equal"] = r["2_scale_recomputed"] == s
        # (3)
        P = g["pixels"]; F = [str(x) for x in g["pixel_fields"]]; fr, vv, uu = (P[:, F.index(k)].astype(int) for k in ("frame", "v", "u"))
        zp = P[:, F.index("z_pred")]; cz = canon[fr, vv, uu]
        r["3_gate9_l_scale_equals_clip_scale"] = float(g["l_scale"]) == s
        r["3_gate9_zpred_equals_cached_x_scale"] = bool(np.array_equal(zp, (cz * s).astype(np.float32)))
        r["3_gate9_zpred_over_cached_median"] = float(np.median(zp / cz)); r["3_gate9_zpred_over_cached_x_scale_max_abs_rel"] = float(np.abs(zp / (cz * s) - 1).max())
        m_forced = VariantMapper(dev, 0.2, 0.2, True); m_forced.scale_state.force(1.0); r["3_mapper_scale_after_force_1"] = float(m_forced.scale_state.scale); del m_forced
        # (4)
        TG = g["T_j_to_t"]; sups = []
        for q, j in enumerate(idx):
            pc = geo.read_velodyne(nat[j]) @ v2c[:3, :3].T + v2c[:3, 3]; zb, _ = zbuffer(pc, Kl); sups.append((zb > D7.MIN_DEPTH_M) & (zb < D7.MAX_DEPTH_M))
        def run(depths, force, trans_div):
            mp = VariantMapper(dev, 0.2, 0.2, True); mp.scale_state.force(force)
            for q in range(5):
                T = TG[q].copy(); T[:3, 3] /= trans_div
                mp.step(FrameInput(index=q, depth_canonical=torch.from_numpy(depths[q].astype(np.float32)).to(dev),
                                   conf=torch.from_numpy(np.where(sups[q], G9.SUPPORT_CONF, 0.0).astype(np.float32)).to(dev), K=Kl, pose_c2w_canonical=T, log_scale_candidate=0.0))
            return mp.query(grid, Tgw)["occupied"].cpu().numpy()
        A = run([canon[q] * s for q in range(5)], 1.0, 1.0); B = run([canon[q] for q in range(5)], s, s); C2 = run([canon[q] * s for q in range(5)], s, s)
        ref = np.zeros(A.shape, bool); ref[g["ref_occ"]] = True
        r["4_gate9_convention_equals_stored_PD_map"] = bool(np.array_equal(np.flatnonzero(A).astype(np.int32), g["occ_PD"]))
        r["4_deployable_convention_xor_voxels"] = int((A ^ B).sum()); r["4_occupied_voxels_A"] = int(A.sum())
        r["4_ref_occ_recall_A_B_double"] = [float((A & ref).sum() / ref.sum()), float((B & ref).sum() / ref.sum()), float((C2 & ref).sum() / ref.sum())]
        r["4_occupied_voxels_double_application"] = int(C2.sum())
        # (5)
        TLc = [np.linalg.inv(z["pred_pose_c2w"][4].astype(np.float64)) @ z["pred_pose_c2w"][q].astype(np.float64) for q in range(5)]
        r["5_clip_translation_t-4_to_t_canonical"] = float(np.linalg.norm(TLc[0][:3, 3])); r["5_clip_translation_x_scale_m"] = float(np.linalg.norm(TLc[0][:3, 3]) * s)
        # (6)
        Gt = geo.rect_cam_to_world(nat[t]); TGr = [TF.inv(Gt) @ geo.rect_cam_to_world(nat[j]) for j in idx]
        r["6_gt_translation_t-4_to_t_m"] = float(np.linalg.norm(TG[0][:3, 3])); r["6_gt_poses_recomputed_equal"] = bool(all(np.array_equal(TG[q], TGr[q]) for q in range(5)))
        # (7)
        zp4 = canon[4] * s; orc = float(np.median(zb4[sup4] / np.maximum(zp4[sup4], 1e-6)))
        r["7_gate9_oracle_ratio_stored"] = float(g["oracle_ratio"]); r["7_recomputed_on_scaled_depth"] = orc; r["7_equal"] = orc == float(g["oracle_ratio"])
        r["7_Gs_total_scale_on_cached_depth"] = s * orc
        rows.append(r); print(json.dumps(r), flush=True)
    an = json.load(open(os.path.join(G9.ART, "analysis.json")))[d]["conditions"]
    old_gap = (an["Gs"]["iou"] - an["PD"]["iou"]) / (an["LD"]["iou"] - an["PD"]["iou"])
    codeloc = {"raw_depth_returned": loc("gates/gate7b/replay.py", '"pred_depth": preds["depth"]'),
               "clip_scale_computed": loc("tools/local_registration_audit/run_local_lingbot.py", "SC.frame_candidate(md, mm, dep[q], cnf[q])"),
               "clip_scale_stored_separately": loc("tools/local_registration_audit/run_local_lingbot.py", "scale=np.float64(np.exp(ls))"),
               "frame_candidate_log_ratio": loc("gates/gate7b/scale.py", "r = np.log(md[v]) - np.log(ld[v])"),
               "gate9_scale_applied": loc("tools/gate9_depth_occupancy/audit.py", 'zp.append(Lz["pred_depth"][q].astype(np.float64) * l_scale)'),
               "gate9_mapper_scale_forced_1": loc("tools/gate9_depth_occupancy/audit.py", "mp.scale_state.force(1.0)"),
               "gate9_gt_pose_input": loc("tools/gate9_depth_occupancy/audit.py", "pose_c2w_canonical=TG[q]"),
               "gate9_oracle_ratio": loc("tools/gate9_depth_occupancy/audit.py", "oracle = float(np.median(zl[4][sup[4]]"),
               "gate9_Gs_depth": loc("tools/gate9_depth_occupancy/audit.py", "dm = zp[q] * oracle"),
               "mapper_depth_times_s": loc("core/mapping/mapper.py", "d_m = s * f.depth_canonical.to(torch.float64)"),
               "mapper_translation_times_s": loc("core/mapping/mapper.py", "T[:3, 3] *= s"),
               "variant_mapper_same": loc("tools/gate9_depth_occupancy/variant_mapper.py", "T[:3, 3] *= s"),
               "deployable_L_canonical_poses": loc("tools/local_registration_audit/geometry_audit.py", "TLc = [np.linalg.inv("),
               "deployable_L_force_clip_scale": loc("tools/local_registration_audit/geometry_audit.py", "m.scale_state.force(sc)"),
               "moge_depth_is_z": loc("moge_gauge/calibrated.py", 'z = points[..., 2]'),
               "moge_depth_z_assert": loc("moge_gauge/calibrated.py", 'depth != points[...,2]')}
    res = {"clips": rows, "code_locations": codeloc,
           "previous_gap_closed_Gs": {"value": old_gap, "Gs_support": "dense LingBot pixels passing the confidence gate (GF), oracle ratio fitted on the anchor frame's LiDAR-supported pixels only",
                                      "PD_support": "LiDAR z-buffer support, no confidence gate", "comparable": False}}
    os.makedirs(D.ART, exist_ok=True); json.dump(res, open(os.path.join(D.ART, "scale_audit.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
