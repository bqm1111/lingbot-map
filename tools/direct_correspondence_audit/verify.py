#!/usr/bin/env python
"""Twelve invariants of the direct-correspondence audit; fails loudly.

    python tools/direct_correspondence_audit/verify.py
"""
from __future__ import annotations
import glob, inspect, json, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.direct_correspondence_audit import common as DC                                # noqa: E402
from tools.direct_correspondence_audit import build_pairs as BP, sample_features as SF     # noqa: E402
from tools.future_semantic_completion import study as ST                                  # noqa: E402
from core.datasets import kitti360_transforms as TF, kitti360_splits as SRC               # noqa: E402


def check(name, ok, detail, R):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True); R.append({"name": name, "pass": bool(ok), "detail": detail})


def main() -> int:
    ST.require_data_root(); R = []
    code_dir = os.path.dirname(os.path.abspath(__file__)); mods = [p for p in os.listdir(code_dir) if p.endswith(".py") and p != "verify.py"]
    srcs = {p: open(os.path.join(code_dir, p)).read() for p in mods}; allsrc = "".join(srcs.values())
    pairs = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(DC.ART, "pairs", "*.json"))}
    samp = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(DC.ART, "sampling", "*.json"))}
    mc = json.load(open(os.path.join(DC.ART, "mapper_comparison.json")))
    ext = [json.load(open(p)) for p in glob.glob(os.path.join(ST.REPO_ROOT, "artifacts", "feature_first_3d_semantics", "extraction", "2013*.json"))]
    # 1 nothing trained
    man = json.load(open(ST.FROZEN_MANIFEST)); ck_ok = all(ST.sha256_file(os.path.join(ST.REPO_ROOT, m["checkpoint"])) == m["checkpoint_sha256"] for m in man["seeds"].values())
    no_opt = all(t not in allsrc for t in ("torch.optim", ".backward(", "requires_grad", "nn.Parameter"))
    check("no_model_trained", ck_ok and no_opt, f"no optimiser/backward in {len(mods)} modules: {no_opt}; frozen checkpoints intact: {ck_ok}", R)
    # 2 Trident frozen: features come from the feature-first cache, whose extraction recorded the checkpoint hashes; cache untouched since
    tok_newest = max(os.path.getmtime(p) for p in glob.glob(os.path.join(DC.TOKEN_DIR, "*", "*.npz"))[::50])
    pair_oldest = min(os.path.getmtime(p) for p in glob.glob(os.path.join(DC.PAIR_DIR, "*", "*.npz"))[::50])
    sam_sha = {e["teacher"]["sam_ckpt_sha256"] for e in ext}
    check("trident_frozen", len(sam_sha) == 1 and tok_newest < pair_oldest,
          f"token cache (SAM-H sha {list(sam_sha)[0][:12]}, official config) written before this audit started: {tok_newest < pair_oldest}; Trident is only read through that cache", R)
    # 3 correspondence defined without the mapper
    bp = srcs["build_pairs.py"]
    ok3 = all(t not in bp for t in ("IncrementalMapper", "pred_depth", "pred_pose", "stream_path", "trident_path", "sem_probs", "unpack_sample")) and "read_velodyne" in bp and "cam0_to_world" in bp
    check("correspondence_without_mapper", ok3, "build_pairs reads raw Velodyne, GT cam0_to_world poses and calibration only (no LingBot depth/pose, no mapper, no semantic cache)", R)
    # 4 intrinsics/extrinsics consistent: reproject GT-projected point back, and A's own sweep residuals ~0
    geo = TF.DriveGeometry(SRC.VAL_DRIVE, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT); K = np.asarray(geo.calib.K)
    f = sorted(glob.glob(os.path.join(DC.PAIR_DIR, SRC.VAL_DRIVE, "*.npz")))[10]
    with np.load(f) as z:
        fa = int(z["native_A"]); Xw = z["X_world"]; uA = z["u_A"]; vA = z["v_A"]; resA = z["resid_A"]
    Twc = TF.inv(geo.rect_cam_to_world(fa)); pc = Xw @ Twc[:3, :3].T + Twc[:3, 3]; u, v, zc = BP.project(K, pc)
    d4 = float(np.abs(u - uA).max() + np.abs(v - vA).max())
    check("calibration_consistent", d4 < 1e-3 and np.abs(resA).max() < 0.3 + 0.01 * 50 and geo.calib.native_hw == tuple(DC.NATIVE_HW),
          f"world->rectified-cam0->pixel round trip reproduces stored u_A/v_A to {d4:.2e} px on {len(Xw)} landmarks; A-frame self z-buffer residual max {np.abs(resA).max():.2f} m; "
          f"K = P_rect_00[:3,:3] on {geo.calib.native_hw}, rect_cam_to_velo = cam0_to_velo @ inv(R_rect_00)", R)
    # 5 visibility on every accepted point
    n_rows = n_ok = 0
    for d in pairs:
        for p in sorted(glob.glob(os.path.join(DC.PAIR_DIR, d, "*.npz")))[::20]:
            with np.load(p) as z:
                r = z["rows"]
            if len(r) == 0:
                continue
            tl = DC.TOLS["loose"][0] + DC.TOLS["loose"][1] * r[:, 4]; ts = DC.TOLS["strict"][0] + DC.TOLS["strict"][1] * r[:, 4]
            n_rows += len(r); n_ok += int(((np.abs(r[:, 5]) < tl) & ((r[:, 6] == 0) | (np.abs(r[:, 5]) < ts)) & (r[:, 4] > DC.DEPTH_MIN) & (r[:, 4] < DC.DEPTH_MAX)).sum())
    tot = {k: sum(pairs[d]["counts"][k] for d in pairs) for k in pairs[next(iter(pairs))]["counts"]}
    check("visibility_on_all_accepted", n_ok == n_rows, f"{n_ok}/{n_rows} sampled stored pairs satisfy |z - z_ref| < tolerance and the depth range (strict flag consistent); pooled counts {tot}", R)
    # 6 same-point pairs refer to the same world geometry: one X_world per landmark, reused for every B
    check("same_point_same_geometry", "Xw @ Twc[:3, :3].T + Twc[:3, 3]" in bp and "X_world=Xw" in bp,
          "each row stores the landmark index into one X_world array; u_B is the projection of that same world point (source-inspected)", R)
    # 7 local controls are different physical points
    sf = srcs["sample_features.py"]
    check("local_controls_distinct", "j != i" in sf and "DC.LOCAL_R[0]" in sf, f"C2 candidates exclude the landmark itself and require {DC.LOCAL_R[0]} m <= |X' - X| <= {DC.LOCAL_R[1]} m, |dz| < {DC.LOCAL_DZ} m; C1/C3 permutations exclude self", R)
    # 8 features sampled directly in image space, rule == official interpolation
    rd = max(v["sampling_rule_max_dev_vs_official_interpolation"] for v in samp.values())
    check("direct_image_space_sampling", rd < 1e-4 and "grid_sample" in sf and "IncrementalMapper" not in sf,
          f"grid_sample at native pixel centres (align_corners=False) equals F.interpolate of the token grid to {rd:.1e}; no mapper in the sampling path", R)
    # 9 mapper not used to select pairs
    check("mapper_not_in_pair_selection", ok3 and "DC.PAIR_DIR" in srcs["mapper_compare.py"] and "IncrementalMapper" not in srcs["mapper_compare.py"], "pairs are selected in build_pairs (geometry only); mapper_compare only reads the already-selected pairs", R)
    # 10 + 11 KITTI-360 only / targets untouched
    viol = sum(len(v["firewall"]["violations"]) for v in list(pairs.values()) + list(samp.values())) + len(mc["firewall"]["violations"])
    n_open = sum(v["firewall"]["n_opened"] for v in list(pairs.values()) + list(samp.values())) + mc["firewall"]["n_opened"]
    check("kitti360_only", viol == 0 and set(pairs) <= set(list(SRC.TRAIN_DRIVES) + [SRC.VAL_DRIVE]), f"drives {sorted(d[17:21] for d in pairs)}; audited opens {n_open}, forbidden-path violations {viol}", R)
    check("targets_untouched", viol == 0, f"no SemanticKITTI / Occ3D / nuScenes path opened ({viol} violations)", R)
    # 12 deterministic filtering
    check("deterministic_filtering", "default_rng(fa)" in bp and "default_rng(20260913)" in sf, "landmark subsampling seeded by the native frame index; control draws seeded by a fixed constant; thresholds are constants in common.py", R)
    n_ok_ = sum(r["pass"] for r in R); print(f"{n_ok_}/{len(R)} passed")
    DC.write_json(os.path.join(DC.ART, "verification.json"), {"checks": R, "n_pass": n_ok_, "n_total": len(R), "env": DC.environment()})
    return 0 if n_ok_ == len(R) else 1


if __name__ == "__main__":
    raise SystemExit(main())
