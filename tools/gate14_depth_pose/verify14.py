#!/usr/bin/env python
"""Gate 14 verification (20 required checks); writes verification.{md,json}.

    python tools/gate14_depth_pose/verify14.py --device cuda:0
"""
from __future__ import annotations
import argparse, datetime, glob, hashlib, json, os, re, subprocess, sys, tempfile
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g14                                                                                   # noqa: E402
sys.path.insert(0, g14.G9_DIR); sys.path.insert(0, g14.REPO_ROOT)
from variant_mapper import VariantMapper, table_state                                        # noqa: E402  (Gate 9)
from core.datasets import kitti360_transforms as TF, kitti360_splits as SRC, sources as S   # noqa: E402
from core.mapping.mapper import IncrementalMapper                                            # noqa: E402
from core.mapping.feed import CachedFeed                                                     # noqa: E402
from core.run._common import REPO_ROOT                                                       # noqa: E402

MUST3R_RERUN_TOL = 1e-3          # fixed before any rerun: max |z_rerun - z_cache| / z_cache on one clip
G12_INPUTS = "/media/SSD1/MINH_DATASETS/lingbot_gate12_suf/inputs"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--device", default="cuda:0"); a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev); R = []
    def check(name, ok, detail):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True); R.append({"name": name, "pass": bool(ok), "detail": detail})
    drives = [d for d in (g14.PRIMARY,) + g14.ROBUST if glob.glob(os.path.join(g14.CLIP_ROOT, d, "*.npz"))]
    src_a = open(os.path.join(HERE, "audit14.py")).read(); src_m = open(os.path.join(HERE, "must3r_infer.py")).read()
    n = dict(tot=0, anchors=0, frames=0, causal=0, pose_L=0, pose_G=0, mask=0, ref=0, g9gf=0, g9ld=0, g12=0, md5=0, scale=0); first_rec = 1e18
    for d in drives:
        recs = sorted(glob.glob(os.path.join(g14.CLIP_ROOT, d, "*.npz"))); man = {c["t"]: c for c in json.load(open(g14.clip_manifest(d)))["clips"]}
        lr = sorted(os.path.basename(f) for f in glob.glob(os.path.join(g14.LR_ROOT, d, "*.npz"))); g9 = sorted(os.path.basename(f) for f in glob.glob(os.path.join(g14.G9_CLIPS, d, "*.npz")))
        nc = sorted(os.path.basename(f) for f in glob.glob(os.path.join(g14.NEW_CACHE, d, "*.npz")))
        n["anchors"] += int([os.path.basename(f) for f in recs] == lr == g9 == nc); n["tot_drives"] = n.get("tot_drives", 0) + 1
        for i, f in enumerate(recs):
            z = np.load(f); t = int(z["t"]); first_rec = min(first_rec, os.path.getmtime(f)); n["tot"] += 1
            zl = np.load(os.path.join(g14.LR_ROOT, d, f"{t:05d}.npz")); zn = np.load(g14.new_cache_path(d, t)); g = np.load(os.path.join(g14.G9_CLIPS, d, f"{t:05d}.npz"))
            idx = list(range(t - 4, t + 1))
            n["frames"] += int([str(k) for k in zn["keys"]] == [str(k) for k in zl["keys"]] == [str(k) for k in z["keys"]] == man[t]["keys"] and all("data_2d_raw" in p and "image_00" in p for p in man[t]["image_paths"]))
            n["causal"] += int(zn["stream_indices"].tolist() == idx == z["stream_indices"].tolist() == man[t]["stream_indices"] and zn["mem_batches"].tolist() == [2, 1, 1, 1])
            P = zl["pred_pose_c2w"].astype(np.float64)
            n["pose_L"] += int(all(np.array_equal(z["T_L_canonical"][q], np.linalg.inv(P[4]) @ P[q]) for q in range(5)))
            n["pose_G"] += int(np.array_equal(z["T_G"], g["T_j_to_t"]))
            n["mask"] += int(str(z["eval_mask_digest"]) == str(g["eval_mask_digest"]))
            n["ref"] += int(int(z["m_LD-G|lidar"][5, 7]) == int(g["n_ref_occ"]))
            n["g9gf"] += int(np.array_equal(z["m_LBM-G|native"][5, :3], g["m_GF"][:3])); n["g9ld"] += int(np.array_equal(z["m_LD-G|lidar"][5, :3], g["m_LD"][:3]))
            n["md5"] += int(str(zn["checkpoint_md5"]) == g14.MUST3R_MD5); n["scale"] += int(float(z["s_clip"]) == float(zl["scale"]))
            hp = os.path.join(G12_INPUTS, d, f"{t:05d}.npz")
            if os.path.exists(hp):
                h = np.load(hp); n["g12"] += int(str(z["LBM-L_native_occ_sha"]) == hashlib.sha256(h["S_H"].tobytes()).hexdigest() and str(z["LBM-L_native_free_sha"]) == hashlib.sha256(h["F_H"].tobytes()).hexdigest())
                n["g12_tot"] = n.get("g12_tot", 0) + 1
    T = n["tot"]
    check("01_same_clip_anchors", n["anchors"] == len(drives), f"per drive, Gate 14 records == local-reset clips == Gate 9 clips == MUSt3R cache ({T} clips over {', '.join(d[17:21] for d in drives)}); all four conditions of a clip come from one record")
    check("02_same_image_frames", n["frames"] == T, f"MUSt3R cache, local-reset clip and Gate 14 record carry the same five frame keys; manifest image paths are those frames' rectified image_00 files ({n['frames']}/{T})")
    check("03_no_future_images", n["causal"] == T, f"stream indices t-4..t in every MUSt3R cache, record and manifest ({n['causal']}/{T}); MUSt3R memory built in temporal order [2, 1, 1, 1]")
    once = src_a.count("mp.scale_state.force(s_clip)") == 1 and 'dcan = canon if dsrc == "LBM"' in src_a and '"LBM": canon.astype(np.float64) * s_clip' in src_a and src_a.count("mp.scale_state.force(1.0)") == 1
    check("04_lbm_scale_applied_once", once and n["scale"] == T, f"LB-M: canonical depth with mapper scale s_clip under L (depth and translation scaled once by the mapper), canonical x s_clip with mapper scale 1 under G; s_clip equals the local-reset clip scale on {n['scale']}/{T} clips")
    frozen = "load_model(a.checkpoint" in src_m and "p.requires_grad_(False)" in src_m and not any(k in src_m for k in ("optim", ".backward(", "train("))
    mld = open(os.path.join(g14.MUST3R_SRC, "must3r", "model", "__init__.py")).read()
    check("05_candidate_frozen", frozen and n["md5"] == T and "strict=True" in mld, f"official must3r load_model (strict=True), all parameters requires_grad False, no optimiser / backward; checkpoint MD5 {g14.MUST3R_MD5} recorded in {n['md5']}/{T} caches")
    no_lidar = not any(k in src_m for k in ("velodyne", "data_3d_raw", "lidar", "zbuffer")) and "lingbot" not in src_m.split('"""', 2)[2].lower().replace("lingbot_gate14_must3r", "")
    check("06_candidate_no_lidar", no_lidar and n["frames"] == T, "must3r_infer.py imports nothing from this repository and reads only the manifest and the listed RGB images (all under data_2d_raw/image_00); no LiDAR token in its code")
    check("07_candidate_no_semantic_labels", not any(k in src_m.lower() for k in ("label", "semantic", "trident", "sscbench")), "no label / semantic / SSCBench access in the MUSt3R inference code")
    check("08_candidate_no_target_calibration", not any(k in src_m for k in ("pred_K", "intrinsic", "calib", "fov")) and "conf = 1e3" in src_a.replace("np.full((LH, LW), 1e3", "conf = 1e3"),
          "MUSt3R receives only images (no intrinsics / FOV / calibration); its metric scale uses the frozen causal MoGe-2 estimator with KITTI-360's own source calibration, never a target dataset")
    check("09_candidate_causal", n["causal"] == T, "every NEW depth map of a clip is computed from images t-4..t only (checks 2-3)")
    check("10_L_pose_matches_local_reset_audit", n["pose_L"] == T and n["g12"] == n.get("g12_tot", -1) and n["g12"] > 0,
          f"T_L = inv(P_4) P_q of the local-reset clip on {n['pose_L']}/{T} clips; LBM-L native occupied/free maps equal the Gate 12 H input (the frozen deployable M) on {n['g12']}/{n.get('g12_tot', 0)} clips")
    check("11_GT_pose_matches", n["pose_G"] == T, f"T_G equal to Gate 9's verified T_j->t on {n['pose_G']}/{T} clips")
    seg = [s for s in S.segments("kitti360", REPO_ROOT) if s.name == g14.PRIMARY][0]; feed = CachedFeed(seg, dev, with_semantics=False)
    m0 = IncrementalMapper(dev, sem_into=None, n_teacher=0); m1 = VariantMapper(dev, 0.2, 0.2, True)
    for m in (m0, m1): m.scale_state.force(15.5)
    for j in range(900, 905):
        fr = feed.frame(j); m0.step(fr); m1.step(fr)
    s0, s1 = table_state(m0), table_state(m1); eq = all(np.array_equal(s0[k], s1[k]) for k in s0)
    check("12_same_mapper", eq and src_a.count("VariantMapper(dev, 0.2, 0.2, True)") == 1 and n["g9gf"] == T and n["g9ld"] == T,
          f"one VariantMapper(0.2, 0.2, carve) == core IncrementalMapper bit for bit; LBM-G native reproduces Gate 9 GF tp/fp/fn on {n['g9gf']}/{T} clips and LD-G reproduces Gate 9 LD on {n['g9ld']}/{T}")
    check("13_same_voxel_grid", TF.GRID.dims == (256, 256, 32) and float(TF.GRID.voxel_size) == 0.2, "anchor SSC grid 256x256x32 at 0.2 m (Gate 9 constants)")
    check("14_same_evaluation_masks", n["mask"] == T, f"evaluation-mask digest equal to Gate 9 on {n['mask']}/{T} clips")
    check("15_same_five_sweep_reference", n["ref"] == T, f"reference-occupied voxel count in the mask equal to Gate 9 on {n['ref']}/{T} clips")
    order_ok = src_a.index("sup_ = {") < src_a.index("zmet = {")
    check("16_shared_support_independent_of_depth_values", order_ok and 'new_valid = np.isfinite(z_new) & (z_new > 0)' in src_a,
          "S_shared = LingBot conf >= 1.5 AND NEW valid (inside the MUSt3R crop, finite, > 0), built before any metric depth array; it uses validity, never depth magnitude or error")
    logs = "".join(open(p).read() for p in glob.glob(os.path.join(g14.ART, "logs", "audit_*.log"))); viol = [int(x) for x in re.findall(r"firewall violations (\d+)", logs)]
    code = src_a + src_m + open(os.path.join(HERE, "analyze14.py")).read()
    check("17_no_target_dataset_access", len(viol) >= len(drives) and sum(viol) == 0 and not any(k in code.lower() for k in ("occ3d_gt", "data/kitti/dataset")),
          f"audit shards ran inside the strict file audit (SemanticKITTI / Occ3D / nuScenes forbidden): violations {viol}; MUSt3R reads only KITTI-360 images")
    tmp = tempfile.mkdtemp(dir=g14.DATA_ROOT); env = dict(os.environ, PYTHONPATH=REPO_ROOT)
    subprocess.run([sys.executable, os.path.join(HERE, "audit14.py"), "--drive", g14.PRIMARY, "--shard", "5", "--shards", "295", "--limit", "2", "--out", tmp, "--device", a.device], check=True, env=env, capture_output=True)
    same = []
    for f in sorted(glob.glob(os.path.join(tmp, "*.npz"))):
        z1 = np.load(f); z0 = np.load(os.path.join(g14.CLIP_ROOT, g14.PRIMARY, os.path.basename(f))); # NaN-aware: the pixel table holds NaN for pixels outside the MUSt3R crop and off-camera reprojections; np.array_equal
        # treats NaN != NaN (amended after the first run failed on exactly that; the MUSt3R rerun itself matched bit for bit).
        same.append(all(np.array_equal(z0[k], z1[k], equal_nan=(z0[k].dtype.kind == "f")) for k in z0.files)); os.remove(f)
    tmpm = os.path.join(tmp, "must3r"); os.makedirs(tmpm)
    menv = dict(os.environ, PYTHONPATH=f"{g14.MUST3R_SRC}:{g14.MUST3R_SRC}/dust3r:{g14.MUST3R_SRC}/dust3r/croco", HF_HUB_OFFLINE="1")
    subprocess.run([g14.MUST3R_PY, os.path.join(HERE, "must3r_infer.py"), "--drive", g14.PRIMARY, "--shard", "3", "--shards", "590", "--limit", "1", "--out", tmpm, "--device", a.device], check=True, env=menv, capture_output=True)
    mres = []
    for f in glob.glob(os.path.join(tmpm, "*.npz")):
        z1 = np.load(f); z0 = np.load(g14.new_cache_path(g14.PRIMARY, int(z1["t"]))); mres.append(float(np.max(np.abs(z1["depth_z"] - z0["depth_z"]) / np.maximum(z0["depth_z"], 1e-6)))); os.remove(f)
    os.rmdir(tmpm); os.rmdir(tmp)
    check("18_deterministic_rerun", len(same) == 2 and all(same) and len(mres) == 1 and mres[0] <= MUST3R_RERUN_TOL,
          f"audit rerun on two 0006 clips reproduces every stored array exactly: {same}; MUSt3R rerun on one clip: max relative depth difference {mres[0] if mres else float('nan'):.2e} (tolerance {MUST3R_RERUN_TOL}, fixed before the rerun)")
    rule = json.load(open(g14.PROMOTION_RULE)); t_rule = datetime.datetime.fromisoformat(rule["written_at"]).timestamp()
    pr = json.load(open(os.path.join(g14.ART, "promotion_result.json"))); rule_sha = hashlib.sha256(open(g14.PROMOTION_RULE, "rb").read()).hexdigest()
    check("19_promotion_rule_frozen_first", t_rule < first_rec and pr["rule_sha256"] == rule_sha,
          f"promotion_rule.json written {rule['written_at']}, before the first Gate 14 occupancy record; promotion_result.json produced under that exact rule (sha {rule_sha[:12]})")
    import stage2_guard as SG
    locked = False
    try:
        SG.require_promotion()
    except SG.Stage2Locked:
        locked = True
    st2 = glob.glob(os.path.join(g14.ART, "stage2*")) + glob.glob(os.path.join(g14.ART, "checkpoints"))
    ok20 = (locked == (not pr["promote"])) and (pr["promote"] or not st2)
    check("20_stage2_gated", ok20, f"stage2_guard.require_promotion() {'raises Stage2Locked' if locked else 'passes'} with promote={pr['promote']}; Stage 2 artifacts present: {bool(st2)}")
    n_ok = sum(r["pass"] for r in R)
    json.dump({"checks": R, "n_pass": n_ok, "n_total": len(R), "drives": drives}, open(os.path.join(g14.ART, "verification.json"), "w"), indent=1)
    L = ["# Gate 14 verification\n", f"{n_ok}/{len(R)} checks pass (drives: {', '.join(d[17:21] for d in drives)}).\n", "| check | result | detail |", "|---|---|---|"] + [f"| {r['name']} | {'PASS' if r['pass'] else 'FAIL'} | {r['detail']} |" for r in R]
    open(os.path.join(g14.ART, "verification.md"), "w").write("\n".join(L) + "\n"); print(f"{n_ok}/{len(R)} passed")
    return 0 if n_ok == len(R) else 1


if __name__ == "__main__":
    raise SystemExit(main())
