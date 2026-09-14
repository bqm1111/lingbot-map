#!/usr/bin/env python
"""Gate 10 verification (15 checks); writes verification.md / verification.json, exits non-zero on failure.
Checks every drive that has Gate 10 clip records.

    python tools/gate10_depth_source/verify_sources.py --device cuda:0
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, re, subprocess, sys, tempfile
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g10 as G10                                                                          # noqa: E402
sys.path.insert(0, G10.G9_DIR); sys.path.insert(0, G10.REPO_ROOT)
import common as G9                                                                        # noqa: E402
from audit import lattice_K                                                                # noqa: E402  (Gate 9)
from variant_mapper import VariantMapper, table_state                                     # noqa: E402  (Gate 9)
from tools.local_registration_audit import common as LR                                   # noqa: E402
from core.datasets import kitti360_transforms as TF, kitti360_splits as SRC, sources as S # noqa: E402
from core.mapping.mapper import IncrementalMapper                                         # noqa: E402
from core.mapping.feed import CachedFeed                                                  # noqa: E402
from core.mapping import depth as D7                                                      # noqa: E402
from core.run._common import REPO_ROOT                                                    # noqa: E402

G9_DATA = os.path.join(G9.DATA_ROOT, "clips")
MOGE_SHA = "3eefd4abb2102f38f12b2d1992e5ff15e4923e5431c67dd494afe157e0111cd5"      # artifacts/gate5*/preflight.json weight_sha256
MOGE_TOL = {"median_rel": 1e-3, "p99_rel": 1e-2}                                   # fixed before re-inference


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--device", default="cuda:0"); a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev); R = []
    def check(name, ok, detail):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True); R.append({"name": name, "pass": bool(ok), "detail": detail})
    drives = [d for d in (G10.PRIMARY,) + G10.ROBUST if glob.glob(os.path.join(G10.DATA_ROOT, "clips", d, "*.npz"))]
    src = open(os.path.join(HERE, "audit_sources.py")).read()
    recs = {}
    for d in drives:
        recs[d] = sorted(glob.glob(os.path.join(G10.DATA_ROOT, "clips", d, "*.npz")))
    # 1 clip ids
    det = []; ok1 = True
    for d in drives:
        t10 = [os.path.basename(f) for f in recs[d]]; tlr = sorted(os.path.basename(f) for f in glob.glob(os.path.dirname(LR.clip_path(d, 0)) + "/*.npz"))
        t9 = sorted(os.path.basename(f) for f in glob.glob(os.path.join(G9_DATA, d, "*.npz")))
        ok1 &= t10 == tlr == t9; det.append(f"{d[17:21]} {len(t10)}")
    check("1_identical_clip_ids", ok1, f"Gate 10 clip set == local-registration clips == Gate 9 clips ({', '.join(det)}); all conditions of a clip come from one record")
    # 2,4,5,6,3 per-clip equality with Gate 9 and the local-reset clips (all clips)
    n_anchor = n_pose = n_mask = n_sup = n_K = n_tot = 0; fov_all = set(); n_match_bad = 0
    for d in drives:
        geo = TF.DriveGeometry(d, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT); Kl = lattice_K(np.asarray(geo.calib.K, np.float64))
        nat = [int(f.order) for f in [s for s in S.segments(SRC.source_of(d), REPO_ROOT) if s.name == d][0].frames]
        for i, f in enumerate(recs[d]):
            z = np.load(f); g = np.load(os.path.join(G9_DATA, d, os.path.basename(f))); lr = np.load(LR.clip_path(d, int(z["t"])))
            n_tot += 1
            n_anchor += int(np.array_equal(z["stream_indices"], lr["stream_indices"]) and np.array_equal(z["keys"], lr["keys"]) and np.array_equal(z["stream_indices"], g["stream_indices"]))
            if i % 10 == 0:
                Gt = geo.rect_cam_to_world(nat[int(z["t"])])
                n_pose += int(all(np.allclose(z["T_j_to_t"][q], TF.inv(Gt) @ geo.rect_cam_to_world(nat[j]), atol=1e-12) for q, j in enumerate(z["stream_indices"])))
            else:
                n_pose += int(np.array_equal(z["T_j_to_t"], g["T_j_to_t"]))
            n_K += int(np.array_equal(z["K_lattice"], g["K_lattice"]) and np.array_equal(z["K_lattice"], Kl))
            n_mask += int(str(z["eval_mask_digest"]) == str(g["eval_mask_digest"]) and int(z["n_eval"]) == int(g["n_eval"]))
            n_sup += int(str(z["support_digest"]) == str(g["support_digest"]))
            n_match_bad += int((z["n_matched_pixels"] > z["n_support_pixels"]).any())
            fov_all |= {round(float(x), 4) for x in z["moge_fov_x_deg"]}
    fov_cal = {}
    from moge_gauge.calibration import calibrated_fov_x_deg
    for d in drives:
        geo = TF.DriveGeometry(d, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT); fov_cal[d[17:21]] = calibrated_fov_x_deg(float(geo.calib.K[0, 0]) * 518 / 1408, 518)
    check("2_identical_anchors", n_anchor == n_tot, f"stream indices and frame keys equal the local-reset clip file and the Gate 9 record on {n_anchor}/{n_tot} clips")
    single_support = src.count("support = sup if sk == \"gate9\" else matched") == 1 and src.count("conf = np.where(support[q], G9.SUPPORT_CONF, 0.0)") == 1
    check("3_identical_matched_pixels", n_sup == n_tot and single_support and n_match_bad == 0,
          f"Gate 9 LiDAR support digest equal on {n_sup}/{n_tot} clips; one matched mask per clip (support AND MoGe-valid) feeds LD, LB, LB-M, M, LB-LiDAR, M-LiDAR through a single support/conf line; depth metrics use the same pixel rows (matched flag) for every source")
    check("4_identical_gt_relative_poses", n_pose == n_tot and src.count("pose_c2w_canonical=TG[q]") == 1,
          f"T_j->t = inv(rect_cam_to_world(t)) rect_cam_to_world(j) recomputed on every 10th clip and equal to the Gate 9 record on the rest ({n_pose}/{n_tot}); one FrameInput constructor for every run")
    fov_ok = all(any(abs(x - v) < 1e-3 for v in fov_cal.values()) for x in fov_all)
    check("5_identical_camera", n_K == n_tot and src.count("K=Kl") == 1 and fov_ok,
          f"lattice K equal to Gate 9's and to lattice_K(calibrated P_rect_00) on {n_K}/{n_tot} clips; extrinsics from the drive calibration; MoGe-2 was given fov_x {sorted(fov_all)} deg == calibrated 2 atan(518 / (2 fx 518/1408)) {({k: round(v, 4) for k, v in fov_cal.items()})} on the same LingBot-preprocessed 140x518 RGB")
    check("6_identical_evaluation_masks", n_mask == n_tot, f"evaluation-mask digest and size equal to Gate 9 on {n_mask}/{n_tot} clips; one mask E per clip for every condition")
    check("7_identical_voxel_grid", G9.DIMS == (256, 256, 32) and G9.VS == 0.2 and G9.ORIGIN == (0.0, -25.6, -2.0) and tuple(TF.GRID.dims) == G9.DIMS,
          f"dims {G9.DIMS}, 0.2 m, origin {G9.ORIGIN} (Gate 9 constants == core GRID) for reference, mapper query and pixel endpoints")
    # 8 mapper code
    freeze9 = json.load(open(os.path.join(REPO_ROOT, "artifacts", "gate9_depth_occupancy", "code_freeze.json")))
    core_diff = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "core/mapping"], cwd=REPO_ROOT).returncode == 0
    seg = [s for s in S.segments("kitti360", REPO_ROOT) if s.name == G10.PRIMARY][0]; feed = CachedFeed(seg, dev, with_semantics=False)
    m0 = IncrementalMapper(dev, sem_into=None, n_teacher=0); m1 = VariantMapper(dev, 0.2, 0.2, True)
    for m in (m0, m1): m.scale_state.force(15.5)
    for j in range(900, 905):
        fr = feed.frame(j); m0.step(fr); m1.step(fr)
    s0, s1 = table_state(m0), table_state(m1); eq = all(np.array_equal(s0[k], s1[k]) for k in s0)
    vm_sha = sha(os.path.join(G10.G9_DIR, "variant_mapper.py")); au_sha = sha(os.path.join(G10.G9_DIR, "audit.py")); cm_sha = sha(os.path.join(G10.G9_DIR, "common.py"))
    check("8_identical_mapper_code", vm_sha == freeze9["variant_mapper.py"] and au_sha == freeze9["audit.py"] and cm_sha == freeze9["common.py"] and core_diff and eq,
          f"Gate 9 variant_mapper.py / audit.py / common.py sha256 equal to Gate 9's code freeze; core/mapping unchanged vs HEAD; VariantMapper(0.2, 0.2, carve) == core IncrementalMapper bit for bit on 5 stream frames ({len(s0['keys']):,} rows)")
    # 9 no mapper parameter change: construction line + exact Gate 9 reproduction
    n_ld = n_pd = 0
    for d in drives:
        for f in recs[d]:
            z = np.load(f); g = np.load(os.path.join(G9_DATA, d, os.path.basename(f)))
            n_ld += int(np.array_equal(z["m_LD_g9"], g["m_LD"])); n_pd += int(np.array_equal(z["m_LBM_g9"], g["m_PD"]))
    one_ctor = src.count("VariantMapper(") == 2 and src.count("VariantMapper(dev, 0.2, 0.2, True)") == 1
    check("9_no_mapper_parameter_change", one_ctor and n_ld == n_tot and n_pd == n_tot,
          f"one mapper construction VariantMapper(band 0.2, margin 0.2, carve on), conf gate / depth range untouched; Gate 9 LD and PD metric vectors reproduced exactly on {n_ld}/{n_tot} and {n_pd}/{n_tot} clips")
    # 10 LingBot weights
    from gates.gate7b import replay
    lsha = sha(os.path.join(REPO_ROOT, replay.CHECKPOINT)); ref_sha = json.load(open(os.path.join(REPO_ROOT, "artifacts", "local_registration_audit", "verification_reset.json")))["weights_identical"]["checkpoint_sha256"]
    all_src = "".join(open(os.path.join(HERE, p)).read() for p in ("audit_sources.py", "analyze_sources.py", "maps_sources.py", "g10.py"))
    check("10_frozen_lingbot", lsha == ref_sha and not any(k in all_src for k in ("torch.optim", ".backward(", "nn.Parameter")),
          f"LingBot checkpoint sha256 {lsha[:16]} == the registration audit's; Gate 10 reads the cached local-reset depth only (Gate 9 PD depth rows reproduced exactly); no optimiser, backward or parameter")
    # 11 MoGe weights + cache reproduction
    from moge_gauge.calibrated import CalibratedMoGe
    from gates.scale_gate.config import load_config
    g5 = load_config("configs/gate5/moge_metric_gauge.yaml")
    moge = CalibratedMoGe(dev, moge_src=g5.moge.src_dir, hf_repo=g5.moge.hf_repo, revision=g5.moge.hf_revision)
    trainable = sum(p.requires_grad for p in moge.model.parameters())
    rel_meds, rel_p99s, done = [], [], []
    for d in drives:
        srcd = SRC.source_of(d); segd = [s for s in S.segments(srcd, REPO_ROOT) if s.name == d][0]; bykey = {f.key: f for f in segd.frames}
        geo = TF.DriveGeometry(d, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT); fov = calibrated_fov_x_deg(float(geo.calib.K[0, 0]) * 518 / 1408, 518)
        index = D7.moge_frame_index("kitti360", REPO_ROOT) if srcd == "kitti360" else None
        for f in recs[d][:: max(len(recs[d]) // 2, 1)][:2]:
            key = str(np.load(f)["keys"][4])
            if srcd == "kitti360":
                cd, cm, _ = D7.load_moge_frame("kitti360", key, index)
            else:
                with np.load(S.moge_path(srcd, key)) as q:
                    cd, cm = q["moge_depth"].astype(np.float32), D7.unpack_mask(q["moge_mask"], q["mask_shape"])
            img = replay.load_images([bykey[key].path])[0].numpy()
            o = moge.infer_calibrated(img, fov_x_deg=fov); v = cm & np.asarray(o.mask, bool) & (cd > 0)
            rel = np.abs(o.depth_z[v].astype(np.float64) - cd[v]) / cd[v]
            rel_meds.append(float(np.median(rel))); rel_p99s.append(float(np.percentile(rel, 99))); done.append(f"{d[17:21]}:{key}")
    ok11 = moge.weight_sha256 == MOGE_SHA and trainable == 0 and max(rel_meds) <= MOGE_TOL["median_rel"] and max(rel_p99s) <= MOGE_TOL["p99_rel"]
    check("11_frozen_moge", ok11, f"MoGe-2 {g5.moge.hf_repo}@{g5.moge.hf_revision[:8]} model.pt sha256 {str(moge.weight_sha256)[:16]} == Gate 5's recorded {MOGE_SHA[:16]}; 0 trainable parameters; re-inference on {done} reproduces the cached depth "
          f"(median rel diff max {max(rel_meds):.1e} <= {MOGE_TOL['median_rel']}, p99 max {max(rel_p99s):.1e} <= {MOGE_TOL['p99_rel']}; caches are float16)")
    del moge; torch.cuda.empty_cache()
    # 12 LB-M scale uses no LiDAR
    from gates.gate7b import scale as SC
    lr_src = open(os.path.join(REPO_ROOT, "tools", "local_registration_audit", "run_local_lingbot.py")).read()
    n_sc = n_sc_tot = 0
    for d in drives:
        srcd = SRC.source_of(d); index = D7.moge_frame_index("kitti360", REPO_ROOT) if srcd == "kitti360" else None
        for f in recs[d][::25]:
            z = np.load(LR.clip_path(d, int(np.load(f)["t"]))); logs = []
            for q in range(5):
                key = str(z["keys"][q])
                if srcd == "kitti360":
                    md, mm, _ = D7.load_moge_frame("kitti360", key, index)
                else:
                    with np.load(S.moge_path(srcd, key)) as qq:
                        md, mm = qq["moge_depth"].astype(np.float32), D7.unpack_mask(qq["moge_mask"], qq["mask_shape"])
                logs.append(SC.frame_candidate(md, mm, z["pred_depth"][q].astype(np.float32), z["pred_depth_conf"][q].astype(np.float32))["log_s"])
            n_sc += int(np.exp(float(np.nanmedian(logs))) == float(z["scale"]) and float(np.load(f)["l_scale"]) == float(z["scale"])); n_sc_tot += 1
    no_lidar = not any(k in lr_src.lower() for k in ("velodyne", "lidar", "rawtarget", "read_velodyne"))
    check("12_lbm_scale_no_lidar", n_sc == n_sc_tot and no_lidar,
          f"LB-M scalar = exp(median_q frame_candidate(MoGe-2 depth, MoGe mask, LingBot depth, LingBot conf)) recomputed from MoGe + LingBot only equals the stored clip scale exactly on {n_sc}/{n_sc_tot} sampled clips; run_local_lingbot.py contains no LiDAR access")
    # 13 oracle isolation
    uses_L = [ln.strip() for ln in src.splitlines() if "s_LBL" in ln]; uses_M = [ln.strip() for ln in src.splitlines() if "s_ML" in ln]
    ok13 = len(uses_L) == 3 and len(uses_M) == 3 and all(("LBL" in ln or "oracle_scale_LBL" in ln) for ln in uses_L[1:]) and "zl_m" in uses_L[0]
    check("13_oracle_isolated", ok13, "the LiDAR oracle scalars are computed once per clip from matched LiDAR/source depth and appear only in the LB-LiDAR / M-LiDAR depth entries and their stored record fields; "
          "no other condition, mask, scale or selection reads them")
    # 14 no target-dataset access
    logs = "".join(open(p).read() for p in glob.glob(os.path.join(G10.ART, "logs", "audit_*.log")))
    viol = [int(x) for x in re.findall(r"firewall violations (\d+)", logs)]; miss = [int(x) for x in re.findall(r"frames without MoGe (\d+)", logs)]
    tokens = [t for t in SRC.FORBIDDEN_PATH_TOKENS if t in all_src] + [t for t in ("semantickitti", "occ3d", "nuscenes") if t in all_src.lower()]
    check("14_no_target_dataset_access", len(viol) >= len(drives) and sum(viol) == 0 and not tokens,
          f"every audit shard ran inside the strict file-access audit (SemanticKITTI, Occ3D, nuScenes, SSCBench labels forbidden): violations {viol}; no forbidden path token in Gate 10 code; frames without MoGe {miss}")
    # 15 deterministic rerun
    tmp = tempfile.mkdtemp(dir=G10.DATA_ROOT); env = dict(os.environ, PYTHONPATH=REPO_ROOT)
    subprocess.run([sys.executable, os.path.join(HERE, "audit_sources.py"), "--drive", G10.PRIMARY, "--shard", "7", "--shards", "295", "--limit", "2", "--out", tmp, "--device", a.device], check=True, env=env, capture_output=True)
    same = []
    for f in sorted(glob.glob(os.path.join(tmp, "*.npz"))):
        z1 = np.load(f); z0 = np.load(os.path.join(G10.DATA_ROOT, "clips", G10.PRIMARY, os.path.basename(f)))
        same.append(all(np.array_equal(z0[k], z1[k]) for k in z0.files))
        os.remove(f)
    os.rmdir(tmp)
    check("15_deterministic_rerun", len(same) == 2 and all(same), f"re-running the audit on two 0006 clips reproduces every stored array (metrics, maps, pixel rows, scalars) exactly: {same}")
    n_ok = sum(r["pass"] for r in R)
    json.dump({"checks": R, "n_pass": n_ok, "n_total": len(R), "drives": drives}, open(os.path.join(G10.ART, "verification.json"), "w"), indent=1)
    L = ["# Gate 10 verification\n", f"{n_ok}/{len(R)} checks pass (drives: {', '.join(d[17:21] for d in drives)}).\n", "| check | result | detail |", "|---|---|---|"]
    L += [f"| {r['name']} | {'PASS' if r['pass'] else 'FAIL'} | {r['detail']} |" for r in R]
    open(os.path.join(G10.ART, "verification.md"), "w").write("\n".join(L) + "\n")
    print(f"{n_ok}/{len(R)} passed")
    return 0 if n_ok == len(R) else 1


if __name__ == "__main__":
    raise SystemExit(main())
