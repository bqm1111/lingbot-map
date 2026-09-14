#!/usr/bin/env python
"""Verification record for the controlled depth-source comparison; writes verification.{md,json},
exits non-zero on any failure. Covers every drive with comparison records.

    python tools/depth_source_comparison/verify.py --device cuda:0
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, re, subprocess, sys, tempfile
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dsc as D                                                                            # noqa: E402
sys.path.insert(0, D.G9_DIR); sys.path.insert(0, D.REPO_ROOT)
import common as G9                                                                        # noqa: E402
from audit import lattice_K                                                                # noqa: E402  (Gate 9)
from variant_mapper import VariantMapper, table_state                                     # noqa: E402  (Gate 9)
from tools.local_registration_audit import common as LR                                    # noqa: E402
from core.datasets import kitti360_transforms as TF, kitti360_splits as SRC, sources as S  # noqa: E402
from core.mapping.mapper import IncrementalMapper                                          # noqa: E402
from core.mapping.feed import CachedFeed                                                   # noqa: E402
from core.run._common import REPO_ROOT                                                     # noqa: E402

G9_DATA = os.path.join(G9.DATA_ROOT, "clips")
MOGE_SHA = "3eefd4abb2102f38f12b2d1992e5ff15e4923e5431c67dd494afe157e0111cd5"      # artifacts/gate5*/preflight.json weight_sha256
MOGE_TOL = {"median_rel": 1e-3, "p99_rel": 1e-2}                                   # fixed before re-inference; caches are float16


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--device", default="cuda:0"); a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev); R = []
    def check(name, ok, detail):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True); R.append({"name": name, "pass": bool(ok), "detail": detail})
    drives = [d for d in (D.PRIMARY,) + D.ROBUST if glob.glob(os.path.join(D.DATA_ROOT, "clips", d, "*.npz"))]
    recs = {d: sorted(glob.glob(os.path.join(D.DATA_ROOT, "clips", d, "*.npz"))) for d in drives}
    src = open(os.path.join(HERE, "compare_sources.py")).read(); sa = json.load(open(os.path.join(D.ART, "scale_audit.json")))
    from moge_gauge.calibration import calibrated_fov_x_deg
    # 1-2 clips and anchors
    ok1, det = True, []
    for d in drives:
        a10 = [os.path.basename(f) for f in recs[d]]; alr = sorted(os.path.basename(f) for f in glob.glob(os.path.dirname(LR.clip_path(d, 0)) + "/*.npz"))
        a9 = sorted(os.path.basename(f) for f in glob.glob(os.path.join(G9_DATA, d, "*.npz"))); ok1 &= a10 == alr == a9; det.append(f"{d[17:21]} {len(a10)}")
    check("01_same_clips", ok1, f"comparison clips == local-reset clips == Gate 9 clips ({', '.join(det)})")
    n = dict(anchor=0, pose=0, K=0, mask=0, sup=0, repro=0, tot=0, s1bad=0); fovs = set(); fov_cal = {}
    for d in drives:
        geo = TF.DriveGeometry(d, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT); Kl = lattice_K(np.asarray(geo.calib.K, np.float64)); fov_cal[d[17:21]] = calibrated_fov_x_deg(float(geo.calib.K[0, 0]) * 518 / 1408, 518)
        nat = [int(f.order) for f in [s for s in S.segments(SRC.source_of(d), REPO_ROOT) if s.name == d][0].frames]
        for i, f in enumerate(recs[d]):
            z = np.load(f); g = np.load(os.path.join(G9_DATA, d, os.path.basename(f))); lr = np.load(LR.clip_path(d, int(z["t"]))); n["tot"] += 1
            n["anchor"] += int(np.array_equal(z["stream_indices"], lr["stream_indices"]) and np.array_equal(z["keys"], lr["keys"]) and np.array_equal(z["stream_indices"], g["stream_indices"]))
            if i % 10 == 0:
                Gt = geo.rect_cam_to_world(nat[int(z["t"])]); n["pose"] += int(all(np.array_equal(z["T_j_to_t"][q], TF.inv(Gt) @ geo.rect_cam_to_world(nat[j])) for q, j in enumerate(z["stream_indices"])))
            else:
                n["pose"] += int(np.array_equal(z["T_j_to_t"], g["T_j_to_t"]))
            n["K"] += int(np.array_equal(z["K_lattice"], g["K_lattice"]) and np.array_equal(z["K_lattice"], Kl))
            n["mask"] += int(str(z["eval_mask_digest"]) == str(g["eval_mask_digest"]) and int(z["n_eval"]) == int(g["n_eval"]))
            n["sup"] += int(str(z["support_digest"]) == str(g["support_digest"]))
            n["s1bad"] += int((z["n_pixels_S1"] > z["n_pixels_S0"]).any() or (z["n_pixels_S0"] > z["n_pixels_g9"]).any())
            n["repro"] += int(all(np.array_equal(z[f"m_{k}"], g[v]) for k, v in D.G9_REPRO.items()))
            fovs |= {round(float(x), 4) for x in z["moge_fov_x_deg"]}
    check("02_same_anchors", n["anchor"] == n["tot"], f"stream indices and frame keys equal the local-reset clip file and Gate 9 record on {n['anchor']}/{n['tot']} clips")
    order_ok = src.index('masks = {"g9"') < src.index('depth = {"lidar"') and src.count("support = masks[mkey]") == 1 and src.count("conf = np.where(support[q], G9.SUPPORT_CONF, 0.0)") == 1
    check("03_identical_pixels_and_weights", n["sup"] == n["tot"] and n["s1bad"] == 0 and order_ok,
          f"Gate 9 LiDAR support digest equal on {n['sup']}/{n['tot']} clips; S1 / S0 built from support, MoGe validity, LingBot finiteness and the frozen conf gate BEFORE any depth array is formed, so the mask cannot depend on the compared depth values; "
          f"one support line and one conf line (SUPPORT_CONF inside the mask, 0 outside) for every run; depth metrics use the same rows (in_S1 / in_S0 flags) for every source")
    check("04_same_gt_relative_poses", n["pose"] == n["tot"] and src.count("pose_c2w_canonical=TG[q]") == 1 and src.count("mp.scale_state.force(1.0)") == 1,
          f"T_j->t recomputed from GT cam0_to_world on every 10th clip, equal to Gate 9 on the rest ({n['pose']}/{n['tot']}); metric translations enter the mapper with scale forced to 1 (translation x 1)")
    fov_ok = all(any(abs(x - v) < 1e-3 for v in fov_cal.values()) for x in fovs)
    moge_src = open(os.path.join(REPO_ROOT, "moge_gauge", "calibrated.py")).read()
    check("05_same_camera_and_moge_convention", n["K"] == n["tot"] and fov_ok and "z = points[..., 2]" in moge_src and "depth != points[...,2]" in moge_src,
          f"lattice K equal to Gate 9 and to lattice_K(P_rect_00) on {n['K']}/{n['tot']} clips; MoGe-2 received fov_x {sorted(fovs)} deg == 2 atan(518 / (2 fx 518/1408)) {({k: round(v, 4) for k, v in fov_cal.items()})} on the LingBot-preprocessed 140x518 RGB, "
          "so MoGe and LingBot depth share the lattice without resampling; MoGe depth is optical-axis z in metres (moge_gauge/calibrated.py asserts depth == points[...,2] at inference), so no ray-to-z conversion and no LingBot scale is applied to it")
    check("06_same_evaluation_masks", n["mask"] == n["tot"], f"evaluation-mask digest and size equal to Gate 9 on {n['mask']}/{n['tot']} clips")
    check("07_same_voxel_grid", G9.DIMS == (256, 256, 32) and G9.VS == 0.2 and G9.ORIGIN == (0.0, -25.6, -2.0) and tuple(TF.GRID.dims) == G9.DIMS, f"dims {G9.DIMS}, 0.2 m, origin {G9.ORIGIN} == core GRID")
    freeze9 = json.load(open(os.path.join(REPO_ROOT, "artifacts", "gate9_depth_occupancy", "code_freeze.json")))
    core_same = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "core/mapping"], cwd=REPO_ROOT).returncode == 0
    seg = [s for s in S.segments("kitti360", REPO_ROOT) if s.name == D.PRIMARY][0]; feed = CachedFeed(seg, dev, with_semantics=False)
    m0 = IncrementalMapper(dev, sem_into=None, n_teacher=0); m1 = VariantMapper(dev, 0.2, 0.2, True)
    for m in (m0, m1): m.scale_state.force(15.5)
    for j in range(900, 905):
        fr = feed.frame(j); m0.step(fr); m1.step(fr)
    s0, s1 = table_state(m0), table_state(m1); eq = all(np.array_equal(s0[k], s1[k]) for k in s0)
    hashes_ok = all(sha(os.path.join(D.G9_DIR, p)) == freeze9[p] for p in ("variant_mapper.py", "audit.py", "common.py"))
    check("08_same_mapper_code", hashes_ok and core_same and eq and src.count("VariantMapper(dev, 0.2, 0.2, True)") == 1 and src.count("VariantMapper(") == 1,
          f"Gate 9 variant_mapper.py / audit.py / common.py match Gate 9's code freeze; core/mapping unchanged vs HEAD; VariantMapper(band 0.2, margin 0.2, carve) == core IncrementalMapper bit for bit ({len(s0['keys']):,} rows); one construction, no parameter changed")
    check("09_gate9_reproduced", n["repro"] == n["tot"], f"Gate 9 LD, PD, LD_gate and PD_gate metric vectors reproduced exactly on {n['repro']}/{n['tot']} clips (original numbers kept separately labelled)")
    from gates.gate7b import replay
    lsha = sha(os.path.join(REPO_ROOT, replay.CHECKPOINT)); ref_sha = json.load(open(os.path.join(REPO_ROOT, "artifacts", "local_registration_audit", "verification_reset.json")))["weights_identical"]["checkpoint_sha256"]
    all_src = "".join(open(os.path.join(HERE, p)).read() for p in ("compare_sources.py", "analyze.py", "dsc.py", "scale_audit.py"))
    check("10_frozen_lingbot", lsha == ref_sha and all(c["1_rerun_depth_bit_identical"] for c in sa["clips"]) and not any(k in all_src for k in ("torch.optim", ".backward(", "nn.Parameter")),
          f"checkpoint sha256 {lsha[:16]} == registration audit's; fresh local-reset re-inference reproduces the cached depth bit for bit on clips {[c['t'] for c in sa['clips']]}; no optimiser / backward / parameter")
    from moge_gauge.calibrated import CalibratedMoGe
    from gates.scale_gate.config import load_config
    g5 = load_config("configs/gate5/moge_metric_gauge.yaml")
    moge = CalibratedMoGe(dev, moge_src=g5.moge.src_dir, hf_repo=g5.moge.hf_repo, revision=g5.moge.hf_revision)
    trainable = sum(p.requires_grad for p in moge.model.parameters()); meds, p99s, done = [], [], []
    for d in drives:
        srcd = SRC.source_of(d); segd = [s for s in S.segments(srcd, REPO_ROOT) if s.name == d][0]; bykey = {f.key: f for f in segd.frames}
        geo = TF.DriveGeometry(d, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT); fov = calibrated_fov_x_deg(float(geo.calib.K[0, 0]) * 518 / 1408, 518); load = D.moge_loader(srcd)
        for f in recs[d][:: max(len(recs[d]) // 2, 1)][:2]:
            key = str(np.load(f)["keys"][4]); cd, cm, _ = load(key)
            o = moge.infer_calibrated(replay.load_images([bykey[key].path])[0].numpy(), fov_x_deg=fov); v = cm & np.asarray(o.mask, bool) & (cd > 0)
            rel = np.abs(o.depth_z[v].astype(np.float64) - cd[v]) / cd[v]; meds.append(float(np.median(rel))); p99s.append(float(np.percentile(rel, 99))); done.append(f"{d[17:21]}:{key}")
    check("11_frozen_moge", moge.weight_sha256 == MOGE_SHA and trainable == 0 and max(meds) <= MOGE_TOL["median_rel"] and max(p99s) <= MOGE_TOL["p99_rel"],
          f"{g5.moge.hf_repo}@{g5.moge.hf_revision[:8]} model.pt sha256 {str(moge.weight_sha256)[:16]} == Gate 5 record; 0 trainable; re-inference on {done} matches the cache (median rel {max(meds):.1e}, p99 {max(p99s):.1e}; tolerances {MOGE_TOL})")
    del moge; torch.cuda.empty_cache()
    from gates.gate7b import scale as SC
    lr_src = open(os.path.join(REPO_ROOT, "tools", "local_registration_audit", "run_local_lingbot.py")).read().lower()
    ns = nst = 0
    for d in drives:
        load = D.moge_loader(SRC.source_of(d))
        for f in recs[d][::25]:
            z = np.load(LR.clip_path(d, int(np.load(f)["t"]))); logs = []
            for q in range(5):
                md, mm, _ = load(str(z["keys"][q])); logs.append(SC.frame_candidate(md.astype(np.float32), mm, z["pred_depth"][q].astype(np.float32), z["pred_depth_conf"][q].astype(np.float32))["log_s"])
            ns += int(np.exp(float(np.nanmedian(logs))) == float(z["scale"]) == float(np.load(f)["l_scale"])); nst += 1
    sa_ok = all(c["2_equal"] and c["2_causal_frames_t-4..t"] and c["3_gate9_zpred_equals_cached_x_scale"] and c["4_gate9_convention_equals_stored_PD_map"] and c["4_deployable_convention_xor_voxels"] == 0 and c["7_equal"] for c in sa["clips"])
    check("12_lbm_scale_once_causal_no_lidar", ns == nst and sa_ok and not any(k in lr_src for k in ("velodyne", "lidar", "rawtarget")) and src.count("canon[q] * l_scale") == 1,
          f"clip scale = exp(median over frames t-4..t of frame_candidate(MoGe-2, LingBot)) recomputed exactly on {ns}/{nst} sampled clips; scale_audit.json: Gate 9 depth == cached x scale, mapper forced 1, Gate 9 map == deployable convention map (0 voxel XOR) on all audited clips; "
          "LB-M applies canon x scale once; run_local_lingbot.py has no LiDAR access")
    lbo = [ln.strip() for ln in src.splitlines() if "s_LBO" in ln]
    check("13_lbo_oracle_isolated", len(lbo) == 2 and "fit(masks" in lbo[0] and lbo[1].startswith('"LBO": [canon[q] * s_LBO'),
          "the LiDAR-fitted scalars are computed once per clip (median LiDAR / cached LingBot over the mask) and used only in the LB-O depth entries; masks, LB-M, MG, LD and the evaluation never read them")
    logs = "".join(open(p).read() for p in glob.glob(os.path.join(D.ART, "logs", "compare_*.log")))
    viol = [int(x) for x in re.findall(r"firewall violations (\d+)", logs)]; miss = [int(x) for x in re.findall(r"frames without MoGe (\d+)", logs)]
    # Amendment (after a first run flagged it): the static scan exempts exactly one line, the printed report
    # disclaimer in analyze.py ("... says nothing about Occ3D numbers."), which is prose, opens no file and
    # reads no data. Every other occurrence of a forbidden token still fails the check.
    ALLOW = ('"evaluation mask and frozen mapper. Not completion IoU; says nothing about Occ3D numbers.\\n",',)
    toks, allowed = [], []
    for ln in all_src.splitlines():
        hit = [t for t in SRC.FORBIDDEN_PATH_TOKENS if t in ln] + [t for t in ("semantickitti", "occ3d", "nuscenes") if t in ln.lower()]
        if hit:
            (allowed if ln.strip() in ALLOW else toks).append(hit)
    check("14_no_target_dataset_access", len(viol) >= len(drives) and sum(viol) == 0 and sum(miss) == 0 and not toks,
          f"all comparison shards ran inside the strict file-access audit (SemanticKITTI / Occ3D / nuScenes / SSCBench labels forbidden): violations {viol}; frames without MoGe {miss}; "
          f"forbidden tokens in code outside the allow-listed report disclaimer: {toks}; allow-listed prose lines: {len(allowed)} (amended after the first run flagged the disclaimer string)")
    tmp = tempfile.mkdtemp(dir=D.DATA_ROOT); env = dict(os.environ, PYTHONPATH=REPO_ROOT)
    subprocess.run([sys.executable, os.path.join(HERE, "compare_sources.py"), "--drive", D.PRIMARY, "--shard", "7", "--shards", "295", "--limit", "2", "--out", tmp, "--device", a.device], check=True, env=env, capture_output=True)
    same = []
    for f in sorted(glob.glob(os.path.join(tmp, "*.npz"))):
        z1 = np.load(f); z0 = np.load(os.path.join(D.DATA_ROOT, "clips", D.PRIMARY, os.path.basename(f))); same.append(all(np.array_equal(z0[k], z1[k]) for k in z0.files)); os.remove(f)
    os.rmdir(tmp)
    check("15_deterministic_rerun", len(same) == 2 and all(same), f"re-running two 0006 clips reproduces every stored array exactly: {same}")
    n_ok = sum(r["pass"] for r in R)
    json.dump({"checks": R, "n_pass": n_ok, "n_total": len(R), "drives": drives}, open(os.path.join(D.ART, "verification.json"), "w"), indent=1)
    L = ["# Verification record\n", f"{n_ok}/{len(R)} checks pass (drives: {', '.join(d[17:21] for d in drives)}).\n", "| check | result | detail |", "|---|---|---|"]
    L += [f"| {r['name']} | {'PASS' if r['pass'] else 'FAIL'} | {r['detail']} |" for r in R]
    open(os.path.join(D.ART, "verification.md"), "w").write("\n".join(L) + "\n"); print(f"{n_ok}/{len(R)} passed")
    return 0 if n_ok == len(R) else 1


if __name__ == "__main__":
    raise SystemExit(main())
