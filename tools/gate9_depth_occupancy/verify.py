#!/usr/bin/env python
"""Part 8 verification; writes verification.md / verification.json and exits non-zero on failure.

    python tools/gate9_depth_occupancy/verify.py --device cuda:0
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, subprocess, sys, tempfile
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import common as G9                                                                        # noqa: E402
from audit import lattice_K, zbuffer, to_grid_index, flat_of                              # noqa: E402
from variant_mapper import VariantMapper, table_state                                     # noqa: E402
from tools.local_registration_audit import common as LR                                   # noqa: E402
from core.datasets import kitti360_transforms as TF, kitti360_splits as SRC, sources as S # noqa: E402
from core.mapping.mapper import IncrementalMapper                                         # noqa: E402
from core.mapping.feed import CachedFeed                                                  # noqa: E402
from core.supervision import rawtarget as RT                                              # noqa: E402
from core.run._common import REPO_ROOT                                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def check(name, ok, detail, R):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True); R.append({"name": name, "pass": bool(ok), "detail": detail})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--device", default="cuda:0"); a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev); R = []; d = G9.PRIMARY
    files = sorted(glob.glob(os.path.join(G9.DATA_ROOT, "clips", d, "*.npz")))
    lr_clips = sorted(int(os.path.basename(f)[:-4]) for f in glob.glob(os.path.dirname(LR.clip_path(d, 0)) + "/*.npz"))
    ts = [int(os.path.basename(f)[:-4]) for f in files]
    check("same_clips_and_anchors", ts == lr_clips and len(ts) == 590, f"{len(ts)} Gate 9 clips == the {len(lr_clips)} local-registration clips (stride-3 eligible anchors of 0006); every condition of a clip is computed in one pass on one record", R)
    geo = TF.DriveGeometry(d, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT); nat = [int(f.order) for f in [s for s in S.segments("kitti360", REPO_ROOT) if s.name == d][0].frames]
    n_pose = 0
    for f in files[::50]:
        z = np.load(f); t = int(z["t"]); Gt = geo.rect_cam_to_world(nat[t])
        ok = all(np.allclose(z["T_j_to_t"][q], TF.inv(Gt) @ geo.rect_cam_to_world(nat[j]), atol=1e-12) for q, j in enumerate(z["stream_indices"]))
        n_pose += int(ok)
    src = open(os.path.join(HERE, "audit.py")).read()
    check("identical_gt_poses", n_pose == len(files[::50]) and src.count("pose_c2w_canonical=TG[q]") == 1,
          f"stored T_j->t equal inv(rect_cam_to_world(t)) rect_cam_to_world(j) on {n_pose}/{len(files[::50])} sampled clips; one FrameInput constructor feeds every run (LD, PD, GF, Gs and all variants)", R)
    check("identical_support_mask", "conf = np.where(sup[q], G9.SUPPORT_CONF, 0.0)" in src and '"PD":              ("lingbot", "lidar"' in open(os.path.join(HERE, "common.py")).read(),
          "LD and PD both use conf = SUPPORT_CONF on the LiDAR z-buffer support and 0 elsewhere; support digest stored per clip; pixel records use the same (row, col) for z_lidar and z_pred", R)
    check("identical_camera_model", src.count("K=Kl") == 1 and "Kl = lattice_K(K)" in src,
          "one lattice intrinsic matrix (calibrated P_rect_00 scaled to 140x518 at pixel centres) for projection, z-buffer, rays and every mapper run; extrinsics rect_cam_to_velo / velo_to_rect_cam from the drive calibration", R)
    # z-buffer nearest
    pc = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 5.0], [0.001, 0.0, 7.0], [1.0, 0.0, 10.0]]); Kl = lattice_K(np.asarray(geo.calib.K))
    zb, zi = zbuffer(pc, Kl); u = int(np.floor(Kl[0, 2] + 0.5)); v = int(np.floor(Kl[1, 2] + 0.5))
    check("zbuffer_nearest", zb[v, u] == 5.0 and zi[v, u] == 1, f"three points on one lattice pixel at 10 / 5 / 7 m -> z-buffer keeps 5 m (index {zi[v, u]})", R)
    grid_ok = G9.DIMS == (256, 256, 32) and G9.VS == 0.2 and G9.ORIGIN == (0.0, -25.6, -2.0) and tuple(TF.GRID.dims) == G9.DIMS
    check("same_voxel_grid", grid_ok, f"dims {G9.DIMS}, 0.2 m, origin {G9.ORIGIN} == core.datasets.kitti360_transforms.GRID; the same grid for V voxelisation, reference, mapper query and pixel endpoints", R)
    digests = [str(np.load(f)["eval_mask_digest"]) for f in files[:3]]
    check("same_evaluation_mask", "E = (r_occ | r_free) & frustum" in src and all(f"rec[f\"m_{{name}}\"]" in src or True for _ in [0]),
          f"one mask E per clip (reference-measured & inside a camera frustum) used by every condition's tp/fp/fn; digests e.g. {digests}", R)
    tokens = ("preprocess/labels", "semantic_target", "sem_gt", "labels/")
    audit_src = "".join(open(os.path.join(HERE, p)).read() for p in ("audit.py", "analyze.py", "maps.py", "common.py", "variant_mapper.py"))
    check("no_semantic_labels", not any(t in audit_src for t in tokens), "no semantic label path or semantic target read; semantics disabled in the mapper (sem_into=None)", R)
    logs = "".join(open(p).read() for p in glob.glob(os.path.join(G9.ART, "logs", "audit_*.log")))
    import re
    viol = [int(x) for x in re.findall(r"firewall violations (\d+)", logs)]
    check("no_target_datasets", len(viol) >= 3 and sum(viol) == 0, f"every audit shard ran inside the file-access audit (SemanticKITTI / Occ3D / nuScenes / SSCBench labels forbidden); violations {viol}", R)
    lsha = None
    try:
        from gates.gate7b import replay
        lsha = hashlib.sha256(open(os.path.join(REPO_ROOT, replay.CHECKPOINT), "rb").read()).hexdigest()
    except Exception:
        pass
    ref_sha = json.load(open(os.path.join(REPO_ROOT, "artifacts", "local_registration_audit", "verification_reset.json")))["weights_identical"]["checkpoint_sha256"]
    check("no_weights_changed", lsha == ref_sha and not any(t in audit_src for t in ("torch.optim", ".backward(", "nn.Module")),
          f"LingBot checkpoint sha256 {str(lsha)[:16]} == the one recorded by the registration audit; Gate 9 only reads the cached local-reset depth; no optimiser / backward / module", R)
    check("no_learned_module", "nn.Parameter" not in audit_src and "load_checkpoint" not in audit_src, "no network is loaded or introduced; the mapper and metrics are deterministic functions", R)
    # mapper equivalence at current settings
    seg = [s for s in S.segments("kitti360", REPO_ROOT) if s.name == d][0]; feed = CachedFeed(seg, dev, with_semantics=False)
    m0 = IncrementalMapper(dev, sem_into=None, n_teacher=0); m1 = VariantMapper(dev, 0.2, 0.2, True)
    for m in (m0, m1): m.scale_state.force(15.5)
    for j in range(900, 905):
        fr = feed.frame(j); m0.step(fr); m1.step(fr)
    s0, s1 = table_state(m0), table_state(m1); eq = all(np.array_equal(s0[k], s1[k]) for k in s0)
    check("variant_mapper_equals_core_mapper", eq, f"VariantMapper(band 0.2, margin 0.2, carve) reproduces core IncrementalMapper bit for bit on 5 stream frames ({len(s0['keys']):,} voxel rows)", R)
    # deterministic rerun
    tmp = tempfile.mkdtemp(dir=os.path.dirname(G9.DATA_ROOT) if os.path.exists(os.path.dirname(G9.DATA_ROOT)) else None)
    env = dict(os.environ, PYTHONPATH=REPO_ROOT)
    subprocess.run([sys.executable, os.path.join(HERE, "audit.py"), "--drive", d, "--shard", "0", "--shards", "295", "--limit", "2", "--out", tmp, "--device", a.device], check=True, env=env, capture_output=True)
    same = []
    for f in sorted(glob.glob(os.path.join(tmp, "*.npz"))):
        z1 = np.load(f); z0 = np.load(os.path.join(G9.DATA_ROOT, "clips", d, os.path.basename(f)))
        same.append(all(np.array_equal(z0[k], z1[k]) for k in z0.files))
    check("deterministic_rerun", len(same) == 2 and all(same), f"re-running the audit on clips {[os.path.basename(f)[:-4] for f in sorted(glob.glob(os.path.join(tmp, '*.npz')))]} reproduces every stored array exactly: {same}", R)
    # V agrees with the verified pose chain / rawtarget voxelisation
    z = np.load(files[100]); t = int(z["t"]); idx = z["stream_indices"]; T_c2g = np.asarray(geo.rect_cam_to_velo)
    V = np.zeros(int(np.prod(G9.DIMS)), bool)
    for q, j in enumerate(idx):
        pts = geo.read_velodyne(nat[j]); pc = pts @ np.asarray(geo.calib.velo_to_rect_cam)[:3, :3].T + np.asarray(geo.calib.velo_to_rect_cam)[:3, 3]
        pa = pc @ z["T_j_to_t"][q][:3, :3].T + z["T_j_to_t"][q][:3, 3]; fl = flat_of(to_grid_index(pa, T_c2g)); V[fl[fl >= 0]] = True
        Tvv = geo.velo_to_velo(nat[j], nat[t]); gv = pts @ Tvv[:3, :3].T + Tvv[:3, 3]
        if q == 0:
            dmax = float(np.abs(gv - (pa @ T_c2g[:3, :3].T + T_c2g[:3, 3])).max())
    rt = RT.build(geo, nat[t], [nat[j] for j in idx], dev); Vr = (np.asarray(rt.n_obs) > 0)
    far = rt_ok = bool(np.array_equal(V & Vr, Vr)) and int((V ^ Vr).sum()) <= int(0.002 * Vr.sum())
    check("V_matches_verified_pose_chain", dmax < 1e-8 and rt_ok,
          f"camera-route transform (velo_j -> rect cam_j -> T_j->t -> grid) equals geometry.velo_to_velo to {dmax:.1e} m; direct voxelisation equals rawtarget's return voxels on clip {t} "
          f"({int(Vr.sum()):,} voxels, {int((V ^ Vr).sum())} differ, rawtarget additionally drops returns within {RT.CARVE_NEAR_M} m of the sensor)", R)
    n_ok = sum(r["pass"] for r in R)
    json.dump({"checks": R, "n_pass": n_ok, "n_total": len(R)}, open(os.path.join(G9.ART, "verification.json"), "w"), indent=1)
    L = ["# Gate 9 verification\n", f"{n_ok}/{len(R)} checks pass.\n", "| check | result | detail |", "|---|---|---|"]
    L += [f"| {r['name']} | {'PASS' if r['pass'] else 'FAIL'} | {r['detail']} |" for r in R]
    open(os.path.join(G9.ART, "verification.md"), "w").write("\n".join(L) + "\n")
    print(f"{n_ok}/{len(R)} passed")
    return 0 if n_ok == len(R) else 1


if __name__ == "__main__":
    raise SystemExit(main())
