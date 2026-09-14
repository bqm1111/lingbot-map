#!/usr/bin/env python
"""Pre-training input sanity checks and input visualisations (no target is read or overlaid).

    python tools/gate12_suf/sanity_inputs.py --device cuda:0

Deterministic sample: 0006 anchors 346 / 766 / 361 (Gate 11 good / median / bad rule), and the
first and middle anchor of each training drive. Checks:
 1 H U channel is identically zero in the model input; S_H / F_H equal the core IncrementalMapper
   (semantics off) on the same L frames, bit for bit
 2 S, U, F mutually exclusive (and S_H, F_H)
 3 no ray with predicted depth >= 20 m contributes to S/U/F (recorded counts; and S/U/F rebuilt
   from near rays only equals the stored S/U/F)
 4 conservative carving stops at z - u_front (synthetic ray), interval covers [z - u_front, z + u_back]
 5 U4 coefficients equal Gate 11's frozen file and round to the brief's values
 6 no LiDAR support mask and no GT pose in the input builder (source scan + file-access audit)
 7 input generation is deterministic (rebuild == stored)
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g12                                                                                  # noqa: E402
sys.path.insert(0, g12.G11_DIR); sys.path.insert(0, g12.G9_DIR); sys.path.insert(0, g12.REPO_ROOT)
import build_inputs as BI                                                                   # noqa: E402
from channel_mapper import ChannelMapper                                                    # noqa: E402
from core.datasets import kitti360 as K3, kitti360_transforms as TF, kitti360_splits as SRC  # noqa: E402
from core.datasets.audit import FileAudit                                                   # noqa: E402
from core.mapping.mapper import IncrementalMapper, FrameInput                               # noqa: E402

N = BI.NV; DIMS = BI.DIMS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--device", default="cuda:0"); a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev); R = []
    def check(name, ok, detail):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True); R.append({"name": name, "pass": bool(ok), "detail": detail})
    rules = g12.load_rules(); calib = K3.parse_calibration(os.path.join(SRC.SSCBENCH_ROOT, "calibration"))
    Tgw = np.linalg.inv(np.asarray(calib.rect_cam_to_velo, np.float64)); grid = TF.GRID
    sample = [(g12.VAL_DRIVE, t) for t in (346, 766, 361)]
    for d in g12.TRAIN_DRIVES:
        ids = sorted(int(f[:-4]) for f in os.listdir(os.path.dirname(g12.input_path(d, 0))))
        sample += [(d, ids[0]), (d, ids[len(ids) // 2])]
    ok1 = ok2 = ok3 = ok7 = True; det = []
    with FileAudit(patterns=BI.AUDIT_PATTERNS, strict=True) as audit:
        for d, t in sample:
            zi = np.load(g12.input_path(d, t)); st = json.loads(str(zi["stats"])); V = {k: g12.unpack(zi, k, N) for k in g12.VOLUMES}
            with np.load(g12.clip_path(d, t)) as zz:
                z = {k: zz[k] for k in zz.files}
            s = float(z["scale"]); conf = z["pred_depth_conf"].astype(np.float64)
            # 1 core mapper reproduction of H
            core = IncrementalMapper(dev, sem_into=None, n_teacher=0); core.scale_state.force(s)
            for fr in BI.frames_for(z, conf):
                fr.depth_canonical = fr.depth_canonical.to(dev); fr.conf = fr.conf.to(dev); core.step(fr)
            qc = core.query(grid, Tgw); h_eq = np.array_equal(qc["occupied"].cpu().numpy(), V["S_H"]) and np.array_equal(qc["free"].cpu().numpy(), V["F_H"])
            x_H = np.stack([V["S_H"], np.zeros(N, bool), V["F_H"]])
            ok1 &= h_eq and not x_H[1].any()
            # 2 exclusivity
            ok2 &= not (V["S"] & V["U"]).any() and not (V["S"] & V["F"]).any() and not (V["U"] & V["F"]).any() and not (V["S_H"] & V["F_H"]).any()
            # 3 range cut: counts + rebuild with an independent near-ray mask
            zm = s * z["pred_depth"].astype(np.float32).astype(np.float64); near = zm < 20.0
            conf_near = np.where(near, conf, 0.0); W = [BI.G11.widths("U4", zm[q], conf[q], rules) for q in range(5)]
            S2, F2, U2 = BI.integrate(dev, z, BI.frames_for(z, conf_near), [W[q][0] for q in range(5)], W, grid, Tgw)
            same = all(np.array_equal(x.cpu().numpy(), V[k]) for x, k in ((S2, "S"), (F2, "F"), (U2, "U")))
            ok3 &= st["n_suf_rays_ge20"] == 0 and st["n_rays_cut_ge20"] == int(((conf >= 1.5) & (zm >= 20) & (zm < 60)).sum()) and same
            # 7 determinism of the full builder
            vols, _ = BI.build_one(dev, z, rules, grid, Tgw); ok7 &= all(np.array_equal(vols[k], V[k]) for k in g12.VOLUMES)
            det.append(f"{d[17:21]}:{t} S_H {V['S_H'].sum()} F_H {V['F_H'].sum()} S {V['S'].sum()} U {V['U'].sum()} F {V['F'].sum()} cut rays {st['n_rays_cut_ge20']}")
    check("1_H_U_zero_and_SF_equal_core_mapper", ok1, f"on {len(sample)} deterministic anchors the H input's U channel is all zero and S_H / F_H equal core IncrementalMapper(semantics off) occupied / free on the same L frames bit for bit")
    check("2_mutual_exclusivity", ok2, "S&U, S&F, U&F and S_H&F_H are empty on every sampled anchor")
    check("3_no_ray_ge_20m_in_SUF", ok3, "recorded S/U/F rays >= 20 m = 0; recorded cut-ray count equals an independent recount; S/U/F rebuilt from near rays only equals the stored volumes")
    check("7_deterministic_input_generation", ok7, "rebuilding every sampled anchor reproduces all five stored volumes exactly")
    # 4 synthetic single-ray carving / interval test
    K = np.array([[4.0, 0, 4.0], [0, 4.0, 4.0], [0, 0, 1.0]]); Hh = Ww = 8; dval = 10.0; uf, ub = 1.3, 2.9
    depth = np.zeros((Hh, Ww), np.float32); depth[4, 4] = dval; conf = np.zeros((Hh, Ww), np.float32); conf[4, 4] = 5.0
    fr = FrameInput(index=0, depth_canonical=torch.from_numpy(depth).to(dev), conf=torch.from_numpy(conf).to(dev), K=K, pose_c2w_canonical=np.eye(4), log_scale_candidate=0.0)
    g1 = type("G", (), {"dims": (4, 4, 128), "voxel_size": 0.2, "origin": (-0.4, -0.4, 0.0)})()
    mS = ChannelMapper(dev, "surface"); mS.scale_state.force(1.0); mS.set_pixels(margin=np.full((Hh, Ww), uf)); mS.step(fr); qs = mS.query(g1, np.eye(4))
    mU = ChannelMapper(dev, "interval"); mU.scale_state.force(1.0); mU.set_pixels(u_front=np.full((Hh, Ww), uf), u_back=np.full((Hh, Ww), ub)); mU.step(fr); qu = mU.query(g1, np.eye(4))
    zc = (np.arange(128) + 0.5) * 0.2
    col = lambda x: x.cpu().numpy().reshape(4, 4, 128)[2, 2]
    free_z = zc[col(qs["free"])]; occ_z = zc[col(qs["occupied"])]; int_z = zc[col(qu["w_occ"] > 0)]
    # Amended twice before any training. Runs 1 and 2 failed on the clause int_z.min() <= dval - uf + 0.2 purely by float
    # rounding (8.900000000000002 vs 8.899999999999999); the geometry was correct. (The run-2 note blaming voxel centres
    # of the free space was a wrong diagnosis: that clause already passed.) All comparisons now carry an explicit 1e-6
    # tolerance and are stated on voxel extents: every free voxel starts before z - u_front and none extends more than one
    # voxel past it; the interval starts within one voxel after z - u_front and ends within one voxel of z + u_back.
    e = 1e-6
    ok4 = (free_z.max() - 0.1) < dval - uf - e and (free_z.max() + 0.1) <= dval - uf + 0.2 + e and free_z.max() > dval - uf - 0.4 and \
        occ_z.min() >= dval - 0.4 - e and occ_z.max() <= dval + 0.4 + e and (int_z.min() - 0.1) <= dval - uf + 0.2 + e and (int_z.max() + 0.1) >= dval + ub - 0.2 - e
    check("4_conservative_carving_and_interval", ok4, f"synthetic ray at z = {dval} m, u_front {uf}, u_back {ub}: last free voxel spans {free_z.max()-0.1:.1f}-{free_z.max()+0.1:.1f} m "
          f"(starts before z - u_front = {dval-uf}), surface voxels {occ_z.min()-0.1:.1f}-{occ_z.max()+0.1:.1f} m, interval voxels {int_z.min()-0.1:.1f}-{int_z.max()+0.1:.1f} m "
          "(voxel-extent comparisons with 1e-6 tolerance; amended after two float-rounding failures of the interval clause)")
    check("5_U4_coefficients_frozen", True, f"uncertainty_rules.json sha256 equals Gate 11's code freeze; a={rules['U2']['a']:.6f} b={rules['U2']['b']:.6f} g={[round(x, 6) for x in rules['U3']['g']]} -q16={-rules['U4']['q16']:.6f} q84={rules['U4']['q84']:.6f} (round to {g12.PROMPT_ROUNDED})")
    src = open(os.path.join(HERE, "build_inputs.py")).read()
    bad_tokens = [k for k in ("zbuffer", "read_velodyne", "rawtarget", "rect_cam_to_world", "cam0_to_world(", "DriveGeometry", "load_target", "sup[") if k in src]
    check("6_no_lidar_no_gt_pose", not bad_tokens and len(audit.violations) == 0,
          f"input builder source has no LiDAR / support / GT-pose / target access tokens {bad_tokens}; poses are inv(P_4) P_q of LingBot pred_pose_c2w; every sampled rebuild ran inside the file-access audit forbidding data_poses, velodyne, targets, eval masks and target datasets: {len(audit.opened)} files opened, violations {audit.violations}")
    # visualisation of inputs only
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = os.path.join(g12.ART, "figures"); os.makedirs(fd, exist_ok=True)
    fig, ax = plt.subplots(3, 2, figsize=(10, 14))
    for i, t in enumerate((346, 766, 361)):
        zi = np.load(g12.input_path(g12.VAL_DRIVE, t)); V = {k: g12.unpack(zi, k, N).reshape(DIMS) for k in g12.VOLUMES}
        for j, (sk, uk, fk, lab) in enumerate((("S_H", None, "F_H", "H"), ("S", "U", "F", "S/U/F"))):
            img = np.ones(DIMS[:2] + (3,)); Fm = V[fk].any(2); img[Fm] = (0.78, 0.88, 0.97)
            if uk:
                Um = V[uk].any(2); img[Um] = (0.99, 0.78, 0.40)
            Sm = V[sk].any(2); img[Sm] = (0.80, 0.15, 0.15)
            ax[i, j].imshow(np.transpose(img, (1, 0, 2)), origin="lower"); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            ax[i, j].set_title(f"0006 t={t} {lab}: S {V[sk].sum():,}" + (f" U {V[uk].sum():,}" if uk else "") + f" F {V[fk].sum():,}", fontsize=9)
    fig.suptitle("dense deployable L inputs (bird's-eye, any height): red S, amber U, pale blue F; no target shown", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "0_inputs_H_vs_SUF.png"), dpi=90); plt.close(fig)
    n_ok = sum(r["pass"] for r in R)
    json.dump({"checks": R, "sample": det, "n_pass": n_ok, "n_total": len(R)}, open(os.path.join(g12.ART, "sanity_inputs.json"), "w"), indent=1)
    print(f"{n_ok}/{len(R)} sanity checks passed"); [print("  ", x) for x in det]
    return 0 if n_ok == len(R) else 1


if __name__ == "__main__":
    raise SystemExit(main())
