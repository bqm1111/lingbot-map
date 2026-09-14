#!/usr/bin/env python
"""Target-construction checks (section 4); exits non-zero if any fails.

    python tools/kitti360_data_scaling_7drives/verify_targets.py --device cuda:0
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drives7                                                                            # noqa: E402
drives7.apply(redirect_artifacts=False)
import build_semantic_targets as BST                                                      # noqa: E402
from core.datasets import kitti360_transforms as TF, kitti360_splits as SRC, sources as S, union_vocab as V8  # noqa: E402
from core.supervision import targets as TG, rawtarget as RT                               # noqa: E402
from core.run.build_targets import load as load_target                                    # noqa: E402
from core.run._common import REPO_ROOT                                                    # noqa: E402
from tools.kitti360_data_scaling import common as C0                                      # noqa: E402

OVERLAY = {"2013_05_28_drive_0006_sync": [4, 604, 1204], "2013_05_28_drive_0002_sync": [400, 2000], "2013_05_28_drive_0000_sync": [900]}
ADOPTED = "/media/SSD1/MINH_DATASETS/lingbot_gate8c1_sky/samples"
#: Run-to-run envelope of the unchanged sample builder on identical caches (drive 0003 rebuilt twice, 193 anchors): the
#: float-accumulated semantic evidence is not bit-reproducible on CUDA -- sem_w relative difference <= 9.8e-4, sem_p
#: (u8-quantised sem / sem_w) differs by at most one step on <= 2 voxels. Every other input array is byte-identical.
#: The bound is on magnitude (not on how many voxels differ, which scales with anchor size). Beyond it = a real change.
SEMW_MAX_REL = 1e-3
SEMP_MAX_STEP = 1


def check(name, ok, detail, R):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True); R.append({"name": name, "pass": bool(ok), "detail": detail})


def seg_of(d):
    seg = [s for s in S.segments(SRC.source_of(d), REPO_ROOT) if s.name == d][0]
    return seg, [int(f.order) for f in seg.frames]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--device", default="cuda:0"); a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev); R = []; n = int(np.prod(BST.GRID_DIMS))
    drives = list(drives7.TRAIN_DRIVES7) + [drives7.VAL_DRIVE]
    # 1-2 LiDAR projection, voxel round trip, visibility of sweep-derived voxels (three drives)
    rt, vr, vis_frac = [], [], {}
    for d in ("2013_05_28_drive_0006_sync", "2013_05_28_drive_0002_sync", "2013_05_28_drive_0005_sync"):
        seg, nat = seg_of(d); geo = TF.DriveGeometry(d, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT); K = np.asarray(geo.calib.K); v2c = np.asarray(geo.calib.velo_to_rect_cam)
        cache = BST.FrameCache(geo, SRC.source_of(d), {nn: f.key for nn, f in zip(nat, seg.frames)}, K, dev); fr = []
        for t in (50, len(nat) // 2, len(nat) - 40):
            j = nat[t + 3]; pts = geo.read_velodyne(j)
            u0, v0, z0 = BST.project(K, pts @ v2c[:3, :3].T + v2c[:3, 3])
            Ta = geo.velo_to_velo(j, nat[t]); pa = pts @ Ta[:3, :3].T + Ta[:3, 3]
            Tb = TF.inv(geo.rect_cam_to_world(j)) @ geo.velo_to_world(nat[t]); u1, v1, z1 = BST.project(K, pa @ Tb[:3, :3].T + Tb[:3, 3])
            ok = (z0 > 1.5) & (u0 >= 0) & (u0 < 1408) & (v0 >= 0) & (v0 < 376); rt.append(float(np.abs(np.stack([u1 - u0, v1 - v0, z1 - z0]))[:, ok].max()))
            idx = np.floor((pa - BST.ORIGIN) / BST.VS).astype(int); ins = (idx >= 0).all(1) & (idx < np.array(BST.GRID_DIMS)).all(1)
            flat = np.unique(np.ravel_multi_index(idx[ins].T, BST.GRID_DIMS)); ctr = (np.stack(np.unravel_index(flat, BST.GRID_DIMS), 1) + 0.5) * BST.VS + BST.ORIGIN
            back = np.floor((ctr - BST.ORIGIN) / BST.VS).astype(int); vr.append(float(np.mean(np.ravel_multi_index(back.T, BST.GRID_DIMS) == flat)))
            u, v, zc = BST.project(K, ctr @ Tb[:3, :3].T + Tb[:3, 3]); zb, _ = cache.get(j); inside, _, _, vis = BST.visible(zb, u, v, zc)
            fr.append(float(vis[inside].mean()))
        vis_frac[d[17:21]] = [round(x, 3) for x in fr]
    allv = [x for v in vis_frac.values() for x in v]
    check("independent_lidar_projection", max(rt) < 1e-6 and min(vr) == 1.0 and min(allv) > 0.7,
          f"sweep j -> anchor grid -> world -> camera j equals direct velodyne->camera projection to {max(rt):.1e}; voxel index/centre round trip exact ({min(vr)}); "
          f"visibility pass rate of voxels built from sweep j itself, seen from camera j: {vis_frac} (misses are occluded sub-voxel structure and range limits)", R)
    # 3 overlays
    od = os.path.join(drives7.ART, "targets", "overlays"); os.makedirs(od, exist_ok=True); cmap = plt.get_cmap("tab20", V8.U); n_ov = 0
    for d, ts in OVERLAY.items():
        seg, nat = seg_of(d); geo = TF.DriveGeometry(d, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT); K = np.asarray(geo.calib.K)
        cache = BST.FrameCache(geo, SRC.source_of(d), {nn: f.key for nn, f in zip(nat, seg.frames)}, K, dev)
        avail = sorted(int(os.path.basename(f)[:-4]) for f in glob.glob(os.path.join(drives7.DATA_ROOT, "samples_gtsem", d, "*.npz")))
        for t0 in ts:
            t = min(avail, key=lambda x: abs(x - t0))
            with np.load(os.path.join(drives7.DATA_ROOT, "samples_gtsem", d, f"{t:05d}.npz")) as z:
                rows = z["fut_rows"].astype(np.int64); cls = z["fut_p"].astype(np.float32).argmax(1); nv = z["fut_n_views"]; fut = z["target_frames"].tolist()
            j = fut[min(3, len(fut) - 1)]; nj = nat[j]
            ctr = (np.stack(np.unravel_index(rows, BST.GRID_DIMS), 1) + 0.5) * BST.VS + BST.ORIGIN
            T = TF.inv(geo.rect_cam_to_world(nj)) @ geo.velo_to_world(nat[t]); u, v, zc = BST.project(K, ctr @ T[:3, :3].T + T[:3, 3])
            zb, _ = cache.get(nj); _, _, _, vis = BST.visible(zb, u, v, zc)
            im = cv2.cvtColor(cv2.imread(seg.frames[j].path), cv2.COLOR_BGR2RGB)
            fig, ax = plt.subplots(2, 1, figsize=(16, 9))
            ax[0].imshow(im); ax[0].scatter(u[vis], v[vis], c=[cmap(int(c)) for c in cls[vis]], s=3); ax[0].axis("off")
            ax[0].set_title(f"{d[17:21]} anchor t={t}: target voxels (GT-registered) projected into future frame {j}; colour = target class; {int(vis.sum())}/{len(rows)} visible here")
            ax[0].legend(handles=[plt.Line2D([], [], marker="o", ls="", color=cmap(k), label=V8.UNION[k]) for k in np.unique(cls[vis])], fontsize=6, ncol=7, loc="lower left")
            ax[1].imshow(im); sc = ax[1].scatter(u[vis], v[vis], c=nv[vis], s=3, cmap="viridis", vmin=1, vmax=8); ax[1].axis("off"); ax[1].set_title("same voxels coloured by number of visible future views"); plt.colorbar(sc, ax=ax[1], fraction=0.02)
            fig.tight_layout(); fig.savefig(os.path.join(od, f"{d[17:21]}_t{t:05d}_f{j:05d}.png"), dpi=90); plt.close(fig); n_ov += 1
    check("deterministic_overlays", n_ov == sum(len(v) for v in OVERLAY.values()), f"{n_ov} overlays at fixed anchors {OVERLAY} in {od}", R)
    # 4 inputs: rebuilt 3-drive + 0006 samples reproduce the adopted inputs; gtsem copies inputs byte for byte; future/GT cannot reach inputs
    in_keys = ("t", "input_frames", "scale", "dims", "rows", "logodds", "w_occ", "w_free", "n_obs", "n_lb", "age", "observed", "sem_rows", "sem_p", "sem_w", "native_frame")
    n_ad = n_ad_ok = n_cp = n_cp_ok = n_er = n_er_ok = n_caus = n_caus_ok = 0; sw_frac, sw_rel, sp_step = [], [], []
    for d in drives:
        files = sorted(glob.glob(os.path.join(drives7.DATA_ROOT, "samples_gtsem", d, "*.npz")))
        for k, f in enumerate(files):
            t = int(os.path.basename(f)[:-4])
            with np.load(f) as g, np.load(os.path.join(drives7.DATA_ROOT, "samples", d, f"{t:05d}.npz")) as s:
                n_cp += 1; n_cp_ok += int(all(np.array_equal(g[x], s[x]) for x in BST.INPUT_KEYS if x in s.files))
                n_caus += 1; n_caus_ok += int(int(g["input_frames"].max()) <= t and int(g["target_frames"].min()) > t and int(g["target_frames"].max()) <= t + 20)
                if k % 50 == 0:
                    G = {x: g[x] for x in g.files}; base = TG.unpack_sample(G, dev)["input"]
                    for x in ("gt_occ", "gt_valid", "fut_observed", "fut_half_agree"):
                        G[x] = np.zeros_like(G[x])
                    for x in ("fut_rows", "fut_p", "fut_w", "fut_conf", "fut_n_views", "occ_rows_target", "n_obs_target", "n_frames_occ_target"):
                        if x in G:
                            G[x] = G[x][:0]
                    n_er += 1; n_er_ok += int(torch.equal(TG.unpack_sample(G, dev)["input"], base))
            ap_ = os.path.join(ADOPTED, d, f"{t:05d}.npz")
            if os.path.exists(ap_) and k % 10 == 0:
                with np.load(ap_) as o, np.load(f) as g:
                    n_ad += 1
                    exact = all(np.array_equal(o[x], g[x]) for x in in_keys if x not in ("sem_w", "sem_p"))
                    wo, wg = o["sem_w"].astype(np.float64), g["sem_w"].astype(np.float64)
                    frac = float((wo != wg).mean()) if len(wo) else 0.0
                    rel = float((np.abs(wo - wg) / np.maximum(np.abs(wo), 1e-9)).max()) if len(wo) else 0.0
                    dp = int(np.abs(o["sem_p"].astype(np.int16) - g["sem_p"].astype(np.int16)).max()) if o["sem_p"].size else 0
                    sw_frac.append(frac); sw_rel.append(rel); sp_step.append(dp)
                    n_ad_ok += int(exact and o["sem_p"].shape == g["sem_p"].shape and rel <= SEMW_MAX_REL and dp <= SEMP_MAX_STEP)
    check("inputs_unchanged", n_cp_ok == n_cp and n_ad_ok == n_ad and n_ad > 0,
          f"{n_cp_ok}/{n_cp} semantic-target files copy every input array of the rebuilt sample byte for byte; {n_ad_ok}/{n_ad} rebuilt samples of drives 0003/0007/0010/0006 "
          f"match the adopted Gate 8C-1 inputs: all geometry/map arrays byte-identical; float-accumulated semantic evidence within the builder's own run-to-run envelope "
          f"(sem_w relative <= {max(sw_rel):.1e}, differing on <= {max(sw_frac):.2%} of evidence voxels; sem_p <= {max(sp_step)} u8 step). Same envelope as the unchanged builder re-run on identical caches "
          f"(artifacts/kitti360_data_scaling_7drives/targets/semw_nondeterminism.json)", R)
    check("future_and_gt_not_in_inputs", n_er_ok == n_er and n_caus_ok == n_caus,
          f"erasing every target array leaves the 32-channel input unchanged on {n_er_ok}/{n_er} sampled anchors; {n_caus_ok}/{n_caus} files have input frames <= t < target frames <= t+20", R)
    # 5 occupancy: rawtarget reproduces the stored targets; conflicts never labelled; no SSCBench label read by the builders
    n_rt = n_rt_ok = 0; n_conf_bad = 0
    for d in ("2013_05_28_drive_0000_sync", "2013_05_28_drive_0004_sync", "2013_05_28_drive_0006_sync"):
        geo = TF.DriveGeometry(d, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT); ancs = {x.stream_index: x for x in SRC.anchors(d, REPO_ROOT)}
        files = sorted(glob.glob(os.path.join(drives7.DATA_ROOT, "targets", d, "*.npz")))
        for f in (files[5], files[len(files) // 2]):
            t = int(os.path.basename(f)[:-4]); anc = ancs[t]; stored = load_target(f)
            b = RT.build(geo, anc.native_frame, [anc.native_frame] + list(anc.future_natives), dev)
            n_rt += 1; n_rt_ok += int(np.array_equal(stored["occupied"], np.asarray(b.occupied)) and np.array_equal(stored["valid"], np.asarray(b.valid)))
            conflict = (np.asarray(b.n_obs) > 0) & (np.asarray(b.n_free) > 0); n_conf_bad += int((conflict & stored["valid"]).sum())
    logs = [json.load(open(p)) for p in glob.glob(os.path.join(drives7.ART, "targets", "semantic_*.json"))]
    viol = sum(len(l["firewall"]["violations"]) for l in logs)
    check("occupancy_targets_trusted", n_rt_ok == n_rt and n_conf_bad == 0 and viol == 0 and len(logs) == 8,
          f"current rawtarget.py (raw Velodyne, GT poses) rebuilds the stored targets exactly on {n_rt_ok}/{n_rt} anchors of 0000/0004/0006; {n_conf_bad} cross-sweep conflicting voxels are supervised; "
          f"semantic-target builds for {len(logs)} drives ran inside the file-access audit with {viol} forbidden opens (no SSCBench label, no SemanticKITTI/Occ3D)", R)
    # 6 semantic target provenance
    tot = {k: sum(l["stats"][k] for l in logs) for k in ("occupied", "target_voxels", "mapper_target_voxels", "visible", "occluded", "no_reference", "contradictions")}
    check("semantic_targets_registered", all(l["rule"]["mapper_used"] is False for l in logs) and tot["target_voxels"] > 0,
          f"all {len(logs)} drives built with GT poses + LiDAR visibility, mapper not used; totals {tot} (target/occupied {tot['target_voxels']/tot['occupied']:.3f}, mapper-built {tot['mapper_target_voxels']/tot['occupied']:.3f})", R)
    n_ok = sum(r["pass"] for r in R); print(f"{n_ok}/{len(R)} passed")
    C0.write_json(os.path.join(drives7.ART, "targets", "verification.json"), {"checks": R, "n_pass": n_ok, "n_total": len(R), "env": C0.environment()})
    return 0 if n_ok == len(R) else 1


if __name__ == "__main__":
    raise SystemExit(main())
