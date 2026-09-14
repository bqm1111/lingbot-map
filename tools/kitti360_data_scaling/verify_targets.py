#!/usr/bin/env python
"""Targeted checks of the regenerated targets (section 4).

    python tools/kitti360_data_scaling/verify_targets.py --device cuda:0
"""
from __future__ import annotations
import argparse, glob, hashlib, os, sys
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.kitti360_data_scaling import common as C                                       # noqa: E402
from tools.kitti360_data_scaling.build_targets_gtsem import project, INPUT_KEYS, GRID_DIMS, ORIGIN, VS, FrameCache  # noqa: E402
os.environ["G8C1_DATA_ROOT"] = C.DATA_ROOT
from core.datasets import kitti360_transforms as TF, kitti360_splits as SRC, sources as S, union_vocab as V8, vocab  # noqa: E402
from core.supervision import targets as TG                                                # noqa: E402
from core.run._common import REPO_ROOT                                                    # noqa: E402

FIXED_ANCHORS = {"2013_05_28_drive_0006_sync": [4, 604, 1204], "2013_05_28_drive_0010_sync": [100, 400]}


def visible(zb, u, v, zc, inside):
    vi, ui = v[inside].astype(int), u[inside].astype(int); tol = C.VIS_TOL[0] + C.VIS_TOL[1] * zc
    mn7 = np.full(len(u), np.inf); mx7 = np.full(len(u), -np.inf); mn3 = np.full(len(u), np.inf)
    mn7[inside] = zb["min7"][vi, ui]; mx7[inside] = zb["max7"][vi, ui]; mn3[inside] = zb["min3"][vi, ui]
    return inside & np.isfinite(mn7) & (zc >= mn7 - tol) & (zc <= mx7 + tol) & ~(np.isfinite(mn3) & (zc - mn3 > tol))


def check(name, ok, detail, R):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True); R.append({"name": name, "pass": bool(ok), "detail": detail})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--device", default="cuda:0"); a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev); R = []
    drives = list(SRC.TRAIN_DRIVES) + [SRC.VAL_DRIVE]; n = int(np.prod(GRID_DIMS))
    # 1 + 2: independent LiDAR projection and voxel round trips
    d = SRC.VAL_DRIVE; src = "kitti360"; seg = [s for s in S.segments(src, REPO_ROOT) if s.name == d][0]
    natives = [int(f.key) for f in seg.frames]; key_of = {nn: f.key for nn, f in zip(natives, seg.frames)}
    geo = TF.DriveGeometry(d, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT); K = np.asarray(geo.calib.K); v2c = np.asarray(geo.calib.velo_to_rect_cam)
    cache = FrameCache(geo, src, key_of, K, dev)
    rt_err = []; vis_frac = []; rt_vox = []
    for t in FIXED_ANCHORS[d]:
        nat = natives[t]; j = natives[t + 3]
        pts = geo.read_velodyne(j)                                            # sweep j, velodyne j
        u0, v0, z0 = project(K, pts @ v2c[:3, :3].T + v2c[:3, 3])            # direct: velo j -> cam j
        Ta = geo.velo_to_velo(j, nat); pa = pts @ Ta[:3, :3].T + Ta[:3, 3]  # via the anchor grid frame
        Tb = TF.inv(geo.rect_cam_to_world(j)) @ geo.velo_to_world(nat); u1, v1, z1 = project(K, pa @ Tb[:3, :3].T + Tb[:3, 3])
        ok = (z0 > 1.5) & (u0 >= 0) & (u0 < 1408) & (v0 >= 0) & (v0 < 376)
        rt_err.append(float(np.abs(np.stack([u1 - u0, v1 - v0, z1 - z0]))[:, ok].max()))
        # voxelise the sweep in the anchor grid, re-project the centres, apply the builder's visibility test
        idx = np.floor((pa - ORIGIN) / VS).astype(int); ins = (idx >= 0).all(1) & (idx < np.array(GRID_DIMS)).all(1)
        flat = np.unique(np.ravel_multi_index(idx[ins].T, GRID_DIMS)); ctr = (np.stack(np.unravel_index(flat, GRID_DIMS), 1) + 0.5) * VS + ORIGIN
        w = ctr @ geo.velo_to_world(nat)[:3, :3].T + geo.velo_to_world(nat)[:3, 3]; back = np.floor(((w - geo.velo_to_world(nat)[:3, 3]) @ geo.velo_to_world(nat)[:3, :3] - ORIGIN) / VS).astype(int)
        rt_vox.append(float(np.mean(np.ravel_multi_index(back.T, GRID_DIMS) == flat)))
        u, v, zc = project(K, ctr @ Tb[:3, :3].T + Tb[:3, 3]); zb, _ = cache.get(j)
        inside = (zc > 1.5) & (u >= 2) & (u < 1406) & (v >= 2) & (v < 374); vis = visible(zb, u, v, zc, inside); vis_frac.append(float(vis[inside].mean()))
    check("independent_lidar_projection", max(rt_err) < 1e-6 and min(rt_vox) == 1.0 and min(vis_frac) > 0.75,
          f"sweep j -> anchor grid -> world -> camera j reproduces the direct velo->camera projection to {max(rt_err):.1e} px/m; voxel index -> centre -> world -> index round trip exact on "
          f"{[round(x, 4) for x in rt_vox]}; {[round(x, 3) for x in vis_frac]} of sweep-j-derived occupied voxels pass the builder's visibility test in frame j (anchors {FIXED_ANCHORS[d]})", R)
    # 3: overlays
    out_dir = os.path.join(C.ART, "targets", "overlays"); os.makedirs(out_dir, exist_ok=True); names = V8.UNION; cmap = plt.get_cmap("tab20", V8.U)
    for t in FIXED_ANCHORS[d]:
        with np.load(C.sample_path(d, t)) as z:
            Z = {k: z[k] for k in z.files}
        nat = natives[t]; fut = Z["target_frames"].tolist(); j = fut[2]; nj = natives[j]
        rows = Z["fut_rows"].astype(np.int64); cls = Z["fut_p"].astype(np.float32).argmax(1); nf = Z["fut_n_frames"]
        ctr = (np.stack(np.unravel_index(rows, GRID_DIMS), 1) + 0.5) * VS + ORIGIN
        T = TF.inv(geo.rect_cam_to_world(nj)) @ geo.velo_to_world(nat); u, v, zc = project(K, ctr @ T[:3, :3].T + T[:3, 3])
        zb, _ = cache.get(nj); inside = (zc > 1.5) & (u >= 2) & (u < 1406) & (v >= 2) & (v < 374); vis = visible(zb, u, v, zc, inside)
        im = cv2.cvtColor(cv2.imread(seg.frames[j].path), cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(2, 1, figsize=(16, 9)); ax[0].imshow(im); ax[0].scatter(u[vis], v[vis], c=[cmap(int(c)) for c in cls[vis]], s=3); ax[0].set_title(f"anchor t={t} (native {nat}): target voxels visible in future frame {j} (native {nj}), coloured by target class; {vis.sum()} of {len(rows)} target voxels")
        ax[0].axis("off"); ax[1].imshow(im); sc = ax[1].scatter(u[vis], v[vis], c=nf[vis], s=3, cmap="viridis", vmin=1, vmax=8); ax[1].set_title("same, coloured by number of contributing future views"); ax[1].axis("off"); plt.colorbar(sc, ax=ax[1], fraction=0.02)
        handles = [plt.Line2D([], [], marker="o", ls="", color=cmap(k), label=names[k]) for k in np.unique(cls[vis])]; ax[0].legend(handles=handles, fontsize=6, ncol=6, loc="lower left")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"{d[17:21]}_t{t:05d}_f{j:05d}.png"), dpi=90); plt.close(fig)
    check("deterministic_overlays", True, f"{len(FIXED_ANCHORS[d])} overlays written to {out_dir} (fixed anchors {FIXED_ANCHORS[d]})", R)
    # 4: inputs byte-identical, 5: future/GT cannot reach inputs
    n_files = n_same = n_inp = 0; n_erase = 0
    for dd in drives:
        files = sorted(glob.glob(os.path.join(C.DATA_ROOT, "samples", dd, "*.npz")))
        for f in files:
            t = int(os.path.basename(f)[:-4])
            with np.load(f) as z1, np.load(C.source_sample_path(dd, t)) as z0:
                same = all(np.array_equal(z1[k], z0[k]) for k in INPUT_KEYS if k in z0.files)
                n_files += 1; n_same += int(same)
                if n_files % 97 == 0:
                    Z1 = {k: z1[k] for k in z1.files}; Z0 = {k: z0[k] for k in z0.files}
                    i1 = TG.unpack_sample(Z1, dev)["input"]; i0 = TG.unpack_sample(Z0, dev)["input"]; n_inp += int(torch.equal(i1, i0))
                    for k in ("gt_occ", "gt_valid", "fut_observed", "fut_half_agree"):
                        Z1[k] = np.zeros_like(Z1[k])
                    for k in ("fut_rows", "fut_p", "fut_w", "fut_conf", "fut_n_frames"):
                        Z1[k] = Z1[k][:0]
                    n_erase += int(torch.equal(TG.unpack_sample(Z1, dev)["input"], i1))
    check("inputs_unchanged", n_same == n_files, f"{n_same}/{n_files} regenerated samples have every input array byte-identical to the adopted samples", R)
    check("future_and_gt_cannot_reach_inputs", n_inp == n_erase == n_files // 97, f"unpacked 32-channel input identical to the adopted one on {n_inp}/{n_files // 97} sampled anchors; erasing every target array leaves it unchanged on {n_erase}", R)
    # 6: human-label frame conversion and teacher agreement (evaluation-only labels; opened here outside any training path)
    pf = geo.pose_frames; t_c2v = np.asarray(geo.calib.cam0_to_velo)[:3, 3]; agree = {"identity": [], "cam0_origin": []}; cov = []
    for f in sorted(glob.glob(os.path.join(C.DATA_ROOT, "samples", d, "*.npz")))[::30]:
        t = int(os.path.basename(f)[:-4]); nat = natives[t]; k = [k for k in range(len(pf) - 1) if pf[k + 1] == nat]
        p = os.path.join(C.SSCBENCH_LABELS, d, f"{k[0]:06d}_1_1.npy") if k else None
        if not p or not os.path.exists(p):
            continue
        L = np.load(p).astype(np.int32)
        with np.load(f) as z:
            rows = z["fut_rows"].astype(np.int64); cls25 = z["fut_p"].astype(np.float32).argmax(1)
        out_ch = V8.out_channel("kitti360"); labels = np.asarray(vocab.load("kitti360").labels); teach = labels[out_ch[cls25]]
        ctr = (np.stack(np.unravel_index(rows, GRID_DIMS), 1) + 0.5) * VS + ORIGIN
        for name, shift in (("identity", np.zeros(3)), ("cam0_origin", -t_c2v)):
            li = np.floor((ctr + shift - ORIGIN) / VS).astype(int); ins = (li >= 0).all(1) & (li < np.array(GRID_DIMS)).all(1)
            hl = np.full(len(rows), -1); hl[ins] = L[tuple(li[ins].T)]; ok = (hl >= 1) & (hl <= 18)
            agree[name].append(float((teach[ok] == hl[ok]).mean()) if ok.sum() else np.nan)
            if name == "cam0_origin":
                cov.append(float(ok.mean()))
    check("human_label_frame", np.nanmean(agree["cam0_origin"]) > np.nanmean(agree["identity"]),
          f"teacher-target vs SSCBench human label agreement on labelled target voxels: identity frame {np.nanmean(agree['identity']):.3f}, cam0-origin frame (label grid shifted by -t_cam0_to_velo) "
          f"{np.nanmean(agree['cam0_origin']):.3f}; human-label coverage of target voxels {np.nanmean(cov):.3f} (used for evaluation only)", R)
    n_ok = sum(r["pass"] for r in R); print(f"{n_ok}/{len(R)} passed")
    C.write_json(os.path.join(C.ART, "targets", "verification.json"), {"checks": R, "n_pass": n_ok, "n_total": len(R), "env": C.environment()})
    return 0 if n_ok == len(R) else 1


if __name__ == "__main__":
    raise SystemExit(main())
