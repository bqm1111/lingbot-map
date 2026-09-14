#!/usr/bin/env python
"""Step 12: visual sanity panels for a deterministic sample of correspondences on 0006.

Each panel: image A with the projected point (and its local-control point), image B with
the same 3D point projected, the LiDAR z-buffer evidence around both projections, the
per-view Trident top-1 classes, the feature cosine, and -- when a mapper comparison row
exists -- the pixels the mapper associated with the same voxel.

    python tools/direct_correspondence_audit/panels.py
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.direct_correspondence_audit import common as DC                                # noqa: E402
from tools.direct_correspondence_audit.sample_features import Frames, sample, classify     # noqa: E402
from tools.direct_correspondence_audit.build_pairs import project, min_depth_image          # noqa: E402
from core.datasets import kitti360_transforms as TF, kitti360_splits as SRC, sources as S, vocab  # noqa: E402
from core.run._common import REPO_ROOT                                                    # noqa: E402


def main() -> int:
    DC.require_data_root(); drive = SRC.VAL_DRIVE; src = "kitti360"; dev = torch.device("cuda:0")
    seg = [s for s in S.segments(src, REPO_ROOT) if s.name == drive][0]; key_of = {int(f.key): f.key for f in seg.frames}; path_of = {int(f.key): f.path for f in seg.frames}
    geo = TF.DriveGeometry(drive, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT); K = np.asarray(geo.calib.K); names = list(vocab.load("kitti360").phrases)
    with np.load(DC.TEXT_PATH) as z:
        text = torch.from_numpy(z["text"].astype(np.float32)).to(dev)
    frames = Frames(src, key_of, dev)
    files = sorted(glob.glob(os.path.join(DC.ROW_DIR, drive, "*.npz")))[::40][:6]
    mp = os.path.join(DC.DATA_ROOT, "mapper", f"{drive}.npz")
    M = None
    if os.path.exists(mp):
        with np.load(mp) as z:
            M = z["rows"]; MF = [str(x) for x in z["fields"]]
    out_dir = os.path.join(DC.ART, "panels"); os.makedirs(out_dir, exist_ok=True); index = []
    rng = np.random.default_rng(0)
    for f in files:
        with np.load(f) as z:
            R = z["rows"]; F = [str(x) for x in z["fields"]]
        fa = int(R[0, F.index("native_A")])
        with np.load(DC.pair_path(drive, fa)) as z:
            P = {k: z[k] for k in z.files}
        strict = R[:, F.index("strict")] > 0; R = R[strict]
        sel = {"easy_consistent": np.flatnonzero((R[:, F.index("top1_same")] > 0) & (R[:, F.index("cos_same")] > 0.97) & (R[:, F.index("angle_deg")] > 10)),
               "hard_inconsistent": np.flatnonzero((R[:, F.index("top1_same")] == 0) & (R[:, F.index("cos_same")] < 0.9) & (np.abs(R[:, F.index("resid_B")]) < 0.15))}
        for kind, ix in sel.items():
            if len(ix) == 0:
                continue
            i = ix[rng.integers(len(ix))]; r = R[i]; fb = int(r[F.index("native_B")]); li = int(r[F.index("landmark")])
            imA = cv2.cvtColor(cv2.imread(path_of[fa]), cv2.COLOR_BGR2RGB); imB = cv2.cvtColor(cv2.imread(path_of[fb]), cv2.COLOR_BGR2RGB)
            uA, vA, uB, vB = r[F.index("u_A")], r[F.index("v_A")], r[F.index("u_B")], r[F.index("v_B")]
            # z-buffer crops
            def zcrop(native, u, v):
                pts = geo.read_velodyne(native); pc = pts @ geo.calib.velo_to_rect_cam[:3, :3].T + geo.calib.velo_to_rect_cam[:3, 3]
                uu, vv, zz = project(K, pc); zb = min_depth_image(uu, vv, zz, DC.NATIVE_HW, DC.ZBUF_WINDOW)
                y0, x0 = int(max(v - 40, 0)), int(max(u - 60, 0)); c = zb[y0:y0 + 80, x0:x0 + 120].copy(); c[~np.isfinite(c)] = np.nan; return c
            zA, zB = zcrop(fa, uA, vA), zcrop(fb, uB, vB)
            # local control point (C2) of this pair: re-derive its projection in B from X'
            fig, ax = plt.subplots(2, 3, figsize=(18, 7.5), gridspec_kw={"width_ratios": [3, 3, 1.2]})
            for k, (im, u, v, nat, ttl) in enumerate(((imA, uA, vA, fa, "A"), (imB, uB, vB, fb, "B"))):
                ax[k, 0].imshow(im); ax[k, 0].scatter([u], [v], s=120, facecolors="none", edgecolors="lime", linewidths=2); ax[k, 0].set_title(f"frame {ttl} = {nat}: X projected (green)"); ax[k, 0].axis("off")
                x0, y0 = max(u - 60, 0), max(v - 40, 0); ax[k, 1].imshow(im[int(y0):int(y0) + 80, int(x0):int(x0) + 120]); ax[k, 1].scatter([u - x0], [v - y0], s=140, facecolors="none", edgecolors="lime", linewidths=2)
                ax[k, 1].set_title(f"crop; Trident top-1 {ttl}: {names[int(r[F.index('cls_' + ttl)])]} (conf {r[F.index('conf_' + ttl)]:.2f}); refined label: {names[int(r[F.index('cls_ref_' + ttl)])]}"); ax[k, 1].axis("off")
                zc = zA if k == 0 else zB; im_ = ax[k, 2].imshow(zc, cmap="viridis"); ax[k, 2].set_title(f"LiDAR z-buffer crop ({ttl}); z_proj {r[F.index('z_' + ttl)]:.1f} m, resid {r[F.index('resid_' + ttl)]:+.2f} m"); ax[k, 2].axis("off")
            if M is not None:
                mm = M[(M[:, MF.index("native_A")] == fa) & (M[:, MF.index("native_B")] == fb) & (M[:, MF.index("landmark")] == li)]
                if len(mm):
                    m0 = mm[0]; note = (f"mapper: {int(m0[MF.index('n_mapper_pix_B')])} B pixels in voxel(X), nearest {m0[MF.index('mapper_pix_dist_B_min_px')]:.0f} px from u_B; "
                                        f"LingBot depth err at u_B {m0[MF.index('lingbot_depth_err_uB_m')]:+.1f} m; pose reproj err {m0[MF.index('pose_reproj_err_px')]:.0f} px")
                    fig.text(0.5, 0.005, note, ha="center", fontsize=9, color="darkred")
            fig.suptitle(f"{kind}: X = {np.round(P['X_world'][li], 1)} | baseline {r[F.index('baseline_m')]:.1f} m, view angle {r[F.index('angle_deg')]:.1f} deg | "
                         f"feature cos {r[F.index('cos_same')]:.3f} (local control C2 {r[F.index('cos_C2')]:.3f} at {r[F.index('d_C2')]:.1f} m) | class agree {int(r[F.index('top1_same')])}", fontsize=10)
            fig.tight_layout(rect=(0, 0.03, 1, 0.96)); pth = os.path.join(out_dir, f"{kind}_{fa:010d}_{fb:010d}_{li}.png"); fig.savefig(pth, dpi=100); plt.close(fig)
            index.append({"file": os.path.basename(pth), "kind": kind, "native_A": fa, "native_B": fb, "landmark": li, "cos_same": float(r[F.index("cos_same")]), "top1_same": int(r[F.index("top1_same")])})
    DC.write_json(os.path.join(out_dir, "index.json"), index)
    print(f"{len(index)} panels -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
