#!/usr/bin/env python
"""Gate 11 figures 6-8 on the predeclared representative clips (Gate 9 PD_gate IoU p10 / p50 / p90):
 6  side-view rays along one anchor image column (Gate 9 column rule) for H / U1_0.4 / U2 / U3
 7  bird's-eye map slices of the voxel states, good / median / bad clip
 8  four example categories on the median clip, each at the anchor image column with the most such rays:
    (a) exact-depth failure   : the LiDAR endpoint voxel is not S under H
    (b) containment success   : outside H's +-0.2 m band, inside the U3 interval, and U under U3
    (c) overly broad interval : U3 interval width u_front + u_back > 4.0 m
    (d) carving prevented     : the LiDAR endpoint voxel is F under U3h (hard carving) but not under U3

    python tools/gate11_uncertainty/figures_u.py [--drive 2013_05_28_drive_0006_sync]
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g11 as G                                                                            # noqa: E402
sys.path.insert(0, G.G9_DIR); sys.path.insert(0, G.REPO_ROOT)
import common as G9                                                                        # noqa: E402
from audit import zbuffer, to_grid_index, flat_of                                          # noqa: E402  (Gate 9)
from tools.local_registration_audit import common as LR                                    # noqa: E402
from core.datasets import kitti360_transforms as TF, kitti360_splits as SRC, sources as S  # noqa: E402
from core.mapping import depth as D7, rays as RY                                           # noqa: E402
from core.run._common import REPO_ROOT                                                     # noqa: E402

DIMS = G9.DIMS; BROAD_M = 4.0
STATE_COL = {0: (1, 1, 1), 1: (0.85, 0.2, 0.2), 2: (0.99, 0.75, 0.35), 3: (0.62, 0.79, 0.92)}


def bev(st, ref, zsl):
    s = st.reshape(DIMS)[:, :, zsl]; r = ref.reshape(DIMS)[:, :, zsl]
    occ_ref = (r == 1).any(2); S_ = (s == 1).any(2); U_ = (s == 2).any(2); F_ = (s == 3).any(2)
    img = np.ones(S_.shape + (3,))
    img[F_] = (0.80, 0.90, 0.97); img[U_] = (0.99, 0.80, 0.45); img[S_] = (0.85, 0.2, 0.2)
    img[occ_ref & ~(S_ | U_)] = (0.6, 0.6, 0.6); img[occ_ref & F_ & ~(S_ | U_)] = (0.2, 0.2, 0.6)
    img[occ_ref & U_ & ~S_] = (0.55, 0.75, 0.25); img[occ_ref & S_] = (0.05, 0.5, 0.15)
    return np.transpose(img, (1, 0, 2))


def column_rays(ax, col, zl, zp, uf, ub, margin, title, st_true=None):
    rows = np.flatnonzero(zl[:, col] > 0)
    for v in rows[::2]:
        ray = np.array([(col - KL[0, 2]) / KL[0, 0], (v - KL[1, 2]) / KL[1, 1], 1.0]); rl = np.linalg.norm(ray)
        pt = lambda z: (ray[2] * z, -ray[1] * z)
        a, b = zp[v, col] - uf[v, col] / rl, zp[v, col] + ub[v, col] / rl
        stop = max(zp[v, col] - margin[v, col] / rl, RY.CARVE_NEAR_M)
        ax.plot(*zip(pt(RY.CARVE_NEAR_M), pt(stop)), color="#9ecae1", lw=0.5)
        if ub[v, col] > 0:
            ax.plot(*zip(pt(max(a, 0.1)), pt(b)), color="#fdd49e", lw=3, solid_capstyle="butt")
        ax.plot(*zip(pt(zp[v, col] - 0.2 / rl), pt(zp[v, col] + 0.2 / rl)), color="#e6550d", lw=2, solid_capstyle="butt")
        c = "k" if st_true is None else {0: "#999999", 1: "#1a9850", 2: "#fdae61", 3: "#3060c0", 4: "#999999"}[int(st_true[v])]
        ax.scatter(*pt(zl[v, col]), color=c, s=9, zorder=4)
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_title(title, fontsize=9); ax.set_xlabel("forward (m)"); ax.set_ylabel("up (m)")


def main() -> int:
    global KL
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--drive", default=G.PRIMARY); a = ap.parse_args(); d = a.drive
    rules = json.load(open(G.RULES)); od = os.path.join(G.ART, "figures", "maps"); os.makedirs(od, exist_ok=True)
    files = sorted(glob.glob(os.path.join(G.DATA_ROOT, "clips", d, "*.npz"))); rep = [int(x) for x in np.load(files[0])["representative"]]
    reps = dict(zip(("bad", "median", "good"), rep))
    seg = [s for s in S.segments(SRC.source_of(d), REPO_ROOT) if s.name == d][0]; nat = [int(f.order) for f in seg.frames]
    geo = TF.DriveGeometry(d, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT); v2c = np.asarray(geo.calib.velo_to_rect_cam); T_c2g = np.asarray(geo.rect_cam_to_velo)
    # 7 maps
    conds = [c for c in ("H", "U1_0.4", "U2", "U3", "U4") if f"map_{c}" in np.load(os.path.join(G.DATA_ROOT, "clips", d, f"{rep[1]:05d}.npz")).files]
    for lab, t in reps.items():
        z = np.load(os.path.join(G.DATA_ROOT, "clips", d, f"{t:05d}.npz")); ref = z["map_ref"]
        fig, ax = plt.subplots(2, len(conds), figsize=(3.9 * len(conds), 8))
        for j, c in enumerate(conds):
            for k, zsl in enumerate((slice(0, 32), slice(10, 14))):
                ax[k, j].imshow(bev(z[f"map_{c}"], ref, zsl), origin="lower"); ax[k, j].set_xticks([]); ax[k, j].set_yticks([])
                cnt = z[f"c_{c}"][5]; R, SR, UR = cnt[0], cnt[3], cnt[4]
                ax[k, j].set_title(f"{c}: band cov {(SR+UR)/max(R,1):.2f}, U/mask {cnt[10]/max(cnt[2],1):.2f}" if k == 0 else f"{c}: height 0-0.8 m", fontsize=9)
        fig.suptitle(f"7. {d[17:21]} {lab} clip t={t}. dark green S on LiDAR-occ, olive U on LiDAR-occ, navy F on LiDAR-occ (false free), grey LiDAR-occ unknown; red S elsewhere, orange U elsewhere, pale blue F", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(od, f"7_{d[17:21]}_{lab}_t{t:05d}.png"), dpi=80); plt.close(fig)
    if d != G.PRIMARY:
        return 0
    # median clip, anchor frame
    t = reps["median"]; z = np.load(os.path.join(G.DATA_ROOT, "clips", d, f"{t:05d}.npz")); lr = np.load(LR.clip_path(d, t)); KL = z["K_lattice"]
    pc = geo.read_velodyne(nat[t]) @ v2c[:3, :3].T + v2c[:3, 3]; zb, _ = zbuffer(pc, KL)
    conf = lr["pred_depth_conf"][4].astype(np.float64); zp = lr["pred_depth"][4].astype(np.float64) * float(lr["scale"])
    gate = (zb > D7.MIN_DEPTH_M) & (zb < D7.MAX_DEPTH_M) & (conf >= D7.CONF_THRESHOLD); zl = np.where(gate, zb, 0.0)
    W = {c: G.widths(None if c == "H" else c, zp, conf, rules) for c in ("H", "U1_0.4", "U2", "U3", "U4")}
    margin = {"H": np.full_like(zp, 0.2), "U1_0.4": W["U1_0.4"][0], "U2": W["U2"][0], "U3": W["U3"][0], "U3h": np.full_like(zp, 0.2)}
    LH, LW = zp.shape; vv, uu = np.meshgrid(np.arange(LH), np.arange(LW), indexing="ij")
    ray = np.stack([(uu - KL[0, 2]) / KL[0, 0], (vv - KL[1, 2]) / KL[1, 1], np.ones_like(uu, dtype=np.float64)], -1); rl = np.linalg.norm(ray, axis=-1)
    fl = flat_of(to_grid_index((ray * zl[..., None]).reshape(-1, 3), T_c2g)).reshape(LH, LW)       # anchor frame: T_j->t = identity
    stv = lambda c: np.where(fl >= 0, z[f"map_{c}"][np.maximum(fl, 0)], 4)
    # 6 side views
    col6 = int(np.argmax((zl > 1).sum(0)[200:320])) + 200
    fig, ax = plt.subplots(4, 1, figsize=(12, 14))
    for i, c in enumerate(("H", "U1_0.4", "U2", "U3")):
        c_ = c if f"map_{c}" in z.files else "H"
        column_rays(ax[i], col6, zl, zp, *W[c], margin[c], f"6. {c}: carve (blue) / +-0.2 m surface band (orange) / interval (tan); LiDAR endpoint coloured by its final voxel state "
                    "(green S, amber U, blue F, grey unknown)" if i == 0 else f"{c}", st_true=stv(c_)[:, col6] if f"map_{c}" in z.files else None)
    fig.suptitle(f"median clip t={t}, anchor image column {col6}, every second LiDAR-supported gated row"); fig.tight_layout(); fig.savefig(os.path.join(G.ART, "figures", "6_side_view_rays.png"), dpi=100); plt.close(fig)
    # 8 example categories
    r = (zb - zp) * rl; uf3, ub3 = W["U3"]
    cats = {"a_exact_depth_failure": gate & (fl >= 0) & (stv("H") != 1),
            "b_containment_success": gate & (np.abs(r) > 0.2) & (r >= -uf3) & (r <= ub3) & (stv("U3") == 2),
            "c_overly_broad": gate & ((uf3 + ub3) > BROAD_M),
            "d_carving_prevented": gate & (stv("U3h") == 3) & (stv("U3") != 3)}
    fig, ax = plt.subplots(4, 2, figsize=(18, 16)); summary = {}
    for i, (name, m) in enumerate(cats.items()):
        colc = int(np.argmax(m.sum(0))); summary[name] = {"n_pixels_anchor_frame": int(m.sum()), "column": colc, "n_in_column": int(m[:, colc].sum())}
        pair = ("U3h", "U3") if name.startswith("d") else ("H", "U3")
        for j, c in enumerate(pair):
            wc = W["U3"] if c.startswith("U3") else W["H"]
            column_rays(ax[i, j], colc, np.where(m | gate, zl, 0.0), zp, *wc, margin[c], f"8{name[0]}. {name[2:]} -- {c} (column {colc}, {int(m[:, colc].sum())} such rays)", st_true=stv(c)[:, colc])
    fig.tight_layout(); fig.savefig(os.path.join(G.ART, "figures", "8_example_categories.png"), dpi=90); plt.close(fig)
    json.dump({"clip": t, "side_view_column": col6, "categories": summary, "broad_threshold_m": BROAD_M}, open(os.path.join(G.ART, "figures", "examples.json"), "w"), indent=1)
    print("figures done", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
