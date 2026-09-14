#!/usr/bin/env python
"""Part 7 items 8-9: V / LD / PD / GF raw maps on the good / median / bad clips (PD IoU 90th /
50th / 10th percentile, from analysis.json) and a side view along one image column of the
anchor frame: LiDAR surface, LingBot endpoint, the carved free segment and the occupied band.

    python tools/gate9_depth_occupancy/maps.py --device cuda:0
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import common as G9                                                                        # noqa: E402
from audit import lattice_K, zbuffer                                                      # noqa: E402
from tools.local_registration_audit import common as LR                                   # noqa: E402
from core.datasets import kitti360_transforms as TF, kitti360_splits as SRC, sources as S # noqa: E402
from core.run._common import REPO_ROOT                                                    # noqa: E402

DIMS = G9.DIMS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--device", default="cuda:0"); a = ap.parse_args()
    d = G9.PRIMARY; an = json.load(open(os.path.join(G9.ART, "analysis.json")))[d]
    od = os.path.join(G9.ART, "figures", "maps"); os.makedirs(od, exist_ok=True)
    seg = [s for s in S.segments("kitti360", REPO_ROOT) if s.name == d][0]; nat = [int(f.order) for f in seg.frames]
    geo = TF.DriveGeometry(d, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT); K = np.asarray(geo.calib.K); Kl = lattice_K(K); v2c = np.asarray(geo.calib.velo_to_rect_cam)
    for lab, t in an["representative_clips"].items():
        z = np.load(os.path.join(G9.DATA_ROOT, "clips", d, f"{t:05d}.npz"))
        grid = lambda k: np.isin(np.arange(np.prod(DIMS)), z[k]).reshape(DIMS)
        ref = grid("ref_occ"); conds = {"V (camera-visible LiDAR)": grid("occ_V_cam"), "LD (LiDAR depth, mapper)": grid("occ_LD"), "PD (LingBot depth, same pixels)": grid("occ_PD"), "GF (dense LingBot)": grid("occ_GF")}
        fig, ax = plt.subplots(2, 4, figsize=(20, 9))
        for j, (name, occ) in enumerate(conds.items()):
            for k, sl in enumerate((slice(0, 32), slice(9, 13))):
                o = occ[:, :, sl].any(2); r = ref[:, :, sl].any(2)
                img = np.ones(o.shape + (3,)); img[r & ~o] = (0.82, 0.82, 0.82); img[o & r] = (0.1, 0.6, 0.2); img[o & ~r] = (0.85, 0.2, 0.2)
                ax[k, j].imshow(np.transpose(img, (1, 0, 2)), origin="lower")
                key = {0: "V_cam", 1: "LD", 2: "PD", 3: "GF"}[j]; tp, fp, fn = z[f"m_{key}"][:3]
                ax[k, j].set_title(f"{name}: IoU {tp/max(tp+fp+fn,1):.3f}" if k == 0 else f"{name}: height 0-0.8 m slice", fontsize=9); ax[k, j].set_xticks([]); ax[k, j].set_yticks([])
        fig.suptitle(f"{lab} clip (0006, anchor stream frame {t}): bird's-eye max over height; green = occupied & LiDAR-occupied, red = occupied elsewhere in the mask or outside, grey = LiDAR-occupied missed")
        fig.tight_layout(); fig.savefig(os.path.join(od, f"{lab}_t{t:05d}_bev.png"), dpi=90); plt.close(fig)
        # side view along one anchor image column
        with np.load(LR.clip_path(d, t)) as c:
            dep = c["pred_depth"][4].astype(np.float64) * float(c["scale"])
        pc = geo.read_velodyne(nat[t]) @ v2c[:3, :3].T + v2c[:3, 3]; zb, _ = zbuffer(pc, Kl)
        col = int(np.argmax((zb > 1).sum(0)[200:320])) + 200
        rows = np.flatnonzero(zb[:, col] > 1)
        fig, ax = plt.subplots(figsize=(12, 5))
        for v in rows[::2]:
            ray = np.array([(col - Kl[0, 2]) / Kl[0, 0], (v - Kl[1, 2]) / Kl[1, 1], 1.0]); rl = np.linalg.norm(ray)
            zl, zp = zb[v, col], dep[v, col]
            stop = max(zp - 0.2 / rl, 1.0)
            ax.plot([ray[2] * 1.0, ray[2] * stop], [-ray[1] * 1.0, -ray[1] * stop], color="#9ecae1", lw=0.6)
            ax.plot([ray[2] * (zp - 0.2 / rl), ray[2] * (zp + 0.2 / rl)], [-ray[1] * (zp - 0.2 / rl), -ray[1] * (zp + 0.2 / rl)], color="#fdae6b", lw=3, solid_capstyle="butt")
            ax.scatter([ray[2] * zp], [-ray[1] * zp], color="#d62728", s=10, zorder=3); ax.scatter([ray[2] * zl], [-ray[1] * zl], color="k", s=10, zorder=4)
        ax.scatter([], [], color="k", label="LiDAR surface (z-buffer)"); ax.scatter([], [], color="#d62728", label="LingBot endpoint"); ax.plot([], [], color="#9ecae1", label="carved free ray (to d - 0.2 m)"); ax.plot([], [], color="#fdae6b", lw=3, label="occupied band (d ± 0.2 m)")
        ax.set_xlabel("forward (m)"); ax.set_ylabel("up (m)"); ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
        ax.set_title(f"{lab} clip t={t}: side view along anchor image column {col} (every second supported row)")
        fig.tight_layout(); fig.savefig(os.path.join(od, f"{lab}_t{t:05d}_side.png"), dpi=110); plt.close(fig)
    print(f"maps -> {od}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
