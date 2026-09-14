#!/usr/bin/env python
"""Representative map slices (good / median / bad clip by Gate 9's rule) for LD, LB, LB-M, M and
the LB-LiDAR oracle: bird's-eye max over all heights and a 0-0.8 m height slice.

    python tools/gate10_depth_source/maps_sources.py [--drive 2013_05_28_drive_0006_sync]
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g10 as G10                                                                          # noqa: E402

PANELS = (("LD", "LD (LiDAR depth)"), ("LB", "LB (raw LingBot)"), ("LBM", "LB-M (LingBot x MoGe scale)"), ("M", "M (MoGe-2 depth)"), ("LBL", "LB-LiDAR (oracle scalar)"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--drive", default=G10.PRIMARY); a = ap.parse_args()
    d = a.drive; tagj = "" if d == G10.PRIMARY else "_all"
    an = json.load(open(os.path.join(G10.ART, f"analysis{tagj}.json")))[d]
    od = os.path.join(G10.ART, "figures", "maps"); os.makedirs(od, exist_ok=True); DIMS = (256, 256, 32); N = int(np.prod(DIMS))
    for lab, t in an["representative_clips"].items():
        z = np.load(os.path.join(G10.DATA_ROOT, "clips", d, f"{t:05d}.npz"))
        grid = lambda k: np.isin(np.arange(N), z[k]).reshape(DIMS)
        ref = grid("ref_occ")
        fig, ax = plt.subplots(2, len(PANELS), figsize=(4.2 * len(PANELS), 9))
        for j, (key, name) in enumerate(PANELS):
            occ = grid(f"occ_{key}"); tp, fp, fn = z[f"m_{key}"][:3]; onfree = z[f"m_{key}"][4] / max(z[f"m_{key}"][3], 1)
            for k, sl in enumerate((slice(0, 32), slice(10, 14))):
                o = occ[:, :, sl].any(2); r = ref[:, :, sl].any(2)
                img = np.ones(o.shape + (3,)); img[r & ~o] = (0.80, 0.80, 0.80); img[o & r] = (0.10, 0.60, 0.20); img[o & ~r] = (0.85, 0.20, 0.20)
                ax[k, j].imshow(np.transpose(img, (1, 0, 2)), origin="lower"); ax[k, j].set_xticks([]); ax[k, j].set_yticks([])
                ax[k, j].set_title(f"{name}\nIoU {tp/max(tp+fp+fn,1):.3f}, occupied in LiDAR-free {onfree:.2f}" if k == 0 else f"{name}: height 0-0.8 m", fontsize=9)
        fig.suptitle(f"{d[17:21]} {lab} clip (anchor stream frame {t}), GT poses, matched pixels. Bird's-eye (x forward = up, 51.2 m).  "
                     "green = occupied & LiDAR-occupied, red = occupied elsewhere, grey = LiDAR-occupied missed", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(od, f"{d[17:21]}_{lab}_t{t:05d}.png"), dpi=85); plt.close(fig)
    print(f"maps -> {od}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
