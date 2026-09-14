#!/usr/bin/env python
"""Deterministic qualitative maps: the same five-frame clip under S / L / G / Gs, bird's-eye
view of the occupied voxels (max over height) coloured against the LiDAR measurement of the
same five sweeps (green = agrees with a LiDAR return, red = LiDAR-free, grey = unmeasured),
plus a height slice. Fixed clips: early / middle / late of 0006 and one of 0007.

    python tools/local_registration_audit/visualize.py
"""
from __future__ import annotations
import glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.local_registration_audit import common as LR                                   # noqa: E402

DIMS = (256, 256, 32); FIXED = {LR.PRIMARY: [100, 700, 1500], "2013_05_28_drive_0007_sync": [400]}
COND = ("S", "L", "G", "Gs")


def main() -> int:
    od = os.path.join(LR.ART, "figures", "maps"); os.makedirs(od, exist_ok=True)
    for d, ts in FIXED.items():
        files = sorted(glob.glob(os.path.join(LR.DATA_ROOT, "geometry", d, "*.npz")))
        avail = [int(os.path.basename(f)[:-4]) for f in files]
        if not avail:
            continue
        for t0 in ts:
            t = min(avail, key=lambda x: abs(x - t0))
            z = np.load(os.path.join(LR.DATA_ROOT, "geometry", d, f"{t:05d}.npz"))
            lid = np.zeros(np.prod(DIMS), bool); lid[z["past5_occ"]] = True; lid = lid.reshape(DIMS)
            fig, ax = plt.subplots(2, len(COND) + 1, figsize=(4 * (len(COND) + 1), 8))
            ax[0, 0].imshow(lid.any(2).T, origin="lower", cmap="Greys"); ax[0, 0].set_title("LiDAR returns, same 5 sweeps (BEV)")
            ax[1, 0].imshow(lid[:, :, 8:14].any(2).T, origin="lower", cmap="Greys"); ax[1, 0].set_title("LiDAR, height 0.2-1.2 m slice")
            for j, c in enumerate(COND):
                occ = np.zeros(np.prod(DIMS), bool); occ[z[f"occmap_{c}"]] = True; occ = occ.reshape(DIMS)
                for k, sl in enumerate((slice(0, 32), slice(8, 14))):
                    o = occ[:, :, sl].any(2); l_ = lid[:, :, sl].any(2)
                    img = np.ones(o.shape + (3,)); img[o & l_] = (0.1, 0.6, 0.2); img[o & ~l_] = (0.8, 0.2, 0.2); img[~o & l_] = (0.85, 0.85, 0.85)
                    ax[k, j + 1].imshow(np.transpose(img, (1, 0, 2)), origin="lower")
                    tp, fp, fn = z[f"{c}_past5_raw"]
                    ax[k, j + 1].set_title(f"{c}{' (oracle scale)' if c == 'Gs' else ''}: IoU {tp/max(tp+fp+fn,1):.3f}" if k == 0 else f"{c}: height slice")
            for a_ in ax.ravel():
                a_.set_xticks([]); a_.set_yticks([])
            fig.suptitle(f"{d[17:21]} clip anchored at stream frame {t}: five-frame map, anchor velodyne grid (x forward up, y left); green = on a LiDAR return, red = elsewhere, grey = LiDAR only")
            fig.tight_layout(); fig.savefig(os.path.join(od, f"{d[17:21]}_t{t:05d}.png"), dpi=90); plt.close(fig)
    print(f"maps -> {od}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
