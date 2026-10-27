#!/usr/bin/env python
"""Phase 1.2 — visual validation of the Waymo arm's input and supervision.

Six panels per anchor, written to ``artifacts/waymo_arm/figures/``:

  A  anchor RGB with projected LiDAR, coloured by range. Uses the DISTORTION-CORRECTED
     projection verified in ``check_transforms.py`` T2 (p99 residual 0.193 m, under one
     voxel); the pipeline's own distortion-free ``project`` is not used here because this
     panel is about whether the image and the LiDAR agree, so the better projection is the
     honest one to look at.
  B  the five clip images in temporal order, t-4 .. t, to eyeball alignment and motion.
  C  hard map (S, F) in bird's eye, with the anchor camera frustum drawn over it.
  D  hard map vertical slice through the grid's centre row.
  E  occupancy target in bird's eye: occupied / free / unknown.
  F  occupancy target vertical slice, same row as D, so input and supervision can be
     compared directly.

    PYTHONPATH=$PWD python tools/waymo_arm/cli/make_figures.py --anchors 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import matplotlib                                                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                         # noqa: E402
from matplotlib.colors import ListedColormap                                            # noqa: E402

from core.supervision.rawtarget import FREE, OCCUPIED, UNKNOWN                          # noqa: E402
from tools.waymo_arm import config as A                                                 # noqa: E402
from tools.waymo_arm import geometry as WAG                                             # noqa: E402
from tools.waymo_arm.cli.check_transforms import _project_distorted                      # noqa: E402
from tools.waymo_pilot import waymo as W                                                 # noqa: E402
from tools.waymo_pilot import waymo_geometry as WG                                        # noqa: E402

N = int(np.prod(A.GRID_DIMS))
FIGDIR = os.path.join(A.ARTIFACT_DIR, "figures")


def _jpeg(seg: WG.WaymoSegment, stream_index: int):
    import PIL.Image
    fr = seg.frames[int(stream_index)]
    return np.asarray(PIL.Image.open(seg.image_path(fr)))


def figure_for(segment: str, anchor: int, p: A.Paths) -> dict:
    seg = WG.WaymoSegment(segment)
    wag = WAG.WaymoAnchorGeometry(segment)
    cam = seg.calibration()
    grid = WG.waymo_grid()
    X, Y, Z = (int(d) for d in grid.dims)
    vs = float(grid.voxel_size)
    org = np.asarray(grid.origin, np.float64)

    with np.load(p.input_path(segment, anchor), allow_pickle=False) as z:
        S = np.unpackbits(z["S_H"])[:N].astype(bool).reshape(X, Y, Z)
        F = np.unpackbits(z["F_H"])[:N].astype(bool).reshape(X, Y, Z)
    with np.load(p.target_path(segment, anchor), allow_pickle=False) as z:
        valid = np.unpackbits(z["valid_packed"])[:N].astype(bool).reshape(X, Y, Z)
        occ = np.unpackbits(z["occ_packed"])[:N].astype(bool).reshape(X, Y, Z)
    state = np.where(occ, OCCUPIED, np.where(valid, FREE, UNKNOWN)).astype(np.uint8)

    fig = plt.figure(figsize=(17, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.05, 1.0, 1.0])

    # --- A: RGB + projected LiDAR -------------------------------------------------
    axA = fig.add_subplot(gs[0, :2])
    img = _jpeg(seg, anchor)
    fr = seg.frames[anchor]
    with np.load(fr.path, allow_pickle=False) as z:
        pts = np.asarray(z["points_vehicle"], np.float64)
    u, v, zc = _project_distorted(pts, cam)
    m = (zc > 1.0) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
    axA.imshow(img)
    sc = axA.scatter(u[m], v[m], c=np.minimum(zc[m], 60.0), s=0.6, cmap="turbo", alpha=0.75)
    fig.colorbar(sc, ax=axA, fraction=0.025, label="range (m)")
    axA.set_title(f"A  anchor RGB + projected LiDAR ({int(m.sum()):,} pts, distortion applied)")
    axA.axis("off")

    # --- B: the five clip frames ---------------------------------------------------
    axB = fig.add_subplot(gs[0, 2])
    strip = [_jpeg(seg, anchor + o)[::8, ::8] for o in A.CLIP_OFFSETS]
    axB.imshow(np.concatenate(strip, axis=0))
    axB.set_title("B  clip t-4 .. t (top to bottom)")
    axB.axis("off")

    # --- C: hard map BEV + frustum -------------------------------------------------
    axC = fig.add_subplot(gs[1, 0])
    bev = np.zeros((X, Y), np.uint8)
    bev[F.any(2)] = 1
    bev[S.any(2)] = 2
    axC.imshow(bev.T, origin="lower", cmap=ListedColormap(["#101217", "#2f6f9f", "#f2c14e"]),
               vmin=0, vmax=2, interpolation="nearest")
    T = np.asarray(WG.camera_to_grid(cam), np.float64)
    c = (T[:3, 3] - org) / vs
    half = np.radians(A.HFOV_WAYMO_DEG / 2.0)
    fwd = T[:3, :3] @ np.array([0.0, 0.0, 1.0])            # pinhole +z in the grid frame
    ang = np.arctan2(fwd[1], fwd[0])
    for s_ in (-1, 1):
        r = 51.2 / vs
        axC.plot([c[0], c[0] + r * np.cos(ang + s_ * half)],
                 [c[1], c[1] + r * np.sin(ang + s_ * half)], "w--", lw=1.0)
    axC.plot(c[0], c[1], "r*", ms=11)
    axC.set_title(f"C  hard map BEV: S={int(S.sum()):,} F={int(F.sum()):,}\n"
                  f"+ {A.HFOV_WAYMO_DEG:.0f}$\\degree$ frustum, camera at voxel "
                  f"({c[0]:.0f}, {c[1]:.0f})")
    axC.set_xlabel("grid x (voxels)"); axC.set_ylabel("grid y (voxels)")

    # --- D: hard map vertical slice -------------------------------------------------
    row = Y // 2
    axD = fig.add_subplot(gs[1, 1])
    sl = np.zeros((X, Z), np.uint8)
    sl[F[:, row, :]] = 1
    sl[S[:, row, :]] = 2
    axD.imshow(sl.T, origin="lower", aspect="auto",
               cmap=ListedColormap(["#101217", "#2f6f9f", "#f2c14e"]), vmin=0, vmax=2,
               interpolation="nearest")
    axD.set_title(f"D  hard map slice, y={row}")
    axD.set_xlabel("grid x"); axD.set_ylabel("grid z")

    # --- E: target BEV --------------------------------------------------------------
    axE = fig.add_subplot(gs[1, 2])
    tb = np.full((X, Y), 0, np.uint8)
    tb[(state == FREE).any(2)] = 1
    tb[(state == OCCUPIED).any(2)] = 2
    axE.imshow(tb.T, origin="lower", cmap=ListedColormap(["#101217", "#3f7f5f", "#e8615d"]),
               vmin=0, vmax=2, interpolation="nearest")
    axE.set_title(f"E  target BEV: occ={int(occ.sum()):,}\n"
                  f"valid={100 * valid.mean():.1f}% of grid")
    axE.set_xlabel("grid x"); axE.set_ylabel("grid y")

    # --- F: target vertical slice ---------------------------------------------------
    axF = fig.add_subplot(gs[2, :])
    sf = state[:, row, :]
    axF.imshow(sf.T, origin="lower", aspect="auto",
               cmap=ListedColormap(["#101217", "#3f7f5f", "#e8615d"]),
               vmin=0, vmax=2, interpolation="nearest")
    axF.set_title(f"F  target slice, y={row}  (dark = UNKNOWN, green = FREE, red = OCCUPIED)"
                  f"   compare with D: supervision must sit where the map sees surface")
    axF.set_xlabel("grid x"); axF.set_ylabel("grid z")

    fig.suptitle(f"{segment[:52]}  anchor t={anchor}  "
                 f"grid {X}x{Y}x{Z} @ {vs} m, origin {np.round(org, 3).tolist()}",
                 fontsize=11)
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, f"anchor_{segment[8:26]}_{anchor:05d}.png")
    fig.savefig(out, dpi=96)
    plt.close(fig)
    return {"segment": segment, "anchor": anchor, "figure": os.path.relpath(out, A.REPO_ROOT),
            "n_lidar_in_image": int(m.sum()), "n_S": int(S.sum()), "n_F": int(F.sum()),
            "n_occupied": int(occ.sum()), "valid_fraction": float(valid.mean()),
            "camera_voxel": [float(c[0]), float(c[1]), float(c[2])]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchors", type=int, default=3)
    a = ap.parse_args()
    A.ensure_dirs()
    p = A.Paths()
    rows, done = [], 0
    for s in sorted(os.listdir(p.targets)):
        if done >= a.anchors:
            break
        d = os.path.join(p.targets, s)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".npz") or done >= a.anchors:
                continue
            t = int(f[:-4])
            if not os.path.exists(p.input_path(s, t)):
                continue
            rows.append(figure_for(s, t, p))
            done += 1
            print(f"  wrote {rows[-1]['figure']}", flush=True)
            break                               # one anchor per segment, for variety
    path = os.path.join(p.logs, "figures.json")
    json.dump({"version": A.VERSION, "n_figures": len(rows), "rows": rows},
              open(path, "w"), indent=2, default=float)
    print(f"\n{len(rows)} figures -> {FIGDIR}\n-> {path}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
