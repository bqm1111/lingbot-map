#!/usr/bin/env python
"""Gate 8C-1: a readable 3D comparison against the released OccAny checkpoint.

The first version of this figure was not interpretable, for three reasons that this one
fixes:

1. **The ground truth was coloured by height.** A rainbow next to two error-coloured panels
   reads as a third error scheme. Here the ground truth is one neutral grey: it is the shape
   both methods are trying to reproduce, and shape is all it needs to convey.
2. **Missed voxels drowned the picture.** ~70 % of the Occ3D ground truth is missed by
   *both* methods (see ``fig_occany_why_missed.png`` for why), so drawing every miss as a
   full opaque blue cube buries the part of the scene that actually distinguishes them.
   Misses are still drawn -- hiding them would flatter both methods -- but as small pale
   ghost cubes, so correct and hallucinated geometry sits on top of them.
3. **Two error maps side by side do not answer "who is better".** A fourth panel does,
   directly: it keeps only the voxels where the two methods *disagree about being right*,
   blue where only we are correct and orange where only OccAny is.

Run after ``compare_occany.py`` -- it reuses that run's ``occany_headtohead.json`` to pick
the same scenes, so no metric sweep is repeated.

    python tools/gate8c1/figure_3d_compare.py --n 3 --device cuda:2
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                  # noqa: E402
from matplotlib.patches import Patch                                             # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, default_device                               # noqa: E402
from gates.gate6 import frames as G6F, grids as G6G, vocab                             # noqa: E402
from gates.gate8 import sources as S, vocab as V8                                      # noqa: E402
from gates.gate8.net import load_checkpoint                                            # noqa: E402
from gates.gate8c1 import render3d as R3                                               # noqa: E402
from tools.gate8c1.compare_occany import (DIMS, FREE, IGNORE, ORIGIN,            # noqa: E402
                                          VOXEL, counts, load_occany,
                                          occany_samples, our_prediction)
from tools.gate8c1.eval_target import MANIFEST, PAST5_OFFSETS                    # noqa: E402

#: nuScenes CAM_FRONT, from the published sensor intrinsics (fx 1266, 1600x900). The
#: frustum is a visual aid, so a documented constant is used rather than the per-frame K.
FOV_X_DEG, FOV_Y_DEG = 64.6, 39.2

#: solid = the model committed to it; ghost = nobody put anything there
C_TP = (0.16, 0.68, 0.34)
C_FP = (0.86, 0.20, 0.24)
C_MISS = (0.62, 0.66, 0.72)
C_GT = (0.66, 0.64, 0.62)
C_OURS = (0.16, 0.42, 0.85)
C_OCC = (0.90, 0.55, 0.13)
C_BOTH = (0.55, 0.57, 0.60)
SOLID = 0.90
#: missed voxels are drawn at the SAME edge length as recovered ones. Shrinking them (an
#: earlier version used 0.30) makes the picture lie: a cube at 1/3 edge covers 1/9 of the
#: screen, so a scene with recall 0.27 reads as ~77 % green. Proportion has to survive.
GHOST = SOLID


def mesh(mask, colour, shrink=SOLID):
    idx = np.argwhere(mask)
    if not len(idx):
        return None
    return R3.cube_mesh(idx, np.tile(colour, (len(idx), 1)), VOXEL, ORIGIN, shrink=shrink)


def recovered_scene(pred, gt, valid):
    """Only what the method got RIGHT, laid over the true shape. No red at all.

    Asked for directly: with hallucinations removed, the eye compares coverage instead of
    trying to net two colours against each other. Green is real geometry the method put
    back; grey ghosts are real geometry it left out.
    """
    return [mesh(~pred & gt & valid, C_MISS, GHOST),
            mesh(pred & gt & valid, C_TP)]


def error_scene(pred, gt, valid):
    """The same panel with the hallucinations added back in red."""
    return [mesh(~pred & gt & valid, C_MISS, GHOST),      # drawn first, sits underneath
            mesh(pred & gt & valid, C_TP),
            mesh(pred & ~gt & valid, C_FP)]


def verdict_scene(a_pred, b_pred, gt, valid):
    """Only where the two methods disagree about being right, plus their common ground."""
    a, b, g = a_pred & valid, b_pred & valid, gt & valid
    return ([mesh(g & a & b, C_BOTH, 0.55),
             mesh(g & b & ~a, C_OURS),
             mesh(g & a & ~b, C_OCC)],
            {"both": int((g & a & b).sum()), "ours_only": int((g & b & ~a).sum()),
             "occany_only": int((g & a & ~b).sum()),
             "neither": int((g & ~a & ~b).sum())})


#: the six panels never change, so the titles are fixed and the renders can be reused
PANEL_TITLES = ("1 · ground truth",
                "2 · OccAny — what it recovered",
                "3 · ours — what it recovered",
                "4 · OccAny — + what it invented",
                "5 · ours — + what it invented",
                "6 · who recovers what")


def compose(rendered, out_path, title, handles, row_h=4.9, dpi=110, titles=PANEL_TITLES):
    """Lay already-rendered panels out as one figure.

    Rendering is the expensive step, so it happens once per scene and the resulting arrays
    are composed into both the full figure and the two-scene ones. Margins are reserved in
    *inches* rather than as fractions, so a 2-row figure and an 8-row figure get the same
    physical space for the title and the legend instead of the short one being crushed.
    """
    n = len(rendered)
    H = row_h * n
    fig, axes = plt.subplots(n, len(titles), figsize=(35, H), squeeze=False)
    for r, row in enumerate(rendered):
        for c, img in enumerate(row["images"]):
            axes[r][c].imshow(img)
            axes[r][c].set_axis_off()
            axes[r][c].set_title(titles[c], fontsize=10.5)
        if row.get("label"):
            axes[r][0].text(0.01, 0.98, row["label"], transform=axes[r][0].transAxes,
                            fontsize=9, va="top", bbox=dict(fc="w", ec="none", alpha=.75))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=11, frameon=False,
               bbox_to_anchor=(0.5, 0.22 / H))
    fig.suptitle(title, fontsize=17, y=1 - 0.34 / H)
    fig.tight_layout(rect=[0, 1.0 / H, 1, 1 - 0.78 / H])
    fig.subplots_adjust(hspace=0.14)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print("wrote", out_path, flush=True)


def write_pairs(rendered, base_path, title, handles, per_fig=2, **kw):
    """The same scenes again, `per_fig` at a time -- one slide's worth each."""
    stem, ext = os.path.splitext(base_path)
    for k in range(0, len(rendered), per_fig):
        chunk = rendered[k:k + per_fig]
        compose(chunk, f"{stem}_pair{k // per_fig + 1}{ext}", title, handles, **kw)


def height_profile(files, preds_fn, nz=DIMS[2]):
    """GT mass and per-method recall as a function of height -- the 'why so blue' figure."""
    gt_h = np.zeros(nz)
    rec_h = {k: np.zeros(nz) for k in ("occany", "ours")}
    for gt, valid, pmap in preds_fn(files):
        idx = np.argwhere(gt & valid)
        gt_h += np.bincount(idx[:, 2], minlength=nz)
        for k, p in pmap.items():
            hit = p[gt & valid]
            rec_h[k] += np.bincount(idx[hit, 2], minlength=nz)
    return gt_h, rec_h


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=3, help="scenes in the 3D figure")
    ap.add_argument("--per-figure", type=int, default=2,
                    help="scenes per extra split figure (0 disables the split figures)")
    ap.add_argument("--n-profile", type=int, default=60,
                    help="samples behind the height-profile figure")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=default_device())
    a = ap.parse_args()

    ds = "occ3d"
    man = json.load(open(MANIFEST))
    fz = man["seeds"][str(a.seed)]
    tau = float(fz["occupancy_threshold"])
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    C = len(vocab.load(ds))
    MAP = G6G.PREDICTION_GRID[ds]
    into = V8.into_matrix(ds)
    net = load_checkpoint(os.path.join(REPO_ROOT, fz["checkpoint"]), dev)
    segs = {s.name: s for s in S.segments(ds, REPO_ROOT)}
    recs = {r.raw["anchor_token"]: r.raw for r in G6F.read_manifest(ds, REPO_ROOT)}
    oc = occany_samples()

    prior = json.load(open(os.path.join(ART, "occany_headtohead.json")))
    rows = prior["per_sample"]
    # one anchor per distinct scene, spread across the split -- several rows can share a
    # scene, and three views of the same street is not three examples
    by_scene, seen = [], set()
    for r in rows:
        if r["scene"] not in seen:
            seen.add(r["scene"])
            by_scene.append(r)
    pick = [by_scene[int(round(q * (len(by_scene) - 1)))]
            for q in np.linspace(0.04, 0.96, a.n)]
    print(f"picked {len(pick)} distinct scenes out of {len(by_scene)} available "
          f"({prior['n_scored']}-sample run): "
          f"{', '.join(p['scene'] for p in pick)}", flush=True)

    def predict(row):
        tok = row["token"]
        d = load_occany(oc[tok])
        label = d["label"]
        gt = (label != FREE) & (label != IGNORE)
        valid = label != IGNORE
        ours, _ = our_prediction(ds, a.seed, dev, net, tau, segs[row["scene"]], row[
            "stream_index"], PAST5_OFFSETS, into, C, MAP)
        return d["pred"], ours, gt, valid, d

    # ------------------------------------------------- why so much is missed at all
    use = rows[:a.n_profile]
    nz = DIMS[2]
    gt_h = np.zeros(nz)
    rec_h = {"occany": np.zeros(nz), "ours": np.zeros(nz)}
    pred_h = {"occany": np.zeros(nz), "ours": np.zeros(nz)}
    # what IS a false positive? distance to the nearest real occupied voxel, and the class
    # of that voxel -- this is what licenses the caption on the 3D figure
    from scipy.ndimage import distance_transform_edt as edt
    names = vocab.load(ds).names
    movable = {2, 3, 4, 5, 6, 7, 9, 10}       # bicycle bus car constr moto ped trailer truck
    fpd = {k: {"n": 0, "dist": np.zeros(4), "cls": np.zeros(18)}
           for k in ("occany", "ours")}
    gt_cls = np.zeros(18)
    for k, row in enumerate(use):
        p_oc, p_us, gt, valid, d = predict(row)
        m = gt & valid
        lab = d["label"]
        gt_cls += np.bincount(lab[m].astype(int), minlength=18)
        dist, ind = edt(~gt, return_indices=True)
        near = lab[tuple(ind)]
        for nm, p in (("occany", p_oc), ("ours", p_us)):
            fp = p & ~gt & valid
            fpd[nm]["n"] += int(fp.sum())
            fpd[nm]["dist"] += np.histogram(dist[fp], bins=[0, 1, 2, 3, 1e9])[0]
            fpd[nm]["cls"] += np.bincount(near[fp].astype(int), minlength=18)
        idx = np.argwhere(m)
        gt_h += np.bincount(idx[:, 2], minlength=nz)
        for nm, p in (("occany", p_oc), ("ours", p_us)):
            rec_h[nm] += np.bincount(idx[p[m], 2], minlength=nz)
            pi = np.argwhere(p & valid)          # everything the method claims, per height
            pred_h[nm] += np.bincount(pi[:, 2], minlength=nz)
        if (k + 1) % 20 == 0:
            print(f"  profiled {k + 1}/{len(use)}", flush=True)

    z = (np.arange(nz) + 0.5) * VOXEL + ORIGIN[2]
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.8))
    share = 100 * gt_h / max(gt_h.sum(), 1)
    ax[0].barh(z, share, height=VOXEL * 0.85, color="#8f8b86")
    ax[0].set_title("where the ground truth actually is", fontsize=12)
    ax[0].set_xlabel("% of all occupied voxels", fontsize=10)
    ax[0].set_ylabel("height above the ego plane (m)", fontsize=10)
    low = share[:4].sum()
    ax[0].axhspan(z[0] - VOXEL / 2, z[3] + VOXEL / 2, color="#c8b8a0", alpha=.35, zorder=0)
    ax[0].text(share.max() * .97, z[3] + 0.3, f"road band: {low:.0f}% of all occupancy",
               ha="right", fontsize=10, color="#6b5b45")
    for nm, col, lab in (("occany", "#e08214", "OccAny (released ckpt)"),
                         ("ours", "#2f6fd0", "ours (causal, 5 past frames)")):
        ax[1].plot(100 * rec_h[nm] / np.maximum(gt_h, 1), z, "-o", ms=4, color=col, label=lab)
        ax[2].plot(100 * rec_h[nm] / np.maximum(pred_h[nm], 1), z, "-o", ms=4,
                   color=col, label=lab)
    ax[1].set_title("how much of it each method recovers", fontsize=12)
    ax[1].set_xlabel("recall at that height (%)  —  higher is better", fontsize=10)
    ax[2].set_title("and how much of what it claims is real", fontsize=12)
    ax[2].set_xlabel("precision at that height (%)  —  higher is better", fontsize=10)
    ax[2].legend(fontsize=10, frameon=False, loc="lower right")
    for x in ax[1:]:
        x.set_xlim(0, 100)
    for x in ax:
        x.grid(alpha=.25, ls=":")
        x.set_ylim(z[0] - VOXEL, z[-1] + VOXEL)
    ax[1].set_ylabel("")
    ax[2].set_ylabel("")
    fig.suptitle(
        "Why so much of the 3D figure is 'missed' — a property of the benchmark, not of "
        "the rendering\n"
        f"Occ3D ground truth is accumulated LiDAR over the whole 40 m box. {low:.0f}% of "
        "it is the road-surface band; the rest is facades and\ntree canopy that a single "
        f"forward camera never observes. Read the two right panels together.  "
        f"({len(use)} validation samples)",
        fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    pw = os.path.join(ART, "fig_occany_why_missed.png")
    fig.savefig(pw, dpi=120)
    plt.close(fig)
    print("wrote", pw, flush=True)

    prof = {"n_samples": len(use),
            "z_centres_m": z.tolist(),
            "gt_share_pct": share.tolist(),
            "recall_pct": {k: (100 * v / np.maximum(gt_h, 1)).tolist()
                           for k, v in rec_h.items()},
            "precision_pct": {k: (100 * rec_h[k] / np.maximum(pred_h[k], 1)).tolist()
                              for k in rec_h},
            "predicted_voxels": {k: v.tolist() for k, v in pred_h.items()},
            "road_band_share_pct": float(low),
            "note": ("GT is accumulated LiDAR over the full 40 m box; a forward monocular "
                     "camera cannot observe most of it, so a large missed fraction is "
                     "expected for any camera-only method.")}
    with open(os.path.join(ART, "occany_height_profile.json"), "w") as f:
        json.dump(prof, f, indent=2)

    gt_share = gt_cls / max(gt_cls.sum(), 1)
    comp = {"n_samples": len(use),
            "question": ("is a red voxel 'nothing is there' or 'something is there but not "
                         "exactly here'?"),
            "distance_bins_m": ["<0.4", "0.4-0.8", "0.8-1.2", ">1.2"],
            "gt_class_share_pct": {names[c]: float(100 * gt_share[c])
                                   for c in range(len(names))},
            "methods": {}}
    for nm, r in fpd.items():
        n = max(r["n"], 1)
        share = r["cls"] / n
        comp["methods"][nm] = {
            "n_false_positive_voxels": int(r["n"]),
            "distance_to_nearest_real_occupied_pct":
                [float(100 * x / n) for x in r["dist"]],
            "within_1_2_m_pct": float(100 * r["dist"][:3].sum() / n),
            "nearest_class_share_pct": {names[c]: float(100 * share[c])
                                        for c in range(len(names)) if share[c] > 0.002},
            "movable_class_enrichment":
                float(share[list(movable)].sum() / max(gt_share[list(movable)].sum(), 1e-9)),
            "reading": ("movable-class enrichment above 1 means false positives cluster "
                        "on cars/trucks/pedestrians relative to their share of the ground "
                        "truth, which is the signature of a moving object smearing across "
                        "the accumulated frames while the target is a single instant")}
        print(f"  {nm}: {r['n']:,} FP, {100*r['dist'][:3].sum()/n:.0f}% within 1.2 m of "
              f"real geometry, movable enrichment "
              f"{comp['methods'][nm]['movable_class_enrichment']:.2f}x", flush=True)
    with open(os.path.join(ART, "occany_fp_composition.json"), "w") as f:
        json.dump(comp, f, indent=2)

    # ------------------------------------------------------------------ 3D figures
    handles = [Patch(color=C_GT, label="ground truth — the shape to be reproduced"),
               Patch(color=C_TP, label="recovered correctly"),
               Patch(color=C_MISS, label="left out by this method"),
               Patch(color=C_FP, label="predicted where the ground truth says free"),
               Patch(color=C_OURS, label="only ours recovers it"),
               Patch(color=C_OCC, label="only OccAny recovers it"),
               Patch(color=C_BOTH, label="both recover it"),
               Patch(color=(0.05, 0.05, 0.05), label="camera frustum at the target frame")]
    TITLE = "Visualization on Occ3D-nuScenes"

    rendered = []
    for row in pick:
        p_oc, p_us, gt, valid, d = predict(row)
        vscene, _vc = verdict_scene(p_oc, p_us, gt, valid)
        # the evaluation grid is ego-centric, so the camera sits at the origin looking +x
        cam = np.eye(4)
        cam[:3, :3] = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        fr = R3.frustum_mesh(cam, FOV_X_DEG, FOV_Y_DEG, depth=10.0, radius=0.24)
        scenes = [[mesh(gt & valid, C_GT)],
                  recovered_scene(p_oc, gt, valid),
                  recovered_scene(p_us, gt, valid),
                  error_scene(p_oc, gt, valid),
                  error_scene(p_us, gt, valid),
                  vscene]
        # one viewpoint for the whole row: centred on the ground truth, not on whatever
        # each panel happens to contain, so the six panels are directly superimposable
        idx = np.argwhere(gt & valid)
        ctr = ((idx + 0.5) * VOXEL + np.asarray(ORIGIN)).mean(0) if len(idx) else None
        rendered.append({"images": [R3.render(ms + [fr], size=(980, 700), look_at=ctr)
                                    for ms in scenes],
                         "label": row["scene"]})
        print(f"  rendered {row['scene']}", flush=True)

    base = os.path.join(ART, "fig_occany_3d.png")
    compose(rendered, base, TITLE, handles)
    write_pairs(rendered, base, TITLE, handles, per_fig=a.per_figure)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
