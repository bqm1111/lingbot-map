#!/usr/bin/env python
"""Qualitative figures, selected by a rule that cannot see a semantic result.

Clips are chosen at fixed percentiles {10, 50, 90} of the **frozen B-D binary occupancy
IoU** published by the reproduction gates (Gate 5.1 for SemanticKITTI and Occ3D-nuScenes,
Gate 5.2 for SSCBench-KITTI-360). That number exists before Gate 6 and knows nothing about
semantics, so the panels cannot be cherry-picked for the teacher's benefit.

Six panels per clip: anchor RGB, ground-truth semantics (BEV), the Trident 2D result,
the raw and dilated 3D semantic predictions (BEV), and the coverage/naming/correct error
map (BEV).

    python tools/gate6/figures.py --dataset kitti360
"""
from __future__ import annotations

import argparse, csv, json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT                                  # noqa: E402
from gates.gate6 import frames as F, grids, targets, vocab                     # noqa: E402

ART = os.path.join(REPO_ROOT, "artifacts", "gate6")
PERCENTILES = (10, 50, 90)
FROZEN_PER_CLIP = {
    "semantickitti": ("artifacts/gate5_1/per_clip_kitti.csv", "G51-B", "group"),
    "occ3d": ("artifacts/gate5_1/per_clip_occ3d.csv", "G51-B", "group"),
    "kitti360": ("artifacts/gate5_2/gate5_2_per_clip.csv", "B", "block"),
}

# Distinguishable palettes; index 0 of each is reserved for "empty".
_BASE = np.array([
    [0, 0, 0], [245, 150, 100], [245, 230, 100], [150, 60, 30], [180, 30, 80],
    [255, 0, 0], [30, 30, 255], [200, 40, 255], [90, 30, 150], [255, 0, 255],
    [255, 150, 255], [75, 0, 75], [175, 0, 75], [255, 200, 0], [255, 120, 50],
    [0, 175, 0], [135, 60, 0], [150, 240, 80], [255, 240, 150], [255, 0, 0],
], np.uint8)


def palette(n):
    p = np.zeros((n + 1, 3), np.uint8)
    for i in range(n + 1):
        p[i] = _BASE[i % len(_BASE)]
    return p


def frozen_iou(dataset):
    rel, cond, _ = FROZEN_PER_CLIP[dataset]
    out = {}
    with open(os.path.join(REPO_ROOT, rel)) as fh:
        for r in csv.DictReader(fh):
            if r["condition"] == cond and r["corrector"] == "dilate_r2":
                out[r["clip_id"]] = float(r["iou"])
    return out


def pick_clips(dataset, available):
    iou = {k: v for k, v in frozen_iou(dataset).items() if k in available}
    if not iou:
        return []
    ids = sorted(iou, key=lambda k: (iou[k], k))
    out = []
    for p in PERCENTILES:
        i = min(len(ids) - 1, max(0, int(round((p / 100.0) * (len(ids) - 1)))))
        out.append((ids[i], p, iou[ids[i]]))
    return out


def bev_labels(vol_flat, dims, empty):
    """Top-down view: the label of the highest occupied voxel in each column."""
    v = vol_flat.reshape(dims)
    occ = v != empty
    out = np.full(dims[:2], empty, v.dtype)
    for z in range(dims[2]):
        m = occ[:, :, z]
        out[m] = v[:, :, z][m]
    return out


def bev_category(cat_flat, dims):
    """Column category: worst-case wins (miss > naming > correct > none)."""
    c = cat_flat.reshape(dims)
    out = np.zeros(dims[:2], np.uint8)
    for k in (3, 2, 1):
        out[np.any(c == k, axis=2)] = k
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=list(vocab.DATASETS))
    a = ap.parse_args()
    ds = a.dataset
    v = vocab.load(ds)
    G = grids.EVAL_GRID[ds]
    n_vox = int(np.prod(G.dims))
    man = json.load(open(os.path.join(ART, f"prediction_manifest_{ds}.json")))
    proot = man["prediction_root"]
    avail = {n[:-4] for n in man["per_file_sha256"]}
    recs = {r.clip_id: r for r in F.read_manifest(ds, REPO_ROOT)}
    sel = pick_clips(ds, avail)
    if not sel:
        print(f"[{ds}] no clips available for figures")
        return 0

    pal = palette(max(max(v.labels), v.empty_label))
    # "empty" must read as background, whatever integer the benchmark uses for it
    # (0 on the KITTI family, 17 on Occ3D-nuScenes)
    pal[v.empty_label] = (0, 0, 0)
    err_colors = np.array([[230, 230, 230], [40, 160, 60], [235, 160, 40], [200, 40, 40]],
                          np.uint8)
    # wide camera panels get wide cells, BEV panels get square ones, so nothing is
    # letterboxed and the row height is set by the BEV squares
    wr = [3.34, 3.34, 1.0, 1.0, 1.0, 1.0]
    cell = 22.0 / sum(wr)
    fig, axes = plt.subplots(len(sel), 6, figsize=(22.0, cell * len(sel) + 1.2),
                             gridspec_kw={"width_ratios": wr, "wspace": 0.04,
                                          "hspace": 0.06})
    axes = np.atleast_2d(axes)

    for row, (cid, pct, fiou) in enumerate(sel):
        rec = recs[cid]
        with np.load(os.path.join(proot, f"{cid}.npz")) as z:
            labels = z["labels"].astype(np.int32)
            raw_f, raw_c = z["raw_flat"], z["raw_channel"]
            dil_f, dil_c = z["dil_flat"], z["dil_channel"]
        target, keep = targets.semantic_target(ds, rec.raw, REPO_ROOT)

        pr = np.full(n_vox, v.empty_label, np.int32)
        pr[raw_f.astype(np.int64)] = labels[raw_c.astype(np.int64)]
        pd = np.full(n_vox, v.empty_label, np.int32)
        pd[dil_f.astype(np.int64)] = labels[dil_c.astype(np.int64)]

        t = target.reshape(-1).copy()
        k = keep.reshape(-1)
        gt_occ = (t != v.empty_label) & k
        pd_occ = (pd != v.empty_label) & k
        cat = np.zeros(n_vox, np.uint8)
        cat[gt_occ & pd_occ & (pd == t)] = 1
        cat[gt_occ & pd_occ & (pd != t)] = 2
        cat[gt_occ & ~pd_occ] = 3

        import cv2
        img = cv2.cvtColor(cv2.imread(os.path.join(F.image_root(ds, REPO_ROOT),
                                                   rec.rel_images[-1])), cv2.COLOR_BGR2RGB)
        with np.load(F.semantic_cache_path(ds, rec.keys[-1])) as z:
            sem2d = labels[z["label"].astype(np.int64)]

        tv = np.where(k, t, v.empty_label).astype(np.int32)
        panels = [
            ("anchor RGB", img, None),
            (f"Trident 2D (frozen teacher)", pal[np.clip(sem2d, 0, len(pal) - 1)], None),
            ("GT semantics", pal[bev_labels(tv, G.dims, v.empty_label)], None),
            ("B-R raw", pal[bev_labels(pr, G.dims, v.empty_label)], None),
            ("B-D dilated", pal[bev_labels(pd, G.dims, v.empty_label)], None),
            ("error map", err_colors[bev_category(cat, G.dims)], None),
        ]
        for col, (title, im, _c) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(np.rot90(im) if col >= 2 else im, aspect="auto",
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4); sp.set_color("#999999")
            if row == 0:
                ax.set_title(title, fontsize=11, pad=6)
            if col == 0:
                ax.set_ylabel(f"p{pct}   frozen B-D IoU {fiou:.3f}", fontsize=9)
    handles = [Patch(color=err_colors[i] / 255, label=l) for i, l in enumerate(
        ["no valid GT occupancy", "correct semantic occupancy", "naming error",
         "coverage miss"])]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f"Gate 6 — frozen Trident-H semantic lifting, {ds}. Clips chosen at "
                 f"percentiles {PERCENTILES} of the FROZEN B-D binary occupancy IoU "
                 f"(prediction-independent). Columns 3-6 are bird's-eye views, "
                 f"sensor at image left.", fontsize=12)
    fig.subplots_adjust(left=0.035, right=0.995, top=1 - 0.55 / (cell * len(sel) + 1.2),
                        bottom=0.055)
    out = os.path.join(ART, f"gate6_qualitative_{ds}.png")
    fig.savefig(out, dpi=110)
    json.dump({"dataset": ds, "rule": f"percentiles {list(PERCENTILES)} of the frozen "
                                      f"B-D binary occupancy IoU",
               "source": FROZEN_PER_CLIP[ds][0],
               "selected": [{"clip_id": c, "percentile": p, "frozen_bd_iou": i}
                            for c, p, i in sel]},
              open(os.path.join(ART, f"figure_selection_{ds}.json"), "w"), indent=2)
    print("wrote", out, [c for c, _, _ in sel])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
