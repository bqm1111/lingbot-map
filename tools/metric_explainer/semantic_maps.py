"""Semantic SSC maps: our labels vs the ground truth, beside OccAny, in the style of
``tools/occany_comparison/figures_3d.py`` (same Open3D voxel renderer, camera and frustum).

Per row: the input camera image, the ground-truth semantics, our predicted class for every voxel, where our label is
right or wrong, our mistakes coloured by their true class, and OccAny's occupancy. Colours are each benchmark's own
class colours (SemanticKITTI ``semantic_kitti.yaml``; Occ3D/nuScenes the table in OccAny's ``nuscenes.py``).

OccAny's released outputs on this machine are **geometry only** (its semantic preset needs GroundingDINO, which cannot
run on this GPU -- ``artifacts/final_zero_shot/logs/occany_semantic_attempt.json``), so its panels show occupancy.
Every voxel of the valid label volume is drawn: no region, distance or field-of-view restriction.

Inputs are read, never recomputed: the frozen canonical predictions, OccAny's stored labels and predictions, and the
frozen per-sample score shards (used only to order the rows).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from gates.gate8c1 import render3d as R3  # noqa: E402
from tools.gate8c1.figure_3d_compare import C_BOTH, C_FP, C_MISS, C_OCC, C_OURS, C_TP, GHOST, SOLID  # noqa: E402
from tools.occany_comparison import config as OC  # noqa: E402
from tools.occany_comparison import scoring as SC  # noqa: E402
from tools.occany_comparison.figures_3d import FRUSTUM, NAMES, VIEW  # noqa: E402
from tools.zero_shot_eval import config as ZC, samples as ZS  # noqa: E402
from tools.zero_shot_eval.canonical.prediction import CanonicalPrediction  # noqa: E402
from tools.zero_shot_eval.practical.protocol import DATASET_SPECS  # noqa: E402

OUT_DIR = os.path.join(ZC.REPO_ROOT, "artifacts", "metric_explainer")
METHOD = "B3_s0"
QUANTILES = (0.15, 0.40, 0.65, 0.90)         # rows at fixed quantiles of (our IoU - OccAny IoU), Track P per-sample counts
IMAGE_HEIGHT = {"semantickitti": 370, "occ3d": 900}
#: Occ3D / nuScenes official class colours, index = label id (OccAny ``occany/datasets/nuscenes.py``)
OCC3D_COLORS = np.array([[0, 0, 0], [112, 128, 144], [220, 20, 60], [255, 127, 80], [255, 158, 0], [233, 150, 70], [255, 61, 99],
                         [0, 0, 230], [47, 79, 79], [255, 140, 0], [255, 98, 70], [0, 207, 191], [175, 0, 75], [75, 0, 75],
                         [112, 180, 60], [222, 184, 135], [0, 175, 0], [135, 206, 235]], float) / 255.0
#: Same plain register as the geometry maps (``tools/occany_comparison/figures_3d.py``): short panel titles, a one-line
#: figure title and a bare sample id per row. Panel 5 still says "occupancy only" because that caveat is load-bearing --
#: OccAny's saved outputs on this machine store no semantics, so panels 5-6 compare occupancy, not class.
PANEL_TITLES = ("0 · input image", "1 · ground-truth semantics", "2 · ours — predicted class",
                "3 · ours — correct vs wrong", "4 · ours — mistakes by true class",
                "5 · OccAny — occupancy only", "6 · who recovers what")


def palette(dataset: str, n_classes: int) -> np.ndarray:
    """Each benchmark's official class colours, RGB in [0, 1], index = label id used by our predictions."""
    if dataset == "occ3d":
        return OCC3D_COLORS[:n_classes]
    spec = yaml.safe_load(open(OC.OCCANY_SK_LEARNING_MAP))
    inv, colours = spec["learning_map_inv"], spec["color_map"]
    return np.asarray([np.asarray(colours[inv[c]], float)[::-1] / 255.0 for c in range(n_classes)])   # stored BGR


def class_meshes(labels: np.ndarray, mask: np.ndarray, pal: np.ndarray, voxel: float, origin, shrink: float = SOLID) -> List:
    """One merged cube mesh per class present in ``mask``."""
    out = []
    for c in np.unique(labels[mask]):
        idx = np.argwhere(mask & (labels == c))
        if len(idx):
            out.append(R3.cube_mesh(idx, np.tile(pal[int(c)], (len(idx), 1)), voxel, origin, shrink=shrink))
    return out


def flat_mesh(mask: np.ndarray, colour, voxel: float, origin, shrink: float = SOLID):
    idx = np.argwhere(mask)
    return None if not len(idx) else R3.cube_mesh(idx, np.tile(colour, (len(idx), 1)), voxel, origin, shrink=shrink)


def iou_from_counts(counts: np.ndarray) -> np.ndarray:
    """[N, 3] TP / FP / FN -> [N] IoU."""
    tp, fp, fn = counts[:, 0].astype(float), counts[:, 1].astype(float), counts[:, 2].astype(float)
    return tp / np.maximum(tp + fp + fn, 1)


def pick_samples(dataset: str) -> List[Tuple]:
    """Uncurated rows: samples at fixed quantiles of (our IoU - OccAny IoU), as in the geometry maps figure.

    Low quantiles are samples where OccAny is better. The counts come from the frozen evaluation shards (Track P),
    so nothing is re-scored here."""
    with np.load(os.path.join(ZC.REPO_ROOT, "artifacts", "final_zero_shot", "scores", f"scores_{dataset}.npz"), allow_pickle=True) as z:
        ids = [str(x) for x in z["sample_id"]]
        delta = iou_from_counts(z["ps__occ__P__all"]) - iou_from_counts(z["ps__occany_raw__P__all"])
    by_id = {s.sample_id: s for s in ZS.evaluated_samples(dataset)}
    order = np.argsort(delta, kind="stable")
    rows = []
    for q in QUANTILES:
        r = int(round(q * (len(order) - 1))); i = int(order[r])
        rows.append((by_id[ids[i]], {"rank_pct": 100.0 * r / (len(order) - 1), "delta_iou_track_p": 100 * float(delta[i])}))
    return rows


def fov_degrees(dataset: str, K: np.ndarray, width: int) -> Tuple[float, float]:
    return (float(np.degrees(2 * np.arctan(width / 2 / K[0, 0]))),
            float(np.degrees(2 * np.arctan(IMAGE_HEIGHT[dataset] / 2 / K[1, 1]))))


def scene_stats(ours_occ, occany_occ, gt_occ, valid, correct) -> Dict[str, float]:
    """Descriptive per-scene numbers over the whole drawn volume (the frozen aggregates use Track P / Track O)."""
    iou = lambda p: int((p & gt_occ).sum()) / max(int((p & gt_occ).sum() + (p & ~gt_occ & valid).sum() + (~p & gt_occ).sum()), 1)
    recovered = int((ours_occ & gt_occ).sum())
    return {"ours_iou_full_volume": 100 * iou(ours_occ), "occany_iou_full_volume": 100 * iou(occany_occ),
            "recovered": recovered, "label_correct_pct": 100 * int(correct.sum()) / max(recovered, 1)}


def compose(rows: List[Dict], out_path: str, title: str, handles: Sequence[Patch], row_h: float = 4.9,
            legend_h: float = 2.6, title_h: float = 1.3, dpi: int = 100) -> None:
    """Panels in a grid with the legend in its own strip underneath, so it can never overlap a panel.

    Heights are reserved in inches, so the title and legend get the same physical space whatever the row count."""
    n, cols = len(rows), len(PANEL_TITLES)
    height = n * row_h + legend_h + title_h
    fig = plt.figure(figsize=(6.0 * cols, height))
    grid = fig.add_gridspec(n + 1, cols, height_ratios=[row_h] * n + [legend_h],
                            top=1 - title_h / height, bottom=0.02, left=0.006, right=0.994, hspace=0.12, wspace=0.02)
    for r, row in enumerate(rows):
        first = None
        for c, img in enumerate(row["images"]):
            ax = fig.add_subplot(grid[r, c]); ax.imshow(img); ax.set_axis_off(); ax.set_title(PANEL_TITLES[c], fontsize=11)
            first = first or ax
        if row.get("label"):
            first.text(0.0, -0.04, row["label"], transform=first.transAxes, fontsize=9, va="top",
                       bbox=dict(fc="w", ec="0.7", alpha=.9))
    legend_ax = fig.add_subplot(grid[n, :]); legend_ax.set_axis_off()
    legend_ax.legend(handles=list(handles), loc="upper center", ncol=8, fontsize=10.5, frameon=False, handlelength=1.4, columnspacing=1.4)
    fig.suptitle(title, fontsize=17, y=1 - 0.30 * title_h / height)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_dataset(dataset: str) -> Dict:
    spec = DATASET_SPECS[dataset]; pal = palette(dataset, spec.n_classes)
    rows, picks, present = [], [], set()
    for s, rank in pick_samples(dataset):
        pred = CanonicalPrediction.load(ZC.prediction_path(dataset, s.sample_id))
        dims, voxel, origin = pred.dims, pred.voxel_size, pred.origin
        oc = SC.load_occany(dataset, s.occany_pkl, dims)
        gt_lab = oc.label.astype(np.int64)
        valid = gt_lab != SC.C.IGNORE
        gt_occ = valid & (gt_lab != spec.empty_class)
        ours_occ = pred.occupied.reshape(dims) & valid
        ours_lab = pred.dense_labels(METHOD, spec.empty_class).reshape(dims)
        occany_occ = oc.pred & valid

        hit = ours_occ & gt_occ
        correct, wrong = hit & (ours_lab == gt_lab), hit & (ours_lab != gt_lab)
        present |= set(np.unique(gt_lab[gt_occ]).tolist()) | set(np.unique(ours_lab[ours_occ]).tolist())
        st = scene_stats(ours_occ, occany_occ, gt_occ, valid, correct)

        ctr = ((np.argwhere(gt_occ) + 0.5) * voxel + np.asarray(origin)).mean(0) if gt_occ.any() else None
        fx, fy = fov_degrees(dataset, pred.K, pred.image_width)
        fr = R3.frustum_mesh(pred.T_cam_to_grid, fx, fy, **FRUSTUM[dataset])
        scenes = [class_meshes(gt_lab, gt_occ, pal, voxel, origin),
                  class_meshes(ours_lab, ours_occ, pal, voxel, origin),
                  [flat_mesh(gt_occ & ~ours_occ, C_MISS, voxel, origin, GHOST), flat_mesh(correct, C_TP, voxel, origin),
                   flat_mesh(wrong, C_FP, voxel, origin)],
                  [flat_mesh(correct, C_MISS, voxel, origin, GHOST)] + class_meshes(gt_lab, wrong, pal, voxel, origin),
                  [flat_mesh(gt_occ & ~occany_occ, C_MISS, voxel, origin, GHOST), flat_mesh(occany_occ & gt_occ, C_TP, voxel, origin),
                   flat_mesh(occany_occ & ~gt_occ & valid, C_FP, voxel, origin)],
                  [flat_mesh(gt_occ & ours_occ & occany_occ, C_BOTH, voxel, origin, 0.55),
                   flat_mesh(gt_occ & ours_occ & ~occany_occ, C_OURS, voxel, origin),
                   flat_mesh(gt_occ & occany_occ & ~ours_occ, C_OCC, voxel, origin)]]
        images = [plt.imread(ZS.history_frames(s)[-1]["path"])]
        images += [R3.render(ms + [fr], size=(980, 700), look_at=ctr, **VIEW[dataset]) for ms in scenes]
        label = s.sample_id
        rows.append({"images": images, "label": label})
        picks.append({"sample_id": s.sample_id, **rank, **st})
        print(f"  [{dataset}] rendered {s.sample_id}", flush=True)

    handles = [Patch(color=C_TP, label="our label is correct"), Patch(color=C_FP, label="wrong class / predicted where GT is free"),
               Patch(color=C_MISS, label="left out by this method"), Patch(color=C_OURS, label="only ours recovers it"),
               Patch(color=C_OCC, label="only OccAny recovers it"), Patch(color=C_BOTH, label="both recover it"),
               Patch(color=(0.05, 0.05, 0.05), label="camera frustum")]
    handles += [Patch(color=pal[c], label=spec.class_names[c]) for c in sorted(present) if c != spec.empty_class]
    title = f"Semantic scene completion on {NAMES[dataset]}"
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"semantic_maps_{dataset}.png")
    compose(rows, out, title, handles)
    for k in range(0, len(rows), 2):
        compose(rows[k:k + 2], f"{os.path.splitext(out)[0]}_pair{k // 2 + 1}.png", title, handles)
    return {"dataset": dataset, "method": METHOD, "figure": out, "samples": picks}


def render_all(datasets: Sequence[str] = ZC.DATASETS) -> Dict:
    rec = {ds: render_dataset(ds) for ds in datasets}
    json.dump(rec, open(os.path.join(OUT_DIR, "semantic_maps_index.json"), "w"), indent=1)
    return rec


if __name__ == "__main__":
    print(json.dumps(render_all(), indent=1))
