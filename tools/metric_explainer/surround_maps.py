"""Occ3D semantic SSC maps for the surround setting, against the 360-degree label.

The Occ3D rows of ``semantic_maps.py`` are drawn against OccAny's ``5frames`` label, whose rear half is ignored. That is
the wrong volume for comparing with OccAny's published nuScenes semantics, which are its ``surround`` setting. This
figure uses the 360-degree label and puts our single-camera and six-camera predictions side by side on it, so the panel
that fills in is exactly the +11.5 pp of occupancy and +2.4 mIoU the surround run measured.

Rows are uncurated: samples at fixed quantiles of the per-sample six-camera minus single-camera occupancy IoU
(``surround_scan.py``), so the first row is near the bottom of the improvement distribution.

OccAny does not appear here. Its saved outputs on this machine are the single-camera ``5frames`` run and store
occupancy only, so there is no six-camera OccAny volume to draw beside ours; its published surround number is a
scalar in ``report.md``, not a map.

    PYTHONPATH=$PWD python -m tools.metric_explainer.surround_maps
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from gates.gate8c1 import render3d as R3  # noqa: E402
from tools.gate8c1.figure_3d_compare import C_BOTH, C_FP, C_MISS, C_OCC, C_OURS, C_TP, GHOST, SOLID  # noqa: E402
from tools.like_for_like import config as LC, surround as SR, surround_predict as SP  # noqa: E402
from tools.metric_explainer.semantic_maps import OUT_DIR, class_meshes, flat_mesh, palette  # noqa: E402
from tools.occany_comparison.figures_3d import FRUSTUM, VIEW  # noqa: E402
from tools.zero_shot_eval.canonical.prediction import CanonicalPrediction  # noqa: E402
from tools.zero_shot_eval.practical.protocol import DATASET_SPECS  # noqa: E402

METHOD = "B3_s0"
QUANTILES = (0.15, 0.40, 0.65, 0.90)
IGNORE = 255
#: nuScenes ring, laid out as the car sees it: forward-facing row on top, rear-facing row beneath.
CANVAS = (("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"), ("CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"))
PANEL_TITLES = ("0 · input — six cameras", "1 · ground-truth semantics (360°)", "2 · ours, 1 camera — predicted class",
                "3 · ours, 6 cameras — predicted class", "4 · ours, 6 cameras — correct vs wrong",
                "5 · ours, 6 cameras — mistakes by true class", "6 · coverage: 1 camera vs 6 cameras")


def ring_canvas(views: List[Dict], cell=(420, 236)) -> np.ndarray:
    """The six inputs on one canvas, forward row above rear row."""
    from PIL import Image
    by_cam = {v["camera"]: v["path"] for v in views}
    rows = []
    for row in CANVAS:
        rows.append(np.hstack([np.asarray(Image.open(by_cam[c]).resize(cell, Image.LANCZOS)) for c in row]))
    return np.vstack(rows)


def pick_samples() -> List[Dict]:
    with np.load(os.path.join(LC.ARTIFACT_DIR, "surround_scan.npz"), allow_pickle=True) as z:
        ids = [str(x) for x in z["sample_id"]]
        iou = lambda c: c[:, 0] / np.maximum(c[:, 0] + c[:, 1] + c[:, 2], 1)
        delta = iou(z["six_cameras"]) - iou(z["one_camera"])
    order = np.argsort(delta, kind="stable")
    out = []
    for q in QUANTILES:
        r = int(round(q * (len(order) - 1))); i = int(order[r])
        out.append({"sample_id": ids[i], "rank_pct": 100.0 * r / (len(order) - 1), "delta_iou_pp": 100 * float(delta[i])})
    return out


def frustums(sample) -> List:
    """One frustum per ring camera. The grid frame is the ego frame, so a camera's to-grid transform is its
    camera-to-ego transform -- the same relation that makes OccAny's stored ``T_cam_to_voxel`` equal to CAM_FRONT's."""
    nusc = SR.nuscenes(); token = SR.sample_token(sample); s = nusc.get("sample", token)
    out = []
    for v in SR.ring_views(token):
        sd = nusc.get("sample_data", s["data"][v["camera"]])
        K = np.asarray(nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])["camera_intrinsic"], float)
        w, h = float(sd["width"]), float(sd["height"])
        fx = float(np.degrees(2 * np.arctan(w / 2 / K[0, 0]))); fy = float(np.degrees(2 * np.arctan(h / 2 / K[1, 1])))
        out.append(R3.frustum_mesh(v["T_cam_to_ego"], fx, fy, **FRUSTUM["occ3d"]))
    return out


def compose(rows: List[Dict], out_path: str, title: str, handles, row_h: float = 4.9, legend_h: float = 2.8,
            title_h: float = 1.3, dpi: int = 100) -> None:
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
    ax = fig.add_subplot(grid[n, :]); ax.set_axis_off()
    ax.legend(handles=list(handles), loc="upper center", ncol=8, fontsize=10.5, frameon=False, handlelength=1.4, columnspacing=1.4)
    fig.suptitle(title, fontsize=17, y=1 - 0.30 * title_h / height)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path, flush=True)


def render() -> Dict:
    spec = DATASET_SPECS["occ3d"]; pal = palette("occ3d", spec.n_classes)
    samples = {s.sample_id: s for s in SR.samples()}
    rows, picks, present = [], [], set()
    for pk in pick_samples():
        s = samples[pk["sample_id"]]
        lab = SR.surround_label(s).astype(np.int64)
        dims = lab.shape
        valid = lab != IGNORE
        gt_occ = valid & (lab != spec.empty_class)
        p1 = CanonicalPrediction.load(LC.prediction_path("occ3d", s.sample_id))
        p6 = CanonicalPrediction.load(SP.prediction_path(s.sample_id))
        voxel, origin = p6.voxel_size, p6.origin
        occ1 = p1.occupied.reshape(dims) & valid
        occ6 = p6.occupied.reshape(dims) & valid
        lab6 = p6.dense_labels(METHOD, spec.empty_class).reshape(dims)
        lab1 = p1.dense_labels(METHOD, spec.empty_class).reshape(dims)
        hit = occ6 & gt_occ
        correct, wrong = hit & (lab6 == lab), hit & (lab6 != lab)
        present |= set(np.unique(lab[gt_occ]).tolist()) | set(np.unique(lab6[occ6]).tolist())

        ctr = ((np.argwhere(gt_occ) + 0.5) * voxel + np.asarray(origin)).mean(0) if gt_occ.any() else None
        fr = frustums(s)
        scenes = [class_meshes(lab, gt_occ, pal, voxel, origin),
                  class_meshes(lab1, occ1, pal, voxel, origin),
                  class_meshes(lab6, occ6, pal, voxel, origin),
                  [flat_mesh(gt_occ & ~occ6, C_MISS, voxel, origin, GHOST), flat_mesh(correct, C_TP, voxel, origin),
                   flat_mesh(wrong, C_FP, voxel, origin)],
                  [flat_mesh(correct, C_MISS, voxel, origin, GHOST)] + class_meshes(lab, wrong, pal, voxel, origin),
                  [flat_mesh(gt_occ & occ1 & occ6, C_BOTH, voxel, origin, 0.55),
                   flat_mesh(gt_occ & occ6 & ~occ1, C_OURS, voxel, origin),
                   flat_mesh(gt_occ & occ1 & ~occ6, C_OCC, voxel, origin)]]
        images = [ring_canvas(SR.ring_views(SR.sample_token(s)))]
        images += [R3.render(ms + fr, size=(980, 700), look_at=ctr, **VIEW["occ3d"]) for ms in scenes]
        rows.append({"images": images, "label": s.sample_id})
        rec = int((occ6 & gt_occ).sum())
        picks.append({**pk, "recovered_6cam": rec, "recovered_1cam": int((occ1 & gt_occ).sum()),
                      "label_correct_pct": 100 * int(correct.sum()) / max(rec, 1)})
        print(f"  [surround] rendered {s.sample_id} (Δ-rank {pk['rank_pct']:.0f} %)", flush=True)

    handles = [Patch(color=C_TP, label="our label is correct"), Patch(color=C_FP, label="wrong class / predicted where GT is free"),
               Patch(color=C_MISS, label="left out"), Patch(color=C_OURS, label="recovered only with 6 cameras"),
               Patch(color=C_OCC, label="recovered only with 1 camera"), Patch(color=C_BOTH, label="recovered by both"),
               Patch(color=(0.05, 0.05, 0.05), label="camera frustums (six)")]
    handles += [Patch(color=pal[c], label=spec.class_names[c]) for c in sorted(present) if c != spec.empty_class]
    title = "Semantic scene completion on Occ3D-nuScenes — surround setting"
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "semantic_maps_occ3d_surround.png")
    compose(rows, out, title, handles)
    for k in range(0, len(rows), 2):
        compose(rows[k:k + 2], f"{os.path.splitext(out)[0]}_pair{k // 2 + 1}.png", title, handles)
    rec = {"setting": "surround", "method": METHOD, "figure": out, "samples": picks}
    json.dump(rec, open(os.path.join(OUT_DIR, "semantic_maps_occ3d_surround_index.json"), "w"), indent=1, default=float)
    return rec


if __name__ == "__main__":
    print(json.dumps(render(), indent=1, default=float))
