"""Gate 15 figures. Plots read analysis.json; example panels recompute B0/B1 on deterministically chosen 0006 anchors.

Example selection (no cherry-picking): per-anchor accuracy gain (B1 - B0, seed 0) on completed_unobserved, anchors with
>= 2000 such voxels; 'success' = 90th percentile, 'median' = 50th, 'failure' = 10th; 'observed-correct / completion-wrong'
= the anchor maximising teacher accuracy on observed voxels minus B1 accuracy on completed_unobserved.
Panels are bird's-eye views: each column shows the label of its highest non-empty voxel.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import torch

from tools.semantic_completion import config as C
from tools.semantic_completion.analysis import load_shards
from tools.semantic_completion.evaluation import baseline_nearest as B0
from tools.semantic_completion.evaluation.evaluate import head_labels, load_head
from tools.semantic_completion.evaluation.regions import PRIMARY
from tools.semantic_completion.evidence.lift import load_evidence
from tools.semantic_completion.evidence.trident import classify, load_text_embeddings, load_token_map, sample_at_lattice
from tools.semantic_completion.frozen_geometry import geometry_input, load_frozen_geometry, load_hard_map, occupancy_and_trunk

FIG = os.path.join(C.ARTIFACT_DIR, "figures")
#: KITTI-360 / SemanticKITTI-style colours for 0 (empty) and classes 1..18
PALETTE = np.array([[255, 255, 255], [245, 150, 100], [245, 230, 100], [150, 60, 30], [180, 30, 80], [255, 0, 0], [30, 30, 255],
                    [255, 0, 255], [255, 150, 255], [75, 0, 75], [175, 0, 75], [255, 200, 0], [255, 120, 50], [0, 175, 0],
                    [150, 240, 80], [255, 240, 150], [255, 0, 0], [50, 120, 255], [0, 150, 255]], np.uint8)


def _plt():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def bev(labels: np.ndarray) -> np.ndarray:
    """[X, Y, Z] labels -> [X, Y] label of the highest non-empty voxel per column (0 if none)."""
    nz = labels > 0
    top = np.where(nz.any(2), (C.GRID_DIMS[2] - 1) - np.argmax(nz[:, :, ::-1], axis=2), -1)
    out = np.take_along_axis(labels, np.clip(top, 0, None)[..., None], 2)[..., 0]
    return np.where(top >= 0, out, 0)


def summary_plots(res: Dict) -> List[str]:
    plt = _plt(); seeds = res["seeds"]; names = res["class_names"]; out = []
    reg = res["regions"]
    labels = ["oracle_observed", "completed_observed", "oracle_unobserved", "completed_unobserved", "ssc_full"]
    fig, ax = plt.subplots(figsize=(8, 3.6)); x = np.arange(len(labels))
    for i, m in enumerate(("B0", "B1")):
        v = [100 * np.mean([reg[r][m][s]["miou"] for s in seeds]) for r in labels]
        ax.bar(x + (i - 0.5) * 0.35, v, 0.35, label=m)
    ax.axhline(100 * res["teacher_observed"]["miou"], ls="--", color="k", lw=1, label="teacher readout, observed")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, fontsize=8); ax.set_ylabel("mIoU % (mean of 3 seeds)"); ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); p = os.path.join(FIG, "6_observed_vs_unobserved_miou.png"); fig.savefig(p, dpi=150); plt.close(fig); out.append(p)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for m in ("B0", "B1"):
        ax.plot(C.DISTANCE_NAMES, [100 * np.mean([res["distance"][m][s][b]["miou"] or 0 for s in seeds]) for b in range(len(C.DISTANCE_NAMES))], marker="o", label=f"{m} mIoU")
        ax.plot(C.DISTANCE_NAMES, [100 * np.mean([res["distance"][m][s][b]["overall_accuracy"] for s in seeds]) for b in range(len(C.DISTANCE_NAMES))], marker="s", ls=":", label=f"{m} accuracy")
    ax.set_xlabel("distance to nearest causal evidence"); ax.set_ylabel("% (completed_unobserved)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); p = os.path.join(FIG, "7_semantics_vs_propagation_distance.png"); fig.savefig(p, dpi=150); plt.close(fig); out.append(p)
    cls = [c for c in range(1, C.N_CLASSES + 1) if reg[PRIMARY]["B0"][seeds[0]]["per_class_iou"][c] is not None]
    fig, ax = plt.subplots(figsize=(10, 3.6)); x = np.arange(len(cls))
    for i, m in enumerate(("B0", "B1")):
        ax.bar(x + (i - 0.5) * 0.35, [100 * np.mean([reg[PRIMARY][m][s]["per_class_iou"][c] for s in seeds]) for c in cls], 0.35, label=m)
    ax.set_xticks(x); ax.set_xticklabels([names[c] for c in cls], rotation=45, ha="right", fontsize=8); ax.set_ylabel("IoU % completed_unobserved"); ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); p = os.path.join(FIG, "8_per_class_iou.png"); fig.savefig(p, dpi=150); plt.close(fig); out.append(p)
    fig, ax = plt.subplots(figsize=(10, 3.6)); x = np.arange(C.N_CLASSES)
    series = [("GT", reg[PRIMARY]["B1"][seeds[0]]["gt_frequency"]), ("B0", np.mean([reg[PRIMARY]["B0"][s]["pred_frequency"] for s in seeds], 0)),
              ("B1", np.mean([reg[PRIMARY]["B1"][s]["pred_frequency"] for s in seeds], 0))]
    for i, (lab, v) in enumerate(series):
        ax.bar(x + (i - 1) * 0.27, 100 * np.asarray(v), 0.27, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([names[c] for c in range(1, C.N_CLASSES + 1)], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% of completed_unobserved voxels"); ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); p = os.path.join(FIG, "9_class_frequency.png"); fig.savefig(p, dpi=150); plt.close(fig); out.append(p)
    return out


def choose_examples(z: Dict[str, np.ndarray]) -> Dict[str, int]:
    def acc(cm):
        cm = cm.astype(np.float64); return np.trace(cm[1:, 1:], axis1=-2, axis2=-1) / np.maximum(cm[:, 1:, :].sum((-2, -1)), 1)
    b1, b0 = z[f"conf__B1_seed0__{PRIMARY}"], z[f"conf__B0_seed0__{PRIMARY}"]
    n = b1[:, 1:, :].sum((1, 2)); ok = n >= 2000
    gain = acc(b1) - acc(b0); ids = z["ids"]; order = np.argsort(gain[ok]); sel = ids[ok][order]
    teach = z["teacher_obs"].astype(np.float64); tacc = np.trace(teach[:, 1:, 1:], axis1=1, axis2=2) / np.maximum(teach[:, 1:, :].sum((1, 2)), 1)
    gap = np.where(ok, tacc - acc(b1), -np.inf)
    return {"success_p90": int(sel[int(0.9 * (len(sel) - 1))]), "median_p50": int(sel[int(0.5 * (len(sel) - 1))]),
            "failure_p10": int(sel[int(0.1 * (len(sel) - 1))]), "observed_correct_completion_wrong": int(ids[int(np.argmax(gap))])}


@torch.no_grad()
def example_panels(examples: Dict[str, int], device: str, paths: C.Paths = C.Paths()) -> List[str]:
    plt = _plt(); dev = torch.device(device); text, _, _ = load_text_embeddings(paths.text_embeddings); text = text.to(dev)
    net = load_frozen_geometry(0, dev, paths); head = load_head(paths.head_checkpoint(0, C.TrainConfig().steps), dev); out = []
    for name, t in examples.items():
        S, F_ = load_hard_map(C.VAL_DRIVE, t, dev, paths); occ, trunk = occupancy_and_trunk(net, geometry_input(S, F_)); occ = occ[0]
        ev = load_evidence(paths.evidence_path(C.VAL_DRIVE, t), dev); E = ev.dense_mask()
        ev_lab = torch.zeros(C.N_VOXELS, dtype=torch.long, device=dev); ev_lab[ev.flat_index] = classify(ev.feature, text)
        near, _ = B0.nearest_evidence(E); b0 = B0.nearest_label(ev_lab, near)
        b1, _ = head_labels(head, trunk, ev.flat_index, ev.feature, text)
        with np.load(paths.eval_mask_path(t)) as z:
            gt = z["sem_gt"].astype(np.int64)
        keys = np.load(paths.evidence_path(C.VAL_DRIVE, t))["keys"]
        tok = load_token_map(paths.token_path(C.VAL_DRIVE, str(keys[-1])), dev)
        img_path = str(np.load(paths.token_path(C.VAL_DRIVE, str(keys[-1])))["image_path"])
        teacher2d = classify(sample_at_lattice(tok, (140, 518)), text).cpu().numpy()
        vol = lambda v: v.reshape(C.GRID_DIMS)
        occ_np, E_np = occ.cpu().numpy(), E.cpu().numpy()
        panels = [("evidence voxels (teacher readout)", bev(vol(np.where(E_np, ev_lab.cpu().numpy(), 0)))),
                  ("completed occupancy (frozen)", bev(vol(occ_np.astype(np.int64) * 11))),
                  ("B1 propagated (on predicted occupied)", bev(vol(np.where(occ_np, b1.cpu().numpy(), 0)))),
                  ("B0 nearest evidence (on predicted occupied)", bev(vol(np.where(occ_np, b0.cpu().numpy(), 0)))),
                  ("GT semantics (evaluation only)", bev(vol(np.where((gt >= 1) & (gt <= 18), gt, 0))))]
        fig = plt.figure(figsize=(16, 7)); gs = fig.add_gridspec(2, 5, height_ratios=[1, 2.3])
        a = fig.add_subplot(gs[0, :3]); a.imshow(plt.imread(img_path), cmap="gray"); a.set_title(f"0006 anchor {t}: RGB frame t"); a.axis("off")
        a = fig.add_subplot(gs[0, 3:]); a.imshow(PALETTE[teacher2d]); a.set_title("Trident-H readout on the LingBot lattice (frame t)"); a.axis("off")
        for i, (title, img) in enumerate(panels):
            a = fig.add_subplot(gs[1, i]); a.imshow(PALETTE[np.rot90(img)], interpolation="nearest"); a.set_title(title, fontsize=8); a.axis("off")
        fig.suptitle(f"example '{name}' (deterministic selection rule, seed 0)"); fig.tight_layout()
        p = os.path.join(FIG, f"10_example_{name}_{t:05d}.png"); fig.savefig(p, dpi=110); plt.close(fig); out.append(p)
    return out


def write_figures(device: str) -> Dict:
    os.makedirs(FIG, exist_ok=True)
    from tools.semantic_completion.analysis import load_analysis
    res = load_analysis()
    plots = summary_plots(res)
    examples = choose_examples(load_shards())
    panels = example_panels(examples, device)
    json.dump({"examples": examples, "files": [os.path.relpath(p, C.REPO_ROOT) for p in plots + panels]}, open(os.path.join(FIG, "index.json"), "w"), indent=1)
    return {"examples": examples, "plots": plots, "panels": panels}


__all__ = ["PALETTE", "bev", "summary_plots", "choose_examples", "example_panels", "write_figures"]
