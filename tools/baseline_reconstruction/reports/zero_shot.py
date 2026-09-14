"""Render zero_shot_results.md, tables/zero_shot.csv and the two evaluation_<dataset>.yaml protocol records."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np
import yaml

from tools.baseline_reconstruction import config as C
from tools.baseline_reconstruction.reports.tables import markdown_table, pct, write_csv
from tools.baseline_reconstruction.targets.evaluate import REGIONS


def load_zero_shot(paths: C.CurrentPaths) -> Dict[str, Dict]:
    out = {}
    for ds in C.HISTORICAL_TARGETS:
        p = os.path.join(paths.target_eval, f"zero_shot_{ds}.json")
        if os.path.exists(p):
            out[ds] = json.load(open(p))
    return out


def bootstrap_ci(paths: C.CurrentPaths, dataset: str, n_boot: int = 2000, seed: int = 0) -> Optional[Dict[str, float]]:
    """Clip-block bootstrap (blocks of 20 anchors in stream order) of the mean-over-seeds full-region IoU."""
    p = os.path.join(paths.target_eval, f"per_anchor_{dataset}.json")
    if not os.path.exists(p):
        return None
    rows = json.load(open(p)); rng = np.random.default_rng(seed)
    cnt = np.array([[[r[f"seed{s}"][k] for k in ("tp", "fp", "fn")] for s in C.SEEDS] for r in rows], np.int64)   # [n, seeds, 3]
    blocks = [cnt[i:i + 20] for i in range(0, len(cnt), 20)]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(blocks), len(blocks)); c = np.concatenate([blocks[b] for b in pick]).sum(0)
        vals.append(np.mean(c[:, 0] / np.maximum(c.sum(1), 1)))
    return {"lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5)), "n_blocks": len(blocks)}


def write_protocol_yamls(zs: Dict[str, Dict], out_dir: str) -> None:
    for ds, d in zs.items():
        rec = {"experiment": C.PROTOCOL_VERSION, "dataset": ds, "protocol": d["protocol"], "clip_offsets": list(C.CLIP_OFFSETS),
               "input_frame_construction": "local-reset LingBot on the five target images, clip-local indices 0..4, KV cache cleared; "
                                           "causal MoGe-2 clip scale from the frozen per-frame MoGe-B caches; anchor camera = identity",
               "camera_preprocessing": "gates/gate7b/replay.load_images: mode='crop', 518 px, patch 14 (identical to the stream and to KITTI-360 clips)",
               "coordinate_transform": "grid -> world(anchor camera) = inv(T_cam_to_grid); " + ("cam_to_velo from calib.txt" if ds == "semantickitti" else "T_camera_to_ego of the anchor sample"),
               "occupancy_threshold": C.OCCUPANCY_THRESHOLD, "checkpoint_rule": "final step 6000, three seeds, mean primary / median secondary",
               "checkpoints": d["checkpoints"], "checkpoint_sha256": d["checkpoint_sha256"], "freeze_frozen_at": d["freeze_frozen_at"],
               "n_anchors_evaluated": d["n_anchors"], "n_skipped_no_label": d["n_skipped_no_label"], "n_skipped_no_scale": d["n_skipped_no_scale"],
               "semantic_miou": "unavailable: geometry-only model (stated, not worked around)",
               "camera_fov_grid_note": ("SemanticKITTI shares the KITTI-360 grid dimensions and voxel size; the camera (image_2, 1226x370) and the stride-5 stream differ from training" if ds == "semantickitti"
                                        else "Occ3D differs from training in grid (400x400x32 canonical, ego-centred, 0.4 m native), camera (CAM_FRONT 1600x900), multi-camera handled by the frozen single-camera x<100 cut, and 2 Hz keyframes"),
               "deterministic_remapping": "none" if ds == "semantickitti" else "any-subvoxel 2x reduction to the native 0.4 m grid (frozen since Gate 4)"}
        yaml.safe_dump(rec, open(os.path.join(out_dir, f"evaluation_{ds}.yaml"), "w"), sort_keys=False, width=120)


def write_figure(zs: Dict[str, Dict], historical: Optional[Dict[str, float]]) -> Optional[str]:
    """IoU by range per dataset (mean over seeds) with the map-only reference and the historical headline."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(zs), figsize=(5.2 * len(zs), 3.6), squeeze=False)
    for ax, (ds, d) in zip(axes[0], zs.items()):
        x = np.arange(len(C.RANGE_NAMES))
        comp = [100 * np.mean([d["per_seed"][str(s)]["full"]["by_range"][n]["iou"] for s in C.SEEDS]) for n in C.RANGE_NAMES]
        mo = [100 * d["map_only"]["full"]["by_range"][n]["iou"] for n in C.RANGE_NAMES]
        ax.bar(x - 0.2, comp, 0.4, label="completion (mean of 3 seeds)"); ax.bar(x + 0.2, mo, 0.4, label="map only (S)")
        ax.axhline(100 * d["seed_statistics"]["iou"]["mean"], color="C0", ls=":", lw=1, label="completion, all ranges")
        if historical and ds in historical:
            ax.axhline(historical[ds], color="k", ls="--", lw=1, label="historical median, all ranges")
        ax.set_xticks(x); ax.set_xticklabels(C.RANGE_NAMES); ax.set_ylabel("SC IoU %"); ax.set_title(ds); ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=7)
    fig.tight_layout(); out = os.path.join(C.ARTIFACT_DIR, "figures", "zero_shot_iou_by_range.png"); os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def write_zero_shot_results(paths: C.CurrentPaths = C.CurrentPaths(), historical: Optional[Dict[str, float]] = None) -> Dict:
    zs = load_zero_shot(paths)
    assert zs, "no zero-shot evaluations found"
    write_protocol_yamls(zs, C.ARTIFACT_DIR)
    cis = {ds: bootstrap_ci(paths, ds) for ds in zs}
    fig = write_figure(zs, historical)
    rows = []
    for ds, d in zs.items():
        for s in C.SEEDS:
            f = d["per_seed"][str(s)]["full"]
            rows.append({"dataset": ds, "seed": s, "region": "full", "sc_iou": f["iou"], "precision": f["precision"], "recall": f["recall"],
                         "tp": f["tp"], "fp": f["fp"], "fn": f["fn"], "n_anchors": d["n_anchors"], "threshold": d["threshold"], "checkpoint_sha256": d["checkpoint_sha256"][str(s)]})
            for r in REGIONS[1:]:
                x = d["per_seed"][str(s)][r]
                rows.append({"dataset": ds, "seed": s, "region": r, "sc_iou": x["iou"], "precision": x["precision"], "recall": x["recall"], "tp": x["tp"], "fp": x["fp"], "fn": x["fn"],
                             "n_anchors": d["n_anchors"], "threshold": d["threshold"], "checkpoint_sha256": ""})
        st = d["seed_statistics"]; m = d["map_only"]["full"]
        rows.append({"dataset": ds, "seed": "mean", "region": "full", "sc_iou": st["iou"]["mean"], "precision": st["precision"]["mean"], "recall": st["recall"]["mean"], "n_anchors": d["n_anchors"], "threshold": d["threshold"]})
        rows.append({"dataset": ds, "seed": "median", "region": "full", "sc_iou": st["iou"]["median"], "precision": st["precision"]["median"], "recall": st["recall"]["median"], "n_anchors": d["n_anchors"], "threshold": d["threshold"]})
        rows.append({"dataset": ds, "seed": "map_only", "region": "full", "sc_iou": m["iou"], "precision": m["precision"], "recall": m["recall"], "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "n_anchors": d["n_anchors"]})
    write_csv(os.path.join(C.ARTIFACT_DIR, "tables", "zero_shot.csv"), rows,
              ["dataset", "seed", "region", "sc_iou", "precision", "recall", "tp", "fp", "fn", "n_anchors", "threshold", "checkpoint_sha256"])
    md = [f"# Zero-shot results -- {C.CURRENT_BASELINE_NAME}\n",
          "First frozen evaluation, run once per dataset after the source freeze (`source_freeze.md`); nothing was changed afterwards. "
          "Protocol records: `evaluation_semantickitti.yaml`, `evaluation_occ3d.yaml`. Semantic mIoU is unavailable (geometry-only model).\n",
          "## Headline (full official mask)\n",
          markdown_table(["dataset", "anchors", "SC IoU % seeds 0/1/2", "**mean**", "median", "sd", "95 % CI of mean (clip-block bootstrap)", "P % mean", "R % mean", "map-only IoU %", "historical (median)"],
                         [[ds, d["n_anchors"], " / ".join(pct(d["per_seed"][str(s)]["full"]["iou"]) for s in C.SEEDS), f"**{pct(d['seed_statistics']['iou']['mean'])}**",
                           pct(d["seed_statistics"]["iou"]["median"]), pct(d["seed_statistics"]["iou"]["sd"]),
                           f"[{pct(cis[ds]['lo'])}, {pct(cis[ds]['hi'])}]" if cis[ds] else "", pct(d["seed_statistics"]["precision"]["mean"]), pct(d["seed_statistics"]["recall"]["mean"]),
                           pct(d["map_only"]["full"]["iou"]), f"{historical[ds]:.2f}" if historical and ds in historical else ""] for ds, d in zs.items()]),
          "\n## By region (mean over seeds)\n",
          markdown_table(["dataset", "region", "IoU %", "P %", "R %", "GT occupied voxels"],
                         [[ds, r, pct(np.mean([d["per_seed"][str(s)][r]["iou"] for s in C.SEEDS])), pct(np.mean([d["per_seed"][str(s)][r]["precision"] for s in C.SEEDS])),
                           pct(np.mean([d["per_seed"][str(s)][r]["recall"] for s in C.SEEDS])), f"{d['per_seed']['0'][r]['n_gt']:,}"] for ds, d in zs.items() for r in REGIONS]),
          "\n`H_evidence` = evaluation voxels where the causal map carries S or F evidence; `no_H_evidence` = the rest (pure completion).\n",
          "## By range (anchor-camera z, mean over seeds, full region)\n",
          markdown_table(["dataset", "range", "IoU %", "P %", "R %", "map-only IoU %", "GT occupied voxels"],
                         [[ds, name, pct(np.mean([d["per_seed"][str(s)]["full"]["by_range"][name]["iou"] for s in C.SEEDS])),
                           pct(np.mean([d["per_seed"][str(s)]["full"]["by_range"][name]["precision"] for s in C.SEEDS])),
                           pct(np.mean([d["per_seed"][str(s)]["full"]["by_range"][name]["recall"] for s in C.SEEDS])),
                           pct(d["map_only"]["full"]["by_range"][name]["iou"]), f"{d['per_seed']['0']['full']['by_range'][name]['n_gt']:,}"]
                          for ds, d in zs.items() for name in C.RANGE_NAMES]),
          "\nSkipped anchors: " + "; ".join(f"{ds}: {d['n_skipped_no_label']} without label, {d['n_skipped_no_scale']} without a causal scale" for ds, d in zs.items()) + ".\n",
          f"\nFigure: `{os.path.relpath(fig, C.REPO_ROOT)}`.\n"]
    out = os.path.join(C.ARTIFACT_DIR, "zero_shot_results.md"); open(out, "w").write("\n".join(md))
    json.dump({"bootstrap_ci": cis}, open(os.path.join(paths.target_eval, "zero_shot_ci.json"), "w"), indent=1)
    return {"path": out, "zero_shot": zs, "ci": cis}


__all__ = ["load_zero_shot", "bootstrap_ci", "write_zero_shot_results"]
