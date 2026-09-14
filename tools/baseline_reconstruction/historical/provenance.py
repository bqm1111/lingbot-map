"""Provenance of the historical post-ceiling-fix baseline: commits, hashes, configs, protocol.

The canonical historical baseline is defined here once, in code, with the reasons:

    the adopted sky-fix model (``frozen_manifest_sky.json``: replicate padding + open-sky-is-FREE
    targets, three seeds retrained, thresholds reselected on KITTI-360 0006), evaluated by the
    gate's own evaluator ``core/run/eval_target.py`` under Protocol A ``past5`` on the complete
    official anchor sets (163 SemanticKITTI / 1182 Occ3D), median of three seeds.

It is canonical because (a) it is the only post-fix evaluation on the full official anchor
sets with the gate's frozen masks and metric; (b) it is what ``reports/METHOD.md`` §8.1
adopts as the headline; (c) the alternatives are subsets scored with OccAny's code on
OccAny's label volumes, or the rejected ``sky2`` variant. Magnitude played no part.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import yaml

from tools.baseline_reconstruction import config as C
from tools.baseline_reconstruction.historical import records as R
from tools.baseline_reconstruction.metrics.occupancy import seed_statistics


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", C.REPO_ROOT, *args], text=True).strip()


def commit_line(ref: str) -> str:
    return git("log", "-1", "--date=iso", "--format=%h %ad %s", ref)


@dataclass(frozen=True)
class FixTimeline:
    """The commits/records that bracket the ceiling fix. Dates are from git and file mtimes."""
    prefix_code_commit: str
    prefix_manifest_frozen_at: str
    fix_applied: str
    fix_first_tracked_commit: str
    sky_rule_measured_return_fix_commit: str
    docs_commit: str
    sky_manifest_frozen_at: str
    first_valid_eval_semantickitti: str
    first_valid_eval_occ3d: str
    superseding_baseline: str


def fix_timeline(paths: C.HistoricalPaths, results: List[R.HistoricalResult]) -> FixTimeline:
    sk = R.select(results, "sky", "semantickitti", C.HISTORICAL_PROTOCOL)
    oc = R.select(results, "sky", "occ3d", C.HISTORICAL_PROTOCOL)
    pre = json.load(open(paths.prefix_manifest)); sky = json.load(open(paths.sky_manifest))
    return FixTimeline(
        prefix_code_commit=commit_line(paths.commit_prefix_code),
        prefix_manifest_frozen_at=pre["frozen_at"],
        fix_applied="2026-09-04 (untracked working tree): configs/gate8c1_sky/*.yaml padding_mode=replicate; "
                    "rawtarget.py open-sky-is-FREE rule; targets rebuilt at /media/SSD1/MINH_DATASETS/lingbot_gate8c1_sky; "
                    "sky_seed{0,1,2}_last.pt trained 19:05-19:15 (file mtimes)",
        fix_first_tracked_commit=commit_line(paths.commit_first_tracked_tree) + "  [tree tracked with SKY_NEIGHBOURHOOD_VOX=7, i.e. the sky2 form]",
        sky_rule_measured_return_fix_commit=commit_line(paths.commit_sky_rule_measured_fix),
        docs_commit=commit_line(paths.commit_docs_update) + "  [documentation + whitespace only; no pipeline change]",
        sky_manifest_frozen_at=sky["frozen_at"],
        first_valid_eval_semantickitti=min(r.source_mtime for r in sk) if sk else "missing",
        first_valid_eval_occ3d=min(r.source_mtime for r in oc) if oc else "missing",
        superseding_baseline="none: Gates 9-14 (2026-09-12..14) were source-side diagnostics and produced no new zero-shot evaluation")


@dataclass(frozen=True)
class HistoricalPipeline:
    """What produced the canonical numbers, read from the checkpoint meta, configs and core/README.md."""
    lingbot_mode: str = ("long-stream causal replay (gates/gate7b/replay.py): one persistent KV cache per stream, "
                         "reset only at a stream boundary; sliding 64-block KV window; per-frame depth/pose cached to "
                         "/media/SSD1/MINH_DATASETS/lingbot_gate7b/stream")
    depth_source: str = "LingBot-Map canonical depth x metric scale (LB-M); MoGe-2 used only as the scale gauge"
    scale_handling: str = ("per stream: exp(median(log_s of the FIRST five stream frames)) from MoGe-2 vs LingBot candidates "
                           "(core/mapping/scale.py); the same frozen stream scale is forced on every past5 window map")
    pose_source: str = "LingBot-Map streaming pose (long-stream), translation scaled by the stream scale"
    mapper: str = ("core/mapping/mapper.py IncrementalMapper: conf >= 1.5, depth 1-60 m, +-0.2 m occupied band, "
                   "free carve to d-0.2 m at stride 4, log-odds accumulation, semantic accumulators on")
    voxel_representation: str = "log-odds occupancy + w_free + n_obs + age + semantic evidence, exported to the benchmark grid"
    input_channels: int = 32
    channel_meaning: str = ("0 logodds/4, 1 w_free/20, 2 observed, 3 1-observed, 4 n_obs/50, 5 age/100, 6 sem_w/20, "
                            "7..31 Trident-H union-vocabulary class probabilities (core/README.md)")
    semantics: str = "present: Trident-H teacher probabilities as input; semantic KL loss w_sem = 0.5; semantic head"
    residual_lock: str = "apply_residual: final = base + residual * [|base| < 2.0] (LOCK_LOGODDS = 2)"
    completion_architecture: str = "core/model/net.py CompletionUNet, width 24, 32-ch input, occupancy residual + semantic logits"
    n_params: int = 986114
    input_shape: str = "[1, 32, X, Y, Z] on the benchmark prediction grid"
    output_shape: str = "[1, 1, X, Y, Z] occupancy residual (+ [1, U, X, Y, Z] semantic logits)"
    padding_mode: str = "replicate (the padding half of the ceiling fix)"
    training_source: str = "KITTI-360 drives 0003/0007/0010 (train), 0006 (validation/selection)"
    supervision: str = ("raw-Velodyne targets t..t+20 with GT poses, open sky FREE (plain column rule, SKY_MARGIN_VOX 1, "
                        "no neighbourhood), targets at /media/SSD1/MINH_DATASETS/lingbot_gate8c1_sky")
    training_steps: int = 6000
    losses: str = "focal (gamma 2, pos-weight cap 8) + Dice, w_unknown 2, + 0.5 semantic KL; AdamW 1e-3 wd 0.01; cosine; clip 1; batch 4; crop 128x128x32"
    checkpoint_selection: str = "manifest rule: highest full-grid occupancy AP on 0006 -> 'last' (step 6000) for all three seeds"
    threshold_selection: str = "per seed, final-log-odds threshold maximising full-grid IoU on 0006: -0.125 / -0.4375 / -0.3125"
    evaluator: str = "core/run/eval_target.py (G8C1_MANIFEST=artifacts/gate8c1/frozen_manifest_sky.json), mode past5"
    zero_shot_window: str = "fresh map over stream indices i-4..i, frozen stream scale, LingBot stream depth/pose"


@dataclass
class CanonicalResult:
    dataset: str
    metric: str
    per_seed_pct: Dict[int, float]
    median_pct: float
    mean_pct: float
    sd_pct: float
    precision_pct_median_seed: float
    recall_pct_median_seed: float
    ssc_miou_pct_median_seed: float
    n_anchors: int
    thresholds: Dict[int, float]
    checkpoints: Dict[int, str]
    checkpoint_sha256: Dict[int, str]
    source_files: List[str]
    protocol: str = C.HISTORICAL_PROTOCOL


def canonical_result(results: List[R.HistoricalResult], dataset: str, paths: C.HistoricalPaths) -> CanonicalResult:
    rows = R.select(results, "sky", dataset, C.HISTORICAL_PROTOCOL)
    assert len(rows) == 3, f"expected three sky {dataset} past5 records, found {len(rows)}"
    seeds = {s.seed: s for s in R.read_manifest_seeds(paths.sky_manifest)}
    ious = {r.seed: 100 * r.sc_iou for r in rows}
    st = seed_statistics(ious.values())
    med = min(rows, key=lambda r: abs(100 * r.sc_iou - st["median"]))
    return CanonicalResult(dataset=dataset, metric="SC IoU (binary occupancy, official mask)", per_seed_pct=ious,
                           median_pct=st["median"], mean_pct=st["mean"], sd_pct=st["sd"],
                           precision_pct_median_seed=100 * med.precision, recall_pct_median_seed=100 * med.recall,
                           ssc_miou_pct_median_seed=100 * med.ssc_miou, n_anchors=rows[0].n_clips,
                           thresholds={r.seed: r.threshold for r in rows}, checkpoints={r.seed: r.checkpoint for r in rows},
                           checkpoint_sha256={s: seeds[s].checkpoint_sha256_recorded for s in ious},
                           source_files=[r.source_file for r in rows])


def provenance_rows(results: List[R.HistoricalResult], paths: C.HistoricalPaths, ceiling_verified: bool) -> List[Dict[str, str]]:
    """The provenance table of the brief, one column per target dataset."""
    out = []
    for ds in C.HISTORICAL_TARGETS:
        c = canonical_result(results, ds, paths)
        out.append({
            "dataset": ds,
            "verified_occupancy_iou_pct": f"{c.median_pct:.2f} (median of {', '.join(f'{c.per_seed_pct[s]:.2f}' for s in sorted(c.per_seed_pct))}; mean {c.mean_pct:.2f})",
            "semantic_miou_pct": f"{c.ssc_miou_pct_median_seed:.2f} (SSC mIoU, median seed; the model carried Trident-H semantics)",
            "artifact_report": "; ".join(c.source_files) + "; reports/METHOD.md §8.1; artifacts/gate8c1/ceiling_fix.md",
            "config": "configs/gate8c1_sky/seed{0,1,2}.yaml",
            "checkpoint": "; ".join(f"seed {s}: {c.checkpoints[s]} sha256 {c.checkpoint_sha256[s][:12]}" for s in sorted(c.checkpoints)),
            "git_commit": f"code first tracked at {paths.commit_first_tracked_tree} (2026-09-07); fix applied 2026-09-04 in an untracked tree; docs at {paths.commit_docs_update}",
            "evaluation_script": f"{paths.evaluator} --dataset {ds} --mode past5 (G8C1_MANIFEST=frozen_manifest_sky.json)",
            "ceiling_fix_active": "yes, verified" if ceiling_verified else "NOT VERIFIED",
            "exact_protocol": f"past5: fresh map over stream indices i-4..i; per-seed thresholds {c.thresholds}; {c.n_anchors} anchors; {C.TARGET_PROTOCOLS[ds].evaluation_mask}",
            "date": ", ".join(sorted({r.source_mtime[:10] for r in R.select(results, 'sky', ds, C.HISTORICAL_PROTOCOL)})),
            "canonical_baseline": "yes",
        })
    return out


def historical_protocol_yaml() -> Dict:
    hp = HistoricalPipeline()
    return {"name": "Gate 8C-1 sky-fix baseline (post ceiling fix)", "protocol_of_record": C.HISTORICAL_PROTOCOL,
            "seed_statistic_of_record": "median of 3 seeds (reports/METHOD.md §8.1); mean also reported here",
            "pipeline": asdict(hp), "targets": {ds: asdict(C.TARGET_PROTOCOLS[ds]) for ds in C.HISTORICAL_TARGETS}}


__all__ = ["git", "commit_line", "FixTimeline", "fix_timeline", "HistoricalPipeline", "CanonicalResult",
           "canonical_result", "provenance_rows", "historical_protocol_yaml"]
