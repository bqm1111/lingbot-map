"""Prove that the recovered historical result is the *fixed* pipeline, not the buggy one.

The ceiling bug had two parts (``artifacts/gate8c1/ceiling_fix.md``): zero padding at the
array boundary, and open sky left UNKNOWN (so unsupervised) in the training targets. A
result "after the fix" must therefore satisfy, on evidence rather than by date:

1. the evaluated checkpoints carry ``padding_mode == "replicate"`` and their hashes match
   the frozen manifest, while the pre-fix checkpoints carry zeros padding;
2. the eval records name exactly those checkpoints and thresholds;
3. the targets that trained them supervise the sky (top layer valid ~22 % vs ~1 % before);
4. those targets are reproduced bit-exactly by the sky rule -- and by *which* variant of it;
5. the model's predictions no longer fill the ceiling (top layer predicted-occupied ~1 %
   against 95 % before), and the stored per-anchor dump re-scores to the recorded IoU.

Checks 1-3 and 5 read only files already on disk. Check 4 rebuilds a few KITTI-360 targets
from raw Velodyne (source data; the targets' own inputs) on a GPU. None of them opens a
target dataset; every check runs inside a strict file-access audit.
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from core.datasets import kitti360_splits as SRC
from core.datasets.audit import FileAudit
from tools.baseline_reconstruction import config as C
from tools.baseline_reconstruction.historical import records as R

#: dataset *directories* (record files legitimately carry dataset names)
TARGET_DATASET_TOKENS = tuple(SRC.FORBIDDEN_PATH_TOKENS)
GRID = C.GRID_DIMS
#: voxel z index of the "+4.3 m" top layer and of the "+2 m" line on the KITTI-360 grid (origin z = -2 m)
TOP_LAYER_Z = GRID[2] - 1
ABOVE_2M_Z = int((2.0 - (-2.0)) / C.VOXEL_SIZE_M)


@dataclass
class Check:
    name: str
    passed: bool
    detail: Dict


def _unpack(packed: np.ndarray, n: int) -> np.ndarray:
    return np.unpackbits(packed)[:n].astype(bool)


# --------------------------------------------------------------------------- #
# 1 + 2: checkpoints and eval records
# --------------------------------------------------------------------------- #
def check_checkpoints(paths: C.HistoricalPaths) -> Check:
    sky = R.read_manifest_seeds(paths.sky_manifest)
    pre = R.read_manifest_seeds(paths.prefix_manifest)
    ok = all(s.padding_mode_in_checkpoint == "replicate" and s.checkpoint_sha256_on_disk == s.checkpoint_sha256_recorded
             and s.input_channels_in_checkpoint == 32 for s in sky)
    ok &= all(s.padding_mode_in_checkpoint == "zeros" and s.checkpoint_sha256_on_disk == s.checkpoint_sha256_recorded for s in pre)
    return Check("sky checkpoints: replicate padding, 32 input channels, sha256 == manifest; pre-fix: zeros padding", ok,
                 {"sky": [asdict(s) for s in sky], "prefix": [asdict(s) for s in pre]})


def check_eval_records(paths: C.HistoricalPaths) -> Check:
    sky_seeds = {s.seed: s for s in R.read_manifest_seeds(paths.sky_manifest)}
    results = [r for r in R.load_variant_results(paths) if r.variant == "sky"]
    logs = R.parse_result_logs(paths.sky_eval_logs)
    detail, ok = [], len(results) == 6
    for r in results:
        m = sky_seeds[r.seed]
        same = r.checkpoint == m.checkpoint and abs(r.threshold - m.occupancy_threshold) < 1e-12
        log_iou = logs.get(r.dataset, {}).get(r.seed)
        log_ok = log_iou is not None and abs(log_iou - round(100 * r.sc_iou, 2)) < 1e-9
        ok &= same and log_ok and r.protocol == C.HISTORICAL_PROTOCOL
        detail.append({"dataset": r.dataset, "seed": r.seed, "json_checkpoint": r.checkpoint, "manifest_checkpoint": m.checkpoint,
                       "json_threshold": r.threshold, "manifest_threshold": m.occupancy_threshold,
                       "json_sc_iou_pct": round(100 * r.sc_iou, 4), "log_sc_iou_pct": log_iou, "n_clips": r.n_clips})
    return Check("sky eval JSONs name the manifest checkpoints/thresholds; RESULT log lines agree; protocol past5", ok, {"rows": detail})


# --------------------------------------------------------------------------- #
# 3: the sky targets supervise the sky
# --------------------------------------------------------------------------- #
def top_layer_valid_fraction(target_root: str, drive: str, n_anchors: int) -> Dict[str, float]:
    files = sorted(glob.glob(os.path.join(target_root, drive, "*.npz")))[:n_anchors]
    n = int(np.prod(GRID)); top = above = 0.0
    for f in files:
        z = np.load(f)
        v = _unpack(z["valid_packed"], n).reshape(GRID)
        top += v[:, :, TOP_LAYER_Z].mean(); above += v[:, :, ABOVE_2M_Z:].mean()
    return {"n_anchors": len(files), "top_layer_valid_pct": 100 * top / len(files), "above_2m_valid_pct": 100 * above / len(files)}


def check_sky_supervision(paths: C.HistoricalPaths, n_anchors: int = 40) -> Check:
    pre = top_layer_valid_fraction(paths.targets_prefix, C.VAL_DRIVE, n_anchors)
    sky = top_layer_valid_fraction(paths.targets_sky, C.VAL_DRIVE, n_anchors)
    cur = top_layer_valid_fraction(paths.targets_current, C.VAL_DRIVE, n_anchors)
    # ceiling_fix.md reports 1.12 % -> 21.9 % on its own sample; the check asks for the same order of change
    ratio = sky["top_layer_valid_pct"] / max(pre["top_layer_valid_pct"], 1e-9)
    ok = ratio >= 4.0 and sky["top_layer_valid_pct"] - pre["top_layer_valid_pct"] >= 15.0
    return Check("sky targets supervise the top layer (>= 4x and >= +15 pp over the pre-fix fraction)", ok,
                 {"prefix": pre, "sky": sky, "current_gate12_13_builder": cur, "ratio_sky_over_prefix": ratio})


# --------------------------------------------------------------------------- #
# 4: which rule variant produced which target root (bit-exact rebuild)
# --------------------------------------------------------------------------- #
def _module_from_git(commit: str, repo_path: str, name: str):
    """Load a file as it was at ``commit`` as an importable module (dataclasses need sys.modules)."""
    src = subprocess.check_output(["git", "-C", C.REPO_ROOT, "show", f"{commit}:{repo_path}"], text=True)
    tmp = os.path.join(C.ARTIFACT_DIR, "logs", f"{name}.py")
    with open(tmp, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(name, tmp)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


def _cached_target(root: str, drive: str, stream_index: int):
    z = np.load(os.path.join(root, drive, f"{stream_index:05d}.npz")); n = int(np.prod(z["dims"]))
    return _unpack(z["valid_packed"], n), _unpack(z["occ_packed"], n)


def check_rule_reproduction(paths: C.HistoricalPaths, device: str, stream_indices: Sequence[int] = (4, 301, 1000)) -> Check:
    """Rebuild anchors of 0006 under three rule variants and compare with the cached roots.

    * ``pre647`` = ``core/supervision/rawtarget.py`` at the first tracked commit (before 647bcf3);
      with ``SKY_NEIGHBOURHOOD_VOX = 0`` it is the plain column rule of the adopted model,
      with 7 it is the rejected ``sky2`` rule;
    * ``current`` = the file on disk today (647bcf3 fix, neighbourhood 7), the Gate 12/13 builder.
    """
    import torch
    from core.datasets import kitti360_transforms as TF
    from core.supervision import rawtarget as RT_now
    RT_pre = _module_from_git(paths.commit_first_tracked_tree, "core/supervision/rawtarget.py", "rawtarget_pre647")
    dev = torch.device(device)
    geo = TF.DriveGeometry(C.VAL_DRIVE, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
    anchors = {a.stream_index: a for a in SRC.anchors(C.VAL_DRIVE, C.REPO_ROOT)}
    expected = {"sky": ("pre647", 0), "sky2": ("pre647", 7), "current": ("current", 7)}
    roots = {"sky": paths.targets_sky, "sky2": paths.targets_sky2, "current": paths.targets_current}
    rows, ok = [], True
    for si in stream_indices:
        anc = anchors[si]; future = [anc.native_frame] + list(anc.future_natives)
        built = {}
        for label, mod in (("pre647", RT_pre), ("current", RT_now)):
            for nb in (0, 7):
                mod.SKY_NEIGHBOURHOOD_VOX = nb
                t = mod.build(geo, anc.native_frame, future, dev)
                built[(label, nb)] = (np.asarray(t.valid, bool).reshape(-1), np.asarray(t.occupied, bool).reshape(-1))
        RT_now.SKY_NEIGHBOURHOOD_VOX = 7      # restore the module constant
        for name, root in roots.items():
            cv, co = _cached_target(root, C.VAL_DRIVE, si)
            matches = [k for k, (bv, bo) in built.items() if np.array_equal(cv, bv) and np.array_equal(co, bo)]
            exp = expected[name]
            ok &= matches == [exp]
            rows.append({"stream_index": si, "native_frame": anc.native_frame, "cached_root": name, "bit_exact_variants": [f"{k[0]}_nb{k[1]}" for k in matches],
                         "expected": f"{exp[0]}_nb{exp[1]}", "occupied_identical_all_variants": all(np.array_equal(co, bo) for (_, bo) in built.values())})
    return Check("cached target roots are reproduced bit-exactly by exactly one rule variant each "
                 "(sky = pre-647bcf3 plain column rule; sky2 = pre-647bcf3 neighbourhood 7; current = today's file)", ok, {"rows": rows})


# --------------------------------------------------------------------------- #
# 5: the ceiling in the stored predictions, and the IoU re-derived from the dump
# --------------------------------------------------------------------------- #
def check_prediction_dump(paths: C.HistoricalPaths, recorded_sc_iou: float) -> Check:
    files = sorted(glob.glob(os.path.join(paths.sky_pred_dump, "*.npz")))
    n = int(np.prod(GRID)); tp = fp = fn = 0; top_pred = top_gt = above_pred = above_gt = 0.0; n_top = n_above = 0
    for f in files:
        z = np.load(f)
        pred = _unpack(z["pred_occ"], n).reshape(GRID); keep = _unpack(z["keep"], n).reshape(GRID)
        gt = (z["gt_label"] != 0) & keep
        p = pred & keep
        tp += int((p & gt).sum()); fp += int((p & ~gt).sum()); fn += int((~p & gt).sum())
        kt = keep[:, :, TOP_LAYER_Z]; top_pred += p[:, :, TOP_LAYER_Z][kt].sum(); top_gt += gt[:, :, TOP_LAYER_Z][kt].sum(); n_top += kt.sum()
        ka = keep[:, :, ABOVE_2M_Z:]; above_pred += p[:, :, ABOVE_2M_Z:][ka].sum(); above_gt += gt[:, :, ABOVE_2M_Z:][ka].sum(); n_above += ka.sum()
    iou = tp / max(tp + fp + fn, 1)
    top_pct = 100 * top_pred / max(n_top, 1)
    ok = len(files) == 163 and abs(iou - recorded_sc_iou) < 2e-4 and top_pct < 2.0
    return Check("seed-2 prediction dump re-scores to the recorded SC IoU and the top layer is no longer filled (< 2 % vs 95.1 % pre-fix)", ok,
                 {"n_anchors": len(files), "sc_iou_from_dump": iou, "recorded_sc_iou": recorded_sc_iou, "tp": tp, "fp": fp, "fn": fn,
                  "top_layer_pred_occupied_pct": top_pct, "top_layer_gt_occupied_pct": 100 * top_gt / max(n_top, 1),
                  "above_2m_pred_occupied_pct": 100 * above_pred / max(n_above, 1), "above_2m_gt_occupied_pct": 100 * above_gt / max(n_above, 1),
                  "prefix_reference": {"top_layer_pred_occupied_pct": 95.1, "source": "artifacts/gate8c1/ceiling_fix.md"}})


# --------------------------------------------------------------------------- #
def run_all(device: Optional[str], paths: C.HistoricalPaths = C.HistoricalPaths()) -> List[Check]:
    results = R.load_variant_results(paths)
    seed2 = R.select(results, "sky", "semantickitti", C.HISTORICAL_PROTOCOL)[2]
    checks: List[Check] = []
    with FileAudit(patterns=TARGET_DATASET_TOKENS, strict=True) as audit:
        checks.append(check_checkpoints(paths))
        checks.append(check_eval_records(paths))
        checks.append(check_sky_supervision(paths))
        checks.append(check_prediction_dump(paths, seed2.sc_iou))
        if device:
            checks.append(check_rule_reproduction(paths, device))
    checks.append(Check("no target-dataset path was opened during the audit", not audit.violations,
                        {"files_opened": len(audit.opened), "violations": audit.violations, "forbidden_tokens": list(TARGET_DATASET_TOKENS)}))
    return checks


def write(checks: List[Check], path: str) -> None:
    json.dump([asdict(c) for c in checks], open(path, "w"), indent=1, default=float)


__all__ = ["Check", "run_all", "write", "check_checkpoints", "check_eval_records", "check_sky_supervision",
           "check_rule_reproduction", "check_prediction_dump", "TOP_LAYER_Z", "ABOVE_2M_Z"]
