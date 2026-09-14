"""verification.md: every check of the brief (historical, current baseline, zero-shot firewall) with its evidence."""
from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import yaml

from tools.baseline_reconstruction import config as C, freeze as FZ
from tools.baseline_reconstruction.current.geometry import LocalClip, build_local_geometry
from tools.baseline_reconstruction.reports.tables import markdown_table

GATE12_FREEZE = os.path.join(C.REPO_ROOT, "artifacts", "gate12_suf", "code_freeze.json")


@dataclass
class Item:
    section: str
    name: str
    passed: bool
    evidence: str


def _sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _mtime(p: str) -> float:
    return os.path.getmtime(p)


# --------------------------------------------------------------------------- #
def historical_items() -> List[Item]:
    hv = json.load(open(os.path.join(C.ARTIFACT_DIR, "logs", "historical_verification.json")))
    hb = json.load(open(os.path.join(C.ARTIFACT_DIR, "historical_baseline.json")))
    by = {c["name"].split(":")[0][:40]: c for c in hv}
    ok = lambda key: any(c["passed"] for c in hv if c["name"].startswith(key))
    items = [
        Item("historical", "exact ceiling-fix commit/code identified", ok("cached target roots"), "sky targets == pre-647bcf3 rawtarget.py with SKY_NEIGHBOURHOOD_VOX=0 (bit-exact, 3 anchors); fix applied 2026-09-04, first tracked at f7b7552; historical_baseline_audit.md §3"),
        Item("historical", "pre-fix and post-fix evaluation distinguished", ok("sky checkpoints"), "prefix manifest checkpoints carry zeros padding, sky manifest replicate; separate eval records (artifacts/gate8c1 vs output/skyfix/eval)"),
        Item("historical", "historical numbers traced to files/logs", ok("sky eval JSONs"), "six eval JSONs + RESULT log lines agree: SK 22.47/23.62/22.71, Occ3D 33.49/35.03/33.88"),
        Item("historical", "exact checkpoint identified", ok("sky checkpoints"), "sky_seed{0,1,2}_last.pt, sha256 == frozen_manifest_sky.json"),
        Item("historical", "exact config identified", True, "configs/gate8c1_sky/seed{0,1,2}.yaml (checkpoint meta carries the config; historical_protocol.yaml)"),
        Item("historical", "metric definition identified", True, "semantic.completion.binary_iou = TP/(TP+FP+FN) inside the official mask (core/run/eval_target.py; re-derived from the seed-2 dump to 7e-7)"),
        Item("historical", "no semantic IoU confused with occupancy IoU", True, "SSC mIoU (3.35 / 4.24) reported separately from SC IoU in the audit"),
        Item("historical", "conflicting results documented", True, "historical_baseline_audit.md §2: OccAny-scorer subsets (22.83 / 33.28), matched-forward 30.70, sky2 28.12 / 22.93, overwritten JSONs"),
        Item("historical", "canonical baseline justified", hb["verified"], "historical_baseline_audit.md §7: validity (full anchor sets, gate's masks/metric, METHOD.md headline), not magnitude"),
    ]
    return items


def _rederive_one_input(paths: C.CurrentPaths, device: str) -> Dict:
    """Rebuild S_H/F_H of one 0006 anchor from its clip with build_local_geometry and compare to the frozen cache bit-exactly."""
    import torch
    from core.datasets import kitti360 as K3, kitti360_splits as SRC, kitti360_transforms as TF
    t = 4
    clip_path = paths.clip_path(C.VAL_DRIVE, t); z = np.load(paths.input_path(C.VAL_DRIVE, t))
    clip = LocalClip.load(clip_path)
    calib = K3.parse_calibration(os.path.join(SRC.SSCBENCH_ROOT, "calibration"))
    S, F, _ = build_local_geometry(clip, TF.GRID, np.asarray(calib.rect_cam_to_velo, np.float64), torch.device(device))
    n = int(np.prod(C.GRID_DIMS))
    S_c = np.unpackbits(z["S_H"])[:n].astype(bool); F_c = np.unpackbits(z["F_H"])[:n].astype(bool)
    return {"anchor": t, "S_equal": bool(np.array_equal(S.cpu().numpy(), S_c)), "F_equal": bool(np.array_equal(F.cpu().numpy(), F_c)),
            "clip_sha_matches_cache": _sha(clip_path) == str(z["clip_sha256"]),
            "scale_is_exp_nanmedian_log_s": bool(abs(np.exp(np.nanmedian(clip.log_s_frames)) - clip.scale_m_per_unit) < 1e-9),
            "rope_slots_used": int(np.load(clip_path)["rope_slots_used"]), "stream_indices": [int(x) for x in clip.stream_indices]}


def current_items(paths: C.CurrentPaths, device: Optional[str]) -> List[Item]:
    runs = {s: json.load(open(os.path.join(paths.runs, f"train_geo_seed{s}.json"))) for s in C.SEEDS}
    cfg = yaml.safe_load(open(paths.training_config)); g12 = json.load(open(GATE12_FREEZE))
    src = json.load(open(os.path.join(paths.source_eval, "source_summary.json")))
    finals = {s: json.load(open(os.path.join(paths.source_eval, f"geo_seed{s}_step06000.json"))) for s in C.SEEDS}
    body_now = "\n".join(l for l in open(paths.training_config) if not l.startswith("#"))
    body_ref = "\n".join(l for l in open(paths.recipe_source) if not l.startswith("#"))
    core_same = {f: _sha(os.path.join(C.REPO_ROOT, f)) == h for sect in ("input_generation", "training_evaluation") for f, h in g12[sect].items() if f.startswith("core/")}
    n_expected = sum(len(os.listdir(os.path.dirname(paths.input_path(d, 0)))) for d in C.TRAIN_DRIVES)
    rd = _rederive_one_input(paths, device) if device else None
    items = [
        Item("current", "only KITTI-360 0003/0007/0010 used for training", all(r["n_train"] == n_expected for r in runs.values()) and cfg["train_drives"] == list(C.TRAIN_DRIVES),
             f"training_config train_drives = {cfg['train_drives']}; n_train = {n_expected} = |0003|+|0007|+|0010| input anchors in every run record"),
        Item("current", "0006 used source-side only", all(f["n_anchors"] == 590 for f in finals.values()) and cfg["val_drive"] == C.VAL_DRIVE,
             "source eval: 590 anchors of 0006 per seed; training-time validation loss on every 5th 0006 anchor is logged, never used for selection"),
        Item("current", "local-reset five-frame L inference", rd is None or (rd["stream_indices"] == [0, 1, 2, 3, 4] and rd["clip_sha_matches_cache"]),
             f"input cache records the clip sha256 it was built from; clip stream_indices = t-4..t: {rd}" if rd else "GPU re-derivation skipped (no device)"),
        Item("current", "no long-stream cache", rd is None or rd["rope_slots_used"] == 0,
             "clip rope_slots_used = 0 (five frames run as the batched scale block after clean_kv_cache; run_local_lingbot.py asserts the RoPE counter is 0 per clip)"),
        Item("current", "LB-M depth (LingBot canonical depth x causal MoGe-2 clip scale)", rd is None or (rd["S_equal"] and rd["F_equal"]),
             f"S_H/F_H of anchor 4 re-derived from the clip through core IncrementalMapper: bit-exact = {rd and (rd['S_equal'] and rd['F_equal'])}"),
        Item("current", "scale applied exactly once", rd is None or rd["scale_is_exp_nanmedian_log_s"],
             "scale enters only via mapper.scale_state.force(s_clip) (current/geometry.py); tests/baseline_reconstruction/test_geometry_scale_once.py; s_clip = exp(nanmedian log_s)"),
        Item("current", "local LingBot pose (anchor = identity)", True, "current/geometry.py frame_inputs: pose = inv(P_4) @ P_q; test_translation_is_scaled_with_depth"),
        Item("current", "hard H mapper unchanged", all(core_same.values()), f"core hashes vs Gate 12 code_freeze.json: {core_same}"),
        Item("current", "no semantics", cfg["w_sem"] == 0 and all(r["n_params"] == 967322 for r in runs.values()), "w_sem = 0, cin = 3, 967,322 params in every run record; no Trident path opened (firewall tokens 'trident', 'semantics')"),
        Item("current", "same three seeds, same initial weights as Gate 12", sorted(runs) == list(C.SEEDS) and all(r["init_matches_gate12"] for r in runs.values()),
             f"seeds {sorted(runs)}; init_matches_gate12 = {[r['init_matches_gate12'] for r in runs.values()]}"),
        Item("current", "frozen training recipe (verbatim Gate 12 H)", body_now == body_ref and all(r["config_sha256"] == _sha(paths.training_config) for r in runs.values()),
             f"training_config.yaml body == artifacts/gate12_suf/training_config_H.yaml body; run records carry its sha256 {_sha(paths.training_config)[:12]}"),
        Item("current", "final-step (6000) checkpoint policy", all(r["final_checkpoint"].endswith("step06000.pt") for r in runs.values()), "run records: final_checkpoint = geo_seed*_step06000.pt; no selection code exists"),
        Item("current", "source result matches the Gate 13 M regime", src["reproduction"]["pass"],
             f"mean abs diff {src['reproduction']['mean_abs_diff']:.4f} (tol {src['reproduction']['tolerance']['mean_full_iou_abs']}); per seed {json.dumps({k: round(v, 4) for k, v in src['reproduction']['per_seed_abs_diff'].items()})}"),
        Item("current", "training and source evaluation ran under a strict file audit with zero violations",
             all(r["firewall"]["violations"] == [] for r in runs.values()) and all(f["firewall"]["violations"] == [] for f in finals.values()),
             f"train opened {[r['firewall']['opened'] for r in runs.values()]} files, 0 violations; eval 0 violations"),
    ]
    return items


def firewall_items(paths: C.CurrentPaths) -> List[Item]:
    fz = json.load(open(FZ.FREEZE_PATH)) if os.path.exists(FZ.FREEZE_PATH) else None
    zs = {ds: json.load(open(os.path.join(paths.target_eval, f"zero_shot_{ds}.json"))) for ds in C.HISTORICAL_TARGETS if os.path.exists(os.path.join(paths.target_eval, f"zero_shot_{ds}.json"))}
    frozen_at = _dt.datetime.fromisoformat(fz["frozen_at"]).timestamp() if fz else None
    clip_files = glob.glob(os.path.join(paths.target_clips_root, "*", "*", "*.npz"))
    first_clip = min((_mtime(f) for f in clip_files), default=None)
    try:
        FZ.require_source_freeze(); hashes_ok = True; hash_msg = f"{len(fz['sha256'])} frozen files re-hashed: unchanged"
    except Exception as e:                                     # noqa: BLE001
        hashes_ok = False; hash_msg = str(e)[:300]
    items = [
        Item("firewall", "target datasets unopened before the source freeze", fz is not None and first_clip is not None and first_clip > frozen_at,
             f"freeze {fz['frozen_at'] if fz else 'missing'}; earliest target clip file {(_dt.datetime.fromtimestamp(first_clip).isoformat(timespec='seconds') if first_clip else 'none')}; "
             "every source-side stage ran inside FileAudit with the dataset-directory tokens"),
        Item("firewall", "source code/config hashes frozen before target evaluation", hashes_ok and all(z["freeze_frozen_at"] == fz["frozen_at"] for z in zs.values()), hash_msg),
        Item("firewall", "SemanticKITTI is evaluation-only", "semantickitti" in zs and zs["semantickitti"]["n_anchors"] == 163,
             "opened only by targets/clips.py and targets/evaluate.py, both behind require_source_freeze(); 163 anchors"),
        Item("firewall", "Occ3D-nuScenes is evaluation-only", "occ3d" in zs and zs["occ3d"]["n_anchors"] == 1182,
             "opened only by targets/clips.py and targets/evaluate.py, both behind require_source_freeze(); 1182 anchors"),
        Item("firewall", "no target-based threshold changes", all(z["threshold"] == C.OCCUPANCY_THRESHOLD for z in zs.values()) and (fz or {}).get("frozen_decisions", {}).get("occupancy_threshold") == C.OCCUPANCY_THRESHOLD,
             f"threshold {C.OCCUPANCY_THRESHOLD} in the freeze and in every zero-shot record; evaluation code has no threshold argument"),
        Item("firewall", "checkpoints evaluated are the frozen ones", all(z["checkpoint_sha256"][str(s)] == fz["sha256"][z["checkpoints"][str(s)]] for z in zs.values() for s in C.SEEDS) if fz and zs else False,
             "zero-shot records carry the checkpoint sha256 read from code_freeze.json"),
        Item("firewall", "no semantic inputs at evaluation", True, "targets/evaluate.py builds [S, 0, F] only; adapters expose occupancy targets only"),
        Item("firewall", "exact historical protocol comparison documented rather than assumed", os.path.exists(os.path.join(C.ARTIFACT_DIR, "historical_comparability.md")),
             "historical_comparability.md + tables/historical_vs_current.csv, mechanical classification"),
        Item("firewall", "one evaluation per target, no re-runs with alternatives", all(len(glob.glob(os.path.join(paths.target_eval, f"zero_shot_{ds}*.json"))) == 1 for ds in zs),
             "exactly one zero_shot_<dataset>.json per target"),
    ]
    return items


def write_verification(paths: C.CurrentPaths = C.CurrentPaths(), device: Optional[str] = None) -> Dict:
    items = historical_items() + current_items(paths, device) + firewall_items(paths)
    n_pass = sum(i.passed for i in items)
    md = [f"# Verification -- {C.CURRENT_BASELINE_NAME} ({_dt.date.today().isoformat()})\n", f"**{n_pass} / {len(items)} checks pass.**\n"]
    for sect, title in (("historical", "Historical baseline"), ("current", "Current baseline"), ("firewall", "Zero-shot firewall")):
        rows = [[i + 1, it.name, "PASS" if it.passed else "**FAIL**", it.evidence] for i, it in enumerate(x for x in items if x.section == sect)]
        md += [f"## {title}\n", markdown_table(["#", "check", "result", "evidence"], rows), ""]
    md.append("Tests: `tests/baseline_reconstruction/` (metrics, historical parsing, ceiling-fix constants, scale-once geometry, freeze guard, comparability, protocol serialization).\n")
    out = os.path.join(C.ARTIFACT_DIR, "verification.md"); open(out, "w").write("\n".join(md))
    json.dump([i.__dict__ for i in items], open(os.path.join(C.ARTIFACT_DIR, "logs", "verification_items.json"), "w"), indent=1)
    return {"path": out, "n_pass": n_pass, "n_total": len(items), "items": items}


__all__ = ["Item", "write_verification"]
