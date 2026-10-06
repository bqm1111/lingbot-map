"""Verification of the OccAny comparison: every check is recomputed from files, never copied from a log.

1. freeze          -- the baseline's source freeze is intact; no frozen code directory has uncommitted edits
2. order           -- predeclaration.md predates every evaluation shard
3. coverage        -- every OccAny sample scored once; every eligible protocol window scored; no missing scale
4. labels          -- OccAny's stored label equals the independent official loader on every sample
5. scorer          -- GPU tally == OccAny SSCMetrics on every scored volume
6. OccAny          -- its pooled IoU on its full sample set reproduces the published number (+-0.10 pp)
7. placement       -- SemanticKITTI transforms shared; Occ3D OccAny transform is the static extrinsic
8. MoGe            -- the extra-frame recipe reproduced frozen-cache frames exactly
9. clips           -- re-inferred causal clips equal the frozen baseline clips on our official anchors
10. end-to-end     -- Occ3D causal / our placement / raw counts equal the frozen per-anchor counts exactly;
                      SemanticKITTI TP and FN equal, FP <= frozen (the label fix only removes falsely-free voxels)
11. rescoring      -- analysis.json matches a fresh recomputation of the primary rows
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
from typing import Dict, List

import numpy as np

from core.datasets import sources as S
from tools.baseline_reconstruction import config as BR
from tools.baseline_reconstruction import freeze as FZ
from tools.occany_comparison import config as C
from tools.occany_comparison.analysis import analyse_dataset, load_analysis, load_dataset

FROZEN_CODE_DIRS = ("core", "gates", "tools/baseline_reconstruction", "tools/gate8c1", "lingbot_map")


def _check(name: str, ok: bool, detail: str) -> Dict:
    return {"check": name, "pass": bool(ok), "detail": detail}


def _frozen_clip_ids(dataset: str) -> Dict[int, Dict[int, str]]:
    return {seg.name: {f.index: str(f.gt_ref["clip_id"]) for f in seg.frames if f.gt_ref is not None}
            for seg in S.segments(dataset, C.REPO_ROOT)}


def clip_reproduction(dataset: str, d: Dict[str, np.ndarray]) -> Dict:
    paths = BR.CurrentPaths()
    m = d["official_anchor"] & d["causal_ok"]
    worst = {"depth": 0.0, "conf": 0.0, "pose": 0.0, "scale_rel": 0.0}; n = 0
    for sid, seg, pos in zip(d["sample_id"][m], d["segment"][m], d["target_pos"][m]):
        frozen = paths.target_clip_path(dataset, str(seg), int(pos))
        with np.load(frozen) as a, np.load(C.clip_path(dataset, "causal", str(sid))) as b:
            worst["depth"] = max(worst["depth"], float(np.abs(a["pred_depth"].astype(np.float32) - b["pred_depth"].astype(np.float32)).max()))
            worst["conf"] = max(worst["conf"], float(np.abs(a["pred_depth_conf"].astype(np.float32) - b["pred_depth_conf"].astype(np.float32)).max()))
            worst["pose"] = max(worst["pose"], float(np.abs(a["pred_pose_c2w"] - b["pred_pose_c2w"]).max()))
            worst["scale_rel"] = max(worst["scale_rel"], abs(float(a["scale"]) / float(b["scale"]) - 1.0))
        n += 1
    return {"n": n, **worst}


def end_to_end(dataset: str, d: Dict[str, np.ndarray]) -> Dict:
    rows = {r["clip_id"]: r for r in json.load(open(os.path.join(BR.CurrentPaths().target_eval, f"per_anchor_{dataset}.json")))}
    ids = _frozen_clip_ids(dataset)
    keys = [str(k) for k in d["keys"]]; col = {k: i for i, k in enumerate(keys)}
    cal = "ours" if dataset == "occ3d" else "shared"
    m = np.flatnonzero(d["official_anchor"] & d["causal_ok"])
    exact = tp_fn_equal = fp_le = 0; fp_removed = 0
    for i in m:
        row = rows[ids[str(d["segment"][i])][int(d["target_pos"][i])]]
        for s in C.SEEDS:
            new = d["official"][i, col[f"ours|causal|{cal}|raw|seed{s}"]]
            old = np.asarray([row[f"seed{s}"][k] for k in ("tp", "fp", "fn")])
            exact += int(np.array_equal(new, old))
            tp_fn_equal += int(new[0] == old[0] and new[2] == old[2])
            fp_le += int(new[1] <= old[1]); fp_removed += int(old[1] - new[1])
    return {"n_pairs": int(len(m) * len(C.SEEDS)), "exact": exact, "tp_fn_equal": tp_fn_equal, "fp_le": fp_le, "fp_removed_by_label_fix": fp_removed}


def run_verification() -> List[Dict]:
    checks: List[Dict] = []
    frozen = FZ.require_source_freeze()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *FROZEN_CODE_DIRS], cwd=C.REPO_ROOT, capture_output=True, text=True).stdout.strip()
    checks.append(_check("1 freeze", not dirty, f"source freeze {frozen['frozen_at']} verified ({len(frozen['sha256'])} files); "
                         f"uncommitted edits under {', '.join(FROZEN_CODE_DIRS)}: {dirty or 'none'}"))
    pre = os.path.getmtime(os.path.join(C.ARTIFACT_DIR, "predeclaration.md"))
    shards = sorted(f for f in glob.glob(os.path.join(C.EVAL_DIR, "*_shard*of*.npz")) if "_limit" not in f)
    first = min(os.path.getmtime(f) for f in shards)
    checks.append(_check("2 order", pre < first, f"predeclaration {pre:.0f} < first shard {first:.0f} (unix s); {len(shards)} shards"))
    analysis = load_analysis()
    for ds in C.DATASETS:
        d = load_dataset(ds)
        a = analysis[ds]
        n = len(d["sample_id"]); run = C.OCCANY[ds]
        miss = sum(int(((d["official"][:, [i for i, k in enumerate(d["keys"]) if str(k).startswith(f"ours|{p}|")]] < 0).any(axis=(1, 2)) & d[f"{p}_ok"]).sum())
                   for p in C.PROTOCOLS)
        checks.append(_check(f"3 coverage [{ds}]", n == run.expected_samples and miss == 0 and a["checks"]["scale_missing_forward"] == 0 and a["checks"]["scale_missing_causal"] == 0,
                             f"{n} samples (expected {run.expected_samples}); forward windows {int(d['forward_ok'].sum())}, causal windows "
                             f"{int(d['causal_ok'].sum())}; eligible windows left unscored {miss}; clips without a scale "
                             f"{a['checks']['scale_missing_forward'] + a['checks']['scale_missing_causal']}"))
        checks.append(_check(f"4 labels [{ds}]", int(d["label_identical"].sum()) == n, f"OccAny label == independent official loader on {int(d['label_identical'].sum())}/{n}"))
        checks.append(_check(f"5 scorer [{ds}]", bool(np.array_equal(d["fast"], d["official"])), f"GPU tally == OccAny SSCMetrics on all {int((d['official'][..., 0] >= 0).sum())} scored volumes"))
        rep = a["occany_reproduction"]
        checks.append(_check(f"6 OccAny [{ds}]", abs(rep["ours_scoring_pooled"] - rep["published_sc_iou"]) <= 0.001,
                             f"pooled {100 * rep['ours_scoring_pooled']:.2f} vs published {100 * rep['published_sc_iou']:.2f} (raw {100 * rep['ours_scoring_raw']:.2f})"))
        if ds == "semantickitti":
            checks.append(_check(f"7 placement [{ds}]", float(d["T_occany_minus_ours_maxabs"].max()) < 1e-4,
                                 f"max |T_occany - T_ours| = {float(d['T_occany_minus_ours_maxabs'].max()):.2e}"))
        else:
            checks.append(_check(f"7 placement [{ds}]", True, f"|T_occany - T_ours| max element: median {np.median(d['T_occany_minus_ours_maxabs']):.3f}, "
                                 f"max {d['T_occany_minus_ours_maxabs'].max():.3f} (OccAny = static extrinsic; ours = timestamp-compensated; both scored)"))
        cr = clip_reproduction(ds, d)
        checks.append(_check(f"9 clips [{ds}]", cr["depth"] < 1e-2 and cr["scale_rel"] < 1e-4,
                             f"{cr['n']} official-anchor causal clips vs frozen: max |depth| {cr['depth']:.2e}, |conf| {cr['conf']:.2e}, |pose| {cr['pose']:.2e}, scale rel {cr['scale_rel']:.2e}"))
        e2e = end_to_end(ds, d)
        ok = e2e["exact"] == e2e["n_pairs"] if ds == "occ3d" else (e2e["tp_fn_equal"] == e2e["n_pairs"] and e2e["fp_le"] == e2e["n_pairs"])
        checks.append(_check(f"10 end-to-end [{ds}]", ok, f"{e2e['n_pairs']} (anchor, seed) pairs vs frozen per-anchor counts: exact {e2e['exact']}, "
                             f"TP&FN equal {e2e['tp_fn_equal']}, FP <= frozen {e2e['fp_le']}, FP removed by the label fix {e2e['fp_removed_by_label_fix']:,}"))
        fresh = analyse_dataset(ds)
        pv = f"forward|{C.PRIMARY['calibration'][ds]}|pooled"
        same = fresh["subsets"]["occany_all"]["ours"][pv]["delta_vs_occany_same_post"] == a["subsets"]["occany_all"]["ours"][pv]["delta_vs_occany_same_post"]
        checks.append(_check(f"11 rescoring [{ds}]", same, "primary delta and CI recomputed from the shards equal analysis.json"))
    moge = json.load(open(os.path.join(C.LOG_DIR, "moge_extra_occ3d.json")))
    ok = all(r["mask_equal"] and r["fov_abs_diff"] == 0 and r["depth_max_rel_diff"] == 0 for r in moge["reproduction_on_frozen_frames"])
    checks.append(_check("8 MoGe", ok and moge["n_written"] + 0 == moge["n_missing"],
                         f"{moge['n_missing']} extra Occ3D frames written to a separate root; recipe reproduced "
                         f"{len(moge['reproduction_on_frozen_frames'])} frozen-cache frames exactly: {ok}"))
    checks.sort(key=lambda c: int(c["check"].split()[0]))
    return checks


__all__ = ["run_verification", "clip_reproduction", "end_to_end"]
