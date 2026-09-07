#!/usr/bin/env python
"""Gate 8C-1: write the frozen manifest that lifts the dataset firewall.

Until this file exists, ``tools/gate8c1/eval_target.py`` refuses to open a target dataset.
It records, per seed, the configuration, the selected checkpoint and its hash, both
thresholds, the source-dataset hashes, the code commit, the seeds, and the runtime
file-audit summaries from training and selection -- the explicit proof that neither
SemanticKITTI nor Occ3D was accessed.

    python tools/gate8c1/freeze_manifest.py
"""
from __future__ import annotations
import argparse, glob, json, os, subprocess, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, SEEDS, sha256                                # noqa: E402
from gates.scale_gate.config import write_json                                         # noqa: E402
from gates.gate8c1 import rawtarget as RT, sources as SRC                              # noqa: E402

FROZEN_CODE = ["gates/gate8/mapper.py", "gates/gate8/net.py", "gates/gate8/feed.py", "gates/gate8/losses.py",
               "gates/gate8/vocab.py", "gates/gate8c0/transforms.py", "gates/gate8c1/rawtarget.py",
               "gates/gate8c1/sources.py", "gates/gate8c1/data.py", "tools/gate8c1/train.py",
               "tools/gate8c1/selection.py", "tools/gate8c1/build_targets.py",
               "tools/gate8c1/build_samples.py",
               "checkpoints/lingbot-map/204754b/lingbot-map.pt"]


def dir_digest(pattern: str, cap: int = 4000) -> dict:
    """Order-independent digest of a cache directory: count, bytes, hash of the name+size list."""
    import hashlib
    fs = sorted(glob.glob(pattern))[:cap]
    h = hashlib.sha256()
    tot = 0
    for f in fs:
        n = os.path.getsize(f); tot += n
        h.update(os.path.basename(f).encode()); h.update(str(n).encode())
    return {"n_files": len(fs), "total_bytes": tot, "listing_sha256": h.hexdigest()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    a = ap.parse_args()
    sel = json.load(open(os.path.join(ART, "selection.json")))
    val = json.load(open(os.path.join(ART, "target_validation.json")))
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                         text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                                             text=True).strip())
    except Exception:
        commit, dirty = None, None
    man = {
        "gate": "8C-1",
        "claim": "target-domain-free transfer of the completion module: trained and selected "
                 "on KITTI-360 alone, evaluated on untouched SemanticKITTI and Occ3D without "
                 "adaptation. NOT whole-system zero-shot: the frozen foundation models "
                 "(LingBot-Map, MoGe-2, Trident-H) have not had their training provenance "
                 "audited, and LingBot-Map's published mixture includes KITTI-360.",
        "frozen_at": time.strftime("%FT%T%z"), "code_commit": commit, "worktree_dirty": dirty,
        "seeds": {}, "training_data": {
            "dataset": "KITTI-360", "train_drives": list(SRC.TRAIN_DRIVES),
            "val_drive": SRC.VAL_DRIVE,
            "occupancy_supervision": "rebuilt from raw Velodyne sweeps t..t+20 with "
                                     "ground-truth poses; endpoints occupied, interiors free, "
                                     "unobserved and cross-sweep-conflicting voxels unknown "
                                     "and excluded from the loss",
            "sscbench_completion_labels_used": False,
            "human_semantic_labels_used": False,
            "semantic_supervision": "frozen Trident-H fused over t+1..t+20, kept only where "
                                    "geometrically supported and where both halves of the "
                                    "future window agree; soft probabilities preserved",
            "future_frames": SRC.FUTURE_FRAMES, "scale_frames": SRC.SCALE_FRAMES,
            "carve_decimation": RT.CARVE_DECIMATION, "band_half_m": RT.BAND_HALF_M},
        "target_datasets_excluded_until_now": ["semantickitti", "occ3d"],
        "target_validation": {k: v.get("pass") for k, v in val["checks"].items()},
        "target_validation_all_pass": val["all_pass"],
        "source_dataset_hashes": {
            "targets_" + d[17:21]: dir_digest(f"{SRC.G8C1_ROOT}/targets/{d}/*.npz")
            for d in list(SRC.TRAIN_DRIVES) + [SRC.VAL_DRIVE]},
        "sample_hashes": {
            "samples_" + d[17:21]: dir_digest(f"{SRC.G8C1_ROOT}/samples/{d}/*.npz")
            for d in list(SRC.TRAIN_DRIVES) + [SRC.VAL_DRIVE]},
        "code_hashes": {p: (sha256(os.path.join(REPO_ROOT, p))
                            if os.path.exists(os.path.join(REPO_ROOT, p)) else None)
                        for p in FROZEN_CODE},
        "config_hashes": {os.path.relpath(p, REPO_ROOT): sha256(p)
                          for p in sorted(glob.glob(os.path.join(REPO_ROOT, "configs",
                                                                 "gate8c1", "*.yaml")))},
        "firewall_proof": {"selection": sel["firewall"], "training": {}},
        "selection_rule": sel["rule"]}
    for s in SEEDS:
        k = str(s)
        if k not in sel["seeds"]:
            continue
        v = sel["seeds"][k]
        tr = json.load(open(os.path.join(ART, f"train_seed{s}.json")))
        assert sha256(os.path.join(REPO_ROOT, v["checkpoint"])) == v["checkpoint_sha256"]
        man["seeds"][k] = {
            "seed": s, "config": f"configs/gate8c1/seed{s}.yaml",
            "config_sha256": man["config_hashes"][f"configs/gate8c1/seed{s}.yaml"],
            "selected": v["selected"], "checkpoint": v["checkpoint"],
            "checkpoint_sha256": v["checkpoint_sha256"],
            "occupancy_threshold": v["occupancy_threshold"],
            "semantic_threshold": v["semantic_threshold"],
            "source_ap": v["source_ap"], "source_ap_over_prevalence": v["source_ap_over_prevalence"],
            "source_auroc": v["source_auroc"],
            "source_iou_at_threshold": v["source_iou_at_threshold"],
            "semantic_agreement_at_threshold": v["semantic_agreement_at_threshold"],
            "train_seconds": tr["seconds"], "gpu_hours": tr["gpu_hours"],
            "peak_gpu_gib": tr["peak_gpu_gib"], "n_params": tr["n_params"],
            "best_step": tr["best_step"]}
        man["firewall_proof"]["training"][k] = tr["firewall"]
    viol = [v for f in [man["firewall_proof"]["selection"]]
            + list(man["firewall_proof"]["training"].values()) for v in f["violations"]]
    man["firewall_proof"]["total_violations"] = len(viol)
    man["firewall_proof"]["neither_target_dataset_was_accessed"] = bool(not viol)
    assert not viol, f"firewall violated: {viol[:4]}"
    p = os.path.join(ART, "frozen_manifest.json")
    write_json(p, man)
    print(f"frozen manifest written: {len(man['seeds'])} seeds, "
          f"{man['firewall_proof']['total_violations']} firewall violations, commit {commit}")
    for k, v in man["seeds"].items():
        print(f"   seed {k}: {v['selected']:4s} AP {v['source_ap']:.4f} "
              f"(x{v['source_ap_over_prevalence']:.2f}) tau {v['occupancy_threshold']:+.4f} "
              f"sem_tau {v['semantic_threshold']:.2f} sha {v['checkpoint_sha256'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
