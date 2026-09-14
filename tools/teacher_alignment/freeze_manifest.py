#!/usr/bin/env python
"""Freeze one manifest per condition -- six checkpoints, thresholds, rho, populations --
before any target dataset is opened. Refuses to overwrite.

    python tools/teacher_alignment/freeze_manifest.py
"""
from __future__ import annotations
import argparse, json, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.teacher_alignment import study as ST                                   # noqa: E402
from tools.feature_study import freeze_manifest as FM                             # noqa: E402
from core.datasets import kitti360_splits as SRC                                  # noqa: E402
from core.model.net import LOCK_LOGODDS                                           # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    a = ap.parse_args()
    ST.require_data_root()
    md = FM.mask_digest()
    rho = json.load(open(ST.rho_path()))
    cfgf = json.load(open(os.path.join(ST.ART, "frozen_config.json")))
    for cond in ST.CONDITIONS:
        out = ST.manifest_path(cond)
        if os.path.exists(out):
            print(f"  {out} exists; leaving it alone"); continue
        seeds = {}
        for s in ST.SEEDS:
            src = json.load(open(os.path.join(ST.ART, "source", f"source_{ST.run_tag(cond, s)}.json")))
            tr = json.load(open(os.path.join(ST.ART, f"train_{ST.run_tag(cond, s)}.json")))
            assert src["checkpoint_sha256"] == tr["checkpoint_sha256"], (cond, s)
            seeds[str(s)] = {
                "seed": s, "condition": cond, "selected": "last",
                "checkpoint": src["checkpoint"], "checkpoint_sha256": src["checkpoint_sha256"],
                "occupancy_threshold": src["occupancy_threshold"],
                "semantic_threshold": src["semantic_threshold"],
                "semantic_agreement_at_threshold": src["semantic_consistency"]["selected"]["agreement"],
                "source_full_iou": src["regions"]["full"]["at_threshold"]["iou"],
                "source_later_iou": src["regions"]["later"]["at_threshold"]["iou"],
                "source_auroc": src["regions"]["full"]["auroc"],
                "aligned_teacher_top1_agreement": src["teacher_agreement"]["all"]["top1_agreement"],
                "n_degenerate_steps": tr["n_degenerate_steps"],
                "achieved_ratio_mean": tr["gradient_matching"]["achieved_ratio_mean"],
                "train_record": os.path.relpath(os.path.join(ST.ART, f"train_{ST.run_tag(cond, s)}.json"),
                                                ST.REPO_ROOT)}
        man = {"study": "teacher-alignment control", "condition": cond,
               "objective": "L_occ + 0.5*L_KL with trunk KL-gradient matched to rho*|g_occ|",
               "auxiliary_target": cfgf["conditions"][cond],
               "rho": rho["rho"], "rho_record": "artifacts/teacher_alignment_study/rho_calibration.json",
               "claim": ("target-domain-free transfer; frozen foundation models not audited "
                         "for training provenance (LingBot-Map, MoGe-2, Trident-H)"),
               "derived_from": ["artifacts/gate8c1/frozen_manifest_sky.json",
                                "artifacts/feature_completion_study/frozen_manifest_B.json"],
               "inherits": cfgf["inherits_from_feature_study_B"] + [
                   f"residual occupancy lock at |log-odds| >= {LOCK_LOGODDS}"],
               "semantic_decoder": {"kind": "the gate's 25-class softmax head", "trained": True},
               "source_partition": dict(md, drive=SRC.VAL_DRIVE),
               "evaluation_populations": {"semantickitti": 163, "occ3d": 1182,
                                          "protocol": "past5"},
               "rule": {"checkpoint": "final step, no selection",
                        "occupancy_threshold": "final-logit threshold maximising full-grid IoU on "
                                               "KITTI-360 drive 0006",
                        "semantic_threshold": "coverage x agreement with the frozen teacher on "
                                              "half-window-agreeing voxels (the gate's rule)",
                        "forbidden": "SemanticKITTI and Occ3D/nuScenes not read before this file"},
               "seeds": seeds, "env": ST.environment()}
        ST.write_json(out, man)
        print(f"  froze {os.path.relpath(out, ST.REPO_ROOT)}  "
              + " ".join(f"seed{s}:tau{seeds[str(s)]['occupancy_threshold']:+.4f}" for s in ST.SEEDS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
