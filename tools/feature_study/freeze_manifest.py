#!/usr/bin/env python
"""Freeze one manifest per condition, then stop reading source data.

Everything a transfer evaluation needs is written here and nowhere else: the final-step
checkpoint and its hash, the occupancy threshold the drive-0006 rule picked, the frozen
feature projection, the input encoding, the masks the source measurement used, and the
semantic-decoder configuration (or the reason there isn't one).

Refuses to overwrite. A manifest that has been used to open a target dataset is evidence,
and evidence does not get rewritten in place.

    python tools/feature_study/freeze_manifest.py
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# study first: it pins G8C1_DATA_ROOT, which core.datasets reads at import time
from tools.feature_study import study as ST                                       # noqa: E402
from core.datasets import kitti360_splits as SRC                                  # noqa: E402
from core.model.net import LOCK_LOGODDS                                           # noqa: E402
from core.supervision.targets import N_INPUT_CHANNELS                             # noqa: E402


def mask_digest() -> dict:
    """A content hash over the frozen drive-0006 partition, so it can be shown unchanged."""
    files = sorted(glob.glob(os.path.dirname(ST.eval_mask_path(SRC.VAL_DRIVE, 0)) + "/*.npz"))
    import hashlib
    h = hashlib.sha256()
    for f in files:
        with np.load(f) as z:
            for k in ("later_measured", "map_observed", "measured_occ", "measured_free"):
                h.update(z[k].tobytes())
    return {"n_files": len(files), "sha256": h.hexdigest()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    ST.require_data_root()
    md = mask_digest()
    with np.load(ST.pca_path()) as z:
        pca_dim = int(z["basis"].shape[1])
    for cond in ST.CONDITIONS:
        out = ST.manifest_path(cond)
        if os.path.exists(out) and not a.force:
            print(f"  {out} exists; leaving it alone")
            continue
        seeds = {}
        for s in ST.SEEDS:
            src = json.load(open(os.path.join(
                ST.ART, "source", f"source_{ST.run_tag(cond, s)}.json")))
            tr = json.load(open(os.path.join(ST.ART, f"train_{ST.run_tag(cond, s)}.json")))
            assert src["checkpoint_sha256"] == tr["checkpoint_sha256"], (cond, s)
            seeds[str(s)] = {
                "seed": s, "condition": cond, "selected": "last",
                "checkpoint": src["checkpoint"],
                "checkpoint_sha256": src["checkpoint_sha256"],
                "occupancy_threshold": src["occupancy_threshold"],
                "semantic_threshold": (src["semantic_threshold"]
                                       if src["semantic_threshold"] is not None else 0.0),
                "source_full_iou": src["regions"]["full"]["at_threshold"]["iou"],
                "source_later_iou": src["regions"]["later"]["at_threshold"]["iou"],
                "source_later_ap": src["regions"]["later"]["average_precision"],
                "source_auroc": src["regions"]["full"]["auroc"],
                "train_record": f"artifacts/feature_completion_study/train_{ST.run_tag(cond, s)}.json",
            }
        man = {
            "study": "feature-completion-study",
            "condition": cond,
            "objective": {"A": "L_occ", "B": "L_occ + 0.5 * L_KL",
                          "C": "L_occ + 0.5 * L_feature"}[cond],
            "claim": ("target-domain-free transfer: trained on KITTI-360 drives "
                      "0003/0007/0010, selected on drive 0006, evaluated on untouched "
                      "SemanticKITTI and Occ3D-nuScenes. NOT whole-system zero-shot -- the "
                      "frozen foundation models (LingBot-Map, MoGe-2, Trident-H, SAM-H) "
                      "have not had their training provenance audited."),
            "derived_from": os.path.relpath(ST.REFERENCE_MANIFEST, ST.REPO_ROOT),
            "inherits": ["replicate padding", "256x256x32 grid at 0.2 m",
                         "frozen five-frame metric scale",
                         f"{N_INPUT_CHANNELS}-channel causal input encoding",
                         f"residual occupancy lock at |log-odds| >= {LOCK_LOGODDS}",
                         "bf16 autocast forward", "past5 inference protocol"],
            "occupancy_targets": {"root": ST.G8C1_DATA_ROOT,
                                  "note": "the adopted ceiling-fixed targets, reused "
                                          "unmodified; the sky supervision is unchanged"},
            "feature_projection": ({"path": ST.pca_path(), "dim": pca_dim,
                                    "sha256": ST.sha256_file(ST.pca_path()),
                                    "record": "artifacts/feature_completion_study/pca_fit.json"}
                                   if cond == "C" else None),
            "semantic_decoder": ({"kind": "the gate's own 25-class softmax head",
                                  "trained": True}
                                 if cond == "B" else
                                 {"kind": None, "trained": False,
                                  "reason": "this condition applies no gradient to the "
                                            "25-class head, so it is at initialisation. No "
                                            "compatible validated decoder exists for SAM "
                                            "features, and this study does not build one; "
                                            "its semantic-label metrics are unavailable."}),
            "source_partition": dict(md, drive=SRC.VAL_DRIVE,
                                     record="artifacts/feature_completion_study/eval_masks"),
            "rule": {"checkpoint": "final step, no selection",
                     "occupancy_threshold": "final-logit threshold maximising full-grid IoU "
                                            "on KITTI-360 drive 0006",
                     "forbidden": "SemanticKITTI and Occ3D/nuScenes are not read before this "
                                  "file exists"},
            "seeds": seeds,
            "env": ST.environment(),
        }
        ST.write_json(out, man)
        print(f"  froze {os.path.relpath(out, ST.REPO_ROOT)}  "
              + " ".join(f"seed{s}:tau{seeds[str(s)]['occupancy_threshold']:+.4f}"
                         for s in ST.SEEDS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
