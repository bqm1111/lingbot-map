#!/usr/bin/env python
"""Write and SHA-256-pin ``configs/gate7a/completion_reachability_precommit.yaml``.

Generated from ``gate7a.config``, which the analysis imports, so the pinned file and the
running code cannot drift: change a radius and the hash changes with it.

    python tools/gate7a/precommit.py
"""
from __future__ import annotations

import hashlib, json, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import yaml                                                              # noqa: E402
from gates.scale_gate.config import REPO_ROOT                                  # noqa: E402
from gates.gate7a import config as C                                          # noqa: E402
from gates.gate6 import frames as G6F, grids as G6G                           # noqa: E402

OUT = "configs/gate7a/completion_reachability_precommit.yaml"


def load_json(rel):
    p = os.path.join(REPO_ROOT, rel)
    return json.load(open(p)) if os.path.exists(p) else {"missing": rel}


def build() -> dict:
    stage0 = load_json("artifacts/gate7a/stage0_audit.json")
    cfg = {}
    cfg["gate"] = {
        "name": "Gate 7A - completion reachability and oracle-envelope diagnosis",
        "kind": ("TARGET-DEPENDENT ORACLE ANALYSIS. Not a deployable prediction, not "
                 "target-free inference. It produces no prediction artifact, and it "
                 "neither alters nor weakens Gate 6's target-free claim: the Gate-6 "
                 "prediction rollup hashes are recorded before and after and must match."),
        "nothing_is_trained": ("no optimizer, no backward pass, no neural network created "
                               "or trained, no radius tuned on any target benchmark"),
        "repo_commit": stage0.get("git", {}).get("commit"),
        "written_before_the_target_dependent_analysis_was_run": True,
        "seeds": {"bootstrap": 0, "global": 0},
        "forbidden": list(C.FORBIDDEN),
        "stop_conditions": list(C.STOP_CONDITIONS),
    }

    cfg["frozen_inputs"] = {
        "occupancy": ("Gate-6 B-R and B-D, recomputed bit-identically from the frozen "
                      "LingBot cache, the frozen G51-B scale table and the frozen Trident "
                      "cache, and verified voxel-for-voxel against the pinned Gate-6 "
                      "prediction files before any analysis"),
        "gauge": "G51-B (frozen MoGe-2, calibrated horizontal FOV, one scalar per clip)",
        "scale_application": "identical scalar on depth and on pose translation",
        "b_d": "B-R plus the frozen dilate_r2 = 0.4 m; unchanged by this gate",
        "primary_condition": C.PRIMARY_CONDITION,
        "teacher": "Trident-H, frozen, not re-run; the Gate-6 cache is read only",
        "manifests": {k: v for k, v in G6F.MANIFESTS.items()},
        "lingbot_cache": {k: v for k, v in G6F.LINGBOT_CACHE.items()},
        "semantic_cache_root": G6F.SEMANTIC_CACHE_ROOT,
        "gate6_prediction_rollups": {
            ds: d["rollup_sha256"]
            for ds, d in stage0.get("gate6", {}).get("predictions", {}).items()},
        "retired": "C3, V3 and every previous learned occupancy corrector remain retired",
    }

    cfg["datasets"] = {}
    for ds in C.DATASETS:
        G = G6G.EVAL_GRID[ds]
        cfg["datasets"][ds] = {
            "evaluation_grid": G.name, "dims": list(G.dims), "voxel_size_m": G.voxel_size,
            "origin_m": list(G.origin), "frame": G.frame,
            "empty_class": int(G.empty_class), "ignore_label": int(G.ignore_label),
            "manifest": G6F.MANIFESTS[ds],
            "native_reduction": ("canonical 0.2 m -> official 0.4 m by the frozen "
                                 "any-subvoxel-occupied rule, probability vectors averaged "
                                 "over the occupied subvoxels"
                                 if G6G.NEEDS_REDUCTION[ds] else "none"),
        }

    cfg["conditions"] = {"base": list(C.CONDITIONS), "primary": C.PRIMARY_CONDITION,
                         "constructions": list(C.CONSTRUCTIONS)}
    cfg["radii_m"] = list(C.RADII_M)
    cfg["range_bands_m"] = [list(b) for b in C.RANGE_BANDS]
    cfg["transport_intervals_m"] = [list(b) for b in C.TRANSPORT_INTERVALS_M]
    cfg["distance"] = dict(C.DISTANCE_DEFINITION)
    cfg["oracle"] = dict(C.ORACLE_DEFINITION)
    cfg["morphology"] = dict(C.MORPHOLOGY_DEFINITION)
    cfg["semantic_propagation"] = dict(C.SEMANTIC_PROPAGATION_RULE)
    cfg["frustum"] = dict(C.FRUSTUM_DEFINITION)
    cfg["aggregation"] = dict(C.AGGREGATION)
    cfg["bootstrap"] = dict(C.BOOTSTRAP)
    cfg["miss_distance_histogram"] = {
        "bin_m": C.MISS_HIST_BIN_M, "max_m": C.MISS_HIST_MAX_M,
        "quantiles": [0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
        "reported_at": "the upper edge of the bin in which the quantile falls"}
    cfg["deliverables"] = {
        "not_a_prediction": ("no artifact of this gate may be consumed as a prediction; "
                             "the oracle envelope in particular is a ceiling produced with "
                             "target knowledge"),
        "reported_separately": ("oracle semantic mIoU is never described as deployable; "
                                "morphology is the only deployable construction here"),
    }
    cfg["future_model_constraint"] = (
        "any later learned model must avoid a fixed 17/18/19-class output head: the three "
        "benchmarks have different ontologies and such a head would weaken the "
        "open-vocabulary transfer claim. Later semantic learning must operate in a "
        "fixed-dimensional language-aligned feature space or use class-agnostic "
        "propagation.")
    return cfg


def main() -> int:
    cfg = build()
    path = os.path.join(REPO_ROOT, OUT)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = yaml.safe_dump(cfg, sort_keys=False, width=100, allow_unicode=True)
    with open(path, "w") as fh:
        fh.write("# Gate 7A precommit. Written and SHA-256-pinned BEFORE the "
                 "target-dependent\n# analysis was run. Generated by "
                 "tools/gate7a/precommit.py from gate7a/config.py,\n# which the analysis "
                 "imports, so config and code cannot drift apart.\n")
        fh.write(text)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    pin = os.path.join(REPO_ROOT, "artifacts", "gate7a", "precommit_pin.json")
    with open(pin, "w") as fh:
        json.dump({"path": OUT, "sha256": digest, "bytes": os.path.getsize(path)},
                  fh, indent=2)
    print(f"{OUT}\n  sha256 {digest}\n  bytes  {os.path.getsize(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
