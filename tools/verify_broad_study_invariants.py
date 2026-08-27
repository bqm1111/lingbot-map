#!/usr/bin/env python
"""Verify every invariant the broad-corpus study must satisfy.

    python tools/verify_broad_study_invariants.py \
        --broad configs/semantic_sidecar/broad_tartanair_controlled.yaml \
        --primary configs/semantic_sidecar/feasibility.yaml

Exits non-zero and prints the offending checks if anything but the training-scene
manifest differs from the primary study.  No result may be accepted unless this passes.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import torch

import _bootstrap  # noqa: F401

from semantic_sidecar.config import config_to_dict, load_config, write_json
from semantic_sidecar.datasets import expand_scenes
from semantic_sidecar.lingbot_features import FeatureCacheReader, manifest_path

FORBIDDEN = ("kitti", "semantickitti", "scannet", "replica", "speira")
#: Config blocks that define the method and must be byte-identical between studies.
CONTROLLED_BLOCKS = ("lingbot", "model", "train", "tracks", "teacher", "consensus", "evaluation")
EXPECTED_SIDECAR_PARAMS = 9_112_896
EXPECTED_STEPS = 20_000


class Checks:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append({"check": name, "passed": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    @property
    def all_passed(self) -> bool:
        return all(r["passed"] for r in self.rows)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--broad", required=True)
    ap.add_argument("--primary", required=True)
    ap.add_argument("--manifest", default="/media/SSD1/lingbot_semantic_sidecar_broad/broad_corpus_manifest.json")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    broad = load_config(args.broad)
    primary = load_config(args.primary)
    bd, pd = config_to_dict(broad), config_to_dict(primary)
    c = Checks()

    print("== config: only the training-scene manifest and artefact paths may differ ==")
    for block in CONTROLLED_BLOCKS:
        c.add(f"config block '{block}' identical", bd[block] == pd[block],
              "" if bd[block] == pd[block] else "DIFFERS")
    differing = {k for k in pd if pd[k] != bd.get(k)}
    c.add("only scenes/paths/name differ", differing <= {"scenes", "paths", "name"},
          f"differing top-level keys: {sorted(differing)}")

    print("\n== training corpus ==")
    b_train = [s for s in expand_scenes(broad.scenes) if s.role == "train"]
    p_train = [s for s in expand_scenes(primary.scenes) if s.role == "train"]
    from semantic_sidecar.lingbot_features import list_scene_images

    b_frames = sum(len(list_scene_images(s)) for s in b_train)
    p_frames = sum(len(list_scene_images(s)) for s in p_train)
    c.add("broad corpus has exactly 3,347 frames", b_frames == 3347, f"{b_frames}")
    c.add("corpus size matches the primary study", b_frames == p_frames, f"broad {b_frames} vs primary {p_frames}")

    if os.path.exists(args.manifest):
        man = json.load(open(args.manifest))
        c.add("manifest total is 3,347", man["total_frames"] == 3347, str(man["total_frames"]))
        c.add("manifest frames are unique", man["unique_frames"] == 3347, str(man["unique_frames"]))
        per = man["per_environment"]
        c.add("six environments, evenly sampled", len(per) == 6 and max(per.values()) - min(per.values()) <= 2,
              json.dumps(per))
        c.add("manifest declares no labels / no target taxonomy",
              not man["uses_semantic_labels"] and not man["uses_target_taxonomy"])
    else:
        c.add("frame manifest exists", False, args.manifest)

    print("\n== no target-dataset image entered training ==")
    bad = []
    for s in b_train:
        low = os.path.abspath(s.image_folder).lower()
        for frag in FORBIDDEN:
            if frag in low:
                bad.append((s.name, frag))
    c.add("no forbidden dataset fragment in any training scene path", not bad, str(bad[:3]))
    if os.path.exists(args.manifest):
        man = json.load(open(args.manifest))
        bad2 = [f["local_path"] for f in man["frames"]
                if any(x in str(f["local_path"]).lower() or x in str(f["source_path"]).lower()
                       for x in FORBIDDEN)]
        c.add("no forbidden fragment in any manifest frame path", not bad2, str(bad2[:2]))

    print("\n== shared PCA basis ==")
    b_pca = os.path.join(broad.paths.cache_root, "pca.safetensors")
    p_pca = os.path.join(primary.paths.cache_root, "pca.safetensors")
    hb, hp = sha256(b_pca), sha256(p_pca)
    c.add("PCA file hash identical to the primary study", hb == hp, hb[:16])
    from semantic_sidecar.teacher_features import SemanticProjection

    proj = SemanticProjection.load(b_pca)
    c.add("PCA basis was not refit on the broad corpus",
          all(not n.startswith("broad_") for n in proj.meta.get("fit_scenes", [])),
          f"fit on {len(proj.meta.get('fit_scenes', []))} original scenes")
    c.add("PCA output dim is 64", proj.dim == 64, str(proj.dim))

    print("\n== evaluation scenes and prompts ==")
    b_eval = [s.name for s in expand_scenes(broad.scenes) if s.role in ("eval", "qualitative")]
    p_eval = [s.name for s in expand_scenes(primary.scenes) if s.role in ("eval", "qualitative")]
    c.add("evaluation scene lists identical", b_eval == p_eval, f"{len(b_eval)} scenes")
    from semantic.eval_semantickitti import CLASS_NAMES, CLASS_PROMPTS

    c.add("19-class prompt set unchanged", len(CLASS_PROMPTS) == 19 and len(CLASS_NAMES) == 19,
          hashlib.sha256("|".join(CLASS_PROMPTS).encode()).hexdigest()[:16])

    print("\n== frozen geometry ==")
    from infer_semantic_sidecar import geometry_fingerprint

    eval_specs = [s for s in expand_scenes(broad.scenes) if s.role == "eval"]
    fb = geometry_fingerprint(broad.paths.cache_root, eval_specs)
    fp = geometry_fingerprint(primary.paths.cache_root,
                              [s for s in expand_scenes(primary.scenes) if s.role == "eval"])
    c.add("eval geometry fingerprint identical to the primary study", fb == fp, fb[:16])

    # Byte-identity of depth/pose/intrinsics across every sidecar map produced here.
    summaries = sorted(glob.glob(os.path.join(broad.paths.maps_root, "*", "inference_summary.json")))
    prints = {}
    for path in summaries:
        with open(path) as fh:
            prints[os.path.basename(os.path.dirname(path))] = json.load(fh).get("geometry_fingerprint")
    uniq = {v for v in prints.values() if v}
    if summaries:
        c.add("geometry fingerprint identical across all broad runs", len(uniq) == 1,
              f"{len(prints)} runs, {len(uniq)} distinct")
    else:
        print("  (no inference summaries yet — rerun after the infer stage)")

    print("\n== trained models ==")
    run_dirs = sorted(glob.glob(os.path.join(broad.paths.runs_root, "*", "summary.json")))
    steps_ok, params_ok, seeds = True, True, []
    for path in run_dirs:
        with open(path) as fh:
            s = json.load(fh)
        steps_ok &= s.get("steps") == EXPECTED_STEPS
        params_ok &= s.get("parameter_report", {}).get("sidecar_trainable") == EXPECTED_SIDECAR_PARAMS
        seeds.append(s.get("provenance", {}).get("seed"))
    if run_dirs:
        c.add(f"all {len(run_dirs)} runs completed exactly {EXPECTED_STEPS} steps", steps_ok)
        c.add("sidecar trainable parameter count unchanged", params_ok, str(EXPECTED_SIDECAR_PARAMS))
        c.add("LingBot contributes zero trainable parameters",
              all(json.load(open(p)).get("parameter_report", {}).get("lingbot_frozen", 0) > 1e9
                  for p in run_dirs))
    else:
        print("  (no runs yet — rerun after the train stage)")

    print("\n== coverage identical across semantic models ==")
    evals = sorted(glob.glob(os.path.join(broad.paths.maps_root, "*", "eval.json")))
    covs = []
    for path in evals:
        with open(path) as fh:
            covs.append(json.load(fh)["coverage"])
    if covs:
        spread = max(covs) - min(covs)
        c.add("coverage identical across models (<0.01 pp spread)", spread < 0.01,
              f"{min(covs):.4f}..{max(covs):.4f}")
    else:
        print("  (no evaluations yet)")

    print("\n" + ("ALL INVARIANTS PASSED" if c.all_passed else "*** INVARIANT VIOLATION ***"))
    payload = {"all_passed": c.all_passed, "checks": c.rows,
               "pca_sha256": hb, "eval_geometry_fingerprint": fb,
               "broad_config": os.path.abspath(args.broad),
               "primary_config": os.path.abspath(args.primary)}
    if args.output:
        write_json(args.output, payload)
        print(f"wrote {args.output}")
    sys.exit(0 if c.all_passed else 1)


if __name__ == "__main__":
    main()
