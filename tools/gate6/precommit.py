#!/usr/bin/env python
"""Write and SHA-256-pin ``configs/gate6/trident_semantic_precommit.yaml``.

Everything the experiment is allowed to decide is decided **here**, before a single
Gate-6 target file is opened. The file is generated from the same modules the experiment
imports (``gate6.vocab``, ``gate6.trident_adapter``), so the config cannot drift from the
code: if a phrase or a hyperparameter changed, the pinned hash would change with it.

    python tools/gate6/precommit.py
"""
from __future__ import annotations

import hashlib, json, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import yaml                                                              # noqa: E402
from gates.scale_gate.config import REPO_ROOT                                  # noqa: E402
from gates.gate6 import vocab                                                  # noqa: E402
from gates.gate6.trident_adapter import (OFFICIAL_CITYSCAPES, OFFICIAL_RESIZE_SCALE)  # noqa: E402

OUT = "configs/gate6/trident_semantic_precommit.yaml"


def load_json(rel):
    p = os.path.join(REPO_ROOT, rel)
    return json.load(open(p)) if os.path.exists(p) else {"missing": rel}


def build() -> dict:
    stage0 = load_json("artifacts/gate6/stage0_audit.json")
    prov = load_json("artifacts/gate6/teacher_provenance.json")

    cfg = {}
    cfg["gate"] = {
        "name": "Gate 6 - frozen Trident semantic lifting and coverage decomposition",
        "written_before_any_gate6_target_was_opened": True,
        "nothing_is_trained": ("no optimizer, no backward pass, no parameter update, no "
                               "prompt/threshold/fusion tuning, no teacher selection"),
        "repo_commit": stage0.get("git", {}).get("commit"),
        "seeds": {"bootstrap": 0, "permutation": 0, "global": 0},
    }

    # ---------------------------------------------------------------- teacher
    cfg["teacher"] = {
        "name": "Trident-H", "status": "primary, predeclared, frozen",
        "provenance": prov,
        "official_config": {
            "source": "configs/cfg_city_scapes.py inheriting configs/base_config.py",
            "why_cityscapes": ("the only official Trident configuration for automotive "
                               "imagery; adopted unchanged for all three automotive "
                               "benchmarks so the teacher runs at one operating point"),
            **{k: v for k, v in OFFICIAL_CITYSCAPES.items()},
        },
        "preprocessing": {
            "load": "cv2.imread of the native rectified image, BGR->RGB",
            "resize": {"op": "mmcv Resize(scale=(2048, 688), keep_ratio=True)",
                       "scale": list(OFFICIAL_RESIZE_SCALE),
                       "uniform_full_extent": True, "crop_or_pad": "none"},
            "normalise": {"mean": [0.48145466, 0.4578275, 0.40821073],
                          "std": [0.26862954, 0.26130258, 0.27577711],
                          "note": "CLIP statistics, as trident_demo.py"},
            "sam_input": ("the native file, read by the official code itself "
                          "(cv2.imread(img_path)); Cityscapes half-split enabled"),
            "ori_shape": ("native (H, W), as mmseg's LoadImageFromFile sets it before "
                          "Resize; the dense output therefore lands at native resolution"),
        },
        "sam_refinement": {
            "enabled": True, "sam_model_type": "vit_h",
            "coarse_thresh": OFFICIAL_CITYSCAPES["coarse_thresh"],
            "minimal_area": OFFICIAL_CITYSCAPES["minimal_area"],
            "sam_iou_thresh": OFFICIAL_CITYSCAPES["sam_iou_thresh"],
            "sam_mask_coff": OFFICIAL_CITYSCAPES["sam_mask_coff"],
            "down_ratio": 2, "precision": "float32",
            "upstream_defect_D1": (
                "trident.py:85 never assigns self.sam when sam_model_type=='vit_h' and "
                "sam_refinement is True (the else branch is missing), so the predeclared "
                "large configuration cannot be constructed. The upstream file is NOT "
                "edited; the float32 SAM-H is installed post-construction, which is the "
                "state the missing branch would have produced (the inline comment cites "
                "segment-anything issue #540: ViT-H overflows in fp16)."),
        },
        "prompt_mechanism": {
            "template": "prompts.imagenet_template.openai_imagenet_template",
            "n_templates": 80, "aggregation": "official: mean of L2-normalised text "
                                              "embeddings, then re-normalised",
            "one_phrase_per_class": True, "synonyms": "none",
            "alternatives_evaluated": "none",
        },
        "dense_score_extraction": {
            "point": ("the tensor the official hard label is taken from: the second "
                      "return value of seg_utils.utils.map_refinement_coarse, i.e. "
                      "refined_whole_logits, immediately before its argmax(0)"),
            "coarse_fallback": ("where refined_whole_logits.sum(0) == 0 the official code "
                                "falls back to the coarse prediction, so the dense tensor "
                                "uses the coarse softmax volume on exactly those pixels"),
            "map_failed_regions": ("runs afterwards and rewrites the returned scores, but "
                                   "provably cannot move the label (it maxes the scores "
                                   "*before* comparing against them); the returned tensor "
                                   "is therefore NOT the one the label came from"),
            "probability_conversion": (
                "coarse volume already carries the official temperature "
                "(logit_scale=40 then softmax over classes); the SAM-refined volume is "
                "non-negative and unnormalised, so it is divided by its per-pixel sum. "
                "Division by a positive scalar is strictly monotone, so the official "
                "argmax is preserved exactly. No second softmax."),
            "parity_requirement": ("argmax of the dense tensor must equal the official "
                                   "Trident prediction pixel-for-pixel on >= 100 "
                                   "deterministic images"),
        },
        "per_frame_independence": ("Trident is run once per unique RGB frame; no video "
                                   "tracking, no temporal smoothing in 2D"),
    }

    # ---------------------------------------------------------- vocabularies
    cfg["vocabulary"] = {d: vocab.load(d).to_dict() for d in vocab.DATASETS}
    cfg["vocabulary"]["rules"] = {
        "every_official_non_empty_class_exactly_once": True,
        "empty_or_free_class_in_vocabulary": False,
        "nuisance_classes_added": "none (no sky, no background)",
        "normalisation": "explicit rewrites then mechanical '-'/'_' -> space",
        "explicit_rewrites": vocab.EXPLICIT_REWRITES,
        "channel_order": "ascending official benchmark label id",
    }

    # ------------------------------------------------------------- geometry
    cfg["frozen_geometry"] = {
        "lingbot": "frozen; checkpoint sha256 "
                   + str(stage0.get("pinned_files", {})
                         .get("checkpoints/lingbot-map/204754b/lingbot-map.pt", {})
                         .get("sha256")),
        "clip_length": 5, "anchor": "last frame", "camera": "single",
        "scale": {"gauge": "G51-B", "teacher": "frozen MoGe-2, calibrated horizontal FOV",
                  "input": "full image, no aspect crop", "granularity": "one scalar per clip",
                  "applied_to": ["lingbot depth", "pose translation"],
                  "rotations": "unchanged", "distillation": "none, MoGe stays at inference"},
        "confidence_threshold": 1.5, "depth_range_m": [1.0, 60.0],
        "correction": {"name": "dilate_r2", "kind": "dilate", "radius_voxels": 2,
                       "canonical_voxel_size_m": 0.2, "physical_radius_m": 0.4},
        "forbidden": ["C3", "V3", "any learned occupancy corrector", "oracle scale"],
    }

    cfg["pixel_transform"] = {
        "chain": ["native rectified image (H_n, W_n)",
                  "-> Trident CLIP input: uniform keep_ratio resize, full extent",
                  "-> Trident dense scores at native (H_n, W_n) via ori_shape",
                  "-> LingBot processed lattice (H_p, W_p) by one align_corners=False "
                  "bilinear resize, then renormalised over classes"],
        "justification": ("both lattices are full-extent resamplings of the same "
                          "rectified image -- LingBot's anisotropic (scale_gate.kitti."
                          "Preprocess: mode='crop' degenerates to a pure resize), "
                          "Trident's uniform -- so a single full-extent resize is the "
                          "exact composition of the two pixel maps"),
        "point_pixel_provenance": ("each fused LingBot point carries (frame, v, u) on the "
                                   "processed lattice, reproducing voxel_gate.c3.c3_points "
                                   "bit-for-bit"),
        "verification": "synthetic pixel coordinates plus real overlays, in tests/gate6",
    }

    # ----------------------------------------------------------------- lifting
    cfg["lifting"] = {
        "per_point": "sample the teacher probability vector at the point's own pixel",
        "voxel_rule": ("arithmetic mean of every contributing probability vector over all "
                       "points and all five frames; NO confidence weighting, NO class "
                       "prior, NO normalisation by frame count"),
        "assignment": "argmax of the fused mean probability vector",
        "outside_occupancy": "predict empty; the frozen occupancy mask is never altered",
        "occ3d_native_reduction": ("predictions are built on the frozen canonical 0.2 m "
                                   "grid; a native 0.4 m voxel is occupied iff any of its "
                                   "eight subvoxels is (frozen rule) and takes the mean "
                                   "probability vector of its OCCUPIED subvoxels"),
        "grid_conventions": {"semantickitti": "SEMANTICKITTI_GRID, velodyne of anchor",
                             "kitti360": "SSCBENCH_KITTI360_GRID, velodyne of anchor",
                             "occ3d": "OCC3D canonical 0.2 m -> native 0.4 m, ego of anchor"},
    }

    cfg["occupancy_conditions"] = {
        "B-R": "raw G51-B reconstruction",
        "B-D": "G51-B + dilate_r2 (0.4 m)",
        "primary": "B-D", "selected_per_dataset": False,
        "dilation_propagation": {
            "sources": "raw-occupied voxels inside the same Chebyshev radius-2 neighbourhood",
            "select": "the nearest source by Euclidean distance in voxel units",
            "ties": "exact-distance ties are averaged (arithmetic mean of probability "
                    "vectors), which is deterministic and order-independent",
            "propagate": "the full probability vector, not the integer label",
            "ground_truth_used": "none",
        },
    }

    # --------------------------------------------------------------- evaluation
    cfg["evaluation"] = {
        "official_masks": {
            "semantickitti": "target != 255; occupied := target != 0",
            "kitti360": "official _1_1.npy: 255 ignore, 0 free, 1..18 occupied",
            "occ3d": ("official labels.npz semantics with mask_camera applied, plus the "
                      "frozen single-camera cut voxel_label[:100] = 255; free = 17"),
        },
        "metrics": ["binary occupancy IoU", "occupancy precision", "occupancy recall",
                    "per-class semantic IoU", "SSC mIoU excluding empty",
                    "per-class counts", "empty prediction rate",
                    "TP-conditioned top-1 semantic accuracy",
                    "TP-conditioned macro/balanced recall", "per-class recall",
                    "confusion matrix", "evaluated voxel count per class"],
        "error_decomposition": {
            "universe": "every valid ground-truth OCCUPIED voxel",
            "categories": ["coverage_miss: prediction empty",
                           "naming_error: prediction occupied, wrong class",
                           "correct: prediction occupied, right class"],
            "must_sum_to_one": True},
        "support_split": {"names": ["reconstruction_support", "dilation_only"],
                          "note": "the first is NOT called visibility"},
        "spatial_diagnostics": {
            "distance_bands_m": [[0, 10], [10, 20], [20, 30], [30, 40]],
            "height_bands": "per-dataset z bands over the grid extent",
            "reported": ["semantic mIoU by distance", "conditional accuracy by distance",
                         "coverage-miss fraction by distance",
                         "naming-error fraction by distance",
                         "raw-supported vs dilation-only accuracy"]},
        "cross_dataset_comparison": "forbidden: grids, masks and protocols differ",
    }

    cfg["negative_controls"] = {
        "vocabulary_permutation": {
            "n_permutations": 100, "seed": 0, "per_dataset": True,
            "what_is_permuted": "the class-to-text assignment only",
            "what_is_held_fixed": ("predicted occupancy, geometry, image features and the "
                                   "fused probability vectors; nothing is recomputed"),
            "reported": ["distribution and 95th percentile of full SSC mIoU",
                         "distribution and 95th percentile of TP-conditioned balanced "
                         "semantic recall",
                         "percentile of the real vocabulary within the distribution"]},
    }

    cfg["statistics"] = {
        "n_resamples": 10000, "seed": 0, "paired": True,
        "units": {"occ3d": "scene (official nuScenes validation scenes)",
                  "semantickitti": "contiguous blocks of 20 clips of sequence 08",
                  "kitti360": "the Gate-5.2 contiguous blocks of 20 clips of sequence 06"},
        "blocks_are_not_scenes": ("blocks come from a single sequence and are NOT "
                                  "described as independent scenes"),
        "b_r_vs_b_d_intervals_for": ["full SSC mIoU", "TP-conditioned semantic accuracy",
                                     "coverage-miss fraction", "naming-error fraction"],
    }

    cfg["decision_rules"] = {
        "dataset_passes_semantic_transfer_iff_all": [
            "correct-vocabulary full SSC mIoU > 95th percentile of the 100 permutations",
            "correct-vocabulary TP-balanced recall > 95th percentile of the permutations",
            "prediction generation is target-independent (auditor clean, tamper test passes)",
            "official class mapping and evaluation pass all tests"],
        "overall": {"FROZEN_TRIDENT_SEMANTICS_TRANSFER": "all three datasets pass",
                    "FROZEN_TRIDENT_SEMANTICS_PARTIAL": "one or two datasets pass",
                    "FROZEN_TRIDENT_SEMANTICS_FAIL": "no dataset passes, or a leakage or "
                                                     "protocol failure invalidates it"},
        "bottleneck": {"COVERAGE_DOMINATES": "B-D coverage-miss fraction > naming-error "
                                             "fraction on at least two datasets",
                       "NAMING_DOMINATES": "the reverse on at least two datasets",
                       "MIXED_BOTTLENECK": "neither"},
        "binary_sanity_tolerance_abs": 0.0005,
        "binary_sanity_targets": {"semantickitti_B_D": 0.1584, "occ3d_B_D": 0.2070,
                                  "kitti360_B_D": 0.1342, "kitti360_B_R": 0.0376},
        "not_rewritable_after_results": True,
    }

    cfg["leakage"] = {
        "prediction_phase_is_target_free": True,
        "auditor_forbids": [".label", ".invalid", "_1_1.npy", "_1_2.npy", "_1_8.npy",
                            "labels.npz", "velodyne", "/voxels/", "oracle_scale",
                            "scale_targets", "lidar", "gts/"],
        "allowed_inputs": ["RGB", "calibration and poses", "frozen LingBot outputs",
                           "frozen G51-B scale caches", "frozen occupancy predictions",
                           "frozen teacher weights", "dataset class schema"],
        "ordering": ("every prediction is written to immutable files and the prediction "
                     "manifest is SHA-256-pinned BEFORE the evaluator may open a target"),
        "tamper_test": ("targets and LiDAR are randomised in a scratch tree and the "
                        "deployable predictions must stay bit-identical"),
    }

    cfg["datasets"] = {
        "semantickitti": {"manifest": "artifacts/scale_gate/manifests/val.jsonl",
                          "sequence": "08", "protocol": "Gate-5.1", "n_clips_expected": 163},
        "occ3d": {"manifest": "manifests/occ3d_zeroshot/val.jsonl",
                  "protocol": "Gate-4/5.1 five-frame single-camera CAM_FRONT",
                  "n_clips_expected": 1182},
        "kitti360": {"manifest": "manifests/gate5_2/val.jsonl",
                     "sequence": "2013_05_28_drive_0006_sync", "protocol": "Gate-5.2",
                     "n_clips_expected": 1753},
        "manifest_hashes": {k: v.get("sha256") for k, v in
                            stage0.get("pinned_files", {}).items()},
        "cache_index_hashes": stage0.get("pinned_trees", {}),
    }

    cfg["technical_fallback"] = {
        "permitted_fallback": "CAT-Seg-L only",
        "status": "NOT triggered",
        "evidence": ("Trident-H official weights obtained, the official large "
                     "configuration runs in a reproducible environment, and the dense "
                     "class scores are exposed without changing the official hard "
                     "prediction (parity verified on >= 100 images)"),
        "cat_seg_used": False,
    }

    cfg["optional_comparator"] = {
        "name": "OccAny-aligned Grounded-SAM semantic readout",
        "repo": "https://github.com/valeoai/OccAny",
        "status": "attempted only AFTER the primary predictions are SHA-256-pinned",
        "may_influence_primary": False,
        "role": "comparator, not the proposed primary teacher",
    }

    cfg["figures"] = {
        "selection_rule": ("prediction-independent: fixed percentiles {10, 50, 90} of the "
                           "per-clip FROZEN B-D binary occupancy IoU of the reproduction "
                           "gate, computed before any semantic metric exists"),
        "panels": ["anchor RGB", "ground-truth semantics", "Trident 2D result",
                   "raw 3D semantic prediction", "dilated 3D semantic prediction",
                   "coverage/naming/correct error map"],
        "never_selected_by_semantic_performance": True,
    }
    return cfg


def main() -> int:
    cfg = build()
    path = os.path.join(REPO_ROOT, OUT)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = yaml.safe_dump(cfg, sort_keys=False, width=100, allow_unicode=True)
    with open(path, "w") as fh:
        fh.write("# Gate 6 precommit. Written and SHA-256-pinned BEFORE any Gate-6 target\n"
                 "# was opened. Generated by tools/gate6/precommit.py from the same modules\n"
                 "# the experiment imports, so config and code cannot drift apart.\n")
        fh.write(text)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    pin = os.path.join(REPO_ROOT, "artifacts", "gate6", "precommit_pin.json")
    with open(pin, "w") as fh:
        json.dump({"path": OUT, "sha256": digest,
                   "bytes": os.path.getsize(path)}, fh, indent=2)
    print(f"{OUT}\n  sha256 {digest}\n  bytes  {os.path.getsize(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
