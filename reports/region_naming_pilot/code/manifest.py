"""Deterministic anchor selection and the experiment manifest. Written before any method score exists.

Selection rule (fixed here, never tuned):

  * the valid source manifest is every anchor of KITTI-360 ``0006`` that has a frozen hard map --
    590 anchors, stream index ``t`` = 4, 7, ..., 1771 (stride 3);
  * the **golden** anchor is ``t = 4``, the one pinned by the legacy repository's own reproduction test;
  * the remaining anchors that share no frame with the golden clip (``t >= 9``) are cut into 28 equal
    contiguous **temporal blocks**; each block contributes its middle anchor;
  * blocks 0, 7, 14, 21 are **calibration** (4 anchors, spread over the drive); the other 24 blocks are
    **evaluation** (24 anchors, one per block, so blocks are disjoint between the two roles);
  * an anchor's clip is stream indices ``t-4..t``, so two anchors share a frame iff ``|t - t'| < 5``;
    the selection asserts this never happens between roles.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

import paths as P
import legacy as L

N_BLOCKS = 28
CALIBRATION_BLOCKS = (0, 7, 14, 21)
GOLDEN = 4
CLIP_SPAN = 5          # an anchor's clip is t-4..t


def select() -> Dict:
    ids: List[int] = L.anchor_ids(P.DRIVE)
    assert ids[0] == GOLDEN and len(ids) == 590, (ids[:2], len(ids))
    pool = [t for t in ids if t >= GOLDEN + CLIP_SPAN]
    n = len(pool)
    blocks = [pool[(k * n) // N_BLOCKS:((k + 1) * n) // N_BLOCKS] for k in range(N_BLOCKS)]
    assert all(blocks), "empty temporal block"
    chosen = {k: b[len(b) // 2] for k, b in enumerate(blocks)}
    calibration = sorted(chosen[k] for k in CALIBRATION_BLOCKS)
    evaluation = sorted(chosen[k] for k in range(N_BLOCKS) if k not in CALIBRATION_BLOCKS)

    assert len(evaluation) == 24 and len(calibration) == 4
    assert not (set(evaluation) & set(calibration)) and GOLDEN not in evaluation and GOLDEN not in calibration
    for a in calibration + [GOLDEN]:
        for b in evaluation:
            assert abs(a - b) >= CLIP_SPAN, f"calibration anchor {a} shares a frame with evaluation anchor {b}"
    return {"drive": P.DRIVE, "n_valid_anchors": len(ids), "anchor_stride": 3, "clip_span": CLIP_SPAN,
            "n_blocks": N_BLOCKS, "calibration_blocks": list(CALIBRATION_BLOCKS),
            "block_ranges": {str(k): [b[0], b[-1]] for k, b in enumerate(blocks)},
            "block_of_anchor": {str(t): k for k, b in enumerate(blocks) for t in b},
            "golden": GOLDEN, "calibration": calibration, "evaluation": evaluation}


def frozen_settings() -> Dict:
    """Everything frozen before evaluation: the model revision, prompt, generation and growth settings."""
    import proposals, extension, recognize
    return {"proposals": proposals.SETTINGS, "recognition": recognize.SETTINGS, "extension": extension.SETTINGS}


def write() -> Dict:
    anchors = select()
    man = {"experiment": "region_naming_pilot",
           "question": "With completed occupancy fixed, can a VLM name regions better than pooled Trident "
                       "evidence, and can those names be extended to occupied voxels without causal "
                       "semantic evidence?",
           "isolation": {"experiment_root": P.EXPERIMENT_ROOT, "legacy_repo": P.LEGACY_REPO,
                         "repro_repo_untouched": P.REPRO_REPO, "python": sys.executable},
           "anchors": anchors, "legacy": L.record(),
           "inputs": {"rgb_root": P.RGB_ROOT, "sam_checkpoint": P.SAM_CKPT,
                      "sam_sha256_prefix": L.sha256_file(P.SAM_CKPT)[:16],
                      "gate12_regions": P.GATE12_REGIONS,
                      "clips": os.path.dirname(L.C.Paths().clip_path(P.DRIVE, 0)),
                      "hard_map": os.path.dirname(L.C.Paths().input_path(P.DRIVE, 0)),
                      "evidence": os.path.dirname(L.C.Paths().evidence_path(P.DRIVE, 0)),
                      "trident_tokens": os.path.dirname(L.C.Paths().token_path(P.DRIVE, "x")),
                      "text_embeddings": L.C.Paths().text_embeddings,
                      "eval_masks (evaluation only)": os.path.dirname(L.C.Paths().eval_mask_path(0)),
                      "geometry_checkpoint": L.C.Paths().geometry_checkpoint(L.GEOMETRY_SEED)},
           "grid": {"dims": list(L.GRID_DIMS), "voxel_m": L.VOXEL_SIZE_M, "origin_m": list(L.GRID_ORIGIN_M),
                    "frame": "SSCBench-KITTI-360 anchor velodyne grid", "n_voxels": L.N_VOXELS},
           "outputs": {k: getattr(P, k) for k in ("PROPOSALS", "CROPS", "QWEN_CACHE", "MEMBERSHIP",
                                                  "PREDICTIONS", "SCORES", "FIGURES", "RESULTS_CSV", "REPORT")}}
    try:
        man["frozen_settings"] = frozen_settings()
    except Exception as e:                                        # settings land as the modules are written
        man["frozen_settings"] = {"pending": repr(e)}
    with open(P.under_experiment(P.MANIFEST), "w") as f:
        json.dump(man, f, indent=2, sort_keys=False)
    return man


if __name__ == "__main__":
    m = write()
    a = m["anchors"]
    print(f"golden      {a['golden']}")
    print(f"calibration {a['calibration']}")
    print(f"evaluation  {a['evaluation']}")
    print(f"blocks      {a['n_blocks']} over anchors {a['block_ranges']['0']} .. {a['block_ranges'][str(a['n_blocks']-1)]}")
    print(f"manifest -> {P.MANIFEST}")
