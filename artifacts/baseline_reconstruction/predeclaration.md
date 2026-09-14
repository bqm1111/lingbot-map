# Predeclaration -- written before any current-baseline training run (2026-09-14)

Phase 1 (historical audit) is complete and verified (`historical_baseline_audit.md`, 6/6 checks).
Everything below is fixed now; nothing in it may change after inspecting a result.

## Phase 2 -- current geometry-only baseline

| item | frozen value | source of truth |
|---|---|---|
| geometry | local-reset five-frame LingBot (L), frames t-4..t, clip-local indexing, no stream cache | frozen clip caches (`config.CurrentPaths.clips_*`) |
| depth | LB-M = s_clip · z_LingBot, s_clip = exp(median of the clip's five MoGe-2 log-scale candidates), applied once by the mapper | `current/geometry.py` |
| pose | local LingBot poses inv(P_4)·P_q, anchor = identity; translation scaled by the same s_clip inside the mapper | `current/geometry.py` |
| mapper | `core.mapping.mapper.IncrementalMapper`, semantics off: conf ≥ 1.5, 1–60 m, ±0.2 m band, carve to d−0.2 m at stride 4 | `core/mapping/*` (unchanged) |
| input | [S, 0, F] — the frozen Gate 12 H cache (`lingbot_gate12_suf/inputs`), no U channel, no semantics | `current/dataset.py` |
| targets | `lingbot_k360_7drives/targets` (current `core/supervision/rawtarget.py`: 647bcf3 fix + neighbourhood-7 sky rule) — the Gate 12/13 targets, unchanged | Gate 12 |
| network | `CompletionUNet(cin=3, width=24, padding_mode="replicate")`, 967,322 params | `core/model/net.py` |
| recipe | `training_config.yaml` = verbatim `artifacts/gate12_suf/training_config_H.yaml` (focal+Dice γ2 cap 8 w_unknown 2; AdamW 1e-3 wd 0.01; cosine; clip 1; batch 4; bf16; uniform 128×128×32 crops; 6000 steps) | `training_config.yaml` |
| seeds | 0, 1, 2 (same as Gates 12/13); no seed added afterwards | `config.SEEDS` |
| checkpoint rule | final step 6000 of every seed; no selection of any kind | `config` / freeze |
| decision threshold | occupied := final log-odds > 0 with base log-odds 0 (Gate 12/13 rule). NOTE: differs from the historical per-seed source-selected thresholds (−0.125 / −0.4375 / −0.3125); this is recorded as a protocol difference, not tuned | `config.OCCUPANCY_THRESHOLD` |
| seed statistic | **mean over the three seeds is primary**; median and sd always reported (the historical headline is a median) | `metrics.occupancy.seed_statistics` |
| source reproduction gate | mean full-region 0006 IoU within 0.003 of Gate 13 M (0.1345) and every seed within 0.005 of its Gate 13 M counterpart (0.1354 / 0.1336 / 0.1346). FAIL ⇒ STOP, diagnose, no target evaluation | `config.SOURCE_REPRODUCTION_TOLERANCE` |

## Phase 3 -- zero-shot protocol (frozen before any target file is opened)

| item | SemanticKITTI | Occ3D-nuScenes |
|---|---|---|
| anchors | the 163 official clips of sequence 08 (stride-5 stream, anchor = last frame) | the 1182 official CAM_FRONT anchors of the 150 validation scenes |
| causal window | stream indices i−4..i (the historical past5 rule; anchors with i < 4 skipped) | same |
| LingBot | local-reset clip inference on the five target images (`targets/clips.py`), same preprocessing as the stream | same |
| scale | causal MoGe-2 clip scale from the frozen per-frame MoGe-B caches of the five frames | same |
| grid | 256×256×32 @ 0.2 m velodyne frame; T_cam_to_grid = official cam→velo calibration | predict 400×400×32 @ 0.2 m ego frame; T_cam_to_grid = T_camera_to_ego; any-subvoxel reduction to 200×200×16 @ 0.4 m |
| mask | official `.invalid == 0` | `mask_camera`, lidar mask off, x-index < 100 ignored (frozen single-camera rule of Gates 4–8) |
| target | label ≠ 0 | semantics ≠ 17 |
| metric | SC IoU, precision, recall; range bins by anchor-camera z; regions full / H_evidence / no_H_evidence; map-only (S) as reference | same |
| seeds | mean of 3 (median, sd reported) | same |
| what may change after seeing a target number | nothing | nothing |

Semantic mIoU is unavailable: the current baseline is geometry-only. This is stated, not worked around.

## Comparability classification (applied after the numbers exist)

* **fully comparable** only if target definition, mask, grid, camera setup and metric computation are identical *and* the two systems differ in no material pipeline property.
* **partially comparable** if the headline metric (SC IoU on the same official anchors/masks/grids) is identical but material pipeline differences remain — this is the expected class for both targets, since the historical model differs in geometry mode (long-stream vs local-reset), scale (stream vs clip), input representation (32-ch + Trident-H semantics vs 3-ch), residual lock, threshold rule, training targets (plain-column sky rule vs neighbourhood-7 + 647bcf3) and validation-window statistic.
* **not comparable** if the evaluation target/mask/grid/anchor set differs.

## Decision rules (exactly one)

* **A** — source reproduces; both target evaluations valid; current mean SC IoU ≥ the canonical historical median on both targets (or within one seed-sd); no unexplained cross-dataset pathology.
* **B** — source reproduces; target evaluations valid; current below historical on at least one target, and the difference is consistent with the documented pipeline differences (semantics/32-ch/residual lock/threshold/targets/geometry mode); the baseline remains internally trustworthy (completion beats its own map-only reference on both targets).
* **C** — source reproduces and protocols are valid, but on at least one target the completion does **not** beat its own map-only reference (completion IoU ≤ map-only IoU), or a cross-dataset asymmetry appears that the documented differences cannot explain.
* **D** — the historical protocol could not be reconstructed with confidence. (Already excluded by Phase 1: it was reconstructed and verified.)

## Known risks declared in advance

1. The current training targets use the neighbourhood-7 sky rule that Gate 8C-1's addendum rejected for target transfer (Occ3D 33.28 → 28.12, single seed, pre-647bcf3). Gates 12/13 adopted it as the repository's builder; it is kept unchanged here. If Occ3D falls relative to the historical number, this is the first documented candidate explanation — it is **not** a licence to rebuild targets after seeing the result.
2. Threshold 0 vs the historical negative thresholds shifts the precision/recall operating point.
3. Local-reset clips on SemanticKITTI span 20 native frames (stride-5 stream) and on Occ3D 2 s of keyframes, versus 5 consecutive frames on KITTI-360 — a larger baseline per clip than the model saw in training. Inherited from the historical protocol; unchanged.
