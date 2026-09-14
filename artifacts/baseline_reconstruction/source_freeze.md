# Source freeze -- Current Geometry-Only Baseline (L + LB-M + H, CompletionUNet-3ch)

Frozen at **2026-09-14T12:21:03+02:00** (git HEAD `0ba3b5deeb23`, 58 uncommitted paths in the working tree). From this moment no code, config, checkpoint, mapper setting, threshold, adapter or evaluation script may change; `require_source_freeze()` re-hashes every file below before any target evaluation and refuses to run on a mismatch.

## Reason

0006 result confirmed against Gate 13 M within the predeclared tolerance

## Frozen decisions

| decision | value |
|---|---|
| seeds | [0, 1, 2] |
| checkpoint_rule | final step 6000 of every seed; no selection |
| occupancy_threshold | 0.0 |
| seed_statistic | mean over seeds primary; median secondary |
| train_drives | ['2013_05_28_drive_0003_sync', '2013_05_28_drive_0007_sync', '2013_05_28_drive_0010_sync'] |
| val_drive | 2013_05_28_drive_0006_sync |
| clip_offsets | [-4, -3, -2, -1, 0] |


## Source result at freeze time (full-region 0006 IoU)

| seed | this run | Gate 13 M | abs diff |
|---|---|---|---|
| 0 | 0.1365 | 0.1354 | 0.0011 |
| 1 | 0.1346 | 0.1336 | 0.0010 |
| 2 | 0.1343 | 0.1346 | 0.0003 |


mean abs diff 0.0006 (tolerance {'mean_full_iou_abs': 0.003, 'per_seed_full_iou_abs': 0.005}) -> PASS


## Hashed files (54)

| file | sha256 |
|---|---|
| tools/baseline_reconstruction/__init__.py | e3b0c44298fc1c14 |
| tools/baseline_reconstruction/cli/__init__.py | e3b0c44298fc1c14 |
| tools/baseline_reconstruction/cli/build_target_clips.py | 8d6e989d747e638e |
| tools/baseline_reconstruction/cli/eval_source.py | 8b024a6acfa1a257 |
| tools/baseline_reconstruction/cli/eval_targets.py | a4568e5fc6eb1f15 |
| tools/baseline_reconstruction/cli/freeze_source.py | ec8ad521f140dc6a |
| tools/baseline_reconstruction/cli/train_current.py | 94cfc3459e5f4134 |
| tools/baseline_reconstruction/config.py | 7d908805c58536d1 |
| tools/baseline_reconstruction/current/__init__.py | e3b0c44298fc1c14 |
| tools/baseline_reconstruction/current/dataset.py | 16db6ff7a11891dc |
| tools/baseline_reconstruction/current/evaluate_source.py | 264e4a9a45f0c383 |
| tools/baseline_reconstruction/current/geometry.py | b6bd60ae78b23731 |
| tools/baseline_reconstruction/current/recipe.py | 1b726d7ae071e40b |
| tools/baseline_reconstruction/current/train.py | fe7acc203a3d0b2a |
| tools/baseline_reconstruction/freeze.py | 9da32c5f4e7fb2bf |
| tools/baseline_reconstruction/metrics/__init__.py | e3b0c44298fc1c14 |
| tools/baseline_reconstruction/metrics/comparability.py | d5d0c723fb2a2974 |
| tools/baseline_reconstruction/metrics/occupancy.py | c2ddc7b22a8059ae |
| tools/baseline_reconstruction/targets/__init__.py | e3b0c44298fc1c14 |
| tools/baseline_reconstruction/targets/adapters.py | 6c50e56c93b6b257 |
| tools/baseline_reconstruction/targets/clips.py | 74c6a86331130260 |
| tools/baseline_reconstruction/targets/evaluate.py | e80299111461303d |
| core/mapping/mapper.py | 9c04e302e0ec0e08 |
| core/mapping/depth.py | ac0429d674404391 |
| core/mapping/rays.py | f46dc371869a226b |
| core/mapping/voxmap.py | 098d19b07dbb3885 |
| core/mapping/scale.py | bbad06548cfc523d |
| core/model/net.py | 8850ba29d42dca0d |
| core/model/losses.py | f6dd7f58648e2f4d |
| core/model/sampler.py | b0374c4ebb76e18c |
| core/run/_gate8a_train.py | aa1330817a43ee33 |
| core/run/_gate8_train.py | 76d6da2bdba8241c |
| core/datasets/sources.py | 76cbeb17c30f4924 |
| core/datasets/streams.py | 0296414ba75ede26 |
| core/datasets/frames.py | ab50655d9411bd5b |
| core/datasets/grids.py | f0f0256d42bda9d7 |
| core/datasets/occ3d_grid.py | 79ad30acbbc39a53 |
| core/datasets/occupancy.py | c722c11db821b1ec |
| core/datasets/kitti360_splits.py | 7b9ac416144103f6 |
| core/datasets/audit.py | f6e9ee191b4ea348 |
| core/evaluation/official_targets.py | c896a893b7ed9f66 |
| core/evaluation/regions.py | 4e201fa1b38bc2a2 |
| core/datasets/occ_datasets.py | 9db1474a80d7f286 |
| core/datasets/nuscenes.py | a129a0c29ea8fc83 |
| core/datasets/kitti_odometry.py | 3b742aec7a8c742a |
| core/supervision/rawtarget.py | b59c9f4d0c862d81 |
| gates/gate7b/replay.py | 9136b474b4077e58 |
| gates/gate7b/scale.py | 2b0b2db98700dbc2 |
| gates/gate7b/depth.py | d9ce5297a3707f7b |
| checkpoints/lingbot-map/204754b/lingbot-map.pt | ee665103348e07e6 |
| artifacts/baseline_reconstruction/training_config.yaml | 4ef021b644485387 |
| artifacts/baseline_reconstruction/checkpoints/geo_seed0_step06000.pt | 7f5e853dd05bdbde |
| artifacts/baseline_reconstruction/checkpoints/geo_seed1_step06000.pt | 2c4d79be4fa7766e |
| artifacts/baseline_reconstruction/checkpoints/geo_seed2_step06000.pt | 6038b1f7d901a751 |


Render-only code excluded from the hash set (may be edited without affecting any prediction or metric): tools/baseline_reconstruction/reports/, tools/baseline_reconstruction/historical/, tools/baseline_reconstruction/cli/audit_history.py, tools/baseline_reconstruction/cli/report.py, tools/baseline_reconstruction/README.md


Target datasets opened before this freeze: **no** (every source-side stage ran under a strict file-access audit; see verification.md).
