# Verification -- Current Geometry-Only Baseline (L + LB-M + H, CompletionUNet-3ch) (2026-09-14)

**32 / 32 checks pass.**

## Historical baseline

| # | check | result | evidence |
|---|---|---|---|
| 1 | exact ceiling-fix commit/code identified | PASS | sky targets == pre-647bcf3 rawtarget.py with SKY_NEIGHBOURHOOD_VOX=0 (bit-exact, 3 anchors); fix applied 2026-09-04, first tracked at f7b7552; historical_baseline_audit.md §3 |
| 2 | pre-fix and post-fix evaluation distinguished | PASS | prefix manifest checkpoints carry zeros padding, sky manifest replicate; separate eval records (artifacts/gate8c1 vs output/skyfix/eval) |
| 3 | historical numbers traced to files/logs | PASS | six eval JSONs + RESULT log lines agree: SK 22.47/23.62/22.71, Occ3D 33.49/35.03/33.88 |
| 4 | exact checkpoint identified | PASS | sky_seed{0,1,2}_last.pt, sha256 == frozen_manifest_sky.json |
| 5 | exact config identified | PASS | configs/gate8c1_sky/seed{0,1,2}.yaml (checkpoint meta carries the config; historical_protocol.yaml) |
| 6 | metric definition identified | PASS | semantic.completion.binary_iou = TP/(TP+FP+FN) inside the official mask (core/run/eval_target.py; re-derived from the seed-2 dump to 7e-7) |
| 7 | no semantic IoU confused with occupancy IoU | PASS | SSC mIoU (3.35 / 4.24) reported separately from SC IoU in the audit |
| 8 | conflicting results documented | PASS | historical_baseline_audit.md §2: OccAny-scorer subsets (22.83 / 33.28), matched-forward 30.70, sky2 28.12 / 22.93, overwritten JSONs |
| 9 | canonical baseline justified | PASS | historical_baseline_audit.md §7: validity (full anchor sets, gate's masks/metric, METHOD.md headline), not magnitude |


## Current baseline

| # | check | result | evidence |
|---|---|---|---|
| 1 | only KITTI-360 0003/0007/0010 used for training | PASS | training_config train_drives = ['2013_05_28_drive_0003_sync', '2013_05_28_drive_0007_sync', '2013_05_28_drive_0010_sync']; n_train = 1358 = |0003|+|0007|+|0010| input anchors in every run record |
| 2 | 0006 used source-side only | PASS | source eval: 590 anchors of 0006 per seed; training-time validation loss on every 5th 0006 anchor is logged, never used for selection |
| 3 | local-reset five-frame L inference | PASS | input cache records the clip sha256 it was built from; clip stream_indices = t-4..t: {'anchor': 4, 'S_equal': True, 'F_equal': True, 'clip_sha_matches_cache': True, 'scale_is_exp_nanmedian_log_s': True, 'rope_slots_used': 0, 'stream_indices': [0, 1, 2, 3, 4]} |
| 4 | no long-stream cache | PASS | clip rope_slots_used = 0 (five frames run as the batched scale block after clean_kv_cache; run_local_lingbot.py asserts the RoPE counter is 0 per clip) |
| 5 | LB-M depth (LingBot canonical depth x causal MoGe-2 clip scale) | PASS | S_H/F_H of anchor 4 re-derived from the clip through core IncrementalMapper: bit-exact = True |
| 6 | scale applied exactly once | PASS | scale enters only via mapper.scale_state.force(s_clip) (current/geometry.py); tests/baseline_reconstruction/test_geometry_scale_once.py; s_clip = exp(nanmedian log_s) |
| 7 | local LingBot pose (anchor = identity) | PASS | current/geometry.py frame_inputs: pose = inv(P_4) @ P_q; test_translation_is_scaled_with_depth |
| 8 | hard H mapper unchanged | PASS | core hashes vs Gate 12 code_freeze.json: {'core/mapping/mapper.py': True, 'core/run/_gate8a_train.py': True, 'core/model/net.py': True, 'core/model/losses.py': True, 'core/model/sampler.py': True} |
| 9 | no semantics | PASS | w_sem = 0, cin = 3, 967,322 params in every run record; no Trident path opened (firewall tokens 'trident', 'semantics') |
| 10 | same three seeds, same initial weights as Gate 12 | PASS | seeds [0, 1, 2]; init_matches_gate12 = [True, True, True] |
| 11 | frozen training recipe (verbatim Gate 12 H) | PASS | training_config.yaml body == artifacts/gate12_suf/training_config_H.yaml body; run records carry its sha256 4ef021b64448 |
| 12 | final-step (6000) checkpoint policy | PASS | run records: final_checkpoint = geo_seed*_step06000.pt; no selection code exists |
| 13 | source result matches the Gate 13 M regime | PASS | mean abs diff 0.0006 (tol 0.003); per seed {"0": 0.0011, "1": 0.001, "2": 0.0003} |
| 14 | training and source evaluation ran under a strict file audit with zero violations | PASS | train opened [5906, 5906, 5906] files, 0 violations; eval 0 violations |


## Zero-shot firewall

| # | check | result | evidence |
|---|---|---|---|
| 1 | target datasets unopened before the source freeze | PASS | freeze 2026-09-14T12:21:03+02:00; earliest target clip file 2026-09-14T12:21:31; every source-side stage ran inside FileAudit with the dataset-directory tokens |
| 2 | source code/config hashes frozen before target evaluation | PASS | 54 frozen files re-hashed: unchanged |
| 3 | SemanticKITTI is evaluation-only | PASS | opened only by targets/clips.py and targets/evaluate.py, both behind require_source_freeze(); 163 anchors |
| 4 | Occ3D-nuScenes is evaluation-only | PASS | opened only by targets/clips.py and targets/evaluate.py, both behind require_source_freeze(); 1182 anchors |
| 5 | no target-based threshold changes | PASS | threshold 0.0 in the freeze and in every zero-shot record; evaluation code has no threshold argument |
| 6 | checkpoints evaluated are the frozen ones | PASS | zero-shot records carry the checkpoint sha256 read from code_freeze.json |
| 7 | no semantic inputs at evaluation | PASS | targets/evaluate.py builds [S, 0, F] only; adapters expose occupancy targets only |
| 8 | exact historical protocol comparison documented rather than assumed | PASS | historical_comparability.md + tables/historical_vs_current.csv, mechanical classification |
| 9 | one evaluation per target, no re-runs with alternatives | PASS | exactly one zero_shot_<dataset>.json per target |


Tests: `tests/baseline_reconstruction/` (metrics, historical parsing, ceiling-fix constants, scale-once geometry, freeze guard, comparability, protocol serialization).
