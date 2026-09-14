# Gate 11 verification

15/15 checks pass (drives: 0006, 0003, 0007, 0010).

| check | result | detail |
|---|---|---|
| 01_same_clips | PASS | Gate 11 clips == local-reset clips == Gate 9 clips == depth-source-comparison clips (0006 590, 0003 65, 0007 190, 0010 199) |
| 02_same_anchor_frames | PASS | stream indices / frame keys equal the local-reset clip and Gate 9 record on 1044/1044 clips |
| 03_same_gt_poses | PASS | T_j->t recomputed from GT on every 10th clip, equal to Gate 9 elsewhere (1044/1044); metric translations, mapper scale forced to 1 in both channel mappers |
| 04_same_lbm_depth | PASS | LB-M depth and LiDAR depth of every stored pixel row equal Gate 9's rows (same deterministic subsample) on 1044/1044 clips; LB-M = cached LingBot x clip scale, once |
| 05_same_moge_clip_scale | PASS | clip scale == local-reset clip file == Gate 9 on 1044/1044 clips (causality / single application audited in artifacts/depth_source_comparison/scale_audit.json) |
| 06_same_confidence | PASS | LingBot confidence and Gate 9 edge class of every stored row equal Gate 9 on 1044/1044 clips; frozen gate conf >= 1.5 |
| 07_same_grid | PASS | 256x256x32, 0.2 m, origin (0, -25.6, -2) == core GRID |
| 08_same_evaluation_masks | PASS | evaluation-mask and LiDAR-support digests equal Gate 9 on 1044/1044 clips |
| 09_same_camera | PASS | lattice K equal Gate 9 and lattice_K(P_rect_00) on 1044/1044 clips |
| 10_widths_from_residuals_not_iou | PASS | derive_rules.py reads only ['pixel_fields', 'pixels'] from Gate 9 records (depth residuals, confidence), never an occupancy array; rules derived at 2026-09-13T21:36:24 before the first Gate 11 occupancy record; re-derivation reproduces a, b, g, q16, q84 and the input-row digest exactly: True; U2 a=0.0658 b=0.1270, g=[1.502, 1.164, 0.978, 0.879], q16/q84=-0.308/1.395 |
| 11_rules_frozen_before_replay | PASS | code_freeze.json at 2026-09-13T21:47:26: all 11 frozen files (code + uncertainty_rules.json) unchanged; earliest 0003/0007/0010 record written after the freeze: True |
| 12_no_target_dataset_access | PASS | every audit shard inside the strict file-access audit: violations [0, 0, 0, 0, 0, 0, 0]; forbidden tokens in Gate 11 code: [] |
| 13_no_training | PASS | no optimiser / backward / parameter / network load in Gate 11 code (only cached depth and confidence are read); LingBot checkpoint sha256 ee665103348e07e6 unchanged |
| 14_H_is_the_unchanged_mapper | PASS | ChannelMapper(surface, margin 0.2) == Gate 9 VariantMapper bit for bit (40,776 rows); H metric vector == Gate 9 PD_gate (and H_LD == LD_gate) on 1044/1044 clips |
| 15_deterministic_rerun | PASS | re-running two 0006 clips reproduces every stored array (counts, metric vectors, pixel states) exactly: [True, True] |
