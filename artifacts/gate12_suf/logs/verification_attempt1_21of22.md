# Gate 12 verification

21/22 checks pass.

| check | result | detail |
|---|---|---|
| 01_same_clips | PASS | training anchor list digest equal for H and S/U/F on every seed (1358 anchors); all six evaluations on the same 590 0006 anchors |
| 02_same_training_targets | PASS | sha256 over every drawn target crop (occupancy, valid, loss-weight observed mask) identical for H and S/U/F on each seed, at every 1000-step checkpoint: seed 0 55abfec10807, seed 1 d1d54bda3b7e, seed 2 c63bd790ce58 |
| 03_same_crop_coordinates | PASS | sha256 over every (anchor, crop origin) draw identical per seed: seed 0 8c81c7be4f20, seed 1 f6e7b8871313, seed 2 41c39b03b97b |
| 04_same_augmentations | PASS | the training pipeline has no augmentation (no augmentation tokens); crops identical per seed |
| 05_same_optimizer | PASS | {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 0.01} for every run |
| 06_same_schedule | PASS | CosineAnnealingLR, identical learning-rate sequence at every step for H and S/U/F |
| 07_same_steps | PASS | 6000 steps, batch 4, every run |
| 08_same_architecture | PASS | CompletionUNet cin 3, width 24, padding replicate, identical parameter names and shapes in all six checkpoints; shared initial weights per seed: True |
| 09_same_parameter_count | PASS | 967,322 parameters in every run |
| 10_same_checkpoint_rule | PASS | the two configs differ only in ['input']; every evaluated checkpoint is the final step; no selection |
| 11_local_reset_L_pose_not_GT | PASS | poses = inv(P_4) P_q of LingBot's local-reset pred_pose_c2w; the input builder never constructs GT geometry and ran inside a file audit forbidding data_poses / cam0_to_world: violations [0, 0, 0, 0, 0] |
| 12_dense_rays_no_lidar_mask | FAIL | rays accepted by the mapper equal every confident dense LingBot pixel in 1-60 m (no LiDAR support mask): ['0006:4 303,082/362,600', '0006:592 302,065/362,600', '0006:1180 317,570/362,600', '0003:4 201,268/362,600', '0003:68 282,863/362,600', '0003:132 162,873/362,600', '0007:4 270,590/362,600', '0007:193 269,532/362,600', '0007:382 324,582/362,600', '0010:4 263,190/362,600', '0010:202 270,463/362,600', '0010:400 262,766/362,600'] |
| 13_causal_moge_scale_once | PASS | input scale == L clip scale on 79/79 sampled anchors; the mapper receives canonical depth and canonical poses and applies s_clip once to both (force(s)); stride-1 L clips regenerated for training are bit-identical to the local-registration audit's stride-3 clips on 454/454 overlapping anchors (the clip scale's causality and single application are audited in artifacts/depth_source_comparison/scale_audit.json) |
| 14_confidence_gate_unchanged | PASS | mapper confidence threshold 1.5 (core constant); raw LingBot confidence passed for H; for S/U/F only the 20 m cut zeroes confidence |
| 15_U4_rules_frozen | PASS | every sampled input records rules sha256 4dfd83dcb443240e == Gate 11's frozen file; coefficients a 0.0658 b 0.1270 g [1.502, 1.164, 0.978, 0.879] u_front 0.3084 u_back 1.3945 (round to the brief) |
| 16_cutoff_input_only | PASS | no S/U/F ray >= 20 m in any of 1948 input files; training and evaluation read the unchanged full target (no range crop); 10,017,445 target-occupied voxels beyond 20 m are scored on 0006 |
| 17_no_semantickitti_access | PASS | training, evaluation and input generation ran inside file audits forbidding the SemanticKITTI path (data/kitti/dataset) and 'semantickitti': no violations |
| 18_no_occ3d_access | PASS | no Occ3D / nuScenes path in Gate 12 code; audits forbid occ3d / nuscenes tokens; no violations |
| 19_no_semantic_supervision | PASS | w_sem = 0; the semantic target passed to the unchanged loss is all zeros with a zero mask; training runs inside an audit forbidding trident / semantics paths; the input has no semantic channel |
| 20_deterministic_input_generation | PASS | sanity_inputs.py rebuilt 9 deterministic anchors from their L clips and reproduced all five stored volumes exactly |
| 21_pre_training_sanity | PASS | 7/7 input sanity checks (H U = 0 and S/F = core mapper; exclusivity; 20 m; carving stop; U4 rules; no LiDAR / GT pose) |
| 22_input_code_frozen | PASS | input-generation code unchanged since code_freeze.json (frozen before training) |
