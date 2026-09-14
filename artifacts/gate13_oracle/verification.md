# Gate 13 verification

22/22 checks pass.

| check | result | detail |
|---|---|---|
| 01_M_is_gate12_H_input | PASS | M reads the Gate 12 H cache (S_H, F_H) unchanged since Gate 12's code freeze (all 1358 training inputs older than the freeze); M's anchor list, draws and target / loss-mask digests equal Gate 12 H's on every seed |
| 02_M_reproduces_gate12_H_validation | PASS | 0006 full IoU Gate 12 H [0.1343, 0.1331, 0.1341] vs Gate 13 M [0.1354, 0.1336, 0.1346], diff [0.0011, 0.0005, 0.0005] (tolerance per seed 0.003, mean 0.0015, fixed before training) |
| 03_identical_anchors | PASS | 1358 training anchors (Gate 12's), identical per seed; all evaluations on the same 590 0006 anchors |
| 04_same_target_tensor | PASS | sha256 over every drawn target crop (occupancy, valid, loss-weight mask) identical for M and O at every 1000-step checkpoint |
| 05_same_crops | PASS | sha256 over every (anchor, crop origin) identical per seed: 8c81c7be4f20, f6e7b8871313, 41c39b03b97b |
| 06_same_augmentations | PASS | no augmentation exists in the training path; crop RNG draws identical |
| 07_same_initial_weights | PASS | per seed M, O and Gate 12 H load the same saved initial weights (identical state digests) |
| 08_same_optimizer | PASS | {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 0.01}, CosineAnnealingLR |
| 09_same_lr_sequence | PASS | identical learning-rate value at every step for M and O |
| 10_same_architecture | PASS | CompletionUNet cin 3 width 24 replicate padding; identical parameter names and shapes in all six checkpoints |
| 11_same_parameter_count | PASS | [967322] parameters |
| 12_same_training_steps | PASS | 6000 steps, batch 4 |
| 13_same_checkpoint_rule | PASS | configs M / O differ only in ['input'] and M differs from Gate 12 H only in ['input']; every evaluated checkpoint is the final step |
| 14_O_only_sweeps_t-4_to_t | PASS | all 1948 oracle inputs (one per Gate 12 anchor, 1948) record stream indices t-4..t and exactly those frames' native sweeps |
| 15_O_no_future_sweeps | PASS | largest recorded input stream index minus anchor = 0; the builder's only sweep list is t-4..t |
| 16_O_no_target_voxels | PASS | the oracle builder never reads a target, evaluation mask, region or label file (strict file audit, violations [0, 0, 0, 0, 0]); its only geometry is the five sweeps' camera z-buffer through the mapper |
| 17_O_GT_relative_poses_input_only | PASS | oracle T_j->t equal GT inv(rect_cam_to_world(t)) rect_cam_to_world(j) on 21/21 sampled inputs; GT poses enter only the oracle input builder (M inputs contain none) |
| 18_O_same_hard_mapper | PASS | O uses VariantMapper(band 0.2, margin 0.2, carving) == core IncrementalMapper bit for bit (the mapper of M's S_H/F_H); O's occupied map equals Gate 9's LD map on [590, 590] 0006 anchors |
| 19_same_grid | PASS | both inputs queried on the SSCBench-KITTI-360 anchor grid 256x256x32 at 0.2 m (TF.GRID) |
| 20_no_semantic_information | PASS | w_sem = 0, zero semantic target and mask, no semantic input channel; training and oracle building audited against trident / semantics paths |
| 21_no_target_dataset_access | PASS | training, evaluation and oracle input generation ran inside file audits forbidding SemanticKITTI / Occ3D / nuScenes paths: no violations |
| 22_deterministic_input_generation | PASS | rebuilding oracle inputs for 0006 anchors 4 / 766 / 1747 reproduces the stored S_O and F_O exactly: [True, True, True] |
