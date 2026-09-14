# Verification record

15/15 checks pass (drives: 0006, 0003, 0007, 0010).

| check | result | detail |
|---|---|---|
| 01_same_clips | PASS | comparison clips == local-reset clips == Gate 9 clips (0006 590, 0003 65, 0007 190, 0010 199) |
| 02_same_anchors | PASS | stream indices and frame keys equal the local-reset clip file and Gate 9 record on 1044/1044 clips |
| 03_identical_pixels_and_weights | PASS | Gate 9 LiDAR support digest equal on 1044/1044 clips; S1 / S0 built from support, MoGe validity, LingBot finiteness and the frozen conf gate BEFORE any depth array is formed, so the mask cannot depend on the compared depth values; one support line and one conf line (SUPPORT_CONF inside the mask, 0 outside) for every run; depth metrics use the same rows (in_S1 / in_S0 flags) for every source |
| 04_same_gt_relative_poses | PASS | T_j->t recomputed from GT cam0_to_world on every 10th clip, equal to Gate 9 on the rest (1044/1044); metric translations enter the mapper with scale forced to 1 (translation x 1) |
| 05_same_camera_and_moge_convention | PASS | lattice K equal to Gate 9 and to lattice_K(P_rect_00) on 1044/1044 clips; MoGe-2 received fov_x [103.7448] deg == 2 atan(518 / (2 fx 518/1408)) {'0006': 103.7448, '0003': 103.7448, '0007': 103.7448, '0010': 103.7448} on the LingBot-preprocessed 140x518 RGB, so MoGe and LingBot depth share the lattice without resampling; MoGe depth is optical-axis z in metres (moge_gauge/calibrated.py asserts depth == points[...,2] at inference), so no ray-to-z conversion and no LingBot scale is applied to it |
| 06_same_evaluation_masks | PASS | evaluation-mask digest and size equal to Gate 9 on 1044/1044 clips |
| 07_same_voxel_grid | PASS | dims (256, 256, 32), 0.2 m, origin (0.0, -25.6, -2.0) == core GRID |
| 08_same_mapper_code | PASS | Gate 9 variant_mapper.py / audit.py / common.py match Gate 9's code freeze; core/mapping unchanged vs HEAD; VariantMapper(band 0.2, margin 0.2, carve) == core IncrementalMapper bit for bit (40,776 rows); one construction, no parameter changed |
| 09_gate9_reproduced | PASS | Gate 9 LD, PD, LD_gate and PD_gate metric vectors reproduced exactly on 1044/1044 clips (original numbers kept separately labelled) |
| 10_frozen_lingbot | PASS | checkpoint sha256 ee665103348e07e6 == registration audit's; fresh local-reset re-inference reproduces the cached depth bit for bit on clips [4, 16, 1549, 1747]; no optimiser / backward / parameter |
| 11_frozen_moge | PASS | Ruicheng/moge-2-vitl@39c4d5e9 model.pt sha256 3eefd4abb2102f38 == Gate 5 record; 0 trainable; re-inference on ['0006:0000000137', '0006:0000004889', '0003:0003_0000000027', '0003:0003_0000000522', '0007:0007_0000000022', '0007:0007_0000001447', '0010:0010_0000000034', '0010:0010_0000001631'] matches the cache (median rel 1.8e-04, p99 4.4e-04; tolerances {'median_rel': 0.001, 'p99_rel': 0.01}) |
| 12_lbm_scale_once_causal_no_lidar | PASS | clip scale = exp(median over frames t-4..t of frame_candidate(MoGe-2, LingBot)) recomputed exactly on 43/43 sampled clips; scale_audit.json: Gate 9 depth == cached x scale, mapper forced 1, Gate 9 map == deployable convention map (0 voxel XOR) on all audited clips; LB-M applies canon x scale once; run_local_lingbot.py has no LiDAR access |
| 13_lbo_oracle_isolated | PASS | the LiDAR-fitted scalars are computed once per clip (median LiDAR / cached LingBot over the mask) and used only in the LB-O depth entries; masks, LB-M, MG, LD and the evaluation never read them |
| 14_no_target_dataset_access | PASS | all comparison shards ran inside the strict file-access audit (SemanticKITTI / Occ3D / nuScenes / SSCBench labels forbidden): violations [0, 0, 0, 0, 0, 0, 0]; frames without MoGe [0, 0, 0, 0, 0, 0, 0]; forbidden tokens in code outside the allow-listed report disclaimer: []; allow-listed prose lines: 1 (amended after the first run flagged the disclaimer string) |
| 15_deterministic_rerun | PASS | re-running two 0006 clips reproduces every stored array exactly: [True, True] |
