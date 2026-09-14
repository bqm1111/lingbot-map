# Gate 14 verification

20/20 checks pass (drives: 0006, 0003, 0007, 0010).

| check | result | detail |
|---|---|---|
| 01_same_clip_anchors | PASS | per drive, Gate 14 records == local-reset clips == Gate 9 clips == MUSt3R cache (1044 clips over 0006, 0003, 0007, 0010); all four conditions of a clip come from one record |
| 02_same_image_frames | PASS | MUSt3R cache, local-reset clip and Gate 14 record carry the same five frame keys; manifest image paths are those frames' rectified image_00 files (1044/1044) |
| 03_no_future_images | PASS | stream indices t-4..t in every MUSt3R cache, record and manifest (1044/1044); MUSt3R memory built in temporal order [2, 1, 1, 1] |
| 04_lbm_scale_applied_once | PASS | LB-M: canonical depth with mapper scale s_clip under L (depth and translation scaled once by the mapper), canonical x s_clip with mapper scale 1 under G; s_clip equals the local-reset clip scale on 1044/1044 clips |
| 05_candidate_frozen | PASS | official must3r load_model (strict=True), all parameters requires_grad False, no optimiser / backward; checkpoint MD5 8854f948a8674fb1740258c1872f80dc recorded in 1044/1044 caches |
| 06_candidate_no_lidar | PASS | must3r_infer.py imports nothing from this repository and reads only the manifest and the listed RGB images (all under data_2d_raw/image_00); no LiDAR token in its code |
| 07_candidate_no_semantic_labels | PASS | no label / semantic / SSCBench access in the MUSt3R inference code |
| 08_candidate_no_target_calibration | PASS | MUSt3R receives only images (no intrinsics / FOV / calibration); its metric scale uses the frozen causal MoGe-2 estimator with KITTI-360's own source calibration, never a target dataset |
| 09_candidate_causal | PASS | every NEW depth map of a clip is computed from images t-4..t only (checks 2-3) |
| 10_L_pose_matches_local_reset_audit | PASS | T_L = inv(P_4) P_q of the local-reset clip on 1044/1044 clips; LBM-L native occupied/free maps equal the Gate 12 H input (the frozen deployable M) on 1044/1044 clips |
| 11_GT_pose_matches | PASS | T_G equal to Gate 9's verified T_j->t on 1044/1044 clips |
| 12_same_mapper | PASS | one VariantMapper(0.2, 0.2, carve) == core IncrementalMapper bit for bit; LBM-G native reproduces Gate 9 GF tp/fp/fn on 1044/1044 clips and LD-G reproduces Gate 9 LD on 1044/1044 |
| 13_same_voxel_grid | PASS | anchor SSC grid 256x256x32 at 0.2 m (Gate 9 constants) |
| 14_same_evaluation_masks | PASS | evaluation-mask digest equal to Gate 9 on 1044/1044 clips |
| 15_same_five_sweep_reference | PASS | reference-occupied voxel count in the mask equal to Gate 9 on 1044/1044 clips |
| 16_shared_support_independent_of_depth_values | PASS | S_shared = LingBot conf >= 1.5 AND NEW valid (inside the MUSt3R crop, finite, > 0), built before any metric depth array; it uses validity, never depth magnitude or error |
| 17_no_target_dataset_access | PASS | audit shards ran inside the strict file audit (SemanticKITTI / Occ3D / nuScenes forbidden): violations [0, 0, 0, 0, 0, 0, 0]; MUSt3R reads only KITTI-360 images |
| 18_deterministic_rerun | PASS | audit rerun on two 0006 clips reproduces every stored array exactly: [True, True]; MUSt3R rerun on one clip: max relative depth difference 0.00e+00 (tolerance 0.001, fixed before the rerun) |
| 19_promotion_rule_frozen_first | PASS | promotion_rule.json written 2026-09-14T00:16:19, before the first Gate 14 occupancy record; promotion_result.json produced under that exact rule (sha d46c5c41dde2) |
| 20_stage2_gated | PASS | stage2_guard.require_promotion() raises Stage2Locked with promote=False; Stage 2 artifacts present: False |
