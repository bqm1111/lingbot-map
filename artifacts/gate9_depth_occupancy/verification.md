# Gate 9 verification

14/14 checks pass.

| check | result | detail |
|---|---|---|
| same_clips_and_anchors | PASS | 590 Gate 9 clips == the 590 local-registration clips (stride-3 eligible anchors of 0006); every condition of a clip is computed in one pass on one record |
| identical_gt_poses | PASS | stored T_j->t equal inv(rect_cam_to_world(t)) rect_cam_to_world(j) on 12/12 sampled clips; one FrameInput constructor feeds every run (LD, PD, GF, Gs and all variants) |
| identical_support_mask | PASS | LD and PD both use conf = SUPPORT_CONF on the LiDAR z-buffer support and 0 elsewhere; support digest stored per clip; pixel records use the same (row, col) for z_lidar and z_pred |
| identical_camera_model | PASS | one lattice intrinsic matrix (calibrated P_rect_00 scaled to 140x518 at pixel centres) for projection, z-buffer, rays and every mapper run; extrinsics rect_cam_to_velo / velo_to_rect_cam from the drive calibration |
| zbuffer_nearest | PASS | three points on one lattice pixel at 10 / 5 / 7 m -> z-buffer keeps 5 m (index 1) |
| same_voxel_grid | PASS | dims (256, 256, 32), 0.2 m, origin (0.0, -25.6, -2.0) == core.datasets.kitti360_transforms.GRID; the same grid for V voxelisation, reference, mapper query and pixel endpoints |
| same_evaluation_mask | PASS | one mask E per clip (reference-measured & inside a camera frustum) used by every condition's tp/fp/fn; digests e.g. ['3571e517bd2043d3', 'da31ff118ee55ceb', '307e80ab0f91cee4'] |
| no_semantic_labels | PASS | no semantic label path or semantic target read; semantics disabled in the mapper (sem_into=None) |
| no_target_datasets | PASS | every audit shard ran inside the file-access audit (SemanticKITTI / Occ3D / nuScenes / SSCBench labels forbidden); violations [0, 0, 0] |
| no_weights_changed | PASS | LingBot checkpoint sha256 ee665103348e07e6 == the one recorded by the registration audit; Gate 9 only reads the cached local-reset depth; no optimiser / backward / module |
| no_learned_module | PASS | no network is loaded or introduced; the mapper and metrics are deterministic functions |
| variant_mapper_equals_core_mapper | PASS | VariantMapper(band 0.2, margin 0.2, carve) reproduces core IncrementalMapper bit for bit on 5 stream frames (40,776 voxel rows) |
| deterministic_rerun | PASS | re-running the audit on clips ['00004', '00889'] reproduces every stored array exactly: [True, True] |
| V_matches_verified_pose_chain | PASS | camera-route transform (velo_j -> rect cam_j -> T_j->t -> grid) equals geometry.velo_to_velo to 8.0e-13 m; direct voxelisation equals rawtarget's return voxels on clip 304 (20,460 voxels, 0 differ, rawtarget additionally drops returns within 1.0 m of the sensor) |
