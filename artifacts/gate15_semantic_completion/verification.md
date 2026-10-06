# Gate 15 verification (2026-09-14)

**18 / 18 checks pass.**

| # | check | result | evidence |
|---|---|---|---|
| 1 | frozen geometry checkpoints unchanged | PASS | sha256 vs baseline code_freeze.json: {0: True, 1: True, 2: True} |
| 2 | frozen geometry inputs unchanged | PASS | S_H/F_H of anchors 4 and 1000 re-derived from their clips bit-exactly: [True, True] |
| 3 | occupancy outputs unchanged after adding the semantic module | PASS | seed 0: TP/FP/FN [6172202, 27026109, 12021922] == baseline [6172202, 27026109, 12021922] (IoU 13.65 %); seed 1: TP/FP/FN [6500344, 30096511, 11693780] == baseline [6500344, 30096511, 11693780] (IoU 13.46 %); seed 2: TP/FP/FN [6129120, 27435715, 12065004] == baseline [6129120, 27435715, 12065004] (IoU 13.43 %) |
| 4 | Trident-H frozen | PASS | Gate 15 never loads the Trident model; it reads token caches written by the feature-first study (newest sampled cache 2026-09-13T13:54, before Gate 15 began) |
| 5 | no future images | PASS | 79 sampled evidence files: frames == t-4..t and == the clip's keys; mismatches 0 |
| 6 | no human semantic training labels | PASS | evidence build and training ran under FileAudit forbidding ['data/kitti/dataset', 'occ3d_gt', 'nuscenes', 'lingbot_occ3d_zeroshot', 'scale_gate/cache', 'preprocess/labels', 'eval_masks', 'samples_gtsem', 'data_3d_semantics', 'velodyne', 'data_3d_raw', 'data_poses', 'targets']; violations 0 |
| 7 | only KITTI-360 0003/0007/0010 used for training | PASS | n_train_anchors = 1358 = anchors of 0003+0007+0010 in every run record |
| 8 | 0006 source-only, no selection | PASS | no validation pass during training; the final step-6000 checkpoint of each seed is evaluated; 0006 used only by evaluation |
| 9 | target datasets unopened before source freeze | PASS | all audits forbid SemanticKITTI/Occ3D/nuScenes paths; violations 0; no target-side output exists |
| 10 | teacher features normalised consistently | PASS | stored evidence |‖f‖-1| max 2.8e-04 (fp16), renormalised on load; tokens and text unit norm |
| 11 | observed/unobserved masks correct | PASS | anchor 4: completed_unobserved ∩ E = ∅; oracle_observed ⊔ oracle_unobserved = GT occupied; tests/semantic_completion/test_regions_and_confusion |
| 12 | semantic evidence exists only at direct causal observations | PASS | anchor 1000 rebuilt: indices identical True, min cosine to cache 1.000000; every evidence voxel ∈ S_H True |
| 13 | no semantic loss on unsupported voxels | PASS | train.head_loss gathers predictions only at the crop's evidence indices; test_loss_supervises_only_given_voxels_and_splits_hidden |
| 14 | trivial baseline fixed before the learned result | PASS | baseline_nearest.py hashed in promotion_rule.json at 2026-09-14T13:43:02+02:00, before the first final head checkpoint; unchanged since |
| 15 | promotion threshold frozen before source evaluation | PASS | rule frozen 2026-09-14T13:43:02+02:00; first source evaluation file 2026-09-14T15:22:18; no results existed at freeze |
| 16 | deterministic evaluation | PASS | anchors [4, 889, 1771] re-evaluated: all 42 confusion arrays identical to the stored run |
| 17 | class IDs / text embeddings mapped correctly | PASS | text embedding rows = vocab.load('kitti360') phrases and label ids 1..18 in order: True |
| 18 | no occupancy gradient from semantic loss | PASS | trunk computed under no_grad; every geometry parameter has grad None after training and the state digest is unchanged; test_head_training_leaves_geometry_bit_identical |
