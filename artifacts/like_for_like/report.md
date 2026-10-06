# Like-for-like semantic evaluation: our heads under OccAny's settings

**What changed from the frozen evaluation.** Only the sample set and the input window. The method, heads, checkpoints, teacher, grids, labels and metric are the frozen ones.

- **Samples:** every sample OccAny scored (805 / 4,819), not just those with a causal window (801 / 4,219).
- **Input:** for SemanticKITTI, OccAny's own five frames starting at the target and running forward (non-causal); for Occ3D, both that window and the six surround cameras of one timestep, because OccAny publishes nuScenes semantics only in the surround setting.
- **Placement:** the comparison's declared primary — SemanticKITTI one shared transform, Occ3D OccAny's own target-camera-to-voxel transform (which is `CAM_FRONT`-to-ego, the transform both OccAny settings use).
- **Metric:** Track O, the official computation OccAny reports (`SSCMetrics` conventions, every valid voxel).

**Which OccAny setting is which** (`occany/datasets/eval_helper.py`). nuScenes `5frames` = one `CAM_FRONT`, five frames at interval 2, `use_surround_label=False`, which sets `voxel_label[:100] = 255` and so ignores the rear half of the grid. nuScenes `surround` = six cameras at one timestep, `use_surround_label=True`, full volume. The saved OccAny outputs on this machine are the `5frames` run, and their stored labels confirm it (no valid voxel at x < 100). Their published nuScenes semantic number is the surround one.

**Verification.** The pipeline reproduces the frozen predictions array for array when run with the causal protocol and placement (`pipeline_equivalence.json`, `all_equal: true`). The surround ground truth reproduces OccAny's stored label voxel for voxel on the front half and restores the rear (`tests/like_for_like/test_surround.py`). The surround MoGe-2 depths reproduce cached frozen frames exactly.

### SemanticKITTI 08

| setting | samples | input | ground truth | post-processing | SC IoU | SSC mIoU | superclass mIoU |
|---|---:|---|---|---|---:|---:|---:|
| **ours, B3 seed mean (forward)** | 805 | 5 forward frames, 1 camera | OccAny's stored label | majority pooling | 30.27 | **5.38** | 9.64 |
| ours, B3 seed mean (forward) | 805 | 5 forward frames, 1 camera | OccAny's stored label | raw | 30.27 | 5.36 | 9.70 |
| ours, B0 (forward) | 805 | 5 forward frames, 1 camera | OccAny's stored label | raw | 30.27 | 5.11 | 9.53 |
| ours, deployed causal setting (B3, frozen) | 801 | 5 causal frames, 1 camera | frozen protocol | raw | 26.40 | 4.43 | 8.42 |
| **OccAny, published (Tab. 5/6)** | 805 | 1 camera, 5 frames starting at the target | its own | majority pooling | — | **7.28** | 13.53 |

The comparable pair is the bold one: OccAny publishes this dataset's semantics only for the *five-frame single-camera* setting.

### Occ3D-nuScenes val

| setting | samples | input | ground truth | post-processing | SC IoU | SSC mIoU | superclass mIoU |
|---|---:|---|---|---|---:|---:|---:|
| ours, B3 seed mean (forward) | 4819 | 5 forward frames, 1 camera | front half (OccAny `5frames`) | majority pooling | 25.45 | 2.62 | 5.08 |
| ours, B3 seed mean (forward) | 4819 | 5 forward frames, 1 camera | front half (OccAny `5frames`) | raw | 25.45 | 2.57 | 4.93 |
| ours, B0 (forward) | 4819 | 5 forward frames, 1 camera | front half (OccAny `5frames`) | raw | 25.45 | 2.50 | 4.81 |
| ours, B3 seed mean (forward_surround_label) | 4819 | 5 forward frames, 1 camera | 360 deg (OccAny `surround`) | majority pooling | 21.38 | 1.67 | 3.04 |
| ours, B3 seed mean (forward_surround_label) | 4819 | 5 forward frames, 1 camera | 360 deg (OccAny `surround`) | raw | 21.38 | 1.63 | 2.94 |
| ours, B0 (forward_surround_label) | 4819 | 5 forward frames, 1 camera | 360 deg (OccAny `surround`) | raw | 21.38 | 1.60 | 2.87 |
| **ours, B3 seed mean (surround)** | 4819 | 6 cameras, 1 timestep | 360 deg (OccAny `surround`) | majority pooling | 32.62 | **4.09** | 7.24 |
| ours, B3 seed mean (surround) | 4819 | 6 cameras, 1 timestep | 360 deg (OccAny `surround`) | raw | 32.62 | 4.03 | 7.10 |
| ours, B0 (surround) | 4819 | 6 cameras, 1 timestep | 360 deg (OccAny `surround`) | raw | 32.62 | 3.90 | 6.86 |
| ours, deployed causal setting (B3, frozen) | 4219 | 5 causal frames, 1 camera | frozen protocol | raw | 31.94 | 3.80 | 6.76 |
| **OccAny, published (Tab. 5/6)** | 4,819 | 6 surround cameras, single timestep, 360-degree label | its own | majority pooling | — | **6.66** | 10.32 |

The comparable pair is the bold one: OccAny publishes this dataset's semantics only for the *surround* setting.

## What is still not like-for-like

- **OccAny's semantics cannot be run here** (its GroundingDINO step fails on this GPU) and its saved outputs on this machine store occupancy only, so its column is the published number, not a re-scored prediction. Only the geometry column is a same-evaluator comparison.
- **Surround view order.** OccAny feeds `CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT` to a model with shared memory across views; we feed a ring (`CAM_FRONT, CAM_FRONT_LEFT, CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT, CAM_FRONT_RIGHT`) because our model is sequential and needs overlap with the previous view. Neither model is given the rig calibration.
- **Pooling:** OccAny's released metric script applies majority pooling by default (`sh/compute_metric.sh`: `USE_MAJORITY_POOLING=1`, `POOLING_MODE=separate`), and its README ties those runs to the published numbers. The semantic branch relabels occupied voxels by a 3x3x3 majority vote and leaves occupancy untouched, so our SC IoU is identical in both rows. It is not described in the paper.
- **The forward window is not our deployed setting**: it sees four frames after the target. It exists here only to match OccAny; the causal rows remain the honest description of our system.

