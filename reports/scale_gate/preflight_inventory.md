# Phase 0 — preflight inventory

Everything here was read off this machine. Reproduce with
`python tools/scale_gate/preflight.py --config configs/scale_gate/semantickitti.yaml`.

## Repository state

| | |
|---|---|
| commit | `9a648322e6dc4e34995f0530650c2fa469779676` ("Add something") |
| pre-existing modifications | `lingbot_map/models/gct_stream.py`, `research/sem_bypass/model.py` — **both whitespace-only** (verified by `git diff`: one trailing space, two trailing blank lines). Left untouched. |
| pre-existing untracked | `lingbot_map_scale_claude_code_plan.md`, three `scripts/*figure*.py`, `research/`, `scale_gate/`, `tests/scale_gate/` |
| repository instructions | `CLAUDE.md` (no `AGENTS.md` exists) |
| python / torch / CUDA | 3.10.19 / 2.7.1+cu128 / 12.8 |
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q, 97 887 MiB |
| required env | `CUDA_HOME=/usr/local/cuda-12.8`, `FLASHINFER_CUDA_ARCH_LIST="12.0a"`, `PYTHONPATH=<repo root>` |

## Checkpoint

`checkpoints/lingbot-map/204754b/lingbot-map.pt`, 4.31 GiB, SHA-256
`ee665103348e07e6b826d529b8e61de8f413d5432a4f2e84970d6c8fd2e1cd72`,
1 157 943 540 parameters. Loaded with `GCTStream`, `model.eval()`, every parameter
`requires_grad=False` (asserted in `cache_lingbot.build_model`, which raises otherwise).

## LingBot outputs, units and frames — **verified, not assumed**

| quantity | key | shape | convention |
|---|---|---|---|
| depth | `predictions["depth"]` | `[S, H, W, 1]` | **optical-axis z depth**, camera frame, arbitrary canonical scale |
| depth confidence | `depth_conf` | `[S, H, W]` | `expp1` = `1 + exp(x)`, so **> 1**; a precision, not an uncertainty |
| pose | `pose_enc` → `pose_encoding_to_extri_intri` | `[S, 3, 4]` | returns **world-to-camera**; this package stores the inverted **camera-to-world** under the explicit key `pred_pose_c2w` |
| intrinsics | same call | `[S, 3, 3]` | pixels of the processed image |
| camera axes | — | — | OpenCV, x right / y down / z forward |
| sequence origin | — | — | first frame of the clip |

The z-depth convention is confirmed two independent ways: `occ_datasets.camera_points_from_depth`
unprojects as `x = (u - cx) d / fx, y = (v - cy) d / fy, z = d`, and the previous study's cache
manifest records `"depth": "z_depth_camera_frame"`. `scale.depth_convention` is set to `z`
explicitly in the config — never auto-detected.

**Preprocessing** is `load_and_preprocess_images(mode="crop", image_size=518, patch_size=14)`.
For every KITTI odometry resolution the resized height stays under 518, so the crop branch
never fires and the mapping is a pure anisotropic resize. `Preprocess.check()` raises if that
ever stops holding rather than silently mis-scaling intrinsics.

| native | processed | grid |
|---|---|---|
| 1226×370 (seqs 04–10) | 518×154 | 37×11 |
| 1241×376 (seqs 00–02) | 518×154 | 37×11 |
| 1242×375 (seq 03) | 518×154 | 37×11 |

## Calibration chain

`p_velo --Tr--> p_cam0 --(+t_cam2)--> p_cam2 --K--> p_pixel(native) --resize--> p_pixel(processed)`

`P2` is a **projection** matrix, not an intrinsic matrix: `K = P2[:3,:3]` and the stereo
baseline becomes `t_cam2 = (P2[0,3]/fx, P2[1,3]/fy, P2[2,3])`. `poses.txt` is cam0-to-world in
metres, so `image_2` frames need that offset applied — `load_poses_cam2_c2w` does it and
`tests/scale_gate/test_kitti.py` pins both facts.

## Reused code

`prompted_lingbot.occupancy` (`SEMANTICKITTI_GRID` 256×256×32 @ 0.2 m, floor-binning
voxeliser, `binary_occupancy_scores`), `prompted_lingbot.occ_datasets`
(`SemanticKittiOccSpec`, `camera_points_from_depth`, `apply_transform`),
`lingbot_map.utils.load_fn` / `pose_enc`. The occupancy evaluator, grid and metric are used
**unmodified**, so only the scalar differs between methods.

## Datasets discovered

| asset | path | status |
|---|---|---|
| SemanticKITTI | `data/kitti/dataset` → `/media/welf/SSD2/MINH_DATASETS/kitti/` | complete: 22 sequences, `voxels/` targets for 00–10 |
| existing LingBot caches | `outputs/prompted_lingbot/cache_semkitti08` (seq 08), `cache_kitti` (09, 10), `cache_occ3d` (150 nuScenes scenes) | **not reused**: streaming mode over 500-frame chunks, `has_gt_depth: false`, and no clip structure. This gate needs direct 5-frame clips, so a fresh cache was built. |
| nuScenes | `/media/welf/MINH/datasets/nuscenes` | images/poses present; **no semantic or occupancy GT** (see `docs/data_setup_scale_gate.md`) |

## Deviations from the plan's suggested defaults

| setting | plan | used | why |
|---|---|---|---|
| `scale.depth_convention` | `auto` | `z` | the convention is verified, and the plan forbids selecting it silently; `auto` would hide it |
| `lingbot.confidence_threshold` | `null` | `1.5` | matches the existing occupancy evaluator so the downstream comparison is like-for-like |
| `lingbot.num_scale_frames` | — | `5` = clip length | makes the whole clip one direct block, satisfying "direct mode, no state reset within a clip" |
| inner selection split | — | seqs 09, 10 | checkpoints must not be chosen on seq 08; the plan forbids tuning on the target |

## Splits

| split | sequences | clips | frames |
|---|---|---:|---:|
| train | 00–07, 09, 10 | 763 | 3 815 |
| val | 08 | 163 | 815 |
| smoke | 08 (first 2) | 2 | 10 |

Clips are 5 frames at stride 5 (≈ 2 Hz), non-overlapping. No sequence appears in both
splits; asserted in `prepare_manifest` and in the tests.

## Blockers

**None for SemanticKITTI.** Occ3D-nuScenes cannot be scored for downstream occupancy
because its labels are absent; the scale-target path would work from LiDAR sweeps alone.
