# Gate 4 — storage and access preflight

## Status: `DATA_PRESENT` — no download was required or performed

Both official datasets were already installed on this machine from earlier work. The
preflight verified them in place, recorded sizes and counts, and re-verified every frozen
checkpoint hash. Nothing was downloaded, and no credential or token was read, stored or
printed.

## Data root resolution

Resolution order per the brief: `$OCC3D_DATA_ROOT`, then `$NUSCENES_DATA_ROOT`, then an
explicit mount configured in the repository.

| step | result |
|---|---|
| `OCC3D_DATA_ROOT` | unset |
| `NUSCENES_DATA_ROOT` | unset |
| repository configuration `data.nuscenes_root` | **`/media/SSD1/MINH_DATASETS/nuscenes`** ← resolved |

The resolved root is checked against a forbidden-prefix list (`$HOME`, the repository root,
`/tmp`, `/var/tmp`, `/dev/shm`) and passed. It lies outside the git repository.

```
nuScenes    /media/SSD1/MINH_DATASETS/nuscenes
Occ3D       /media/SSD1/MINH_DATASETS/nuscenes/occ3d_gt/Occupancy3D-nuScenes-trainval
```

## Filesystem

| field | value |
|---|---|
| device / type | `/dev/nvme0n1p1`, **ext4** |
| capacity | 1 833 GiB |
| free | **595 GiB (32 %)** |
| inodes free | 119 346 171 of 122 101 760 (98 %) |

The repository volume `/` is at **99 % capacity (24 GiB free)**, so no bulk data was
written there. Frozen LingBot predictions were written outside the repository to
`/media/SSD1/MINH_DATASETS/lingbot_occ3d_zeroshot/cache_lingbot` (1 182 files, 1.74 GiB);
only summaries, manifests and figures (25 MiB total) live under `artifacts/`.

Because nothing was downloaded, the 20 % free-space headroom rule was not binding. Had a
download been required, the estimate would have been ~29 GiB compressed for
`imgs.tar.gz` plus ~2.6 GiB for `gts.tar.gz` (per OccAny `dataset_setup/occ3d_nuscenes.md`),
against 595 GiB available — comfortably clear.

## Present and verified

| requirement | path | status |
|---|---|---|
| nuScenes metadata | `v1.0-trainval/` (14 tables incl. `calibrated_sensor`, `ego_pose`, `sample_data`) | present |
| camera images | `samples/CAM_FRONT/` — **34 149 files, 4.7 GiB** | present |
| LiDAR sweeps | `samples/LIDAR_TOP/` | present (diagnostics only) |
| Occ3D annotations | `annotations.json` — 150 463 571 bytes | present |
| Occ3D labels | `gts/` — **850 scene directories, 2.71 GiB** | present |
| Occ3D archive | `gts.tar.gz` retained alongside the extraction | present |

Official split from `annotations.json`: **700 train / 150 val / 850 scene_infos**. All 150
official validation scenes are present. Only the validation split is used in this task.

## License and access

nuScenes and Occ3D-nuScenes are covered by the nuScenes non-commercial license
(`LICENSE` is present at the nuScenes root). Access was granted and the data downloaded in
earlier work on this machine; no login, click-through or credential was exercised in this
task, and none is stored in the repository.

## Frozen component verification

Every hash re-verified against the Gate-3.1 record before any evaluation ran:

| component | path | SHA-256 (first 16) |
|---|---|---|
| LingBot | `checkpoints/lingbot-map/204754b/lingbot-map.pt` | `ee665103348e07e6` |
| Gate-2 depth head | `artifacts/depth_gate/runs/depth_cnn/best.pt` | `2a91822ea5d1182a` |
| Gate-3.1 `full_s0` | `artifacts/voxel_gate_validation/runs/full_s0/best.pt` | `05730ac7addb101d` |
| Gate-3.1 `occ_only_s0` | `artifacts/voxel_gate_validation/runs/occ_only_s0/best.pt` | `b90a6e4ee9cdeccd` |
| Gate-3.1 `c0_corrector_s0` | `artifacts/voxel_gate_validation/runs/c0_corrector_s0/best.pt` | `6590049302ef2044` |
| Gate-3.1 config | `configs/voxel_gate_validation/clean_infill.yaml` | `a548ed08fcbe1ea6` |
| Gate-2 config | `configs/depth_gate/refine.yaml` | `b7692022114830f9` |
| Gate-0 config | `configs/scale_gate/semantickitti.yaml` | `472b45311a5e03ba` |

All eight match the values recorded in `reports/voxel_gate/local_infill_validation_report.md`.

## Artifacts produced

```
artifacts/occ3d_zeroshot/preflight.json          machine-readable preflight record
manifests/occ3d_zeroshot/val.jsonl               60a74604…f18e1463
configs/occ3d_zeroshot/frozen_transfer.yaml      d82a1350…873f953f8
```

Command: `python tools/occ3d_zeroshot/preflight.py`
