# Zero-shot results -- Current Geometry-Only Baseline (L + LB-M + H, CompletionUNet-3ch)

First frozen evaluation, run once per dataset after the source freeze (`source_freeze.md`); nothing was changed afterwards. Protocol records: `evaluation_semantickitti.yaml`, `evaluation_occ3d.yaml`. Semantic mIoU is unavailable (geometry-only model).

## Headline (full official mask)

| dataset | anchors | SC IoU % seeds 0/1/2 | **mean** | median | sd | 95 % CI of mean (clip-block bootstrap) | P % mean | R % mean | map-only IoU % | historical (median) |
|---|---|---|---|---|---|---|---|---|---|---|
| semantickitti | 163 | 26.25 / 26.45 / 25.82 | **26.17** | 26.25 | 0.32 | [24.61, 27.69] | 35.83 | 49.31 | 8.82 | 22.71 |
| occ3d | 1182 | 31.56 / 32.81 / 29.30 | **31.22** | 31.56 | 1.78 | [29.17, 33.21] | 77.07 | 34.41 | 10.04 | 33.88 |


## By region (mean over seeds)

| dataset | region | IoU % | P % | R % | GT occupied voxels |
|---|---|---|---|---|---|
| semantickitti | full | 26.17 | 35.83 | 49.31 | 18,762,930 |
| semantickitti | H_evidence | 33.23 | 39.56 | 67.79 | 2,928,267 |
| semantickitti | no_H_evidence | 24.74 | 34.97 | 45.90 | 15,834,663 |
| occ3d | full | 31.22 | 77.07 | 34.41 | 11,957,590 |
| occ3d | H_evidence | 39.03 | 88.06 | 41.21 | 4,014,775 |
| occ3d | no_H_evidence | 27.50 | 71.18 | 30.97 | 7,942,815 |


`H_evidence` = evaluation voxels where the causal map carries S or F evidence; `no_H_evidence` = the rest (pure completion).

## By range (anchor-camera z, mean over seeds, full region)

| dataset | range | IoU % | P % | R % | map-only IoU % | GT occupied voxels |
|---|---|---|---|---|---|---|
| semantickitti | 0-5 m | 34.04 | 43.56 | 60.93 | 15.40 | 1,968,755 |
| semantickitti | 5-10 m | 35.45 | 44.42 | 63.74 | 16.19 | 1,905,089 |
| semantickitti | 10-20 m | 32.91 | 41.46 | 61.48 | 12.57 | 4,075,038 |
| semantickitti | 20-30 m | 28.49 | 36.60 | 56.31 | 7.45 | 3,823,726 |
| semantickitti | >30 m | 16.44 | 25.88 | 31.19 | 1.68 | 6,990,322 |
| occ3d | 0-5 m | 29.51 | 82.44 | 31.46 | 11.83 | 2,362,576 |
| occ3d | 5-10 m | 29.59 | 79.76 | 31.95 | 13.05 | 1,769,558 |
| occ3d | 10-20 m | 30.93 | 76.89 | 34.08 | 12.68 | 3,188,868 |
| occ3d | 20-30 m | 34.85 | 75.01 | 39.44 | 8.04 | 2,732,496 |
| occ3d | >30 m | 29.93 | 72.96 | 33.69 | 3.15 | 1,904,092 |


Skipped anchors: semantickitti: 0 without label, 0 without a causal scale; occ3d: 0 without label, 0 without a causal scale.


Figure: `artifacts/baseline_reconstruction/figures/zero_shot_iou_by_range.png`.
