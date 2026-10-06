# Final zero-shot results (numbers; interpretation in `final_zero_shot_results.md` / `paper_decision.md`)


## SemanticKITTI 08 — 801 samples, 33 bootstrap blocks

### Track P semantic (mIoU %)
| population | GT voxels | B0 | B3 mean (seeds) | Δ B3−B0 [CI] | accuracy B0 / B3 |
|---|---|---|---|---|---|
| SSC (predicted occupancy) | 50,029,541 | 5.45 | 5.64 (5.66 / 5.62 / 5.65) | +0.19 [+0.05, +0.34] | 24.94 / 25.91 |
| SSC, no-evidence voxels | 45,160,732 | 4.77 | 4.93 (4.95 / 4.91 / 4.94) | +0.16 [+0.02, +0.32] | 22.47 / 23.55 |
| GT occupied (diagnostic) | 50,029,541 | 13.66 | 14.17 (14.21 / 14.10 / 14.21) | +0.52 [+0.11, +0.87] | 44.64 / 46.37 |
| observed with evidence | 4,868,809 | 26.47 | 26.47 (26.47 / 26.47 / 26.47) | +0.00 [+0.00, +0.00] | 64.99 / 64.99 |
| no evidence (GT occupancy) | 45,160,732 | 12.39 | 12.85 (12.89 / 12.77 / 12.89) | +0.46 [+0.03, +0.87] | 42.45 / 44.37 |
| completed, no evidence (CENTRAL) | 24,140,570 | 12.33 | 12.58 (12.65 / 12.49 / 12.59) | +0.25 [-0.08, +0.57] | 42.04 / 44.06 |
| completed, geometrically unobserved | 20,768,806 | 11.39 | 11.74 (11.82 / 11.64 / 11.76) | +0.36 [+0.03, +0.65] | 40.57 / 42.67 |


### Track P occupancy (%)
| source / partition | IoU | precision | recall |
|---|---|---|---|
| ours all | 32.65 | 43.92 | 55.98 |
| ours observed | 34.73 | 40.73 | 70.23 |
| ours unobserved | 31.98 | 45.15 | 52.29 |
| occany_raw all | 31.28 | 48.87 | 46.49 |
| occany_raw observed | 37.08 | 46.05 | 65.56 |
| occany_raw unobserved | 29.39 | 50.13 | 41.54 |
| occany_pooled all | 31.05 | 39.28 | 59.72 |
| occany_pooled observed | 32.39 | 35.36 | 79.40 |
| occany_pooled unobserved | 30.58 | 40.99 | 54.61 |


### By range (Track P)
| range | ours SC IoU | OccAny raw SC IoU | B0 central mIoU | B3 central mIoU |
|---|---|---|---|---|
| 0-10 m | 50.51 | 23.34 | 18.18 | 19.77 |
| 10-20 m | 41.06 | 33.91 | 15.14 | 15.34 |
| 20-30 m | 32.37 | 32.12 | 10.85 | 10.79 |
| 30-40 m | 23.23 | 30.08 | 7.22 | 7.20 |


### Central population by distance to nearest evidence
| distance | voxels | B0 mIoU | B3 mIoU | B0 acc | B3 acc |
|---|---|---|---|---|---|
| 0-1 m | 13,765,736 | 15.26 | 15.56 | 46.93 | 49.20 |
| 1-2 m | 4,645,792 | 8.55 | 8.93 | 40.98 | 43.01 |
| 2-4 m | 3,829,048 | 5.81 | 5.97 | 36.88 | 38.14 |
| 4-8 m | 1,334,801 | 3.67 | 3.84 | 20.01 | 21.40 |
| >8 m | 565,193 | 3.82 | 4.25 | 18.76 | 21.23 |


### Track O (official)

SC IoU / P / R ours 26.40 / 37.45 / 47.22; OccAny released geometry raw 25.21, pooled 25.93. SSC mIoU B0 4.32, B3 4.43 (seeds 4.44 / 4.41 / 4.43); superclass mIoU B0 8.33, B3 8.42. OccAny published (GSAM2): 7.28.

* ours − OccAny raw SC IoU: Track P +1.37 [+0.39, +2.30]; official +1.19 [+0.35, +2.09]

* ours − OccAny pooled SC IoU: Track P +1.60 [+0.87, +2.31]; official +0.47 [-0.42, +1.36]

* Track O confusions identical to Track P full-region confusions: {'B0': True, 'B3_s0': True, 'B3_s1': True, 'B3_s2': True}


## Occ3D-nuScenes val — 4219 samples, 150 bootstrap blocks

### Track P semantic (mIoU %)
| population | GT voxels | B0 | B3 mean (seeds) | Δ B3−B0 [CI] | accuracy B0 / B3 |
|---|---|---|---|---|---|
| SSC (predicted occupancy) | 20,747,706 | 4.36 | 4.54 (4.54 / 4.52 / 4.54) | +0.18 [+0.09, +0.30] | 16.15 / 16.61 |
| SSC, no-evidence voxels | 18,412,529 | 3.65 | 3.79 (3.80 / 3.78 / 3.80) | +0.15 [+0.05, +0.29] | 14.07 / 14.55 |
| GT occupied (diagnostic) | 20,747,706 | 12.59 | 13.16 (13.15 / 13.14 / 13.19) | +0.56 [+0.42, +0.75] | 38.76 / 39.39 |
| observed with evidence | 2,335,177 | 20.16 | 20.56 (20.52 / 20.57 / 20.58) | +0.39 [+0.30, +0.50] | 49.92 / 50.32 |
| no evidence (GT occupancy) | 18,412,529 | 11.75 | 12.30 (12.30 / 12.28 / 12.33) | +0.55 [+0.41, +0.75] | 37.34 / 38.00 |
| completed, no evidence (CENTRAL) | 6,933,742 | 8.18 | 8.72 (8.75 / 8.68 / 8.74) | +0.54 [+0.29, +0.86] | 37.35 / 38.63 |
| completed, geometrically unobserved | 4,514,614 | 7.23 | 7.73 (7.75 / 7.69 / 7.74) | +0.50 [+0.20, +0.84] | 31.58 / 32.88 |


### Track P occupancy (%)
| source / partition | IoU | precision | recall |
|---|---|---|---|
| ours all | 36.87 | 76.98 | 41.44 |
| ours observed | 44.44 | 89.43 | 46.90 |
| ours unobserved | 31.95 | 68.37 | 37.49 |
| occany_raw all | 30.05 | 42.60 | 50.51 |
| occany_raw observed | 39.57 | 66.34 | 49.51 |
| occany_raw unobserved | 25.73 | 34.08 | 51.23 |
| occany_pooled all | 30.78 | 35.56 | 69.62 |
| occany_pooled observed | 45.71 | 58.01 | 68.32 |
| occany_pooled unobserved | 25.06 | 27.98 | 70.56 |


### By range (Track P)
| range | ours SC IoU | OccAny raw SC IoU | B0 central mIoU | B3 central mIoU |
|---|---|---|---|---|
| 0-10 m | 46.17 | 33.24 | 10.67 | 11.69 |
| 10-20 m | 35.77 | 39.97 | 11.10 | 12.00 |
| 20-30 m | 39.38 | 30.54 | 8.92 | 9.52 |
| 30-40 m | 33.15 | 24.67 | 6.16 | 6.47 |


### Central population by distance to nearest evidence
| distance | voxels | B0 mIoU | B3 mIoU | B0 acc | B3 acc |
|---|---|---|---|---|---|
| 0-1 m | 3,628,664 | 11.89 | 12.62 | 46.97 | 48.38 |
| 1-2 m | 857,143 | 8.25 | 8.59 | 30.46 | 31.40 |
| 2-4 m | 922,881 | 6.49 | 7.00 | 28.33 | 28.89 |
| 4-8 m | 804,860 | 3.39 | 3.39 | 24.21 | 25.02 |
| >8 m | 720,194 | 2.86 | 2.78 | 23.35 | 25.80 |


### Track O (official)

SC IoU / P / R ours 31.94 / 77.87 / 35.13; OccAny released geometry raw 20.66, pooled 23.55. SSC mIoU B0 3.71, B3 3.80 (seeds 3.80 / 3.79 / 3.80); superclass mIoU B0 6.63, B3 6.76. OccAny published (GSAM2): 6.66.

* ours − OccAny raw SC IoU: Track P +6.82 [+3.93, +9.93]; official +11.28 [+9.25, +13.37]

* ours − OccAny pooled SC IoU: Track P +6.08 [+3.57, +8.64]; official +8.39 [+6.57, +10.24]

* Track O confusions identical to Track P full-region confusions: {'B0': True, 'B3_s0': True, 'B3_s1': True, 'B3_s2': True}


## Criteria

```
{
 "generalises": {
  "semantickitti": false,
  "occ3d": true
 },
 "meaningful_gain": {
  "semantickitti": false,
  "occ3d": false
 },
 "competitive_with_published_occany": {
  "semantickitti": false,
  "occ3d": false
 },
 "occany_published_miou": {
  "semantickitti": 7.28,
  "occ3d": 6.66
 }
}
```

**PAPER C**
