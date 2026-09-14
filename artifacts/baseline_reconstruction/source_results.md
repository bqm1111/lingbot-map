# Source results -- Current Geometry-Only Baseline (L + LB-M + H, CompletionUNet-3ch)

KITTI-360 0006, 590 anchors, final-step (6000) checkpoints, occupied := final log-odds > 0.0. Regions and range bins as in Gates 12/13.

## Headline (full region)

| seed | SC IoU % | precision % | recall % | map-only IoU % | checkpoint sha256 |
|---|---|---|---|---|---|
| 0 | 13.65 | 18.59 | 33.92 | 3.99 | 7f5e853dd05b |
| 1 | 13.46 | 17.76 | 35.73 | 3.99 | 2c4d79be4fa7 |
| 2 | 13.43 | 18.26 | 33.69 | 3.99 | 6038b1f7d901 |
| **mean** | **13.51** | 18.20 | 34.45 |  |  |
| median | 13.46 |  |  |  |  |
| sd | 0.12 |  |  |  |  |


## Reproduction of Gate 13 M (predeclared tolerance)

| seed | this run | Gate 13 M | abs diff | tolerance |
|---|---|---|---|---|
| 0 | 0.1365 | 0.1354 | 0.0011 | 0.005 |
| 1 | 0.1346 | 0.1336 | 0.0010 | 0.005 |
| 2 | 0.1343 | 0.1346 | 0.0003 | 0.005 |
| mean | 0.1351 | 0.1345 | 0.0006 | 0.003 |


**Verdict: PASS -- the current pipeline reproduces the Gate 13 M regime.**

## By region and range (mean over seeds; per-seed in tables/source_per_range.csv)

| region | range | IoU % mean | median | sd | P % mean | R % mean | GT occupied voxels |
|---|---|---|---|---|---|---|---|
| full | 0-5 m | 17.93 | 18.08 | 0.47 | 20.45 | 59.28 | 2,249,650 |
| full | 5-10 m | 18.78 | 18.70 | 0.27 | 21.83 | 57.40 | 1,987,861 |
| full | 10-20 m | 16.19 | 16.12 | 0.15 | 19.75 | 47.39 | 3,939,168 |
| full | 20-30 m | 11.25 | 11.29 | 0.19 | 16.10 | 27.21 | 3,654,698 |
| full | >30 m | 7.45 | 7.49 | 0.37 | 13.20 | 14.64 | 6,362,747 |
| full | all | 13.51 | 13.46 | 0.12 | 18.20 | 34.45 | 18,194,124 |
| past5 | 0-5 m | 18.70 | 19.10 | 0.72 | 19.98 | 74.51 | 1,141,653 |
| past5 | 5-10 m | 20.03 | 20.33 | 0.55 | 21.97 | 69.51 | 799,471 |
| past5 | 10-20 m | 15.43 | 15.48 | 0.18 | 17.40 | 57.74 | 1,019,502 |
| past5 | 20-30 m | 9.20 | 9.25 | 0.23 | 10.78 | 38.58 | 531,503 |
| past5 | >30 m | 7.16 | 7.19 | 0.06 | 9.65 | 21.81 | 513,260 |
| past5 | all | 15.48 | 15.68 | 0.41 | 17.47 | 57.72 | 4,005,389 |
| unobs5_later | 0-5 m | 21.19 | 21.17 | 0.34 | 29.21 | 43.58 | 1,107,997 |
| unobs5_later | 5-10 m | 21.14 | 21.19 | 0.08 | 27.04 | 49.25 | 1,188,390 |
| unobs5_later | 10-20 m | 18.88 | 18.88 | 0.04 | 24.93 | 43.77 | 2,919,666 |
| unobs5_later | 20-30 m | 12.93 | 12.88 | 0.16 | 20.94 | 25.28 | 3,123,195 |
| unobs5_later | >30 m | 7.60 | 7.62 | 0.42 | 14.30 | 14.01 | 5,849,487 |
| unobs5_later | all | 13.79 | 13.74 | 0.10 | 21.45 | 27.88 | 14,188,735 |
| observed20 | all | 15.49 | 15.67 | 0.37 | 17.70 | 55.48 | 4,504,234 |
| unobserved_later20 | all | 13.70 | 13.65 | 0.11 | 21.44 | 27.53 | 13,689,890 |
| H_evidence | all | 22.44 | 22.63 | 0.74 | 26.16 | 61.52 | 1,989,019 |
| no_H_evidence | all | 12.33 | 12.31 | 0.14 | 16.96 | 31.12 | 16,205,105 |


`past5` = observed by the five causal sweeps; `unobs5_later` = unobserved by them, resolved by the future sweeps (the completion region).

## Learning curves

| step | seed 0 IoU % | seed 1 IoU % | seed 2 IoU % | mean |
|---|---|---|---|---|
| 1000 | 11.87 | 12.30 | 11.33 | 11.83 |
| 2000 | 12.92 | 12.71 | 12.29 | 12.64 |
| 3000 | 13.35 | 12.96 | 13.15 | 13.15 |
| 4000 | 13.03 | 13.49 | 12.96 | 13.16 |
| 5000 | 13.35 | 13.26 | 13.30 | 13.30 |
| 6000 | 13.47 | 13.34 | 13.38 | 13.40 |


Figures: artifacts/baseline_reconstruction/figures/source_learning_curve_iou.png, artifacts/baseline_reconstruction/figures/source_training_loss.png
