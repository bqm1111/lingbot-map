# Streaming-validation accuracy tables (generated — do not hand-edit)

Source: `/home/minh/workspace/lingbot-map_fork/artifacts/method_vs_data/streaming_scale_validation/scores`

## semantickitti

### semantickitti — global SC IoU, 801 anchors (canonical population 801, exclusions 0)

| arm | raw IoU | raw P | raw R | pooled IoU | pooled P | pooled R |
|---|---|---|---|---|---|---|
| clip-reset / k360_s0 | 26.40 | 37.45 | 47.22 | 26.34 | 33.08 | 56.39 |
| clip-reset / k360_s1 | 26.57 | 35.90 | 50.55 | 25.66 | 31.23 | 59.01 |
| clip-reset / k360_s2 | 25.99 | 35.84 | 48.59 | 25.66 | 31.65 | 57.57 |
| clip-reset / waymo_s0 | 23.55 | 34.55 | 42.53 | 23.37 | 29.87 | 51.81 |
| clip-reset / waymo_s1 | 22.54 | 33.69 | 40.52 | 22.31 | 28.94 | 49.34 |
| clip-reset / waymo_s2 | 22.84 | 33.88 | 41.21 | 22.96 | 29.47 | 50.95 |
| clip-reset / once_s0 | 18.20 | 31.14 | 30.45 | 20.26 | 28.01 | 42.26 |
| clip-reset / once_s1 | 19.03 | 31.57 | 32.39 | 20.82 | 28.10 | 44.55 |
| clip-reset / once_s2 | 17.69 | 31.74 | 28.55 | 20.24 | 28.85 | 40.40 |
| clip-reset / map_only | 8.79 | 32.75 | 10.72 | 15.04 | 29.25 | 23.65 |
| streaming / k360_s0 | 24.87 | 35.76 | 44.94 | 25.23 | 32.01 | 54.34 |
| streaming / k360_s1 | 25.32 | 34.34 | 49.09 | 24.74 | 30.22 | 57.67 |
| streaming / k360_s2 | 25.00 | 35.01 | 46.66 | 24.97 | 31.14 | 55.74 |
| streaming / waymo_s0 | 21.30 | 36.16 | 34.13 | 22.67 | 31.65 | 44.39 |
| streaming / waymo_s1 | 20.38 | 34.38 | 33.35 | 21.44 | 30.06 | 42.79 |
| streaming / waymo_s2 | 20.25 | 35.42 | 32.09 | 21.78 | 31.11 | 42.06 |
| streaming / once_s0 | 16.51 | 32.49 | 25.12 | 19.84 | 30.08 | 36.83 |
| streaming / once_s1 | 17.23 | 32.13 | 27.09 | 20.26 | 29.44 | 39.39 |
| streaming / once_s2 | 16.68 | 32.92 | 25.27 | 20.00 | 30.33 | 37.00 |
| streaming / map_only | 8.84 | 33.87 | 10.69 | 14.74 | 28.45 | 23.41 |
| occany_1f | 20.32 | 55.64 | 24.25 | 24.06 | 46.08 | 33.49 |
| occany_5f_fwd | 25.21 | 45.64 | 36.03 | 25.93 | 36.83 | 46.69 |

### Streaming minus clip-reset, within each source and seed

Paired block bootstrap, 2,000 resamples, seed 0. `*` = interval excludes zero; `ns` = no difference detected, which is **not** demonstrated equivalence.

| source | seed | raw Δ IoU [95% CI] | pooled Δ IoU [95% CI] |
|---|---|---|---|
| k360 | 0 | -1.53 [-1.85, -1.17] * | -1.12 [-1.38, -0.84] * |
| k360 | 1 | -1.24 [-1.51, -0.95] * | -0.93 [-1.16, -0.68] * |
| k360 | 2 | -0.99 [-1.26, -0.69] * | -0.69 [-0.94, -0.44] * |
| waymo | 0 | -2.26 [-2.73, -1.74] * | -0.71 [-1.01, -0.40] * |
| waymo | 1 | -2.16 [-2.59, -1.70] * | -0.87 [-1.17, -0.56] * |
| waymo | 2 | -2.59 [-3.04, -2.07] * | -1.18 [-1.52, -0.82] * |
| once | 0 | -1.69 [-2.06, -1.31] * | -0.42 [-0.71, -0.12] * |
| once | 1 | -1.80 [-2.18, -1.41] * | -0.56 [-0.81, -0.28] * |
| once | 2 | -1.01 [-1.31, -0.69] * | -0.24 [-0.48, +0.00] ns |
| map-only (no head) | — | +0.06 [-0.49, +0.67] ns | -0.31 [-0.98, +0.40] ns |

### Per-seed IoU and the three-seed spread (a different quantity from the CI)

| source | mode | seed 0 | seed 1 | seed 2 | mean | max−min |
|---|---|---|---|---|---|---|
| **raw** | | | | | | |
| k360 | clip-reset | 26.40 | 26.57 | 25.99 | 26.32 | 0.58 |
| k360 | streaming | 24.87 | 25.32 | 25.00 | 25.06 | 0.46 |
| waymo | clip-reset | 23.55 | 22.54 | 22.84 | 22.98 | 1.01 |
| waymo | streaming | 21.30 | 20.38 | 20.25 | 20.64 | 1.05 |
| once | clip-reset | 18.20 | 19.03 | 17.69 | 18.31 | 1.34 |
| once | streaming | 16.51 | 17.23 | 16.68 | 16.81 | 0.73 |
| **pooled** | | | | | | |
| k360 | clip-reset | 26.34 | 25.66 | 25.66 | 25.89 | 0.68 |
| k360 | streaming | 25.23 | 24.74 | 24.97 | 24.98 | 0.49 |
| waymo | clip-reset | 23.37 | 22.31 | 22.96 | 22.88 | 1.06 |
| waymo | streaming | 22.67 | 21.44 | 21.78 | 21.96 | 1.22 |
| once | clip-reset | 20.26 | 20.82 | 20.24 | 20.44 | 0.58 |
| once | streaming | 19.84 | 20.26 | 20.00 | 20.04 | 0.42 |

### Source differences, held at one fixed inference mode

| contrast | mode | seed | raw Δ IoU [95% CI] | pooled Δ IoU [95% CI] |
|---|---|---|---|---|
| k360 − waymo | clip-reset | 0 | +2.84 [+2.26, +3.41] * | +2.97 [+2.40, +3.53] * |
| k360 − waymo | clip-reset | 1 | +4.03 [+3.41, +4.68] * | +3.35 [+2.75, +4.00] * |
| k360 − waymo | clip-reset | 2 | +3.15 [+2.54, +3.76] * | +2.70 [+2.09, +3.30] * |
| k360 − waymo | streaming | 0 | +3.57 [+2.85, +4.21] * | +2.56 [+1.97, +3.16] * |
| k360 − waymo | streaming | 1 | +4.94 [+4.16, +5.67] * | +3.30 [+2.61, +4.01] * |
| k360 − waymo | streaming | 2 | +4.76 [+3.91, +5.52] * | +3.19 [+2.43, +3.95] * |
| k360 − once | clip-reset | 0 | +8.20 [+7.32, +9.02] * | +6.08 [+5.25, +6.96] * |
| k360 − once | clip-reset | 1 | +7.54 [+6.57, +8.43] * | +4.84 [+3.97, +5.74] * |
| k360 − once | clip-reset | 2 | +8.30 [+7.37, +9.12] * | +5.42 [+4.50, +6.32] * |
| k360 − once | streaming | 0 | +8.36 [+7.45, +9.18] * | +5.38 [+4.50, +6.27] * |
| k360 − once | streaming | 1 | +8.09 [+7.11, +8.99] * | +4.47 [+3.57, +5.37] * |
| k360 − once | streaming | 2 | +8.32 [+7.43, +9.13] * | +4.96 [+4.07, +5.86] * |
| waymo − once | clip-reset | 0 | +5.36 [+4.86, +5.81] * | +3.12 [+2.66, +3.55] * |
| waymo − once | clip-reset | 1 | +3.51 [+3.01, +3.99] * | +1.49 [+1.03, +1.93] * |
| waymo − once | clip-reset | 2 | +5.15 [+4.60, +5.63] * | +2.72 [+2.22, +3.20] * |
| waymo − once | streaming | 0 | +4.79 [+4.16, +5.37] * | +2.82 [+2.33, +3.28] * |
| waymo − once | streaming | 1 | +3.15 [+2.65, +3.60] * | +1.18 [+0.79, +1.57] * |
| waymo − once | streaming | 2 | +3.56 [+3.08, +4.00] * | +1.77 [+1.35, +2.15] * |

### Completion minus map-only, under each mode

| source | seed | mode | raw Δ IoU [95% CI] | pooled Δ IoU [95% CI] |
|---|---|---|---|---|
| k360 | 0 | clip-reset | +17.61 [+16.55, +18.68] * | +11.30 [+10.11, +12.53] * |
| k360 | 0 | streaming | +16.02 [+15.17, +16.90] * | +10.49 [+9.53, +11.47] * |
| k360 | 1 | clip-reset | +17.78 [+16.76, +18.82] * | +10.62 [+9.43, +11.88] * |
| k360 | 1 | streaming | +16.48 [+15.61, +17.37] * | +10.00 [+9.02, +10.95] * |
| k360 | 2 | clip-reset | +17.20 [+16.18, +18.25] * | +10.62 [+9.45, +11.85] * |
| k360 | 2 | streaming | +16.16 [+15.30, +17.02] * | +10.23 [+9.29, +11.20] * |
| waymo | 0 | clip-reset | +14.77 [+13.83, +15.77] * | +8.33 [+7.42, +9.24] * |
| waymo | 0 | streaming | +12.45 [+11.63, +13.35] * | +7.93 [+7.12, +8.79] * |
| waymo | 1 | clip-reset | +13.76 [+12.93, +14.59] * | +7.27 [+6.44, +8.10] * |
| waymo | 1 | streaming | +11.54 [+10.92, +12.23] * | +6.71 [+6.07, +7.41] * |
| waymo | 2 | clip-reset | +14.05 [+13.07, +15.05] * | +7.92 [+6.99, +8.84] * |
| waymo | 2 | streaming | +11.40 [+10.67, +12.18] * | +7.04 [+6.30, +7.83] * |
| once | 0 | clip-reset | +9.41 [+8.55, +10.28] * | +5.21 [+4.31, +6.14] * |
| once | 0 | streaming | +7.66 [+7.03, +8.32] * | +5.11 [+4.35, +5.93] * |
| once | 1 | clip-reset | +10.24 [+9.27, +11.24] * | +5.78 [+4.80, +6.76] * |
| once | 1 | streaming | +8.39 [+7.67, +9.17] * | +5.53 [+4.74, +6.38] * |
| once | 2 | clip-reset | +8.90 [+8.00, +9.83] * | +5.20 [+4.24, +6.15] * |
| once | 2 | streaming | +7.84 [+7.17, +8.57] * | +5.27 [+4.46, +6.16] * |

### Against the fixed OccAny comparators (seed 0)

OccAny's forward configuration consumes **four future frames (~4 s)**; both of our modes are causal and consume none. These rows are not temporally matched and are never described as such.

| comparator | source | mode | raw Δ IoU [95% CI] | pooled Δ IoU [95% CI] |
|---|---|---|---|---|
| causal 1-frame | k360 | clip-reset | +6.08 [+4.82, +7.33] * | +2.28 [+1.23, +3.33] * |
| causal 1-frame | k360 | streaming | +4.55 [+3.35, +5.77] * | +1.16 [+0.20, +2.22] * |
| causal 1-frame | waymo | clip-reset | +3.23 [+2.26, +4.21] * | -0.69 [-1.52, +0.10] ns |
| causal 1-frame | waymo | streaming | +0.98 [-0.13, +2.06] ns | -1.40 [-2.28, -0.53] * |
| causal 1-frame | once | clip-reset | -2.12 [-3.06, -1.13] * | -3.80 [-4.69, -2.98] * |
| causal 1-frame | once | streaming | -3.81 [-4.92, -2.61] * | -4.22 [-5.17, -3.30] * |
| forward 5-frame | k360 | clip-reset | +1.19 [+0.35, +2.09] * | +0.41 [-0.34, +1.20] ns |
| forward 5-frame | k360 | streaming | -0.34 [-1.28, +0.63] ns | -0.70 [-1.56, +0.15] ns |
| forward 5-frame | waymo | clip-reset | -1.66 [-2.52, -0.79] * | -2.55 [-3.52, -1.65] * |
| forward 5-frame | waymo | streaming | -3.91 [-5.00, -2.82] * | -3.26 [-4.32, -2.28] * |
| forward 5-frame | once | clip-reset | -7.01 [-7.97, -6.04] * | -5.67 [-6.68, -4.67] * |
| forward 5-frame | once | streaming | -8.70 [-9.80, -7.58] * | -6.09 [-7.19, -4.99] * |

### Frustum regions (global map-evidence-free cut)

| region | share of valid | share of GT occupied |
|---|---|---|
| all | 100.00% | 100.00% |
| in_frustum | 69.34% | 74.53% |
| out_frustum | 30.66% | 25.47% |

| arm | in-frustum raw | out-frustum raw | in-frustum pooled | out-frustum pooled |
|---|---|---|---|---|
| clip-reset / k360_s0 | 25.16 | 29.98 | 25.76 | 28.01 |
| clip-reset / k360_s1 | 25.77 | 28.98 | 25.23 | 26.93 |
| clip-reset / k360_s2 | 25.01 | 28.86 | 25.24 | 26.87 |
| clip-reset / waymo_s0 | 22.38 | 26.58 | 23.05 | 24.19 |
| clip-reset / waymo_s1 | 21.50 | 25.16 | 22.00 | 23.07 |
| clip-reset / waymo_s2 | 21.46 | 26.32 | 22.55 | 23.97 |
| clip-reset / once_s0 | 16.72 | 21.55 | 20.27 | 20.22 |
| clip-reset / once_s1 | 17.84 | 21.66 | 21.20 | 20.04 |
| clip-reset / once_s2 | 15.85 | 21.90 | 19.92 | 20.91 |
| clip-reset / map_only | 8.51 | 9.50 | 15.42 | 14.18 |
| streaming / k360_s0 | 23.96 | 27.39 | 24.95 | 25.98 |
| streaming / k360_s1 | 24.90 | 26.56 | 24.63 | 25.03 |
| streaming / k360_s2 | 24.42 | 26.63 | 24.84 | 25.33 |
| streaming / waymo_s0 | 20.08 | 24.41 | 22.26 | 23.67 |
| streaming / waymo_s1 | 19.28 | 23.12 | 21.01 | 22.47 |
| streaming / waymo_s2 | 18.68 | 24.15 | 21.10 | 23.39 |
| streaming / once_s0 | 14.45 | 21.25 | 19.25 | 21.11 |
| streaming / once_s1 | 15.46 | 21.15 | 20.14 | 20.51 |
| streaming / once_s2 | 14.41 | 21.94 | 19.23 | 21.66 |
| streaming / map_only | 9.19 | 7.95 | 15.68 | 12.50 |
| occany_1f | 24.16 | 7.43 | 27.33 | 11.85 |
| occany_5f_fwd | 29.55 | 8.04 | 29.01 | 11.59 |

### SemanticKITTI block-length sensitivity (one physical stream)

SemanticKITTI is a single stream, so no block length turns these into between-stream intervals. The widening shows how much the canonical 25-anchor block understates within-stream correlation; the 801 row is one block and its interval is degenerate.

| contrast | L=25 | L=50 | L=100 | L=200 | L=400 | L=801 |
|---|---|---|---|---|---|---|
| k360_s0 | -1.38,-0.84* | -1.41,-0.80* | -1.43,-0.89* | -1.41,-0.89* | -1.22,+1.81 | -1.12,-1.12* |
| k360_s1 | -1.16,-0.68* | -1.17,-0.66* | -1.16,-0.68* | -1.19,-0.62* | -1.19,+2.54 | -0.93,-0.93* |
| k360_s2 | -0.94,-0.44* | -0.93,-0.45* | -0.88,-0.50* | -0.89,-0.42* | -0.81,+2.95 | -0.69,-0.69* |
| waymo_s0 | -1.01,-0.40* | -0.98,-0.44* | -0.94,-0.50* | -1.05,-0.47* | -1.01,+3.22 | -0.71,-0.71* |
| waymo_s1 | -1.17,-0.56* | -1.14,-0.60* | -1.03,-0.71* | -1.08,-0.75* | -1.04,+3.96 | -0.87,-0.87* |
| waymo_s2 | -1.52,-0.82* | -1.49,-0.83* | -1.51,-0.88* | -1.61,-0.87* | -1.57,+2.85 | -1.18,-1.18* |
| once_s0 | -0.71,-0.12* | -0.66,-0.17* | -0.62,-0.17* | -0.61,-0.26* | -0.56,+0.83 | -0.42,-0.42* |
| once_s1 | -0.81,-0.28* | -0.83,-0.27* | -0.79,-0.29* | -0.76,-0.41* | -0.59,+1.15 | -0.56,-0.56* |
| once_s2 | -0.48,+0.00 | -0.46,-0.01* | -0.44,-0.02* | -0.34,-0.13* | -0.34,+2.60 | -0.24,-0.24* |
| map_only | -0.98,+0.40 | -0.96,+0.48 | -0.92,+0.39 | -0.88,+0.18 | -0.63,+6.08 | -0.31,-0.31* |

## occ3d

### occ3d — global SC IoU, 4219 anchors (canonical population 4219, exclusions 0)

| arm | raw IoU | raw P | raw R | pooled IoU | pooled P | pooled R |
|---|---|---|---|---|---|---|
| clip-reset / k360_s0 | 31.94 | 77.87 | 35.13 | 41.56 | 66.49 | 52.57 |
| clip-reset / k360_s1 | 33.21 | 76.47 | 36.99 | 41.41 | 64.22 | 53.83 |
| clip-reset / k360_s2 | 29.77 | 72.77 | 33.51 | 39.77 | 62.75 | 52.06 |
| clip-reset / waymo_s0 | 22.85 | 66.46 | 25.84 | 38.05 | 62.09 | 49.57 |
| clip-reset / waymo_s1 | 22.93 | 65.09 | 26.15 | 36.82 | 60.33 | 48.58 |
| clip-reset / waymo_s2 | 22.65 | 67.58 | 25.42 | 38.20 | 63.35 | 49.04 |
| clip-reset / once_s0 | 18.67 | 52.66 | 22.44 | 30.52 | 49.60 | 44.25 |
| clip-reset / once_s1 | 19.56 | 50.04 | 24.31 | 30.86 | 46.78 | 47.56 |
| clip-reset / once_s2 | 19.27 | 52.66 | 23.30 | 30.66 | 49.07 | 44.97 |
| clip-reset / map_only | 9.80 | 60.06 | 10.49 | 22.16 | 53.76 | 27.38 |
| streaming / k360_s0 | 32.93 | 76.84 | 36.56 | 41.03 | 64.71 | 52.85 |
| streaming / k360_s1 | 33.79 | 75.42 | 37.98 | 40.72 | 62.67 | 53.75 |
| streaming / k360_s2 | 30.79 | 71.55 | 35.09 | 39.24 | 60.89 | 52.47 |
| streaming / waymo_s0 | 21.91 | 69.82 | 24.20 | 37.49 | 65.09 | 46.93 |
| streaming / waymo_s1 | 22.20 | 68.99 | 24.66 | 36.35 | 63.68 | 45.85 |
| streaming / waymo_s2 | 21.67 | 70.91 | 23.78 | 37.52 | 66.41 | 46.30 |
| streaming / once_s0 | 17.72 | 56.56 | 20.52 | 30.33 | 53.41 | 41.25 |
| streaming / once_s1 | 18.87 | 54.30 | 22.43 | 31.13 | 50.75 | 44.60 |
| streaming / once_s2 | 18.92 | 57.35 | 22.02 | 30.91 | 53.17 | 42.47 |
| streaming / map_only | 9.33 | 62.96 | 9.87 | 20.96 | 56.25 | 25.04 |
| occany_5f_fwd | 20.66 | 42.96 | 28.47 | 23.55 | 36.28 | 40.16 |

### Streaming minus clip-reset, within each source and seed

Paired block bootstrap, 2,000 resamples, seed 0. `*` = interval excludes zero; `ns` = no difference detected, which is **not** demonstrated equivalence.

| source | seed | raw Δ IoU [95% CI] | pooled Δ IoU [95% CI] |
|---|---|---|---|
| k360 | 0 | +0.99 [+0.18, +1.79] * | -0.53 [-0.90, -0.15] * |
| k360 | 1 | +0.58 [-0.23, +1.39] ns | -0.69 [-1.07, -0.33] * |
| k360 | 2 | +1.02 [+0.36, +1.67] * | -0.53 [-0.86, -0.18] * |
| waymo | 0 | -0.95 [-1.79, -0.10] * | -0.56 [-1.29, +0.19] ns |
| waymo | 1 | -0.73 [-1.58, +0.10] ns | -0.48 [-1.14, +0.17] ns |
| waymo | 2 | -0.99 [-1.87, -0.15] * | -0.68 [-1.40, +0.06] ns |
| once | 0 | -0.95 [-1.50, -0.43] * | -0.19 [-0.68, +0.30] ns |
| once | 1 | -0.69 [-1.23, -0.19] * | +0.27 [-0.24, +0.77] ns |
| once | 2 | -0.35 [-0.88, +0.14] ns | +0.25 [-0.21, +0.74] ns |
| map-only (no head) | — | -0.47 [-1.19, +0.22] ns | -1.20 [-2.18, -0.17] * |

### Per-seed IoU and the three-seed spread (a different quantity from the CI)

| source | mode | seed 0 | seed 1 | seed 2 | mean | max−min |
|---|---|---|---|---|---|---|
| **raw** | | | | | | |
| k360 | clip-reset | 31.94 | 33.21 | 29.77 | 31.64 | 3.44 |
| k360 | streaming | 32.93 | 33.79 | 30.79 | 32.50 | 3.00 |
| waymo | clip-reset | 22.85 | 22.93 | 22.65 | 22.81 | 0.28 |
| waymo | streaming | 21.91 | 22.20 | 21.67 | 21.93 | 0.53 |
| once | clip-reset | 18.67 | 19.56 | 19.27 | 19.17 | 0.89 |
| once | streaming | 17.72 | 18.87 | 18.92 | 18.50 | 1.20 |
| **pooled** | | | | | | |
| k360 | clip-reset | 41.56 | 41.41 | 39.77 | 40.91 | 1.79 |
| k360 | streaming | 41.03 | 40.72 | 39.24 | 40.33 | 1.78 |
| waymo | clip-reset | 38.05 | 36.82 | 38.20 | 37.69 | 1.38 |
| waymo | streaming | 37.49 | 36.35 | 37.52 | 37.12 | 1.17 |
| once | clip-reset | 30.52 | 30.86 | 30.66 | 30.68 | 0.34 |
| once | streaming | 30.33 | 31.13 | 30.91 | 30.79 | 0.79 |

### Source differences, held at one fixed inference mode

| contrast | mode | seed | raw Δ IoU [95% CI] | pooled Δ IoU [95% CI] |
|---|---|---|---|---|
| k360 − waymo | clip-reset | 0 | +9.09 [+8.00, +10.19] * | +3.51 [+2.51, +4.47] * |
| k360 − waymo | clip-reset | 1 | +10.28 [+9.22, +11.29] * | +4.59 [+3.55, +5.60] * |
| k360 − waymo | clip-reset | 2 | +7.12 [+6.11, +8.12] * | +1.57 [+0.57, +2.54] * |
| k360 − waymo | streaming | 0 | +11.02 [+9.92, +12.18] * | +3.54 [+2.44, +4.64] * |
| k360 − waymo | streaming | 1 | +11.60 [+10.52, +12.66] * | +4.37 [+3.27, +5.47] * |
| k360 − waymo | streaming | 2 | +9.12 [+8.04, +10.18] * | +1.73 [+0.56, +2.91] * |
| k360 − once | clip-reset | 0 | +13.27 [+11.94, +14.61] * | +11.04 [+9.90, +12.13] * |
| k360 − once | clip-reset | 1 | +13.65 [+12.41, +14.90] * | +10.55 [+9.40, +11.60] * |
| k360 − once | clip-reset | 2 | +10.51 [+9.48, +11.51] * | +9.11 [+7.98, +10.19] * |
| k360 − once | streaming | 0 | +15.21 [+13.90, +16.55] * | +10.69 [+9.45, +11.84] * |
| k360 − once | streaming | 1 | +14.93 [+13.62, +16.30] * | +9.59 [+8.31, +10.80] * |
| k360 − once | streaming | 2 | +11.87 [+10.80, +12.92] * | +8.33 [+7.08, +9.54] * |
| waymo − once | clip-reset | 0 | +4.18 [+3.51, +4.89] * | +7.53 [+6.76, +8.35] * |
| waymo − once | clip-reset | 1 | +3.37 [+2.75, +4.04] * | +5.96 [+5.27, +6.71] * |
| waymo − once | clip-reset | 2 | +3.39 [+2.71, +4.06] * | +7.54 [+6.78, +8.35] * |
| waymo − once | streaming | 0 | +4.19 [+3.48, +4.94] * | +7.16 [+6.31, +8.09] * |
| waymo − once | streaming | 1 | +3.33 [+2.67, +4.06] * | +5.22 [+4.45, +6.06] * |
| waymo − once | streaming | 2 | +2.75 [+2.03, +3.48] * | +6.61 [+5.74, +7.57] * |

### Completion minus map-only, under each mode

| source | seed | mode | raw Δ IoU [95% CI] | pooled Δ IoU [95% CI] |
|---|---|---|---|---|
| k360 | 0 | clip-reset | +22.14 [+20.60, +23.75] * | +19.40 [+17.74, +21.10] * |
| k360 | 0 | streaming | +23.60 [+22.04, +25.22] * | +20.07 [+18.29, +21.88] * |
| k360 | 1 | clip-reset | +23.41 [+21.95, +24.92] * | +19.25 [+17.53, +20.95] * |
| k360 | 1 | streaming | +24.46 [+22.95, +25.96] * | +19.76 [+17.93, +21.59] * |
| k360 | 2 | clip-reset | +19.97 [+18.64, +21.40] * | +17.61 [+15.93, +19.30] * |
| k360 | 2 | streaming | +21.46 [+20.07, +22.87] * | +18.28 [+16.45, +20.12] * |
| waymo | 0 | clip-reset | +13.05 [+12.16, +13.98] * | +15.90 [+14.51, +17.33] * |
| waymo | 0 | streaming | +12.58 [+11.65, +13.52] * | +16.53 [+15.05, +18.06] * |
| waymo | 1 | clip-reset | +13.13 [+12.27, +14.01] * | +14.66 [+13.38, +16.04] * |
| waymo | 1 | streaming | +12.87 [+11.95, +13.81] * | +15.38 [+13.97, +16.86] * |
| waymo | 2 | clip-reset | +12.85 [+11.97, +13.79] * | +16.04 [+14.71, +17.44] * |
| waymo | 2 | streaming | +12.34 [+11.48, +13.25] * | +16.56 [+15.16, +18.00] * |
| once | 0 | clip-reset | +8.87 [+8.15, +9.59] * | +8.37 [+6.95, +9.79] * |
| once | 0 | streaming | +8.39 [+7.56, +9.25] * | +9.37 [+7.68, +11.01] * |
| once | 1 | clip-reset | +9.76 [+9.02, +10.51] * | +8.70 [+7.21, +10.20] * |
| once | 1 | streaming | +9.53 [+8.72, +10.40] * | +10.17 [+8.43, +11.87] * |
| once | 2 | clip-reset | +9.46 [+8.69, +10.25] * | +8.50 [+7.02, +10.00] * |
| once | 2 | streaming | +9.59 [+8.70, +10.53] * | +9.95 [+8.18, +11.64] * |

### Against the fixed OccAny comparators (seed 0)

OccAny's forward configuration consumes **four future frames (~4 s)**; both of our modes are causal and consume none. These rows are not temporally matched and are never described as such.

| comparator | source | mode | raw Δ IoU [95% CI] | pooled Δ IoU [95% CI] |
|---|---|---|---|---|
| forward 5-frame | k360 | clip-reset | +11.28 [+9.25, +13.37] * | +18.01 [+16.67, +19.34] * |
| forward 5-frame | k360 | streaming | +12.27 [+10.21, +14.36] * | +17.48 [+16.20, +18.79] * |
| forward 5-frame | waymo | clip-reset | +2.20 [-0.02, +4.35] ns | +14.51 [+12.92, +16.06] * |
| forward 5-frame | waymo | streaming | +1.25 [-0.96, +3.51] ns | +13.94 [+12.32, +15.58] * |
| forward 5-frame | once | clip-reset | -1.99 [-3.94, -0.08] * | +6.98 [+5.57, +8.42] * |
| forward 5-frame | once | streaming | -2.94 [-4.93, -1.04] * | +6.79 [+5.25, +8.31] * |

### Frustum regions (global map-evidence-free cut)

| region | share of valid | share of GT occupied |
|---|---|---|
| all | 100.00% | 100.00% |
| in_frustum | 50.95% | 45.51% |
| out_frustum | 49.05% | 54.49% |

| arm | in-frustum raw | out-frustum raw | in-frustum pooled | out-frustum pooled |
|---|---|---|---|---|
| clip-reset / k360_s0 | 36.18 | 28.25 | 46.45 | 37.09 |
| clip-reset / k360_s1 | 38.05 | 28.96 | 46.05 | 37.11 |
| clip-reset / k360_s2 | 33.46 | 26.55 | 43.82 | 36.06 |
| clip-reset / waymo_s0 | 22.86 | 22.85 | 40.55 | 35.71 |
| clip-reset / waymo_s1 | 23.94 | 22.02 | 39.72 | 34.08 |
| clip-reset / waymo_s2 | 23.05 | 22.30 | 40.94 | 35.64 |
| clip-reset / once_s0 | 17.64 | 19.54 | 31.88 | 29.37 |
| clip-reset / once_s1 | 18.45 | 20.48 | 33.11 | 29.02 |
| clip-reset / once_s2 | 17.95 | 20.37 | 31.92 | 29.60 |
| clip-reset / map_only | 11.59 | 8.26 | 26.88 | 17.86 |
| streaming / k360_s0 | 37.69 | 28.81 | 46.14 | 36.41 |
| streaming / k360_s1 | 38.98 | 29.29 | 45.51 | 36.34 |
| streaming / k360_s2 | 35.14 | 27.01 | 43.55 | 35.32 |
| streaming / waymo_s0 | 22.01 | 21.82 | 40.86 | 34.45 |
| streaming / waymo_s1 | 23.27 | 21.26 | 40.06 | 32.97 |
| streaming / waymo_s2 | 22.20 | 21.21 | 41.29 | 34.15 |
| streaming / once_s0 | 16.48 | 18.76 | 31.79 | 29.12 |
| streaming / once_s1 | 17.46 | 20.02 | 33.61 | 29.14 |
| streaming / once_s2 | 17.20 | 20.36 | 32.23 | 29.82 |
| streaming / map_only | 10.93 | 7.97 | 25.89 | 16.58 |
| occany_5f_fwd | 29.98 | 8.94 | 30.72 | 12.55 |

