# OccAny comparison — results

Pooled SC IoU / precision / recall in %, OccAny's `SSCMetrics`, OccAny's stored labels. Ours: three frozen seeds, each pooled separately; mean and median over seeds. Δ = seed-mean IoU(ours) − IoU(OccAny) on the same samples, paired block bootstrap 95 % CI (2,000 draws; blocks: Occ3D scenes, SemanticKITTI runs of 25 target frames).

## SemanticKITTI 08 — `occany_all` (805 samples, 33 blocks)

_every sample OccAny scored (805 / 4,819)_

| method | protocol | placement | post | IoU seeds 0/1/2 | IoU mean | IoU median | P | R | pred/GT | Δ vs OccAny (same post) | reading | map-only IoU |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|---:|
| OccAny (released) | forward | occany | raw | — | 25.20 | 25.20 | 45.59 | 36.05 | 0.79 | — | — | — |
| OccAny (released) | forward | occany | pooled | — | 25.92 | 25.92 | 36.80 | 46.70 | 1.27 | — | — | — |
| ours (frozen) | forward | shared | raw | 30.27 / 29.70 / 29.49 | 29.82 | 29.70 | 39.91 | 54.11 | 1.36 | +4.61 [+3.73, +5.50] | higher | 11.40 |
| ours (frozen) | forward | shared | pooled | 29.05 / 28.35 / 28.57 | 28.66 | 28.57 | 34.45 | 63.01 | 1.83 | +2.74 [+2.05, +3.43] | higher | 17.52 |

## SemanticKITTI 08 — `causal_eligible` (801 samples, 33 blocks)

_samples with four earlier stream frames (the causal protocol needs them)_

| method | protocol | placement | post | IoU seeds 0/1/2 | IoU mean | IoU median | P | R | pred/GT | Δ vs OccAny (same post) | reading | map-only IoU |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|---:|
| OccAny (released) | forward | occany | raw | — | 25.21 | 25.21 | 45.64 | 36.03 | 0.79 | — | — | — |
| OccAny (released) | forward | occany | pooled | — | 25.93 | 25.93 | 36.83 | 46.69 | 1.27 | — | — | — |
| ours (frozen) | forward | shared | raw | 30.27 / 29.70 / 29.49 | 29.82 | 29.70 | 39.94 | 54.07 | 1.35 | +4.61 [+3.72, +5.50] | higher | 11.40 |
| ours (frozen) | forward | shared | pooled | 29.05 / 28.36 / 28.58 | 28.66 | 28.58 | 34.47 | 62.98 | 1.83 | +2.73 [+2.04, +3.42] | higher | 17.53 |
| ours (frozen) | causal | shared | raw | 26.40 / 26.57 / 25.99 | 26.32 | 26.40 | 36.40 | 48.79 | 1.34 | +1.11 [+0.30, +1.93] | higher | 8.79 |
| ours (frozen) | causal | shared | pooled | 26.34 / 25.66 / 25.66 | 25.89 | 25.66 | 31.99 | 57.66 | 1.80 | -0.04 [-0.77, +0.70] | not distinguishable | 15.04 |

## SemanticKITTI 08 — `official_anchors` (161 samples, 33 blocks)

_our reported anchors (163 / 1,182) that OccAny also scored and both protocols can run_

| method | protocol | placement | post | IoU seeds 0/1/2 | IoU mean | IoU median | P | R | pred/GT | Δ vs OccAny (same post) | reading | map-only IoU |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|---:|
| OccAny (released) | forward | occany | raw | — | 25.28 | 25.28 | 45.49 | 36.26 | 0.80 | — | — | — |
| OccAny (released) | forward | occany | pooled | — | 25.92 | 25.92 | 36.68 | 46.93 | 1.28 | — | — | — |
| ours (frozen) | forward | shared | raw | 30.30 / 29.63 / 29.44 | 29.79 | 29.63 | 39.72 | 54.37 | 1.37 | +4.51 [+3.61, +5.43] | higher | 11.42 |
| ours (frozen) | forward | shared | pooled | 29.03 / 28.23 / 28.53 | 28.60 | 28.53 | 34.30 | 63.26 | 1.84 | +2.67 [+1.95, +3.39] | higher | 17.55 |
| ours (frozen) | causal | shared | raw | 26.47 / 26.66 / 26.05 | 26.39 | 26.47 | 36.32 | 49.19 | 1.36 | +1.12 [+0.35, +1.89] | higher | 8.87 |
| ours (frozen) | causal | shared | pooled | 26.29 / 25.64 / 25.63 | 25.85 | 25.64 | 31.85 | 57.92 | 1.82 | -0.07 [-0.74, +0.61] | not distinguishable | 15.17 |

## Occ3D-nuScenes val — `occany_all` (4,819 samples, 150 blocks)

_every sample OccAny scored (805 / 4,819)_

| method | protocol | placement | post | IoU seeds 0/1/2 | IoU mean | IoU median | P | R | pred/GT | Δ vs OccAny (same post) | reading | map-only IoU |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|---:|
| OccAny (released) | forward | occany | raw | — | 20.67 | 20.67 | 42.67 | 28.61 | 0.67 | — | — | — |
| OccAny (released) | forward | occany | pooled | — | 23.56 | 23.56 | 36.11 | 40.39 | 1.12 | — | — | — |
| ours (frozen) | forward | occany | raw | 25.45 / 26.40 / 23.53 | 25.13 | 25.45 | 79.07 | 26.91 | 0.34 | +4.46 [+2.30, +6.62] | higher | 6.94 |
| ours (frozen) | forward | occany | pooled | 40.63 / 40.70 / 38.94 | 40.09 | 40.63 | 71.59 | 47.68 | 0.67 | +16.54 [+15.05, +18.03] | higher | 17.94 |
| ours (frozen) | forward | ours | raw | 25.49 / 26.44 / 23.56 | 25.16 | 25.49 | 79.08 | 26.95 | 0.34 | +4.50 [+2.34, +6.65] | higher | 7.01 |
| ours (frozen) | forward | ours | pooled | 40.60 / 40.66 / 38.91 | 40.06 | 40.60 | 71.52 | 47.66 | 0.67 | +16.50 [+15.02, +17.99] | higher | 18.09 |

## Occ3D-nuScenes val — `causal_eligible` (4,219 samples, 150 blocks)

_samples with four earlier stream frames (the causal protocol needs them)_

| method | protocol | placement | post | IoU seeds 0/1/2 | IoU mean | IoU median | P | R | pred/GT | Δ vs OccAny (same post) | reading | map-only IoU |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|---:|
| OccAny (released) | forward | occany | raw | — | 20.66 | 20.66 | 42.96 | 28.47 | 0.66 | — | — | — |
| OccAny (released) | forward | occany | pooled | — | 23.55 | 23.55 | 36.28 | 40.16 | 1.11 | — | — | — |
| ours (frozen) | forward | occany | raw | 25.26 / 26.22 / 23.37 | 24.95 | 25.26 | 79.14 | 26.70 | 0.34 | +4.29 [+2.07, +6.46] | higher | 6.91 |
| ours (frozen) | forward | occany | pooled | 40.59 / 40.64 / 38.92 | 40.05 | 40.59 | 71.77 | 47.54 | 0.66 | +16.50 [+14.99, +18.00] | higher | 17.91 |
| ours (frozen) | forward | ours | raw | 25.29 / 26.26 / 23.40 | 24.98 | 25.29 | 79.15 | 26.74 | 0.34 | +4.32 [+2.11, +6.49] | higher | 6.98 |
| ours (frozen) | forward | ours | pooled | 40.56 / 40.60 / 38.89 | 40.02 | 40.56 | 71.70 | 47.52 | 0.66 | +16.47 [+14.96, +17.97] | higher | 18.06 |
| ours (frozen) | causal | occany | raw | 31.93 / 33.20 / 29.75 | 31.63 | 31.93 | 75.65 | 35.20 | 0.47 | +10.97 [+8.99, +12.96] | higher | 9.82 |
| ours (frozen) | causal | occany | pooled | 41.53 / 41.39 / 39.74 | 40.88 | 41.39 | 64.45 | 52.80 | 0.82 | +17.34 [+16.02, +18.63] | higher | 22.19 |
| ours (frozen) | causal | ours | raw | 31.94 / 33.21 / 29.77 | 31.64 | 31.94 | 75.70 | 35.21 | 0.47 | +10.98 [+9.01, +12.97] | higher | 9.80 |
| ours (frozen) | causal | ours | pooled | 41.56 / 41.41 / 39.77 | 40.91 | 41.41 | 64.48 | 52.82 | 0.82 | +17.36 [+16.04, +18.66] | higher | 22.16 |

## Occ3D-nuScenes val — `official_anchors` (900 samples, 150 blocks)

_our reported anchors (163 / 1,182) that OccAny also scored and both protocols can run_

| method | protocol | placement | post | IoU seeds 0/1/2 | IoU mean | IoU median | P | R | pred/GT | Δ vs OccAny (same post) | reading | map-only IoU |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|---:|
| OccAny (released) | forward | occany | raw | — | 20.64 | 20.64 | 42.67 | 28.57 | 0.67 | — | — | — |
| OccAny (released) | forward | occany | pooled | — | 23.56 | 23.56 | 36.11 | 40.40 | 1.12 | — | — | — |
| ours (frozen) | forward | occany | raw | 25.27 / 26.30 / 23.48 | 25.02 | 25.27 | 78.99 | 26.79 | 0.34 | +4.37 [+2.15, +6.55] | higher | 6.98 |
| ours (frozen) | forward | occany | pooled | 40.56 / 40.60 / 38.90 | 40.02 | 40.56 | 71.59 | 47.57 | 0.66 | +16.46 [+14.97, +17.99] | higher | 17.76 |
| ours (frozen) | forward | ours | raw | 25.32 / 26.35 / 23.51 | 25.06 | 25.32 | 78.94 | 26.85 | 0.34 | +4.42 [+2.19, +6.61] | higher | 7.06 |
| ours (frozen) | forward | ours | pooled | 40.53 / 40.54 / 38.85 | 39.97 | 40.53 | 71.47 | 47.56 | 0.67 | +16.41 [+14.93, +17.94] | higher | 17.94 |
| ours (frozen) | causal | occany | raw | 31.90 / 33.13 / 29.72 | 31.59 | 31.90 | 75.48 | 35.19 | 0.47 | +10.94 [+8.98, +12.90] | higher | 9.76 |
| ours (frozen) | causal | occany | pooled | 41.50 / 41.26 / 39.68 | 40.81 | 41.26 | 64.29 | 52.79 | 0.82 | +17.25 [+15.99, +18.56] | higher | 22.03 |
| ours (frozen) | causal | ours | raw | 31.90 / 33.13 / 29.72 | 31.58 | 31.90 | 75.53 | 35.17 | 0.47 | +10.94 [+8.97, +12.91] | higher | 9.76 |
| ours (frozen) | causal | ours | pooled | 41.53 / 41.27 / 39.71 | 40.84 | 41.27 | 64.33 | 52.80 | 0.82 | +17.28 [+16.01, +18.58] | higher | 22.00 |
