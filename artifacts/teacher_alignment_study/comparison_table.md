# Teacher-alignment control — R (aligned) vs S (shuffled)

Same init, same 24 000 paired data draws, same occupancy targets, same auxiliary
gradient budget at the shared trunk (ρ = 3.5644, achieved on 5 999/6 000 steps of every run;
the one degenerate step is step 1, where the zero-initialised occupancy head gives
‖g_occ‖ = 0). Bars = per-seed values; ± = 95 % bootstrap CI of the paired R−S difference
(contiguous 20-anchor blocks on drive 0006, whole scenes on the targets, 2 000 resamples).
A/B rows are the feature study's nominal-weight runs, reference only (different gradient budget).

## Occupancy (SC IoU at the frozen source-only threshold)

| | seed 0 | seed 1 | seed 2 | median |
|---|---|---|---|---|
| **drive 0006 later-measured** | | | | |
| R | 0.1019 | 0.1023 | 0.1051 | 0.1023 |
| S | 0.1081 | 0.1047 | 0.1000 | 0.1047 |
| R−S | −0.0063 [−0.0081, −0.0045] | −0.0024 [−0.0039, −0.0009] | +0.0050 [+0.0036, +0.0065] | mean −0.0012 |
| A (ref) | 0.1080 | 0.1118 | 0.1114 | 0.1114 |
| B (ref) | 0.1037 | 0.1069 | 0.1031 | 0.1037 |
| **drive 0006 observed** | | | | |
| R | 0.1056 | 0.1037 | 0.1073 | 0.1056 |
| S | 0.1055 | 0.1042 | 0.1036 | 0.1042 |
| R−S | +0.0000 [−0.0008, +0.0009] | −0.0005 [−0.0010, +0.0000] | +0.0038 [+0.0031, +0.0045] | mean +0.0011 |
| **SemanticKITTI past5 (163)** | | | | |
| R | 0.2067 | 0.2126 | 0.2252 | 0.2126 |
| S | 0.2284 | 0.2316 | 0.2118 | 0.2284 |
| R−S | −0.0218 [−0.0263, −0.0190] | −0.0190 [−0.0232, −0.0175] | +0.0134 [+0.0094, +0.0179] | mean −0.0091 |
| A (ref) | 0.2195 | 0.2255 | 0.2270 | 0.2255 |
| B (ref) | 0.2295 | 0.2377 | 0.2284 | 0.2295 |
| **Occ3D past5 (1 182)** | | | | |
| R | 0.2867 | 0.3309 | 0.3165 | 0.3165 |
| S | 0.3226 | 0.3322 | 0.2780 | 0.3226 |
| R−S | −0.0360 [−0.0411, −0.0310] | −0.0012 [−0.0061, +0.0038] | +0.0385 [+0.0322, +0.0460] | mean +0.0004 |
| A (ref) | 0.2598 | 0.2854 | 0.2941 | 0.2854 |
| B (ref) | 0.2999 | 0.3134 | 0.3317 | 0.3134 |

Frozen thresholds τ: R +0.0000 / −0.3125 / −0.3125; S −0.2188 / −0.2812 / −0.0938.
The sign of the transfer difference follows the threshold gap (more negative τ → more recall
on the sparser targets), not the arm.

## Threshold-free occupancy ranking on the targets

| | metric | R (s0/s1/s2) | S (s0/s1/s2) | R−S |
|---|---|---|---|---|
| SemanticKITTI | AUROC | 0.803 / 0.789 / 0.797 | 0.809 / 0.807 / 0.796 | −0.006 / −0.018 / +0.001 |
| SemanticKITTI | best-τ IoU (oracle) | 0.224 / 0.215 / 0.229 | 0.236 / 0.238 / 0.229 | −0.011 / −0.023 / +0.001 |
| Occ3D | AUROC | 0.723 / 0.728 / 0.738 | 0.746 / 0.728 / 0.736 | −0.023 / −0.000 / +0.002 |
| Occ3D | best-τ IoU (oracle) | 0.356 / 0.361 / 0.361 | 0.379 / 0.379 / 0.365 | −0.023 / −0.018 / −0.004 |

## Semantics

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| aligned-teacher top-1 agreement, drive 0006 (R / S) | 0.412 / 0.399 | 0.409 / 0.392 | 0.398 / 0.383 |
| SemanticKITTI SSC mIoU (R / S) | 0.0329 / 0.0316 | 0.0333 / 0.0305 | 0.0320 / 0.0266 |
| SemanticKITTI naming acc. on common set (R / S), coverage | 0.394 / 0.379, 33.0 % | 0.365 / 0.355, 38.2 % | 0.378 / 0.355, 33.8 % |
| Occ3D SSC mIoU (R / S) | 0.0395 / 0.0392 | 0.0402 / 0.0405 | 0.0422 / 0.0460 |
| Occ3D naming acc. on common set (R / S), coverage | 0.316 / 0.297, 30.7 % | 0.297 / 0.295, 33.6 % | 0.373 / 0.500 *, 28.6 % |

\* Occ3D seed 2: S's higher threshold (−0.094) restricts the common set to the
highest-confidence voxels, which are `driveable_surface` (S IoU 0.309 vs R 0.249 on that class);
every other class is within ±0.003. The common-set accuracy there measures the set's road
fraction, not naming skill. Per-class IoU/support for every run: `semantics.json`.

## Manipulation and budget (training, all steps)

| | R | S |
|---|---|---|
| supervision vectors changed by the shuffle | 0 % | 46.4 / 46.4 / 46.8 % |
| argmax class changed | 0 % | 30.5 / 30.5 / 30.8 % |
| mean L1 between aligned and delivered vector | 0 | 0.584 / 0.586 / 0.589 |
| supervision inside class-homogeneous strata | — | 14.0 / 13.7 / 14.3 % |
| supervision dropped (strata < 2) | 0.02 % | 0.02 % |
| raw ‖g_aux‖/‖g_occ‖ median (c = ρ·n_occ/n_aux) | 3.72 / 3.95 / 3.74 (c 0.96 / 0.90 / 0.95) | 3.37 / 3.60 / 3.39 (c 1.06 / 0.99 / 1.05) |
| achieved ratio | 3.564 on 5 999/6 000 steps | 3.564 on 5 999/6 000 steps |
| cos(g_occ, g_aux) median | +0.012 / +0.016 / +0.027 | +0.031 / +0.050 / +0.033 |
| final L_occ / L_KL (train) | 1.060 / 1.040 · 1.067 / 0.989 · 1.064 / 1.090 | 1.059 / 1.071 · 1.067 / 1.019 · 1.066 / 1.129 |
| validation top-1 agreement with the aligned teacher | 0.445 / 0.444 / 0.430 | 0.418 / 0.445 / 0.406 |
