# Gate 13 — oracle observed-geometry completion ceiling

**Decision A: observed geometry is the dominant completion bottleneck.**

The frozen completion model and recipe from Gate 12 were trained twice: on the realistic
monocular hard input M, and on an oracle input O. O is the five causal LiDAR sweeps with GT
relative poses, fed through the same hard mapper. Everything else was byte-identical per seed.

O lifts KITTI-360 0006 completion IoU from **0.135 to 0.333 (+0.199, +148 %)**:
- **Seeds:** every seed gains, with a tight CI.
- **Range:** every range bin gains.
- **Unobserved space:** space never observed by the five causal sweeps gains **+0.128**.
- **Plateaus:** O settles on a clearly higher plateau, rather than converging faster.

This is a source-side diagnostic only. SemanticKITTI and Occ3D-nuScenes were not accessed, and
O is not deployable. Verification passes 22/22 (`verification.md`).

| 0006 completion IoU (590 anchors, final step 6000) | seed 0 | seed 1 | seed 2 | mean | precision | recall |
|---|---|---|---|---|---|---|
| M: monocular hard (Gate 12 H input) | 0.1354 | 0.1336 | 0.1346 | **0.1345** | 0.183 | 0.337 |
| O: oracle LiDAR-depth hard | 0.3337 | 0.3332 | 0.3331 | **0.3334** | 0.532 | 0.472 |
| O − M | +0.1983 | +0.1997 | +0.1986 | **+0.1988** [+0.1909, +0.2076] | +0.349 | +0.135 |

The CI is a paired clip-block bootstrap over 0006 anchors (blocks of 20, 2000 resamples) of the
mean-over-seeds difference. The relative gain is +147.8 %.

## Setup (frozen before training; `code_freeze.json`)

**M.** The Gate 12 H input cache, unchanged.
- **Geometry:** local-reset five-frame LingBot depth, confidence ≥ 1.5, LingBot local poses and
  camera, causal MoGe-2 clip scale applied once.
- **Mapper:** the hard mapper (±0.2 m band, hard carving).
- **Channels:** [S, 0, F].

**O.** The Gate 9 LD path, built for the same 1,948 anchors.
- **Depth:** for each anchor t, only sweeps t−4..t are projected into their own rectified cameras
  and nearest-depth z-buffered on the calibrated lattice camera, keeping 1–60 m.
- **Poses:** KITTI-360 GT relative poses.
- **Mapper:** the same `VariantMapper(0.2, 0.2, carve)`, which equals the core mapper bit for bit.
- **Channels:** [S, 0, F] on the same grid.
- **Exact match to Gate 9:** O's occupied map equals Gate 9's LD map on all 590 0006 anchors.
- **Firewall:** the builder never reads targets, evaluation masks, regions, labels or later sweeps
  (strict file audit, 0 violations).

**Everything downstream is Gate 12's frozen recipe:**
- `CompletionUNet` (width 24, 3 channels, replicate padding, 967,322 parameters);
- focal + Dice loss;
- AdamW 1e-3, cosine schedule, batch 4, uniform 128×128×32 crops, 6000 steps, final-step checkpoint;
- geometry only;
- the same 1,358 training anchors, the same raw-LiDAR target, and Gate 12's evaluation regions.

**Per-seed pairing.** M and O share the same initial weights (Gate 12's saved ones), anchor draws,
crop origins, target and loss-mask crops, and learning-rate sequence, all verified by digests.
The configs differ only in `input`.

## Answers

1. **Does M reproduce the Gate 12 hard baseline?** Yes.
   - **Validation IoU:** M gives 0.1354 / 0.1336 / 0.1346, against Gate 12 H's 0.1343 / 0.1331 /
     0.1341. Differences are +0.0011 / +0.0005 / +0.0005, within the tolerance fixed before
     training (per seed ≤ 0.003, mean ≤ 0.0015).
   - **Pairing:** M's anchor draws, target digests and initial weights equal Gate 12 H's exactly.
   - **Remaining difference:** only GPU nondeterminism.
2. **How much better is the oracle observed geometry before completion?** Against the five-sweep
   LiDAR measurement inside the camera frusta on 0006:

   | input | IoU | precision | recall | occupied in LiDAR-free space |
   |---|---|---|---|---|
   | M | 0.091 | 0.187 | 0.152 | 81 % |
   | O | 0.438 | 0.532 | 0.711 | 47 % |

   O is about 4.8× better, and it is still a lossy camera-visible map.
3. **Completion IoU, M:** 0.1345 (precision 0.183, recall 0.337).
4. **Completion IoU, O:** 0.3334 (precision 0.532, recall 0.472).
5. **Absolute geometry headroom, O − M:** +0.1988 IoU [+0.1909, +0.2076], a relative gain of +148 %.
6. **Consistent across seeds?** Yes: +0.1983 / +0.1997 / +0.1986 (sd 0.0007). O's three seeds
   agree to within 0.0006.
7. **Does accurate geometry improve genuinely unobserved / future-measured voxels?** Yes, strongly.

   | region | M | O | O − M [95 % CI] |
   |---|---|---|---|
   | Not measured by the five causal sweeps, resolved by the future sweeps (78 % of target-occupied voxels) | 0.137 | 0.265 | **+0.128** [+0.121, +0.137] |
   | Same, 20-sweep past-window masks | 0.136 | 0.260 | +0.124 |
   | Voxels with no oracle input evidence at all | 0.110 | 0.250 | +0.140 |
   | Measured by the five causal sweeps | 0.157 | 0.598 | +0.441 |

   - **Unobserved precision and recall:** precision 0.216 → 0.493, recall 0.273 → 0.365.
   - **Observed region:** its gain is expected, since O is a direct measurement there.
   - **Conclusion:** the unobserved-space gain shows that noisy input suppresses learned
     completion, not only reconstruction.
8. **Which range bins benefit most?** O − M in the full region:

   | 0–5 m | 5–10 m | 10–20 m | 20–30 m | > 30 m |
   |---|---|---|---|---|
   | +0.236 (0.179 → 0.415) | +0.266 (0.188 → 0.454) | +0.247 (0.161 → 0.409) | +0.209 (0.112 → 0.321) | +0.160 (0.074 → 0.233) |

   - **Absolute gains:** largest at 5–10 m.
   - **Relative gains:** grow with range, +132 % to +217 %.
   - **Unobserved space:** the gain grows from +0.057 (0–5 m) to about +0.14–0.15 (10–30 m and beyond 30 m).
   - **Implication:** good nearby geometry propagates into far and unobserved completion.
9. **Do O and M converge to different plateaus?** Yes. Mean full IoU on every 5th anchor:
   - **M:** 0.119 → 0.125 → 0.131 → 0.130 → 0.132 → 0.133 (steps 1000–6000).
   - **O:** 0.311 → 0.323 → 0.327 → 0.328 → 0.331 → 0.332.
   - **Gap:** 0.192 at step 1000 and 0.199 at step 6000, essentially constant.
   - **Tail:** both arms change by less than 0.002 from step 5000 to 6000.

   O is already about 0.19 higher after 1000 steps and holds that margin. It is a different
   plateau, not faster convergence.
10. **Primary remaining bottleneck: geometry, completion, or both?** Geometry, by a wide margin.
    - **Geometry:** swapping monocular for accurate observed geometry, with no other change, adds
      +0.199 IoU. Representation changes on monocular depth added +0.0024 (Gate 12).
    - **Completion headroom:** the oracle model still reaches only 0.333 overall and 0.265 in
      unobserved space. The completion model is not "solved"; it is simply not what currently
      limits the system.
11. **Research direction to pursue next:** a stronger deployable causal geometry source. Stop
    modifying the completion network, the uncertainty representation and the mapper for now.

## Decision

**A — geometry is the dominant completion bottleneck.**
- **Magnitude:** the gain (+0.199) is four times the guide's +0.05 bar.
- **Consistency:** every seed gains, with a tight CI.
- **Completion, not just reconstruction:** the gain holds in genuinely unobserved space (+0.128)
  and in every range bin.
- **Plateau:** the learning curves show a higher plateau, not faster convergence.

B and C are ruled out by magnitude, and D is ruled out by the unobserved-space gain.

**Next experiment (one; source side; target datasets stay locked).** Screen a camera-only,
causal, stronger observed-geometry source that also exists on both target datasets:
- **Candidate:** frozen multi-view depth over the same five local-reset frames, using the frozen
  local-reset poses.
- **Stage 1 (observed map):** run it through the frozen Gate 9 observed-map audit on 0006, the same
  measurement that gave M 0.091 and O 0.438.
- **Stage 2 (completion):** only if it clearly beats M there, train it once with the frozen
  Gate 13 recipe (same anchors, draws, targets, seeds) against M.

Nothing about the completion model, mapper or representation changes.

## Limitations

- **Two quality factors are bundled.** O improves depth (LiDAR) and pose (GT) together. The
  registration audit found pose is secondary, but this gate does not separate them.
- **O shares the target's sensor.** O's input comes from the same LiDAR that builds the target, so
  the network may exploit sensor-specific structure. O is therefore an optimistic ceiling for any
  camera-derived source. The unobserved-space gain is the least exposed to this effect.
- **O is sparser.** O uses the LiDAR-supported lattice pixels (about 20k per frame) rather than
  dense rays. Being sparser, it is not an unfair density advantage over M.
- **Single validation drive.** Evaluation is on one source drive (0006). The CI covers anchor
  resampling, and the seeds cover training variance.
- **Not comparable to earlier headline numbers.** These are KITTI-360 geometry-only
  source-validation IoUs. They are not comparable to the historical 30.89 % Occ3D-nuScenes score
  (different dataset, protocol, architecture, input representation and semantic configuration),
  and they are not placed beside it.

## Verification and process notes

Verification is 22/22. The first run failed check 15 only because a word scan for "future"
matched `from __future__ import annotations`. The substantive check held: on all 1,948 oracle
inputs the largest recorded input index minus the anchor is 0. The scan now looks for future-sweep
identifiers in code, and the failed record is `logs/verification_attempt1_21of22.md`.

## Figures and tables

- `figures/1_2_per_seed_and_learning_curves.png`
- `figures/3_4_5_range_region_pr.png`: range differences, observed vs unobserved regions,
  precision / recall.
- `figures/maps/6_inputs_M_vs_O.png`
- `figures/maps/7_completion_{good,median,bad}_t*.png`: M seed-0 anchor IoU at the 90th / 50th /
  10th percentile, anchors 421 / 1030 / 1159.
- Tables: `results_table.md`, `tables/per_seed.csv`, `tables/per_range.csv`,
  `tables/per_region.csv`, `tables/learning_curves.csv`, `tables/paired.csv`.
- Configs: `training_config_M.yaml`, `training_config_O.yaml`.
- Records: `input_quality.json`, `analysis.json`, `runs/`, `logs/`.

## Reproduce

```bash
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1
PY=/home/minh/anaconda3/envs/cu128/bin/python; T=tools/gate13_oracle
# prerequisite: Gate 12 inputs, regions and initial weights (tools/gate12_suf/README.md)
for d in 0006 0003 0007 0010; do $PY $T/build_oracle.py --drive 2013_05_28_drive_${d}_sync --device cuda:0; done
$PY $T/quality13.py --device cuda:0
bash $T/run_all13.sh        # freeze -> 6 paired runs -> final + learning-curve evaluation -> analyze13.py -> figures13.py
$PY $T/verify13.py --device cuda:0
```
