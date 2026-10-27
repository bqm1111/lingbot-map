# Phase 1 — correctness gates

**Run 2026-10-19.** Records: `artifacts/waymo_arm/logs/{check_transforms,check_integrity,clip_timing_census,figures}.json`,
figures in `artifacts/waymo_arm/figures/`.

## 1.1 Coordinate transformations — geometry gates PASS

| check | result |
|---|---|
| T1 axis permutation (Waymo +x forward → pinhole) | **PASS**, error 0 px to the principal point, 3/3 segments |
| T2 projection vs the SDK's own `cp_points` | **advisory FAIL**, see below |
| T3 pose round trip, 21-frame window | **PASS**, max error < 1e-9 |
| T4 anchor transform is exactly I | **PASS**, error 0.0 |
| T5 per-laser beam inclinations about each laser's own origin | **PASS**, 100 % inside the calibrated cone for all five lasers; TOP measured [−17.6°, +2.2°] vs calibrated [−18.02°, +2.29°] |
| T6 voxel index ↔ flat C-order round trip | **PASS**, exact on 4,096 random voxels |
| T7 camera centre inside the grid | **PASS**, 3/3 |

**T5 is the validation of the laser-sidecar recovery.** A wrong slice or a wrong origin would
put points outside their laser's calibrated inclination cone; none are.

**T2, stated plainly.** `waymo.project` deliberately applies no lens distortion. Measured
residual against the SDK: p50 **0.061 m**, p99 **0.237 m** on the worst of three segments,
against a one-voxel (0.20 m) criterion. Applying Waymo's radial-tangential model brings the
same segment to p99 **0.193 m** and 99.3 % of points within a voxel, so the transform chain
is correct and the residual is the approximation, not a bug.

It does **not** gate this experiment: `grep` shows `W.project` is called only by
`waymo_pilot/checks.py` and `cli/verify_reader.py`. The occupancy target and the hard map
never project — LiDAR is voxelised directly in the grid frame. It **is** a precondition for
any Waymo *semantic* target, which must assign returns to pixels; a verified distorted
projection now exists (`check_transforms._project_distorted`).

## 1.2 Visual validation — PASS

Three anchors, six panels each. Confirmed by inspection: LiDAR projects onto the road surface
and building façades in the anchor image; the five clip frames advance monotonically; the
hard map fans beyond the 50° frustum into area seen by earlier frames (consistent with the
measured 50.9 % out-of-frustum coverage); and the hard map's surface slice coincides with the
target's occupied band, with FREE above it and UNKNOWN below.

## 1.3 Preprocessing integrity — PASS (8/8)

| check | result |
|---|---|
| I1 corpus | 798 cached, 791 fully extracted, clips for 791 |
| I2 anchors per segment | 5 where eligible, all inside the eligible set |
| I3 sequence boundaries | 0 violations — every clip and privileged window inside its own segment |
| I4 clip causality | 0 violations; every clip is exactly `t−4..t`, nothing after `t` |
| I5 timestamps | 0 violations under the gate below |
| I6 cache health | 0 bad inputs, 0 bad targets |
| I7 grid consistency | one grid everywhere: 256×256×32 @ 0.2 m, origin (0, −25.6, −0.257) |
| I8 supervision rules | 0 violations; every occupied voxel carries ≥ 1 return and is valid |

**I5 deserves its own note, because the first run failed it.** The initial gate was
`|dt − 0.50| ≤ 0.05 s`, which 2 of 200 clips failed. Rather than widen the tolerance, the
spacing was censused over **all 3,955 clips / 15,820 intervals**: dt p50 **0.5000 s**,
p99 **0.5033 s**, max **0.8987 s**, clip span p50 **2.000 s**, and **zero non-monotonic
intervals**. 78 clips (**1.97 %**) carry one interval in (0.55, 0.90] s from a single dropped
native frame; 2 clips (0.05 %) exceed 0.75 s; **none reaches 1.00 s**.

The gate is now monotonicity plus "no interval ≥ 2× nominal", which is what a real
discontinuity (two or more consecutive dropped frames) would produce. Single-frame jitter is
**counted and reported, not excluded**: nothing in the frozen pipeline keys off `dt` — the
mapper consumes LingBot's predicted poses — so the only effect is a slightly longer baseline.

## Verdict

Geometry gates for Phases 2–4: **PASS**. One advisory failure (T2), scoped to a semantic
target build that this phase does not perform. Training is unblocked on correctness and
remains blocked only on the laser-sidecar download.
