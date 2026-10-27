# ONCE-only geometry training — BLOCKED, not executed

**2026-10-20.** Phase C of the brief. **No ONCE training run was executed and no ONCE adapter
was built.** This file records the blocked stage rather than leaving a gap, and states exactly
what would be needed to unblock it.

---

## 1. Why it is blocked

Phase C's stated precondition is *"Execute this phase only if ONCE passes all mandatory
gates."* It does not. The full audit is `once_feasibility.md`; the verdict is **CONDITIONAL
GO**, and the unmet gate is the one the brief itself labels hard:

> "Our training protocol requires five causal camera frames and privileged LiDAR supervision
> over a defined temporal window. Verify that ONCE provides reliable motion transformations for
> both. … This is a hard gate, not an optional diagnostic."

**ONCE's released point clouds are 2 FPS**, downsampled from a 10 FPS recording — from its own
documentation: *"We further downsample those original data with the sampling rate of 2 FPS."*
The frozen supervision accumulates ~21 sweeps spanning ~2.1 s (measured 20.94 on KITTI-360,
21.00 on Waymo, both at 10 Hz). On ONCE:

| choice | sweeps | span | consequence |
|---|---:|---:|---|
| match the sweep **count** | 21 | **10.5 s** | ~105 m of travel into a 51.2 m grid; most sweeps contribute from outside it, dynamic objects smear |
| match the time **span** | **5** | 2.1 s | 4× fewer sweeps carving free space, directly moving `valid_fraction` and the conflict rate |

Neither preserves the frozen policy. The brief forbids changing occupancy targets to force
statistical similarity, so this is a protocol decision for the user, not a judgement for me to
make silently.

Three further conditions are also unmet, none of them individually fatal:

* **The origin claim is documented, not measured.** ONCE states the LiDAR coordinate is "placed
  at the center of the LiDAR sensor" — exactly what our carve rule needs and exactly what DDAD
  lacked. But DDAD's prose was also accurate about its sensors; the problem was what its
  *release* preserved. No ONCE bytes were inspected (§2).
* **Axis convention** is x-left / y-back / z-up, a fixed known rotation away from our
  x-forward grid. A condition to implement and gate T1-style, not missing information.
* **Possible double-undistortion.** ONCE says released images already have distortions removed,
  yet ships a `distortion` vector and OccAny's reader calls `cv2.undistort` on the loaded JPEG.
  Unresolved, and it would corrupt the hard map.

---

## 2. What could not be verified, and why

ONCE is distributed only through **Google Drive folders** and **Baidu Pan**. Verified by
listing with `gdown` 5.2.1: the three training-split Drive folders resolve and contain
`train_cam01.tar` … `train_cam09.tar`. But Drive serves whole files, not byte ranges.

That is the difference from the DDAD audit, which cost 1.2 GiB out of 256 GiB because
`DDAD.tar` sits on public S3, honours HTTP range requests, and tar is sequential — so one
complete scene could be recovered without downloading the dataset. ONCE offers no equivalent
path: reading one sequence's calibration or one point cloud means pulling a multi-gigabyte tar.

**The cheapest unblocking step is `train_infos.tar` alone** — the per-sequence JSONs, no point
clouds, no images. It would settle camera HFOV from the real intrinsics, timestamp spacing,
sequence and frame counts, and the `cam_to_velo` translations. Only the empirical origin check
and the target statistics need `train_lidar.tar`. Only **1 of 7** camera tars is needed, since
the protocol uses a single forward camera.

---

## 3. What exists and would be reused unchanged

Recorded so the work is not redone:

| component | status |
|---|---|
| frozen single-sensor carve `core/supervision/rawtarget.py` | applies directly if the origin check passes — ONCE is structurally the KITTI-360 case |
| multisensor carve `core/supervision/rawtarget_multisensor.py` | not needed for ONCE (one sensor); already validated bit-for-bit for PandaSet later |
| feasibility probe `tools/method_vs_data/ddad_probe.py` | duck-types the three members `rawtarget.build` touches (`cam0_to_world`, `read_velodyne`, `velo_to_velo`); needs only a thin ONCE reader pointed at it. **Run this before writing an adapter** — it is what turned a multi-day task into an afternoon's audit on DDAD |
| training recipe | `tools/waymo_arm/train.py`, unchanged: 6,000 steps, batch 4, 128×128×32 crops, focal+Dice, AdamW, cosine, seeds 0/1/2 |
| acceptance bands | predeclared in `predeclaration.md` §6 **before** any ONCE data was inspected: occupied prevalence [0.015, 0.040], conflict-to-unknown ≤ 0.12, valid fraction [0.45, 0.75], occupancy in ≥30 of 32 z-levels |
| evaluation | the frozen zero-shot protocol, untouched |

Output directory `artifacts/method_vs_data/once_arm/` was **not created**, because creating an
empty experiment directory would imply an experiment.

---

## 4. Ordered prerequisites

1. **User decision on the §1 sweep policy.** Every available resolution departs from the recipe
   A0 and A1 were trained under. This is the blocking item.
2. `train_infos.tar` → camera HFOV, timestamps, sequence counts. Record HFOV **before**
   training (`predeclaration.md` §6): DDAD's only forward camera is 47.9° and Waymo's are
   50.0°, and the Waymo arm failed. A narrow ONCE camera would weaken the arm's power to
   separate dataset identity from training field of view.
3. `train_lidar.tar` → run the probe: z of near points (expect road at ≈ −1.8 m, **not** ≈ 0 as
   DDAD had) and elevation spanning well below 0°.
4. Target statistics against the predeclared bands.
5. Axis rotation implemented and gated; distortion question resolved.
6. Only then: adapter, smoke test, three seeds.

---

## 5. What is not claimed

* **No ONCE number of any kind appears in this directory.** Nothing was trained, nothing
  evaluated.
* **No claim that ONCE will fail.** On the decisive origin gate it looks like the best candidate
  in OccAny's pool, and its 0.5 s cadence matches our five-frame input contract exactly. The
  block is the supervision window, not the dataset's quality.
* **No claim that the sweep-policy conflict is unresolvable** — only that resolving it is a
  protocol change requiring authorization, and that it must be decided on measured target
  statistics rather than preference.


---

# Addendum — 2026-10-27, under the authorized 2.1 s policy

The October 20 report above is preserved as a historical record. It blocked Phase C on ONCE's
2 FPS cadence. **That blocker was my error and is retracted** (`once_feasibility.md` §12): the
frozen window is 21 sweeps at 0.523 s spacing spanning 10.46 s, not 2.1 s, so ONCE's native
0.500 s cadence is not the obstacle the report claimed.

Phase B has since been executed on real data under the authorized ~2.1 s native-cadence policy.
Full result: `once_supervision_validation.md`. Summary:

* **Every correctness gate passes** — sensor origin confirmed from data (road plane −1.73 m,
  elevation p1 −24.97°), axis convention verified against travel direction, pose round-trip
  3.7e-13, 100 % pose availability, windows built from 5 real native sweeps spanning 2.000 s,
  conflict fraction 0.020 against a 0.12 ceiling, 32 of 32 z-levels.
* **Verdict: CONDITIONAL GO**, not GO. The predeclared occupied-prevalence band is not met
  (median 0.0130 against a 0.015 floor, 2 of 6 sequences inside).
* **The band cannot adjudicate this protocol.** Under the same authorized 2.1 s window,
  KITTI-360 — the corpus the bands were calibrated from — also falls out of bands, on vertical
  coverage (21 of 32 against a floor of 30). Each corpus fails a different band.
* **Sparsity is falsified as the cause**: removing 85 % of KITTI-360's LiDAR returns leaves it
  in band. The cause is scene content — ONCE's scenes hold less occupied surface in the grid.
* **Supervision density, stated explicitly**: ONCE carries ~34 % fewer occupied voxels than
  duration-matched KITTI-360, structured — ratio 0.34–0.49 in the 0.8–3.2 m height band and
  0.44 beyond 40 m, but ONCE is the only one of the two with any occupancy above 2 m.

**Phase C remains not executed**, now by brief §10 (*"Execute this phase only if Phase B returns
GO"*) rather than by the retracted cadence blocker. **No ONCE model exists, no ONCE number
exists, and none is projected.** `artifacts/method_vs_data/once_arm/` is still deliberately not
created. §9 of the validation report specifies the remaining port and its cost.

---

# Addendum 2 — 2026-10-27: Phase C EXECUTED

The October 20 report and Addendum 1 above are preserved as historical records. Both described
Phase C as blocked. **It is no longer blocked: the ONCE-only CompletionUNet was trained and
evaluated.** Full detail in `once_training_report.md`; this is the results record.

**Status: exploratory.** The predeclared occupied-prevalence gate is **failed, not passed**
(median 0.0131 against [0.015, 0.040]) and is recorded as failed in every manifest the arm
produced. It was run because the same band also rejects KITTI-360 under the authorized ~2.1 s
window, on vertical coverage (`once_supervision_validation.md` §5). No band was altered.

## What was built and trained

1,580 anchors over **6 independent sequences** (1,259 train / 321 val, sequence-disjoint val
000121), manifest digest `aab448cc…`; 1,580 causal clips, 1,568 occupancy targets, 1,580
hard-map inputs, **0 failures**. Three seeds at the frozen recipe (6,000 steps, batch 4,
128×128×32, focal+Dice, AdamW, cosine, 967,322 parameters, recipe sha256 `4ef021b6…`),
7.7 min each, three distinct checkpoints.

## Frozen zero-shot results, raw convention, three seeds

| arm | SemanticKITTI (163) | Occ3D (1,182) |
|---|---:|---:|
| KITTI-360 | **26.17** | **31.22** |
| Waymo | 23.14 | 22.86 |
| **ONCE** | **18.50** ± 0.68 | **19.29** ± 0.47 |
| ONCE per-seed | 18.37 / 19.23 / 17.89 | 18.78 / 19.70 / 19.38 |
| ONCE − KITTI-360 | **−7.67** | **−11.93** |
| ONCE − Waymo | **−4.64** | **−3.57** |

By region:

| benchmark | region | KITTI-360 | Waymo | ONCE |
|---|---|---:|---:|---:|
| SemanticKITTI | observed | 33.23 | 28.26 | **18.77** |
| SemanticKITTI | completed | 24.74 | 21.46 | **18.39** |
| Occ3D | observed | 39.03 | 35.05 | **25.34** |
| Occ3D | completed | 27.50 | 15.86 | **15.10** |

**Control:** `map_only` SC IoU is **8.8205** / **10.0351**, bit-identical to both other arms, so
the only thing that differs is the completion checkpoint.

## Two corrections this phase forced

* **The camera-to-grid extrinsic was wrong** (`A·T·A⁻¹` where only the target frame rotates;
  correct is `A·T`). Caught by the arm's own input/target alignment check at IoU 0.22 %; the
  gates could not catch it because G1/G2 use LiDAR only and the distortion gate's two
  projections cancelled the error.
* **ONCE's images are already rectified** — the §15 conclusion that they are raw is
  **retracted**. Plain pinhole beats a distortion-applied projection 0.857 vs 0.698 on
  LiDAR-depth correlation (12/12 frames) and 3.26 % vs 1.44 % on map-vs-target IoU (11/12
  anchors). The gradient test that misled us is centre-biased.

**Verdict: ONCE-only training does not reproduce the KITTI-360 arm's geometry.** It is the
weakest of the three source arms on both benchmarks. The OccAny comparison requires the
canonical protocol and is in `once_vs_occany.md`; the numbers above may not be placed in a
column with OccAny's.
