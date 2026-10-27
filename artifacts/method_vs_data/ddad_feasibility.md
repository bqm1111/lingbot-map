# DDAD feasibility — can it support our occupancy supervision?

**Audited 2026-10-20.** Raw: `artifacts/method_vs_data/results/ddad_probe.json`. Probe code:
`tools/method_vs_data/ddad_probe.py`. No training was launched.

> ## Verdict: **NO-GO for A2 as specified.**
>
> Two independent blockers. The first is hard: DDAD does not publish the information our
> carve rule needs, and unlike Waymo there is **no recovery path**. The second is that even
> if it did, a DDAD front-camera arm would reproduce the input condition the Waymo arm
> already failed under, so it would be a weak test of the question being asked.
>
> **Recommended substitute: ONCE (first choice) or PandaSet (second).** Both are OccAny
> source datasets, so the subset logic of the brief is preserved. §6 gives the evidence.

---

## 1. How this was audited without a 256 GiB download

`DDAD.tar` is 275,266,242,560 bytes (256.4 GiB), publicly readable (HTTP 200, last modified
2020-10-13). tar is sequential and the archive honours HTTP range requests, so two ranged
reads (0–200 MB and 1.00–2.80 GB) recovered **one complete scene**: `ddad_train_val/000020`
with its scene metadata, its calibration, and all 100 LiDAR sweeps. Everything below is
measured on that real data, with the frozen builder called unchanged.

Total fetched: ≈1.2 GiB instead of 256.4 GiB.

---

## 2. Verified dataset facts

### 2.1 Structure and scale (official README + archive)

| item | verified value |
|---|---|
| scenes | 150 train (12,650 samples) + 50 val (3,950 samples) |
| scene length | 5 s or 10 s, 50 or 100 samples at 10 Hz |
| cameras | 6, 1936×1216, global shutter, 60° intervals, 360° coverage |
| LiDAR | 4 × Luminar-H2 at 90° intervals, 250 m range |
| **LiDAR as released** | **one aggregated cloud per sample, datum name `LIDAR`** |
| scene 000020 | 100 samples, 38.6 m path, median step 0.374 m (≈3.7 m/s) |
| points per sweep | **229,726** (vs KITTI-360 Velodyne ≈120 k, Waymo ≈180 k) |
| range | p50 18.4 m, p90 71.7 m, max 237.0 m |
| licence / access | public S3, no credentials required |

### 2.2 The on-disk layout confirms the aggregation

```
ddad_train_val/000020/
├── scene_c81887aa....json          971,417 B   100 samples × 7 datums = 700 entries
├── calibration/66f67995....json      3,277 B   7 sensors
├── point_cloud/LIDAR/<ts>.npz      ≈4.9 MB     ← ONE directory, not four
└── rgb/CAMERA_{01,05,06,07,08,09}/
```

* `point_format` in the scene JSON: **`["X", "Y", "Z", "INTENSITY"]`**. The npz holds a single
  `data` array of shape (N, 4), float64. No ring index, no sensor index, no per-point
  timestamp.
* Calibration gives `LIDAR` **translation (0, 0, 0), rotation identity, intrinsics all zero**.
  The published "sensor frame" *is* the vehicle frame, whose origin the README places "on the
  ground below the center of the vehicle rear axle".
* Ego poses are present and usable: per-sample sensor→world pose, verified by reconstructing
  a 38.6 m trajectory with a clean 0.374 m median step.

### 2.3 Camera intrinsics (scene 000020 calibration, measured)

| camera | fx | fy | cx | cy | HFOV at 1936 px |
|---|---:|---:|---:|---:|---:|
| **CAMERA_01 (front)** | 2181.53 | 2181.60 | 928.02 | 615.96 | **47.9°** |
| CAMERA_05 | 1057.07 | 1055.98 | 964.68 | 588.66 | 84.8° |
| CAMERA_06 | 1060.76 | 1059.26 | 946.56 | 611.41 | 84.8° |
| CAMERA_07 | 1058.95 | 1056.78 | 966.03 | 615.20 | 84.8° |
| CAMERA_08 | 1057.29 | 1060.15 | 966.42 | 619.27 | 84.8° |
| CAMERA_09 | 1063.46 | 1065.22 | 944.67 | 612.70 | 84.7° |

Camera extrinsics are published and plausible (all six at z ≈ 1.53–1.58 m). **No distortion
parameters exist in the calibration** — only `fx, fy, cx, cy, skew` — so DDAD images are
supplied as ideal pinhole. That is one thing in DDAD's favour: it removes the lens-distortion
problem that Phase 1's T2 gate raised for Waymo.

---

## 3. Blocker 1 — the carve origin is unrecoverable

Our supervision is `core/supervision/rawtarget.py`, the frozen Gate 8C-1 rule: ray endpoints
are OCCUPIED, ray **interiors** are FREE, nothing past an endpoint, cross-sweep disagreement
resolves to UNKNOWN. Every FREE voxel is decided by a ray cast **from the sensor origin**. The
origin is not a detail of this recipe; it is what the recipe is made of.

DDAD merges four physically separated sensors before release and publishes the result in a
frame whose origin is a ground point under the rear axle. So:

* **No per-point identity exists.** Channels are X/Y/Z/INTENSITY. Intensity takes 174 distinct
  values in [0, 0.784] with a smooth distribution — a reflectance measurement, not a 2-bit
  device tag.
* **No per-sensor counts exist** in the release. This is the difference from Waymo, where the
  SDK's deterministic laser ordering plus per-laser point counts and calibrations let us
  reconstruct identity exactly — and we verified it, with every recovered TOP beam landing
  inside the calibrated inclination cone. DDAD offers no equivalent handle.
* **Geometry does not permit inferring it.** If four sensors covered four 90° sectors,
  azimuthal seams or density steps should appear. Over 72 five-degree bins the counts vary
  with a max/min ratio of 3.49 in a pattern that tracks scene content (buildings, a vehicle
  cluster), with **no 4-fold periodicity and no seams**. Sector assignment is not inferable.

### 3.1 The pathology, measured

From the published origin the sensor sits **on the road**. Measured elevation angles of one
sweep: **p1 −1.35°, p50 +1.43°, p99 +12.39°** — essentially nothing below the horizon, and
only 23 of 180 one-degree elevation bins occupied. Road returns therefore sit at ≈0°
elevation, so every ray to a road point **grazes along the road plane** for its whole length.
That is precisely the configuration the frozen rule's docstring warns about, and the reason it
protects same-sweep returns from being carved.

Independent confirmation that the origin is at ground level: for points within 6 m
horizontally, the z distribution has mode **+0.08 m** and p01 **+0.02 m**. The road is at
z ≈ 0 in the published frame. KITTI-360's velodyne frame puts the road at z ≈ −1.7 m.

### 3.2 What the frozen builder actually produces on DDAD

`rawtarget.build` called unchanged on real DDAD sweeps, SSCBench-KITTI-360 grid
(256×256×32 @ 0.2 m, origin (0, −25.6, −2)), 21 sweeps per anchor, three anchors of scene
000020:

| arm | valid fraction | occupied prevalence in valid | conflict / touched | occupied z-levels |
|---|---:|---:|---:|---:|
| DDAD, published origin (0, 0, 0) | 0.583 – 0.616 | **0.0042 – 0.0094** | 0.062 – 0.097 | 25–31 / 32 |
| DDAD, assumed roof mount (1.6, 0, 1.6) | 0.784 – 0.810 | 0.0085 – 0.0157 | 0.043 – 0.067 | 32 / 32 |
| **KITTI-360 drive 0000 (reference)** | **0.5703** | **0.0276** | **0.0865** | — |
| **Waymo, 3,955 anchors (reference)** | **0.6616** | **0.0222** | **0.0679** | — |

Two readings:

1. **With the published origin the target is out of family.** Occupied prevalence is
   **3–6× lower** than either existing arm. The positive class our focal + Dice objective is
   balanced against would be a different quantity, on a 128×128×32 crop, with no change to
   the loss being legitimate under the brief.
2. **The target is materially sensitive to the constant we would have to invent.** Assuming a
   roof mount moves the valid fraction by ~20 percentage points and roughly **doubles**
   prevalence. The grid's own z extent (−2 to +4.4 m) presumes a roof-mounted sensor frame;
   with the ground-level origin, 2 m of the grid falls below the road and 1–7 of 32 z-levels
   carry no occupancy at all.

The roof-mount arm is **not** a fix. It conflates a better grid placement with a different
carve origin, and both come from the same unpublished quantity. It is reported to size the
uncertainty, not to propose a value.

### 3.3 Why this blocks the experiment rather than merely complicating it

A2 exists to answer "does the Waymo outcome depend on the chosen source dataset". If we
trained on DDAD with an invented mount height, then:

* a **negative** result could be our constant rather than DDAD, and
* a **positive** result could be a lucky choice of that constant,

because §3.2 shows the target moves by 20 pp of valid fraction and 2× in prevalence across
plausible values. The arm would not be a control. That is the brief's own standard — report
the blocker rather than silently altering the training recipe.

---

## 4. Blocker 2 — a DDAD front-camera arm repeats the condition that already failed

DDAD's front camera is **47.9° HFOV** (§2.3). Waymo's cameras are **50.0°**. The arm the brief
calls a completed negative control was trained at 50°, and the Phase 3 analysis found its
failure mode to be a less aggressive completion prior — recall rising in observed regions,
falling everywhere it must complete.

The brief requires "a single configurable front camera for causal input". DDAD's only
forward-facing camera is the 47.9° one; the five ~84.8° cameras point ±60°, ±120° and 180°.
So a protocol-conforming DDAD arm sits at essentially the same input geometry as the arm that
already failed, and would be a weak discriminator between "Waymo specifically" and "narrow
training FOV generally".

This is a scientific-value objection, not a correctness one. On its own it would argue for
running A2 anyway and interpreting carefully. Combined with §3 it argues for substitution.

---

## 5. What would unblock DDAD, stated for completeness

**Camera-ray carving.** The six camera extrinsics and intrinsics *are* published and exact
(§2.3), and DDAD images carry no distortion. One could project the aggregated cloud into each
camera, build a depth map, and carve free space along **camera** rays from known camera
centres — fully determined, no invented constant. Our hard map already does something close
to this (the Gate 7B pixel-decimated camera-ray carve).

But this is a **different supervision rule** from the one E0 and A1 were trained under, so an
arm built this way would not be comparable with them — which is the one thing the brief
forbids doing silently. It is listed as an option requiring explicit authorization, with the
cost stated: it would need its own baseline re-run on KITTI-360 to restore comparability,
roughly doubling the experiment.

---

## 6. Recommended substitutes, with evidence

The brief directs that on a DDAD blocker we evaluate ONCE or PandaSet. Both are OccAny source
datasets (verified in `occany_training_audit.md` §1.2 at 10,000 draws/epoch each).

### 6.1 ONCE — first choice

From OccAny's own `dataset_setup/once/preprocess.py`:

* `self.lidar_name = "lidar_roof"` — **a single roof-mounted LiDAR.** One true sensor origin,
  exactly the KITTI-360 situation. Our frozen single-sensor `rawtarget.build` applies with
  **no new assumption at all**.
* `np.fromfile(...).reshape(-1, 4)` — x, y, z, intensity per point.
* Per-frame `pose` (quaternion + translation), converted by `pose_matrix`.
* Per-camera `cam_to_velo`, `cam_intrinsic` **and `distortion`** are published — better
  calibration metadata than Waymo gave us.

This is the cleanest match to our recipe of any OccAny source dataset.

### 6.2 PandaSet — second choice

From OccAny's own `dataset_setup/pandaset/preprocess.py`, which documents the devkit schema:

```
# index        x           y         z        i         t       d
# - `i`: float: Reflection intensity in a range [0,255]
# - `t`: float: Recorded timestamp for specific point
# - `d`: int: Sensor ID. 0 -> mechanical 360° LiDAR, 1 -> forward-facing LiDAR
```

**PandaSet publishes a per-point sensor ID.** That is exactly the field DDAD lacks. With two
devices and per-device poses from the devkit, our already-validated
`core/supervision/rawtarget_multisensor.py` — proven bit-for-bit identical to the frozen
single-sensor rule during the Waymo work — applies directly. Per-point timestamps are a bonus
for motion compensation. OccAny discards all of this (`[:, :3]`, world frame).

### 6.3 Both sources reachable

ONCE project site and `pandaset-devkit` both return HTTP 200. Download sizes and exact split
contents are **not yet verified** and must be before any commitment.

---

## 7. Correctness gates — status against brief §6

| gate | status |
|---|---|
| 1. camera/LiDAR projection alignment | **not run** — blocked at §3 |
| 2. round-trip coordinate transforms | **passed incidentally**: pose composition reconstructed a 38.6 m trajectory with 0.374 m median step and no discontinuity |
| 3. five-frame causal ordering | **feasible**: 10 Hz, so stride 5 gives 0.5 s spacing and a 2 s span, matching the frozen contract; a 100-sample scene supports it, a 50-sample scene is tight |
| 4. no sequence-boundary crossings | **feasible**: scenes are explicit directories |
| 5. voxel indexing | **exercised**: the frozen builder voxelized 2.3–2.5 M points per anchor into the grid without error |
| 6. occupied/free/unknown assignment | **FAILS comparability** — §3.2 |
| 7. comparable target statistics | **FAILS** — prevalence 3–6× low — §3.2 |
| 8. no target-domain contamination | satisfied by construction (DDAD is disjoint from SemanticKITTI and Occ3D) |
| 9. deterministic sampling / manifests | not reached |
| 10. visualizations | not reached |

Gates 6 and 7 are the ones that matter, and they fail for a reason no amount of
implementation effort removes.

---

## 8. Unverified / not claimed

* **Only one scene was measured** (000020, 100 sweeps, 3 anchors). The prevalence figures are
  from one urban scene and would move across DDAD's 150 training scenes. They are not offered
  as DDAD's corpus statistics — they are offered as evidence that the published origin puts
  the target out of family, and the **sensitivity** result in §3.2 does not depend on which
  scene it is measured on.
* **The true Luminar mount positions are unknown**, not merely un-looked-up. (1.6, 0, 1.6) was
  chosen to match the published camera mount heights and is explicitly an assumption used for
  a sensitivity bound.
* **ONCE and PandaSet download sizes, licences and split contents are not verified.** Only
  their sensor geometry and published fields are, from OccAny's own readers.
* **No claim that DDAD is a poor dataset.** It is excellent for its purpose — long-range
  monocular depth, which needs camera-projected depth maps, exactly what it provides and what
  OccAny consumes. The mismatch is with *our* ray-carving supervision, not with DDAD.
* **No claim about what a DDAD-trained model would score.** Nothing was trained.

---

## 9. Decision and next action

**A2 (DDAD-only) is not launched.** Gates 6 and 7 of brief §6 fail, and the brief's
instruction in that case is explicit.

The requested substitution assessment is §6: **ONCE** is the strongest candidate because it
needs no new supervision machinery and no invented constant; **PandaSet** is viable because it
publishes the per-point sensor ID that would let the already-validated multisensor carve run
unchanged.

Both require a separate authorization under brief §5 — neither is A2, and the brief authorizes
only A2 for training. The decision of which to pursue, and whether to pursue the §5
camera-ray route on DDAD instead, is the user's.

Work that does **not** depend on this blocker has been carried out and is reported separately:
`occany_training_audit.md`, `fair_evaluation_protocol.md` and
`observed_vs_completed_analysis.md`.
