# ONCE metadata audit — Phase A

**2026-10-27.** Deliverable for brief §4–§5. Every figure below is measured from the released
ONCE archives on disk, not read from documentation; where documentation and data disagree the
disagreement is recorded and the data wins. Code:
`tools/method_vs_data/{once_infos_audit,once_reader,once_distortion_gate}.py`. Raw:
`artifacts/method_vs_data/results/{once_infos_audit,once_distortion_gate}.json`.

## 1. Data on disk — no new download was needed

| archive | size | state |
|---|---:|---|
| `train_infos.tar` | 57.6 MB | downloaded, extracted |
| `train_lidar.tar` | 10.4 GB | downloaded, extracted |
| `train_cam01.tar` | 1.5 GB | downloaded, extracted |
| `train_cam02..09.tar` | — | **not downloaded**; one forward camera suffices |

Extracted to `/media/minh/MINH1/once_data/extracted/data`: **6 sequences, 9,977 point clouds,
9,977 cam01 images**, 100 % pairing. Fetched with `gdown`; no access control was bypassed and
no credential was used.

**Correction to the earlier audit.** §9 of `once_feasibility.md` recorded that ONCE is "Google
Drive / Baidu only, whole files", implying no cheap probe was possible. That was wrong and the
error was mine: `train_infos.tar` came down in about 2 s. The real obstacle had been that the
download page places each label *before* its href, which defeated my first parse; pairing by
document offset fixed it. The brief's staged download order was followed.

## 2. Sequences, frames and cadence

| sequence | frames | clouds | cam01 images |
|---|---:|---:|---:|
| 000076 | 2,055 | 2,055 | 2,055 |
| 000080 | 1,922 | 1,922 | 1,922 |
| 000092 | 1,165 | 1,165 | 1,165 |
| 000104 | 1,230 | 1,230 | 1,230 |
| 000113 | 1,891 | 1,891 | 1,891 |
| 000121 | 1,714 | 1,714 | 1,714 |
| **total** | **9,977** | **9,977** | **9,977** |

* **Poses: 100 % of frames**, as `(qx, qy, qz, qw, tx, ty, tz)` relative to each scene's first
  frame. No frame in the supervision window lacks a pose.
* **Timestamp spacing: p50 0.500 s**, i.e. **2.00 FPS** — confirmed, not assumed. Max interval
  **13.001 s**: 1.7 % of intervals are gaps, so the window builder rejects any window
  containing an interval over 0.75 s rather than assuming continuity.
* Timestamps are strictly increasing in every sequence (gate G6).
* Ego speed ≈ 5.5 m/s, so a 2.0 s window covers ≈ 11 m of travel.
* Calibration is **identical across all six sequences**.

## 3. Camera — intrinsics, HFOV and the input condition

`HFOV = 2·arctan(W / 2f_x)`, with the released `cam_intrinsic` and the true image size
**1920 × 1020** (measured from the JPEGs, not taken from the docs):

| camera | HFOV | role |
|---|---:|---|
| **cam01** (forward) | **90.1°** | the arm's input camera |
| cam03 | 52.7° | available as a within-dataset FOV control at no extra download cost |

**Recorded before any training, as the predeclaration requires.** This is the favourable
finding of the audit: DDAD's only forward camera is 47.9° and Waymo's are 50.0°, and the Waymo
arm's failure could not be separated from that narrow training FOV. At 90.1°, cam01 is **wider**
than either evaluation camera (SemanticKITTI 1226×370, Occ3D 64.5°), so an ONCE arm does not
re-run the narrow-camera condition A1 already failed under. The predeclared
input-geometry-disclosure gate therefore **passes**, and passes in the direction that preserves
the experiment's power to discriminate dataset identity from training FOV.

## 4. Coordinate convention — verified, not assumed

Documentation states x-left / y-back / z-up. Verified against the vehicle's own direction of
travel: the measured travel vector is `[0.0003, −1.0000, −0.0026]` in ONCE's native frame,
i.e. straight down −y, which is exactly what x-left/y-back/z-up predicts for forward motion.
**The documented convention is correct.**

The map to this project's x-forward / y-left / z-up grid is therefore an exact 90° yaw, not a
fitted quantity:

```
ours_x (forward) = −once_y     ours_y (left) = once_x     ours_z (up) = once_z

R = [[0, −1, 0],
     [1,  0, 0],
     [0,  0, 1]]            det R = +1
```

Points are rotated by `R`; pose transforms are conjugated as `T_ours = A T_once A⁻¹` with
`A = diag(R, 1)`, the only composition that leaves relative geometry unchanged. In our frame the
measured travel vector becomes `[1.0000, 0.0003, −0.0026]` — forward, as required.
**Transform round-trip error: 3.7e-13** over all sequences.

Camera-to-LiDAR extrinsics are published per camera as `cam_to_velo` and are conjugated the
same way.

## 5. Lens distortion — the documentation is wrong

The brief asks this be resolved geometrically rather than by reading, because ONCE's docs say
images are already undistorted while OccAny's reader calls `cv2.undistort` on them. Two
measurements on real ONCE images and clouds, 6 frames × 6 sequences:

**Does the choice matter?** Projecting every in-image LiDAR point with the plain pinhole `K`
and again with the published distortion, and converting the pixel disagreement to a metric
offset at that point's own depth (`d_metric ≈ d_pixel · Z / f_x`):

| | value |
|---|---:|
| disagreement, p99 | **51 px** |
| as metric offset at depth | **2.16 m** |
| one voxel | 0.20 m |

**Material — about 10× one voxel.** It cannot be waved away.

**Which projection is right?** LiDAR depth discontinuities fall on object silhouettes, so they
should land on image gradients. Scoring mean gradient magnitude at projected depth-edge points:
the distortion-applied projection wins in **6 of 6 sequences**.

**Conclusion: the released images are RAW. The documentation's claim that distortion was removed
is false.** So `cv2.undistort` must be applied once, as OccAny's reader does; there is no
double-undistortion risk and the ambiguity is resolved in favour of undistorting.

**A further documentation error:** the shipped `distortion` is **5 values**
(`[−0.3121, 0.1024, 0.00033, −0.000398, −0.0158]`), not the documented 1×7. OpenCV rejects a
true 7-vector and accepts these 5. Verified before M1/M2 were trusted.

## 6. LiDAR sensor — measured, and it is not KITTI-360's

Measured on 5 clouds in each of 3 sequences (and 2 KITTI-360 drives for reference):

| | points / sweep | distinct 0.1° elevation bins | r99 | r max | landing in our grid |
|---|---:|---:|---:|---:|---:|
| KITTI-360 (HDL-64E) | 120,055 | 250–291 | 52–68 m | **80 m** | ~48,800–63,400 |
| **ONCE (`lidar_roof`)** | **66,563** | **exactly 40** | 93–132 m | **~198 m** | ~23,700–28,500 |

ONCE is a **40-beam, ~200 m** sensor against KITTI-360's **64-beam, 80 m**. The same 66 k
returns are spread over roughly 15× the volume, so only ~25 k land inside the
51.2 × 51.2 × 6.4 m grid against KITTI-360's ~56 k. This is recorded here because it is the
first candidate explanation for the supervision statistics in
`once_supervision_validation.md` — where it is tested and **falsified**.

## 7. Phase A verdict

Every Phase A item the brief lists is measured and passes:

| item | result |
|---|---|
| sequences / frames | 6 / 9,977, all with clouds and cam01 images |
| pose availability across the window | **100 %** |
| LiDAR cadence | **2.00 FPS measured**, gaps to 13 s present and rejected |
| camera intrinsics | published, identical across sequences |
| **camera HFOV** | **90.1° (cam01)** — wider than either evaluation camera |
| camera-to-LiDAR extrinsics | published per camera, conjugated into our frame |
| coordinate convention | x-left / y-back / z-up **verified against travel direction** |
| ego-motion transforms | round-trip **3.7e-13** |
| distortion ambiguity | **resolved: images are raw, undistort once** |

**No Phase A blocker.** Proceed to Phase B.
