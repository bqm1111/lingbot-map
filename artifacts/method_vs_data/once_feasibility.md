# ONCE feasibility — can it support the frozen occupancy supervision?

**Audited 2026-10-20.** Sources: ONCE's official documentation and download pages, and
OccAny's own ONCE reader (`/home/minh/workspace/OccAny/dataset_setup/once/preprocess.py`).
**No ONCE data was downloaded** — §6 explains why that was not possible on the same terms as
the DDAD audit, and what it costs this verdict.

> ### ⚠ SUPERSEDED IN PART, 2026-10-21
>
> The user has **authorized** the protocol decision that blocked this audit. §1–§9 below are
> the original report and are preserved unchanged, including the predeclared acceptance bands.
> **§10 records the authorized decision** and **§11 replaces the "NOT verified" rows with
> measurements from the real `train_infos.tar`.** Read §10–§11 for current status.
>
> One §1 row is now known to have been wrong: item 3's HFOV. ONCE's forward wide camera is
> **90.1°**, not narrow — see §11.

> ## Verdict (original, 2026-10-20): **CONDITIONAL GO** — and **Phase C is not launched.**
>
> The gate DDAD failed, ONCE passes on authoritative documentation: *"The LiDAR coordinate is
> **placed at the center of the LiDAR sensor**"*, with a single `lidar_roof` sensor. The ray
> origin our carve rule needs is published, not invented.
>
> But two conditions are unmet, and one of them is the brief's **hard** temporal gate:
>
> 1. **ONCE's released point clouds are 2 FPS, downsampled from 10 FPS.** The frozen recipe
>    accumulates ~21 sweeps spanning ~2.1 s. On ONCE, 21 sweeps span **10.5 s**, and a 2.1 s
>    window gives only **5 sweeps**. The frozen "same temporal sweep policy" cannot be
>    preserved; one or the other must give. §4.
> 2. **The origin claim is documented, not measured.** DDAD's prose was also reasonable and
>    its data still failed. §6.
>
> The brief states the temporal gate is "a hard gate, not an optional diagnostic", so training
> on ONCE is not authorized by this audit.

---

## 1. Audit results against brief §6's fourteen items

| # | item | status | value |
|---|---|---|---|
| 1 | version and split | **verified** | labelled train / val / test, plus unlabeled small / medium / large; 16 k annotated scenes of 1 M total |
| 2 | RGB cameras and calibration | **verified** | 7 cameras `cam01, cam03, cam05–cam09`; per-sequence `calib[cam]` with `cam_to_velo` (4×4), `cam_intrinsic` (3×3), `distortion` (1×7) |
| 3 | camera HFOV and resolution | **resolution verified** 1920×1020; **HFOV NOT verified** — needs the intrinsics, which ship only inside the data |
| 4 | LiDAR identity and origin | **verified** | one `lidar_roof`; *"The LiDAR coordinate is placed at the center of the LiDAR sensor"* |
| 5 | LiDAR→camera / →vehicle | **verified** | `cam_to_velo` published per camera; the LiDAR frame **is** the reference frame, so no vehicle-frame hop is needed |
| 6 | temporal ego poses | **verified** | per frame, `pose = (quat_x, quat_y, quat_z, quat_w, trans_x, trans_y, trans_z)`, relative to the scene's first frame |
| 7 | timestamp consistency | **verified as a design** | `frame_id` is the timestamp; OccAny's reader computes `round((t − t0)/100)`, i.e. 100 ms units. Not confirmed on data |
| 8 | five-frame causal clip | **PASSES, and fits unusually well** | released cadence is 2 FPS = **0.5 s**, exactly our clip spacing; 5 consecutive frames span 2.0 s, matching the frozen causal contract |
| 9 | future sweeps for supervision | **PARTIAL — the blocker** | sweeps exist, but at 0.5 s spacing; see §4 |
| 10 | 256×256×32 grid compatibility | **CONDITIONAL** | needs a fixed axis rotation; see §3 |
| 11 | occupied/free/unknown statistics | **NOT measured** | requires data; predeclared acceptance bands are in `predeclaration.md` §6 |
| 12 | sequence count and anchors | **NOT verified** | the per-split sequence counts live in `*_infos.tar` |
| 13 | licence and access | **verified** | public, but Google Drive / Baidu Pan only; Baidu extraction code `1234`. §6 |
| 14 | preprocessing and training cost | **partly estimable** | only **1 of 7** camera tars is needed (single forward camera), plus `train_lidar.tar` and `train_infos.tar` |

---

## 2. Why ONCE is the right candidate: the origin gate

Our supervision (`core/supervision/rawtarget.py`) decides every FREE voxel by casting a ray
**from the sensor origin**. DDAD failed because it merges four Luminars into a vehicle frame
whose origin sits on the ground, leaving the true origins unrecoverable — measured elevation
p1 of −1.35° confirmed rays graze along the road.

ONCE, per its own documentation:

> "There are 3 types of coordinate systems on the ONCE dataset... The LiDAR coordinate is
> **placed at the center of the LiDAR sensor**, with x positive to the left, y-axis positive
> to the back and the z-axis positive upwards. We additionally provide a transformation matrix
> (vehicle pose) between each two adjacent LiDAR coordinates, which enables the fusion of
> multiple point clouds."

and

> "Each point cloud is represented as an N×4 matrix... The 3D coordinate (x, y, z) is based on
> the LiDAR coordinate, and r denotes the reflection intensity."

OccAny's reader corroborates the structure: a single `self.lidar_name = "lidar_roof"`,
`np.fromfile(...).reshape(-1, 4)`, `calib[cam]['cam_to_velo']`, and
`lidar_pose = pose_matrix(frame_anno['pose'])`.

**This is structurally the KITTI-360 situation** — one sensor, points in that sensor's own
frame, sensor→world poses per frame. The frozen single-sensor `rawtarget.build` applies with
**no new assumption and no invented constant**, which is precisely what DDAD could not offer.

---

## 3. Axis convention — a condition, not a blocker

ONCE uses **x positive to the left, y positive to the back, z up**. This is right-handed
(x×y = z checks out) but it is a 90° yaw rotation away from the x-forward convention our grid
assumes: the SSCBench-KITTI-360 grid is 256×256×32 at 0.2 m with origin (0, −25.6, −2), i.e.
x forward 0…51.2 m and y lateral ±25.6 m. Dropping ONCE points in unrotated would lay the
grid across the road instead of along it.

The fix is a **fixed, exactly known permutation** — the same class of transform as the Waymo
arm's T1 axis gate, which passed at 0 px error. It is a condition to implement and verify, not
missing information. It must be gated the same way (T1-style round-trip plus a visual check),
never assumed.

---

## 4. The hard blocker: 2 FPS against a ~21-sweep policy

From the documentation:

> "The original LiDAR data is recorded at a speed of 10 frames per second (FPS). We further
> **downsample those original data with the sampling rate of 2 FPS**, since most adjacent
> frames are quite similar thus redundant."

So the released cadence is **0.5 s**. That is excellent for the *input* contract — 5
consecutive frames span exactly our 2.0 s causal window (§1 item 8). It is a problem for the
*supervision* contract.

The frozen recipe accumulates ~21 sweeps per anchor: measured 20.94 on KITTI-360 and 21.00 on
Waymo, both at 10 Hz, i.e. a **~2.1 s** accumulation window. On ONCE:

| choice | sweeps | span | consequence |
|---|---:|---:|---|
| match the **sweep count** | 21 | **10.5 s** | at ~10 m/s that is ~105 m of travel into a 51.2 m grid: most sweeps contribute from outside it, and dynamic objects smear across many voxels, which the conflict rule turns into UNKNOWN |
| match the **time span** | **5** | 2.1 s | 4× fewer sweeps carving free space and supporting `n_frames_occ`, which directly moves `valid_fraction` and the conflict rate |

**Neither preserves the frozen policy.** This is not a tuning knob — the brief forbids changing
occupancy targets to force statistical similarity, and either choice is a documented deviation
from the recipe that A0 and A1 were trained under. It therefore requires an explicit decision,
informed by measurement, and it is why this audit does not return GO.

The measurement that would settle it is already predeclared, before any ONCE data was
inspected (`predeclaration.md` §6): build targets both ways on ≥3 sequences and require
occupied prevalence in **[0.015, 0.040]**, conflict-to-unknown ≤ **0.12**, valid fraction in
**[0.45, 0.75]**, and occupancy in ≥30 of 32 z-levels. If one choice lands inside those bands
and the other does not, the deviation is bounded and defensible; if neither does, it is NO-GO.

---

## 5. A discrepancy worth flagging before anyone builds the adapter

The documentation says the released images are already corrected:

> "The camera data is also downsampled along with the LiDAR data for synchronization, and then
> **the distortions are removed** to enhance the quality of the images."

Yet the per-sequence `calib` still ships a 1×7 `distortion` vector, and OccAny's reader calls
`cv2.undistort(image, cam_intrinsic, distortion)` on the loaded JPEG. If the shipped images are
already undistorted, that is a **second** undistortion, which would corrupt the geometry our
hard map depends on.

This is unresolved and **must be checked on real images before any adapter is trusted** — it is
exactly the class of error the Waymo T2 gate caught (p99 residual 0.237 m against a 0.20 m
one-voxel criterion). Stated here so it is not inherited silently from OccAny's code.

---

## 6. Access: why no data was probed, and what that costs

The DDAD audit was cheap because `DDAD.tar` sits on public S3, honours HTTP range requests, and
tar is sequential — 1.2 GiB of ranged reads recovered a complete scene out of 256 GiB.

ONCE has no equivalent path. It is distributed only through **Google Drive folders** and
**Baidu Pan** (extraction code `1234`). Verified by listing: the three training-split Drive
folders resolve and contain `train_cam01.tar` … `train_cam09.tar`; `gdown` 5.2.1 is installed
and can enumerate them. But Drive serves whole files, not byte ranges, so there is no way to
read one sequence's calibration or one point cloud without pulling a multi-gigabyte tar.

**What this costs:** every "verified" row in §1 rests on documentation and on OccAny's reader,
not on ONCE bytes. The origin claim in §2 is the decisive one, and DDAD is the standing warning
that plausible prose can still disagree with shipped data — DDAD's README also described its
sensors accurately, and the problem was what the *release* preserved.

**The cheapest honest next step** is `train_infos.tar` alone (the per-sequence JSONs: no point
clouds, no images). It would settle items 3, 7 and 12 — camera HFOV from the real intrinsics,
timestamp spacing, sequence and frame counts, and the `cam_to_velo` translations — for a small
download. Only the origin check and the §4 target statistics need `train_lidar.tar`.

---

## 7. What must pass before Phase C is authorized

1. **Empirical origin check.** On one sequence: z distribution of points within 6 m
   horizontally (expect the road at z ≈ −1.8 m, *not* ≈ 0 as DDAD had), and elevation angles
   spanning well below 0°. `tools/method_vs_data/ddad_probe.py` runs this against the frozen
   builder through a duck-typed geometry object and needs only a thin reader to be pointed at
   ONCE — it is the tool that turned a multi-day adapter task into an afternoon's audit.
2. **The §4 sweep-policy decision**, resolved against the predeclared bands, and recorded as an
   explicit protocol deviation with the user's authorization.
3. **The §3 axis rotation**, implemented and gated T1-style.
4. **The §5 distortion question**, resolved on real images.
5. **HFOV recorded before training** (`predeclaration.md` §6). This matters: DDAD's only
   forward camera is 47.9° and Waymo's are 50.0°, and the Waymo arm failed. If ONCE's forward
   camera is also narrow, the arm's power to separate "dataset identity" from "narrow training
   FOV" is weak, and that must be known in advance rather than discovered afterwards.

---

## 8. PandaSet, at feasibility-audit level only

Per the brief, assessed but not pursued. From OccAny's own reader, which documents the devkit
schema verbatim:

```
# index        x           y         z        i         t       d
# - `i`: float: Reflection intensity in a range [0,255]
# - `t`: float: Recorded timestamp for specific point
# - `d`: int: Sensor ID. 0 -> mechanical 360° LiDAR, 1 -> forward-facing LiDAR
```

**PandaSet publishes a per-point sensor ID** — exactly the field DDAD lacks. With two devices
and per-device poses from `pandaset-devkit`, our already-validated
`core/supervision/rawtarget_multisensor.py` (proven bit-for-bit identical to the frozen
single-sensor rule during the Waymo work) applies directly. Per-point timestamps would also
allow motion compensation. OccAny discards all of it (`[:, :3]`, world frame).

**Unverified:** PandaSet's release cadence (the 2 FPS issue may or may not recur), camera HFOV,
sequence counts, download size, and whether per-device poses are published for both sensors.
`pandaset-devkit` on GitHub returns HTTP 200. **No audit beyond this was performed**, and
PandaSet is not recommended over ONCE on present evidence — ONCE's single sensor is still the
simpler fit, and its only hard problem (§4) may well recur in PandaSet too.

---

## 9. Summary

| gate | result |
|---|---|
| sensor origin published and usable | **PASS** (documented), **unmeasured** |
| single LiDAR, points in sensor frame | **PASS** |
| per-frame ego poses | **PASS** |
| five-frame causal input contract | **PASS** — 0.5 s cadence matches exactly |
| same temporal sweep policy | **FAIL as specified** — 2 FPS cannot give 21 sweeps in 2.1 s |
| voxel grid conventions | **CONDITIONAL** — fixed known axis rotation required |
| distortion handling | **UNRESOLVED** — possible double-undistortion in OccAny's path |
| target statistics in predeclared bands | **NOT MEASURED** |
| data access | **PASS but awkward** — Drive/Baidu only, no ranged reads |

**CONDITIONAL GO. Phase C not launched.** The blocking item is §4, and it needs a decision from
the user rather than a judgement from me, because every available resolution is a departure
from the frozen recipe that A0 and A1 were trained under.

---

# ADDENDUM, 2026-10-21

§1–§9 above are the original 2026-10-20 report, preserved unchanged, including the
predeclared acceptance bands referenced from `predeclaration.md` §6. Nothing in them has been
edited except the superseded-notice at the top.

## 10. Authorized protocol decision — recorded before training

The blocking item in §4 was that ONCE's 2 FPS release cannot reproduce the frozen ~21-sweep /
~2.1 s supervision window. The user has resolved it:

> **Use a fixed approximately 2.1-second LiDAR supervision window, with the number of real
> sweeps determined by the dataset's native acquisition frequency.** This means approximately
> 21 sweeps for KITTI-360/Waymo and approximately 5 sweeps for ONCE at 2 FPS. Do not
> interpolate or synthesize extra LiDAR sweeps. Do not use 21 ONCE sweeps spanning
> approximately 10.5 seconds.

**This is an explicitly authorized dataset adaptation, not a claim of identical supervision.**
It is recorded here, before any ONCE training, as a protocol difference that must be carried
into every comparison involving an ONCE-trained arm:

| | KITTI-360 (A0) | Waymo | **ONCE (authorized)** |
|---|---|---|---|
| native LiDAR rate | 10 Hz | 10 Hz | **2 Hz** |
| supervision window | ~2.1 s | ~2.1 s | **~2.1 s** (matched) |
| real sweeps in that window | 20.94 | 21.00 | **~5** (4× fewer) |
| sweeps interpolated or synthesised | none | none | **none** |

**What the matched window preserves:** the temporal extent over which geometry is accumulated,
and therefore which parts of the scene the target can cover and how much parallax and
dynamic-object motion it contains.

**What it does not preserve:** the number of independent free-space carves per voxel. With 4×
fewer sweeps, `n_free` and `n_frames_occ` are built from 4× fewer observations, so
`valid_fraction` and the cross-sweep conflict rate are expected to move. §12 measures whether
they move outside the **predeclared** bands, which are **not** adjusted for this decision.

**Consequence for the research claim.** Any ONCE-vs-KITTI-360 or ONCE-vs-Waymo difference is
confounded by this supervision difference. An ONCE arm therefore cannot, on its own, separate
"different source dataset" from "4× fewer carves per window". The brief anticipates this and
asks (step 10) for a **KITTI-360 2 Hz control** — KITTI-360 supervision decimated to 2 Hz with
matched ~2.1 s windows — which is the control that would isolate it. That control is prepared
but not launched unless needed to interpret the ONCE result.

## 11. Measured from the real `train_infos.tar` — §1 rows now verified

`train_infos.tar` (57,620,480 bytes) obtained via `gdown` from the Training-split
*annotations* Drive folder `14cI2vleBokHEtSLjZAgzmjL1cFOkoYwD`, file id
`1Qa6JYbp7SEZtq7wI0Sw7bvhLc-CZdX8I`. Audit code `tools/method_vs_data/once_infos_audit.py`,
output `artifacts/method_vs_data/results/once_infos_audit.json`.

**Access note, stated because the original report got this wrong.** §6 concluded no ONCE data
could be probed because Drive serves whole files. That was an overstatement of the obstacle:
`gdown` retrieved `train_infos.tar` in about two seconds, and the only real difficulty was
locating the right folder ids among the download page's 32 links. The labels precede their
hrefs in the page source, which is what defeated the first parse.

### Corpus

| quantity | measured |
|---|---|
| sequences (labelled train split) | **6** |
| frames | **9,977** |
| frames with an ego pose | **9,977 (100 %)** |
| frames with 3D annotations | 4,961 (49.7 %) — irrelevant to us; we use no labels |
| frame interval `dt` | p50 **0.500 s**, min 0.499 s, **max 13.001 s** |
| implied rate | **2.00 FPS** — confirms the documented downsampling |
| `dt` monotonic everywhere | **yes** |
| intervals within 5 % of 0.500 s | **98.3 %** |
| ego step between frames | p50 **2.746 m**, p99 10.350 m → ≈ **5.5 m/s** |
| five-frame causal clips available | 9,953 |
| **independent sequences** | **6** |

Per sequence: 1,165–2,055 frames, spanning **608–1,107 s** (10–18 minutes) and **2.8–7.1 km**.

Two consequences:

* **ONCE sequences are long drives, not short clips.** Unlike Waymo's 20 s segments (798
  segments, 1 independent anchor each), a single ONCE sequence covers kilometres, so
  within-sequence scene diversity is high. But **independence is still only 6**, far fewer than
  Waymo's 798 and only twice KITTI-360's 3. Any ONCE result must quote 6 independent
  sequences, never 9,953 clips, as its independent unit.
* **1.7 % of intervals are gaps**, up to 13 s. Window construction must verify the realised
  span and reject any window containing a gap, rather than assume 5 consecutive frames span
  2.0 s. §12 records realised spans.

### Cameras — item 3, and the finding that changes the outlook

Calibration is **identical across all six sequences** (max absolute intrinsic difference
0.000 px), so one calibration governs the corpus. Image size 1920 × 1020.

| camera | fx | **HFOV** | VFOV | `cam_to_velo` translation | distortion all zero? |
|---|---:|---:|---:|---|---|
| **cam01** | 958.83 | **90.1°** | 55.8° | (−0.136, −0.373, −0.303) | **no** |
| cam03 | 1938.35 | 52.7° | 29.5° | (0.042, −0.415, −0.308) | no |
| cam05 | 956.18 | 90.2° | 56.1° | (0.935, −1.250, −0.990) | no |
| cam06 | 960.17 | 90.0° | 55.9° | (0.826, 1.637, −0.777) | no |
| cam07 | 958.72 | 90.1° | 56.0° | (−0.828, 1.663, −0.773) | no |
| cam08 | 961.42 | 89.9° | 55.9° | (−0.941, −1.246, −0.991) | no |
| cam09 | 957.39 | 90.2° | 55.9° | (−0.003, 0.765, −0.226) | no |

**`cam01` is 90.1° HFOV.** The original §1 listed HFOV as unverified and §7 warned that a
narrow forward camera would weaken the arm — because DDAD's only forward camera is 47.9° and
Waymo's are 50.0°, and the Waymo arm failed. **That concern does not apply to ONCE.** 90.1°
sits between SemanticKITTI's 82.0° test condition and KITTI-360's 103.9° training condition,
and is far from the narrow regime. On input geometry, ONCE is the closest match to our existing
arms of any dataset in OccAny's pool.

This is recorded **before** training, per `predeclaration.md` §6's requirement to declare HFOV
in advance. No directional prediction is attached to it: the Waymo arm already falsified
reasoning from input-statistic similarity, and that lesson stands.

ONCE also ships a second forward camera, `cam03` at 52.7°, which would permit a within-dataset
FOV control at no extra download cost. Not authorized; noted.

### Distortion — item 7's ambiguity sharpened, not yet resolved

**All seven distortion vectors are non-zero**, while the documentation states the released
images have had distortions removed. Those two facts are in tension, and OccAny's reader
applies `cv2.undistort` on top. The ambiguity is therefore real and must be settled
geometrically, not by reading. §13 does that.

### Still requiring point clouds

| item | status |
|---|---|
| empirical sensor-origin check | pending `train_lidar.tar` (10.4 GiB, downloading) |
| occupancy target statistics vs the predeclared bands | pending |
| distortion resolution by reprojection | pending |
| axis-rotation verification | pending |

---

# ADDENDUM 2, 2026-10-21 — a correction to my own blocker, and the gate results

## 12. CORRECTION: the "2 FPS problem" in §4 was my error

**§4 is wrong and is retracted.** It claimed the frozen supervision window is ~2.1 s with ~21
sweeps, so ONCE's 2 FPS release could not reproduce it. I conflated two different windows:

| window | what it is | span |
|---|---|---|
| the **camera input clip** | 5 causal frames feeding LingBotMap | **~2.0 s** |
| the **LiDAR supervision window** | `[anchor] + future_natives` fed to `rawtarget.build` | **~10.5 s** |

Measured, not inferred:

* **KITTI-360**: `core/run/build_targets.py` calls
  `RT.build(geo, anc.native_frame, [anc.native_frame] + list(anc.future_natives), dev)` — 21
  sweeps. Stream frames are every **5th** native frame and the velodyne period is 0.1046 s, so
  the sweep spacing is **0.523 s** and the span is **10.461 s**.
* **Waymo**: `STREAM_STRIDE = 5 native frames = 0.50 s`, `n_frames_used = 21.0` → **10.0 s**.

**ONCE's native cadence is 0.500 s.** That is the same sweep spacing both existing arms already
use. So **21 ONCE sweeps spanning 10.0 s reproduces the frozen recipe exactly**, and the
adaptation §4 demanded was never necessary.

The authorization in §10 was granted on this false premise. Its stated intent — match the
frozen supervision — is served by the *opposite* of its letter: the letter said "do not use 21
ONCE sweeps spanning approximately 10.5 seconds", which is precisely what KITTI-360 and Waymo
do. **Both configurations were therefore measured** (§13) rather than either being chosen
unilaterally, so the decision rests on numbers instead of on my premise. The predeclared bands
were not touched.

## 13. Gate results on real ONCE point clouds

Data: `train_infos.tar` (57.6 MB), `train_lidar.tar` (10.4 GB), `train_cam01.tar` (1.5 GB),
all via `gdown`; 9,977 clouds and 9,977 cam01 images across 6 sequences. Code
`tools/method_vs_data/once_gates.py`, `once_reader.py`. Raw
`artifacts/method_vs_data/results/once_gates{,_21sweeps}.json`.

### Geometry and calibration gates — PASS 6/6, and G1 decisively

| gate | result |
|---|---|
| **G1 sensor origin** | **PASS 6/6** |
| **G2 axis rotation** | **PASS 6/6** |
| **G3 transform round-trip** | **PASS 6/6** |
| **G4 window construction** | **PASS 6/6** |
| **G6 temporal ordering** | **PASS 6/6** |

**G1 is the gate DDAD failed, and ONCE passes it conclusively.** Road surface mode at
**−1.730 to −1.762 m** below the frame origin — a roof-mounted sensor — with elevation p1 at
**−24.97°** in every sequence. DDAD gave **+0.08 m** and **−1.35°**, which is what made its rays
graze along the road. 64,128–67,328 points per sweep.

**G2** verified against the vehicle's own motion rather than asserted: the direction of travel
is `[1.0000, 0.0003, −0.0026]` in our x-forward convention and `[0.0003, −1.0000, −0.0026]` in
ONCE's, confirming the exact 90° yaw `R = [[0,−1,0],[1,0,0],[0,0,1]]`. **G3** round-trips to
3.7e-13. **G4** realises exactly 21 sweeps at 10.000 s (or 5 at 2.000 s), with 6–22 of 32–56
candidate anchors rejected for containing a timing gap — the 1.7 % gap population is handled,
not assumed away.

### G5 occupancy statistics — FAIL, on occupied prevalence only

Frozen-matched protocol (21 sweeps, 10.0 s), 172 targets over 6 sequences, judged against the
**unchanged** predeclared bands:

| sequence | targets | prevalence | valid | conflict | z-levels | G5 |
|---|---:|---:|---:|---:|---:|---|
| 000076 | 23 | **0.0120** | 0.6971 | 0.0394 | 32 | FAIL |
| 000080 | 23 | 0.0195 | 0.6392 | 0.0394 | 32 | PASS |
| 000092 | 30 | **0.0138** | 0.7023 | 0.0369 | 32 | FAIL |
| 000104 | 25 | **0.0144** | 0.7044 | 0.0336 | 32 | FAIL |
| 000113 | 37 | **0.0133** | 0.6909 | 0.0328 | 32 | FAIL |
| 000121 | 34 | **0.0128** | 0.5728 | 0.0404 | 32 | FAIL |
| band | | **[0.015, 0.040]** | [0.45, 0.75] | ≤ 0.12 | ≥ 30 | |

**Only occupied prevalence fails**, in 5 of 6 sequences, at a median of ≈0.0133 against a floor
of 0.015. Valid fraction, conflict rate and vertical coverage all pass everywhere, and the
conflict rate (0.033–0.040) is comfortably below KITTI-360's 0.0752–0.0725.

Under the authorized 5-sweep / 2.0 s protocol the same quantity is 0.0097–0.0245, median
0.0142 — 3 of 6 sequences pass. Marginally better, still a failure, and §14 shows why the
sweep count is not the cause.

## 14. The KITTI-360 control (brief step 10) — measurement only, no training

`tools/method_vs_data/k360_sweep_control.py`, 25 anchors per drive, identical code path,
identical bands. Raw `artifacts/method_vs_data/results/k360_sweep_control.json`.

| drive | sweeps | sweep dt | span | prevalence | valid | conflict | in bands |
|---|---:|---:|---:|---:|---:|---:|---|
| 0003 | 21 | 0.523 s | 10.46 s | 0.0217 | 0.6581 | 0.0752 | yes |
| 0007 | 21 | 0.522 s | 10.44 s | 0.0256 | 0.5910 | 0.0725 | yes |
| 0010 | 21 | 0.521 s | 10.42 s | 0.0223 | 0.6528 | 0.0725 | yes |
| 0003 | **5** | 2.615 s | 10.46 s | **0.0212** | 0.6229 | 0.0219 | yes |
| 0007 | **5** | 2.610 s | 10.44 s | **0.0208** | 0.5714 | 0.0134 | yes |
| 0010 | **5** | 2.605 s | 10.42 s | **0.0213** | 0.5956 | 0.0239 | yes |

Two things this establishes.

**The measurement path is validated.** KITTI-360 reproduces 0.0217–0.0256 prevalence through
the same code that gives ONCE 0.0120–0.0195, against the 0.0276 recorded for drive 0000. So the
ONCE numbers are not an artifact of my reader or my gate.

**Sweep count is not the cause.** Decimating KITTI-360 from 21 sweeps to 5 moves prevalence by
less than 0.005 (0.0217→0.0212, 0.0256→0.0208, 0.0223→0.0213). It collapses the conflict rate,
as expected, and barely touches prevalence. **ONCE's prevalence shortfall is therefore a
property of ONCE, not of the window protocol.**

Plausible mechanism, not separately verified: ONCE's LiDAR returns ~65 k points per sweep
against KITTI-360's ~120 k, and ONCE's valid fraction is *higher* (0.57–0.70 vs 0.59–0.66) —
more of the grid is resolved, mostly as free space. Occupied over valid falls accordingly.

## 15. Distortion — resolved, and the documentation is wrong

`tools/method_vs_data/once_distortion_gate.py`, 36 frames over 6 sequences. Raw
`artifacts/method_vs_data/results/once_distortion_gate.json`.

First, a data correction: the documentation describes `distortion` as 1×7, but the shipped
vector has **5** values — `[−0.3121, 0.1024, 0.00033, −0.000398, −0.0158]` — a standard OpenCV
(k1, k2, p1, p2, k3) model. OpenCV rejects a true 7-vector, so any code passing one would have
failed loudly; 5 is accepted and is what is there.

**M1 — the choice is MATERIAL.** Projecting LiDAR with the plain pinhole `K` versus with the
distortion model disagrees by **p99 51 px, max 94 px**, which at each point's own depth is
**p99 2.16 m** — more than **ten times** the 0.20 m one-voxel criterion the Waymo T2 gate used.
The disagreement grows with radius as a radial model should: 1.7 px median inside r<300 px,
21.5 px at r 300–700 px. This cannot be ignored.

**M2 — the images are still distorted.** Gradient alignment at LiDAR depth-edge points favours
applying the undistortion in **6 of 6 sequences** (mean 37.45 vs 35.91). The margin is small
(~4 %), so on its own M2 is weak; two further facts point the same way and are not weak:
**k1 = −0.3121** is far too large to appear in the calibration of an already-rectified image,
and DDAD — which genuinely ships rectified images — carries **no distortion field at all**,
only `fx, fy, cx, cy, skew`.

**Conclusion: ONCE's released cam01 images are NOT undistorted**, contrary to the
documentation's *"the distortions are removed"*. OccAny's `cv2.undistort` call is correct, and
any adapter of ours must undistort before the hard map sees the image. This is a required
implementation step, not a blocker — but trusting the documentation here would have silently
corrupted every projection by metres.

## 16. Verdict

| gate | result |
|---|---|
| G1 sensor origin | **PASS 6/6** — road at −1.73 to −1.76 m, elevation p1 −24.97° |
| G2 axis rotation | **PASS 6/6** — exact 90° yaw, verified against ego motion |
| G3 transform round-trip | **PASS 6/6** — 3.7e-13 |
| G4 window construction | **PASS 6/6** — 21 sweeps / 10.000 s, gaps rejected |
| G5 occupancy statistics | **FAIL 5/6** — occupied prevalence ≈0.0133 vs floor 0.015 |
| G6 temporal ordering | **PASS 6/6** |
| distortion | **resolved** — images are raw; undistortion required |
| camera HFOV | **90.1°** — the best match in OccAny's pool, not the narrow regime |

**ONCE is geometrically and calibrationally sound. It fails the occupancy-supervision gate on
one quantity: occupied prevalence is about 0.6× KITTI-360's, and the KITTI-360 control proves
this is the dataset rather than the protocol.**

Per the brief — *"If ONCE fails the occupancy-supervision or calibration gates, stop before
full training, report the failure, and identify whether PandaSet is a more suitable next
candidate"* — **training is not launched.** §17 of `once_geometry_results.md` assesses
PandaSet.

**What the user may want to weigh.** The failure is one-sided and modest: 0.0133 against a 0.015
floor, with 000080 passing at 0.0195. The band exists so that a training result is attributable
to the dataset's content rather than to a different positive-class balance seen by focal+Dice
on a 128×128×32 crop; a ~1.6× rarer positive class is a real optimization difference, which is
why the gate is not a formality. Widening the band is a protocol change and is **not** made
here. The bands remain exactly as predeclared.

---

# Addendum 3 — what the prevalence shortfall actually is

**2026-10-27.** §1–§16 are unmodified and the predeclared bands are unchanged. This addendum
adds measurements only; it changes no threshold and launches no training.

§16 left the G5 prevalence failure as a judgement call for the user: is 0.0133 against a 0.015
floor acceptable? That framing was incomplete. A gate failure is only interpretable once its
cause is known, and the cause was not measured. It is measured here. New code:
`tools/method_vs_data/k360_beam_control.py`; raw:
`artifacts/method_vs_data/results/{k360_beam_control,k360_point_budget_control}.json`.

## 17. The shortfall is in absolute occupied voxels, not a denominator effect

`prevalence_in_valid` is a ratio, so a larger `valid` would depress it mechanically. It does
not: ONCE's valid fraction is **higher** than KITTI-360's, which makes the deficit worse rather
than explaining it.

| | valid (median) | prevalence (median) | **occupied voxels per target** |
|---|---:|---:|---:|
| KITTI-360, 3 train drives | 0.653 | 0.0233 | **30,529** |
| ONCE, 6 sequences | **0.694** | 0.0133 | **19,798** |

Of 2,097,152 voxels. **ONCE targets carry ~35 % fewer occupied voxels in absolute terms.** The
shortfall is real supervision, not an artefact of normalization.

## 18. It is NOT LiDAR sparsity — the hypothesis is falsified

ONCE's sensor is visibly sparser, measured on real clouds (5 per sequence):

| | points / sweep | distinct 0.1° elevation bins | r99 | r max | points landing in our grid |
|---|---:|---:|---:|---:|---:|
| KITTI-360 (HDL-64E) | **120,055** | 250–291 | 52–68 m | **80 m** | ~48,800–63,400 |
| ONCE (`lidar_roof`) | **66,563** | **exactly 40** | 93–132 m | **~198 m** | ~23,700–28,500 |

ONCE is a 40-beam, ~200 m sensor: the same 66 k returns are spread over roughly 15× the volume,
so only ~25 k land inside the 51.2 × 51.2 × 6.4 m grid against KITTI-360's ~56 k. That looked
like a sufficient explanation. **It is not.** Decimating KITTI-360's own sweeps through the same
frozen builder:

| KITTI-360 condition | points kept | prevalence (median of 3 drives) | in band? |
|---|---:|---:|---|
| unmodified, 64 beams | 1.000 | **0.0233** | ✓ |
| 40 beams — ONCE's beam count | 0.625 | 0.0212 | ✓ |
| 24 beams | 0.375 | 0.0188 | ✓ |
| uniform 45 % — matches ONCE's in-grid point count | 0.450 | 0.0225 | ✓ |
| uniform 25 % | 0.250 | 0.0222 | ✓ |
| uniform 15 % | 0.150 | **0.0214** | ✓ |
| **ONCE, actual** | — | **0.0133** | ✗ |

**Removing 85 % of KITTI-360's LiDAR returns leaves it comfortably in band.** The 21-sweep
accumulated target saturates at 0.2 m voxels: surviving returns hit the voxels the removed ones
would have. Prevalence elasticity to point count is about 0.1 — an 85 % cut costs 8 %. No
achievable sparsity reproduces ONCE's 0.0133.

Two causes are therefore excluded: the adapter (G1–G4 and G6 pass 6/6, round-trip 3.7e-13) and
sensor sparsity (this section). What remains is **scene content**: ONCE's scenes simply contain
less occupied surface inside the grid. The free-space figures confirm it directly — ONCE
resolves **more** of the grid as known while **less** of it is occupied (free fraction 0.685
against 0.638; occupied fraction of all voxels 0.0092 against 0.0152). That is the signature of
more open road: wide carriageways, fewer close buildings, less roadside vegetation than
KITTI-360's German suburbia.

## 19. What that means for the gate — stated, not acted on

The band was predeclared to catch **supervision-recipe errors**, and it worked: DDAD's
0.0042–0.0094 was 3–6× out because its ray origin is wrong, which is a fault. ONCE's 0.0133 has
no fault behind it. So on ONCE the band is **not** measuring what it was designed to measure;
it is measuring how built-up the scenery is.

This matters because the two readings lead opposite ways:

* **The band's letter.** 0.0133 < 0.015, 5 of 6 sequences fail, the gate fails, stop. The
  underlying worry survives the diagnosis intact: focal+Dice on a 128×128×32 crop would see a
  ~1.6× rarer positive class, so an ONCE arm's result is confounded by class balance.
* **The band's purpose.** The confound is now known to be *inseparable from the dataset*. It is
  not a bug to fix, not a parameter to tune, and no protocol change reaches it — §18 shows even
  an 85 % sparsity change does not. Applied here the band says "never train on a dataset whose
  scenes are more open than KITTI-360's", which forecloses the experiment permanently rather
  than conditionally, and forecloses it on the one axis the experiment exists to vary.

**No threshold is changed and no training is launched.** Recording the choice precisely: the
decision is not *"is 0.0133 close enough to 0.015?"* but *"is a verified 35 % scene-content
difference an acceptable, stated confound, or a disqualification?"* That is the user's call,
and it is a different question from the one §16 put to them.

**What an ONCE arm would and would not test, after this diagnosis.** It would test *our method
trained on an OccAny source dataset whose scenes are materially more open than our current
training corpus*. It would **not** be a clean dataset-identity test, and the write-up must say
so. One point of prior evidence bears on it: the Waymo arm (in band at 0.0222) still lost 3.0
and 9.1 SC IoU, so corpus change alone is already known to be costly in our system.

**What would make it clean, and the cost.** A density-and-content-matched KITTI-360 control
cannot be built by decimation (§18 falsifies the mechanism), so matching would require
selecting KITTI-360 anchors by scene openness until prevalence matches ONCE's 0.0133 — feasible
from the existing cache, but a new training arm, not a measurement. Not proposed here.
