# ONCE occupancy-supervision validation — Phase B

**2026-10-27.** Deliverable for brief §6–§9, under the authorized ~2.1 s native-cadence
supervision policy. Code: `tools/method_vs_data/{once_reader,once_gates,k360_sweep_control,
k360_beam_control,supervision_density_compare}.py`. Raw:
`artifacts/method_vs_data/results/{once_gates,once_gates_21sweeps,k360_duration_matched,
k360_beam_control,k360_point_budget_control,supervision_density_compare}.json`.

**No predeclared band was changed. No threshold was tuned. No training was launched.**

---

## 1. Verdict

> ## CONDITIONAL GO
>
> Every mandatory **correctness** gate passes: the published sensor origin is confirmed
> empirically, the axis convention is verified against the vehicle's own motion, pose
> round-trip is 3.7e-13, poses exist for 100 % of window frames, windows are built from real
> native sweeps only, and ray carving is physically plausible. ONCE **can** provide valid
> occupancy targets under the authorized policy.
>
> The bounded issue is **supervision density**: ONCE targets carry about **34 % fewer occupied
> voxels** than duration-matched KITTI-360, concentrated at mid height and long range. This is
> a stated confound, not a defect, and it requires no protocol change.
>
> The predeclared occupied-prevalence band is **not met** (median 0.0130 against a 0.015
> floor). §5 shows that band cannot adjudicate this protocol: applied to the authorized 2.1 s
> window it **also rejects KITTI-360**, the corpus it was calibrated from.

Per brief §10 — *"Execute this phase only if Phase B returns GO"* — Phase C is **not launched**
on a CONDITIONAL GO. §8 states what it would cost and what is already built.

## 2. The minimal reader (brief §6)

`tools/method_vs_data/once_reader.py` reads **no labels and no images**. It exposes exactly the
three members `core.supervision.rawtarget.build` touches, so the frozen builder runs unmodified:

| member | meaning |
|---|---|
| `cam0_to_world` | membership only; mapped to `T_sensor_to_world` |
| `read_velodyne(nf)` | `(N,3)` xyz in the sensor frame, rotated into our convention |
| `velo_to_velo(a,b)` | `inv(T[b]) @ T[a]` |

No separate occupancy-supervision implementation was written. All transformations are documented
in `once_metadata_audit.md` §4. The authorized window is built by `OnceSequence.window(anchor,
span_s=2.1)`, which consumes only real frames at the native 0.500 s cadence, **interpolates
nothing and duplicates nothing**, and rejects any window containing an interval over 0.75 s.

## 3. Mandatory gates — results

| gate | criterion | result | pass |
|---|---|---|---|
| **G1 sensor origin** | near-field road returns at the published sensor height; elevation p1 well below horizontal | road plane **−1.730 to −1.762 m**; elevation p1 **−24.97°** | **6/6 ✓** |
| **G2 axis** | travel direction forward after the 90° yaw | `[1.0000, 0.0003, −0.0026]` | **6/6 ✓** |
| **G3 pose round-trip** | `inv(T) @ T = I` | **3.7e-13** | **6/6 ✓** |
| **G4 windows** | real sweeps, authorized span, no interpolation | **5 sweeps**, span p50 **2.000 s** (min 1.998, max 2.003) | **6/6 ✓** |
| **G6 temporal ordering** | timestamps strictly increasing, no reordering | holds | **6/6 ✓** |
| **G5 target statistics** | the four predeclared bands | prevalence below floor; see §4 | **✗** |

G1 is the gate DDAD failed, and it is the one that matters most: ONCE's elevation p1 of −24.97°
places the ray origin on the **roof**, not on the road. DDAD measured −1.35°, which is what a
ground-level origin looks like. ONCE's origin is published and confirmed; **no mounting height
was invented and no vehicle-origin carving was used.**

## 4. Target statistics against the predeclared bands (brief §8)

Authorized window, 40 anchors per sequence, bands exactly as predeclared.

| sequence | prevalence | valid | conflict | z-levels | anchors with prevalence in band |
|---|---:|---:|---:|---:|---:|
| 000076 | 0.0104 | 0.606 ✓ | 0.016 ✓ | 32 ✓ | 27 % |
| 000080 | **0.0184 ✓** | 0.520 ✓ | 0.022 ✓ | 32 ✓ | 53 % |
| 000092 | 0.0122 | 0.634 ✓ | 0.014 ✓ | 32 ✓ | 31 % |
| 000104 | **0.0152 ✓** | 0.569 ✓ | 0.021 ✓ | 32 ✓ | 50 % |
| 000113 | 0.0128 | 0.573 ✓ | 0.022 ✓ | 32 ✓ | 44 % |
| 000121 | 0.0133 | 0.555 ✓ | 0.019 ✓ | 32 ✓ | 44 % |
| **median** | **0.0130** ✗ | **0.569 ✓** | **0.020 ✓** | **32 ✓** | **44 %** |

**Three of four bands pass on every sequence.** Vertical coverage passes at the maximum (32 of
32). Conflict is far inside its ceiling. Only occupied prevalence falls short, and it does so by
a small margin: 0.0130 against a 0.015 floor, with two of six sequences inside the band.

Note also that the authorized 2.1 s window is **better** on prevalence than the frozen 21-sweep
window would be (0.0130 against 0.0133 median, but 2 of 6 sequences in band against 1 of 6, and
all 6 at 32 z-levels). Shortening the window shrinks the free-carved `valid` denominator faster
than it shrinks the occupied numerator.

## 5. Why the band cannot adjudicate this protocol — the decisive control

The bands were predeclared from KITTI-360's 0.0276 and Waymo's 0.0222, both measured under the
**frozen 21-sweep / 10.46 s** recipe. The authorized policy replaces that recipe. So the
reference values must be re-measured under the authorized policy before the band means anything.

**A consequence of the authorized policy that the brief's own table does not state:** at
KITTI-360's frozen **0.523 s** sweep spacing, a 2.1 s window holds exactly **5 sweeps** — the
same number ONCE gets at 2 FPS. The policy therefore makes the two corpora **sweep-matched as
well as duration-matched**, which is a stronger match than the brief claims for it. (The brief
states KITTI-360 gets "approximately 21 sweeps over 2.1 seconds"; measured, KITTI-360's 21
sweeps span **10.46 s**, so 2.1 s yields 5. This is the same factual correction recorded in
`once_feasibility.md` §12, and it strengthens rather than weakens the authorized policy.)

Measured, 30 anchors per drive, same builder, same bands:

| corpus, authorized 2.1 s window | prevalence | valid | conflict | **z-levels** | in bands? |
|---|---:|---:|---:|---:|---|
| KITTI-360 drive 0003 | 0.0189 ✓ | 0.633 ✓ | 0.059 ✓ | **21** ✗ | **OUT** |
| KITTI-360 drive 0007 | 0.0272 ✓ | 0.598 ✓ | 0.072 ✓ | **22** ✗ | **OUT** |
| KITTI-360 drive 0010 | 0.0170 ✓ | 0.667 ✓ | 0.051 ✓ | **21** ✗ | **OUT** |
| **ONCE, all 6 sequences** | 0.0130 ✗ | 0.569 ✓ | 0.020 ✓ | **32 ✓** | **OUT** |

**Under the authorized protocol, KITTI-360 fails the predeclared bands too — on vertical
coverage, 21 of 32 against a floor of 30.** Each corpus fails a different band, and the corpus
the bands were calibrated from is one of them.

That settles the status of the gate. The band set is a valid decision rule for the frozen
21-sweep recipe and is **inapplicable** to the authorized 2.1 s recipe. The predeclaration says
as much in its own words: it requires the statistics "**the frozen carve** must produce". The
authorized policy is, by the user's own description, "an authorized dataset-specific
supervision-cadence adaptation" — so it is not the frozen carve, and the frozen carve's bands do
not transfer to it. **The bands are preserved unchanged, as instructed; they are simply not a
test this protocol can be put to.**

A secondary observation on the strict reading: the predeclaration says "per anchor". Only
68–72 % of KITTI-360's own anchors sit inside the prevalence band under the authorized window
(and 44 % of ONCE's). Read literally per anchor, no corpus has ever passed, which confirms the
operative reading is central tendency across ≥3 scenes — which is how `once_gates.py` was
written, before any ONCE data was inspected.

## 6. The supervision-density difference, reported explicitly (brief §2.6, §8)

This is the real finding and the brief requires it be stated plainly. All figures are medians
under the authorized window, 40 anchors per sequence.

| quantity | KITTI-360 @2.1 s | ONCE @2.0 s | ratio |
|---|---:|---:|---:|
| contributing sweeps | 5 | 5 | 1.00 |
| temporal coverage | 2.09 s | 2.00 s | 0.96 |
| LiDAR points landing in grid | ~56,000 | ~25,000 | 0.45 |
| **occupied voxels** | **24,050** | **15,572** | **0.65** |
| free voxels | 1,302,602 | 1,225,638 | 0.94 |
| unknown voxels | 759,540 | 840,720 | 1.11 |
| z-levels occupied | 21 | **32** | 1.52 |

**Occupancy by height** (8 bins of 0.8 m, from the grid floor at z = −2 m):

| bin | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| KITTI-360 | 8,094 | 6,377 | 5,012 | 3,612 | 1,040 | 4 | 0 | 0 |
| ONCE | 6,161 | 3,140 | 1,870 | 1,215 | 526 | 484 | 375 | 289 |
| ONCE/K360 | 0.76 | 0.49 | 0.37 | 0.34 | 0.51 | — | — | — |

**Occupancy by range from the anchor:**

| range | 0–10 m | 10–20 m | 20–30 m | 30–40 m | 40–51.2 m |
|---|---:|---:|---:|---:|---:|
| KITTI-360 | 2,472 | 6,492 | 5,989 | 4,738 | 3,677 |
| ONCE | 1,799 | 4,080 | 4,406 | 2,540 | 1,628 |
| ONCE/K360 | 0.73 | 0.63 | 0.74 | 0.54 | 0.44 |

The deficit is **structured, not uniform**. ONCE loses most in the 0.8–3.2 m height band (ratio
0.34–0.49) and at long range (0.44 beyond 40 m), while it is the **only** corpus of the two with
any occupancy above 2 m above the grid floor — its 200 m, roof-mounted, 40-beam sensor reaches
tall structure that KITTI-360's 80 m HDL-64E does not in a 2.1 s window. So ONCE supervision is
not simply "less"; it is differently distributed, worse in the mid-height band that dominates
the metric and better in vertical extent.

## 7. What was excluded as a cause — sparsity is falsified

ONCE's sensor is visibly sparser (40 beams and 66 k points against 64 and 120 k), so sparsity is
the obvious explanation for the occupied deficit. It was tested by decimating KITTI-360's own
sweeps through the same frozen builder, at the frozen 21-sweep window:

| KITTI-360 condition | points kept | prevalence | in band? |
|---|---:|---:|---|
| unmodified, 64 beams | 1.000 | 0.0233 | ✓ |
| 40 beams — ONCE's beam count | 0.625 | 0.0212 | ✓ |
| 24 beams | 0.375 | 0.0188 | ✓ |
| uniform 45 % — matches ONCE's in-grid point count | 0.450 | 0.0225 | ✓ |
| uniform 25 % | 0.250 | 0.0222 | ✓ |
| uniform 15 % | 0.150 | 0.0214 | ✓ |

**Removing 85 % of KITTI-360's returns leaves it in band.** Accumulated targets saturate at
0.2 m voxels; prevalence elasticity to point count is about 0.1. Sparsity therefore does **not**
explain ONCE's statistics.

With the adapter excluded (§3) and sparsity excluded here, what remains is **scene content**:
ONCE's scenes hold less occupied surface inside the grid. The free-space figures confirm it —
ONCE resolves a comparable fraction of the grid as known while less of it is occupied. That is
the signature of more open road: wide carriageways, fewer close buildings, less roadside
vegetation than KITTI-360's German suburbia. It is a property of the dataset, which is the
variable the experiment exists to vary, and no protocol change reaches it.

## 8. Geometric alignment and ray plausibility (brief §8, closing paragraph)

The brief is explicit that the bands are *necessary diagnostics, not sufficient proof*, and that
actual alignment and physically plausible carving must be checked. They were:

* **Ray origin on the sensor, confirmed from data** — road plane at −1.73 to −1.76 m with
  elevation p1 at −24.97°, so rays descend onto the road rather than grazing along it (G1).
* **Projection consistency with imagery** — LiDAR depth edges align with image gradients in 6 of
  6 sequences once distortion is applied, which is also what resolved the distortion ambiguity.
* **Carve rule unmodified** — endpoints occupied, ray interiors free from `CARVE_NEAR_M` to
  `BAND_HALF_M`, nothing beyond an endpoint, same-sweep returns never carved, cross-sweep
  conflict → unknown, sky rule unchanged (`SKY_MARGIN_VOX=1`, `SKY_NEIGHBOURHOOD_VOX=7`).
* **Conflict fraction 0.014–0.022**, far under the 0.12 ceiling — a misregistered pose chain
  raises this sharply, so a low value is positive evidence that the poses are right.
* **Full vertical extent used** — 32 of 32 z-levels, so the grid's z range suits ONCE's sensor
  frame.

Nothing in the geometry is wrong. The supervision is valid; there is less of it in one region.

## 9. What Phase C needs, if it is authorized

Built and verified already: the reader, the window builder, the gate harness, the frozen carve
(runs unmodified), and the data on disk. **Not built:** the ONCE arm of the training chain,
which is the port of `tools/waymo_arm/cli/` — `config.py` (paths and grid), a clip builder for
ONCE cam01 (the analogue of `waymo_pilot/cli/build_waymo_clips.py`, running frozen MoGe-2 and
frozen LingBotMap over 5 causal frames), `build_inputs.py` (frozen
`build_local_geometry`, npz contract unchanged so `HInputDataset` reads it), and
`build_targets.py` (frozen `rawtarget`, single sensor — ONCE needs no multisensor path).

Cost estimate: the clip build is the long pole, a few hours of inference over the eligible
anchors; targets are comparable to the Waymo arm's; training is 3 seeds × ~9 min at the frozen
recipe; evaluation reuses the frozen scripts unchanged. Output would go to
`artifacts/method_vs_data/once_arm/`, which **has deliberately not been created**, so nothing
can be mistaken for a result that does not exist.

Scope the arm would have, stated in advance: *our method trained on an OccAny source dataset
whose supervision carries ~34 % fewer occupied voxels, concentrated at mid height and long
range, and whose camera is 90.1° rather than 50°.* Not a clean dataset-identity test. The
relevant prior is that the Waymo arm, fully **in band** at 0.0222, still lost 3.0 and 9.1 SC IoU.

## 10. Not claimed

* No ONCE performance number, estimate or projection.
* No claim that any band was met that was not.
* No claim that ONCE would succeed or fail if trained.
* No change to any predeclared band, threshold, carve constant or grid.
* No use of target-domain labels.
