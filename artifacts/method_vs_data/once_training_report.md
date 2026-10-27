# ONCE-only CompletionUNet — training report

**2026-10-27.** The ONCE arm was implemented, preprocessed, validated and trained. Three
checkpoints exist with logs. Code: `tools/method_vs_data/once_arm/`. Caches:
`/media/minh/MINH1/lingbot_once_arm/`. Manifests, logs and checkpoints:
`artifacts/method_vs_data/once_arm/`.

> ## Status: EXPLORATORY
>
> The predeclared occupied-prevalence gate is **FAILED, not passed**: ONCE's median is
> **0.0131** against a band of **[0.015, 0.040]**. It is recorded as failed in
> `config.PREVELANCE_BAND_STATUS`, written into every manifest and log this arm produced, and
> printed at the head of the training run. Nothing in this report may be read as the arm having
> passed the original preregistration.
>
> Why it was nevertheless run: under the authorized ~2.1 s duration-matched window the band
> also rejects **KITTI-360**, the corpus it was calibrated from, on vertical coverage (21 of 32
> z-levels against a floor of 30). Full argument and measurements in
> `once_supervision_validation.md` §5. No band was altered.

---

## 1. Two bugs the validation found, both of which would have invalidated the result

Reported first because both change conclusions recorded earlier in this directory.

### 1.1 The camera-to-grid extrinsic was wrong (fixed)

`once_reader.cam_to_velo` returned `A·T·A⁻¹`. Conjugation is correct for `velo_to_velo` and for
the poses, where **both** endpoints are ONCE frames being re-expressed; it is wrong here,
because this transform's source is ONCE's **camera** and only its target is being rotated. The
correct map is `A·T`. The conjugation spun the camera frame by an extra −90° of yaw.

Effect, and why no gate caught it: map-vs-target IoU was **0.22 %** against KITTI-360's ~3 %.
G1 and G2 use LiDAR only, and the distortion gate compared two projections of the *same*
points, so an extrinsic error cancelled exactly. The arm's own input/target alignment check is
what exposed it. Fixed and documented at the call site.

### 1.2 The distortion conclusion is REVERSED (retracted)

`once_metadata_audit.md` §5 concluded that ONCE's released images are **raw** and must be
undistorted once, on the strength of a gradient-alignment test. **That is wrong. The images are
already rectified and ONCE's documentation is correct.** Two direct measurements, both with the
extrinsic fixed:

| test | plain pinhole | distortion applied |
|---|---:|---:|
| log-corr(LiDAR depth, monocular depth), 12 frames | **0.857** — better in **12/12** | 0.698 |
| map-vs-target IoU, clips built both ways, 12 anchors | **3.26 %** — better in **11/12** | 1.44 % |

The gradient test is **centre-biased**: a barrel projection pulls points toward the image
centre, where a street scene carries more gradient, so it rewards the distorted model whatever
the true alignment. With the broken extrinsic it also gave 6/6 in the wrong direction.

The arm therefore feeds the **shipped JPEGs** straight through and never calls `cv2.undistort`.
Recorded in `config.DISTORTION_EVIDENCE`. Applying it costs more than half the map quality.

## 2. What was reused unchanged, and what is new

**Unchanged, imported not copied:** the incremental mapper and network input
(`baseline_reconstruction.current.geometry.build_local_geometry`); the occupancy rule
(`core/supervision/rawtarget.py` — ONCE has one roof LiDAR, so the single-sensor function
applies and the multisensor path Waymo needed is unused); the grid
(`kitti360_transforms.GRID`, asserted at import); the recipe yaml
(`artifacts/baseline_reconstruction/training_config.yaml`, **sha256 `4ef021b6…` identical
across all three seeds**); `CompletionUNet`; `core.run._gate8a_train.step_loss`;
`HInputDataset`; the training loop via `tools.waymo_arm.train`, itself gated against the frozen
KITTI-360 trainer; and the zero-shot evaluation harness.

**New:** `once_arm/{config,geometry,anchors,train}.py` and
`once_arm/cli/{build_clips,build_targets,build_inputs,validate,train,evaluate,predict_canonical,
score_vs_occany,compare}.py`.

The frozen recipe file still names the three KITTI-360 drives in its `train_drives` field. That
field is **not** read by this arm — the scene list is passed as an argument, exactly as the
Waymo arm does — and the file is untouched so its hash still verifies.

## 3. Data: manifest, clips, targets, inputs

Manifest digest **`aab448ccf7cee085e6bace9665d884c26b3d13dc69be72ea11458e6d47dce6f3`**,
deterministic (even per-sequence stride, no RNG).

| | value |
|---|---|
| independent source sequences | **6** |
| eligible anchors | 1,771 |
| selected anchors | **1,580** (1,259 train / 321 val) |
| train sequences | 000076, 000080, 000092, 000104, 000113 |
| **val sequence (disjoint)** | **000121** |
| input clip | 5 frames `t-4..t`, stride 1 native, span **2.000 s**, cam01 only |
| future camera frames used | **0 of 1,580** |
| supervision window | **5 real native sweeps, span 2.000 s**, every anchor |
| interpolated / duplicated sweeps | **none** |

Per-anchor input and supervision frame ids, timestamps, spans and sweep counts are all recorded
in `once_arm/manifests/anchors.json`, as brief §5.E requires.

| stage | built | failed | time |
|---|---:|---:|---:|
| causal clips | **1,580** | 0 | 14.5 min |
| occupancy targets | **1,568** | 0 | 2.8 min |
| hard-map inputs | **1,580** | 0 | 1.2 min |

**Target statistics, preserved and not corrected toward KITTI-360:**

| | min | p50 | max |
|---|---:|---:|---:|
| occupied prevalence in valid | 0.0035 | **0.0131** | 0.0485 |
| valid fraction | 0.2029 | 0.5698 | 0.9206 |
| conflict of touched | 0.0017 | 0.0167 | 0.0647 |
| occupied voxels | 4,871 | 13,331 | 44,294 |

## 4. Pre-training validation (brief §6)

| # | check | result |
|---|---|---|
| 1, 6 | camera/LiDAR alignment, target agreement | ONCE map-vs-target IoU **4.26 %** vs KITTI-360's **3.23 %** through one function (400 vs 320 anchors, 8 drives) — **1.32×, in family** |
| 2 | five-frame causal ordering | **0** of 1,580 misordered, **0** using a future camera frame, span 2.000 s |
| 3 | poses, metric scale | **0** non-finite scales; p50 45.4 m/unit; MoGe candidates present on every clip |
| 4 | representative samples | recorded in `logs/validate.json` |
| 5 | tensor contract | `[S, 0, F]` float, 256×256×32, keys `S_H`/`F_H` and `valid_packed`/`occ_packed` |
| 7 | train/val separation | **sequence-disjoint**, no overlap |
| 8, 9 | smoke test | finite losses (1.126 / focal 0.176 / dice 0.950), IoU rising, **967,322 params** |
| 10 | frozen KITTI-360 pipeline | `require_source_freeze()` **PASS** over all frozen sources |

Also verified: **0** files touched in the KITTI-360 input cache, the KITTI-360 target cache or
the Waymo arm, and the same 9 modified tracked files as at task start.

### A measured dataset-adapter difference, stated not tuned

The frozen MoGe-2 metric-scale procedure **over-estimates depth on ONCE by ~1.35×**
(p50 of LingBot×scale ÷ LiDAR depth, over in-image LiDAR returns). The same diagnostic on
**KITTI-360's own clips gives ~1.20×**. So the bias is a property of the frozen procedure on
both corpora and is ~15 % worse on ONCE. Per brief §7 nothing was adjusted to compensate.

## 5. Training

Frozen recipe, nothing changed:

| parameter | value |
|---|---|
| model | CompletionUNet, 3 input channels, width 24 |
| parameters | **967,322** |
| loss | focal + Dice (`w_focal` 1.0, `w_dice` 1.0, γ 2.0, `pos_weight_cap` 8.0, `w_unknown` 2.0) |
| optimiser / schedule | AdamW, lr 1e-3, wd 0.01, grad clip 1.0, cosine |
| steps | **6,000** |
| batch size | **4** |
| crop | **128×128×32** |
| seeds | **0, 1, 2** |
| checkpoint rule | final step, **no selection** |
| recipe sha256 | `4ef021b6…` — identical for all three seeds |

| seed | minutes | final train IoU | final val IoU | best val IoU | checkpoint sha256 |
|---|---:|---:|---:|---:|---|
| 0 | 7.7 | 0.1539 | 0.1474 | 0.1617 | `03fc0d26fc83…` |
| 1 | 7.7 | 0.1134 | 0.1563 | 0.1627 | `f469f80d0cb6…` |
| 2 | 7.7 | 0.2405 | 0.1605 | 0.1666 | `2057b05e2d23…` |

All three checkpoints are distinct, and each seed has a distinct `init_state_digest`,
`draw_digest` and `target_digest`, so the seeds really differ in initialization and in sampling.

Source-domain IoU is reported here and **was not used for anything**: the checkpoint rule is
the final step, no target-domain label was read during training, and no SemanticKITTI or Occ3D
number influenced selection or hyperparameters.

**Checkpoint locations.** The shared trainer writes to the canonical
`paths.checkpoint_path` only when `tag_prefix == "waymo_geo"`; any other tag goes to a
subdirectory. That guard exists so the Waymo arm's trainer-equivalence runs cannot clobber its
real checkpoints, and it was deliberately left alone — editing that file would invalidate the
gate proving this loop reproduces the frozen KITTI-360 trainer. Per-step checkpoints therefore
live in `checkpoints/by_step/once_geo_seed{s}_step{N}.pt`, and `train.publish_final` copies
each seed's final checkpoint to the canonical `checkpoints/geo_seed{s}_step06000.pt`, verifying
the copy by sha256. That is what the frozen evaluator reads.

## 6. Results: frozen zero-shot evaluation

Frozen harness unchanged; only the `checkpoints` field of `CurrentPaths` was replaced. 163
SemanticKITTI and 1,182 Occ3D anchors, raw convention, three seeds.

**The control that proves only the checkpoint changed:** `map_only` SC IoU is
**8.8205** on SemanticKITTI and **10.0351** on Occ3D — bit-identical to the KITTI-360 and Waymo
arms, as it must be since the hard map does not depend on the completion checkpoint.

### SemanticKITTI sequence 08 — 163 anchors

| arm | full | observed | completed | precision | recall |
|---|---:|---:|---:|---:|---:|
| KITTI-360 | **26.17** | 33.23 | 24.74 | — | — |
| Waymo | 23.14 | 28.26 | 21.46 | — | — |
| **ONCE** | **18.50** ± 0.68 | **18.77** | **18.39** | 31.43 | 31.05 |

Per-seed: 18.37 / 19.23 / 17.89. **ONCE − KITTI-360 = −7.67; ONCE − Waymo = −4.64.**

### Occ3D-nuScenes validation — 1,182 anchors

| arm | full | observed | completed | precision | recall |
|---|---:|---:|---:|---:|---:|
| KITTI-360 | **31.22** | 39.03 | 27.50 | — | — |
| Waymo | 22.86 | 35.05 | 15.86 | — | — |
| **ONCE** | **19.29** ± 0.47 | **25.34** | **15.10** | 53.15 | 23.26 |

Per-seed: 18.78 / 19.70 / 19.38. **ONCE − KITTI-360 = −11.93; ONCE − Waymo = −3.57.**

### The behavioural signature

On SemanticKITTI the ONCE arm's loss is **concentrated in the observed region** (−14.46 against
KITTI-360) rather than in the completed region (−6.35), and its observed-region precision is
**21.82** against recall **57.41** — it over-predicts occupancy heavily wherever the hard map
resolved something. That is the behaviour a rarer positive class would produce: with prevalence
0.0131 against KITTI-360's 0.0233, the Dice term pushes recall up and precision down. It is a
**description confirmed on two benchmarks**, not a demonstration of cause; the intervention that
would establish cause is §7.

Note the contrast with the Waymo arm, which lost mainly in the **completed** region on Occ3D
(−11.64). The two failing arms fail differently, so "training on another corpus hurts" is not
one mechanism.

## 7. What this report does not establish

* **No causal attribution of the deficit to supervision density.** The prevalence difference,
  the scale bias, the 90.1°-vs-103.9° camera and ONCE's scene content all differ at once.
* **No comparison with OccAny yet.** That requires the canonical protocol (801 / 4,219 anchors,
  OccAny's labels and scorer) and is reported in `once_vs_occany.md`. The frozen-evaluator
  numbers above may **not** be placed in a column with OccAny's.
* **Six independent scenes.** 1,580 anchors is the training volume, not the independent count.
* **The prevalence gate remains failed.** This arm is exploratory.
