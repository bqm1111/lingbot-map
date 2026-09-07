# Gate 8D — Dense Future-Visibility and Semantic Surface Completion

*Executed end to end 2026-09-05/06. Every number was measured on this machine; the script
that produced each one is named. Protocol frozen before any target was built; the final
target evaluation was run once, after the manifest was frozen, and nothing was tuned after
seeing it.*

---

## 1. Executive verdict

**FAIL.** One source-selected checkpoint was trained on KITTI-360 alone and evaluated once
on both targets. It **meets the Occ3D criterion by a wide margin and misses the
SemanticKITTI criterion by more than half.**

| success criterion | required | achieved | verdict |
|---|---|---|---|
| SemanticKITTI SC IoU ≥ reproduced OccAny | 25.28 | **10.96** | **FAIL** |
| Occ3D matched-forward raw SC IoU | ≥ 30.0 | **38.60** | **PASS** |
| no gain by volume inflation (pred/GT ≤ 1.6) | ≤ 1.60 | SemKITTI **3.58** / Occ3D 0.70 | **FAIL** on SemanticKITTI |
| SSC mIoU improved on both targets | — | **not computed** | **not established** |
| newly-completed naming accuracy improved | — | source-side only | **not established** |
| causal streaming, fixed scale preserved | yes | yes | PASS |

Gate 8D is **not** an improvement over the Gate 8C-1 plain-column baseline overall. It is a
large improvement on Occ3D (33.28 → 41.33 causal, 30.70 → 38.60 matched-forward) and a large
regression on SemanticKITTI (22.61 → 10.96).

**The SemanticKITTI number is mostly an operating-point failure, not a model failure**
(§9.1): at the best possible threshold the same checkpoint scores 20.34, so roughly 9.4 of
the 11.6-point gap is threshold transfer and 2.3 is genuine regression. That does not change
the verdict -- 20.34 is an oracle number and still below the baseline -- but it changes the
diagnosis and what should be fixed next (§9.2).

Per the brief, the failure is reported and the run stops. No target threshold, class, height
band or visualisation was used to tune anything.

## 2. The exact scientific claim

> A single 0.99 M-parameter completion module, trained and selected on KITTI-360 drives
> 0003/0007/0010 with source validation on 0006, using no target-domain optimisation,
> fine-tuning, threshold calibration or checkpoint selection, evaluated once on
> SemanticKITTI 08 and Occ3D-nuScenes val.

**This is not a claim of untouched benchmarks.** Both targets were inspected during earlier
development in this project. The defensible claim is the one above: *target-domain-free
training and selection*, not *target-domain-naive experimentation*.

The frozen foundation models (LingBot-Map, MoGe-2, Trident-H) have not had their training
provenance audited, and LingBot-Map's published mixture includes KITTI-360, so nothing here
is whole-system zero-shot.

## 3. Honest protocol description

`artifacts/gate8d/protocol.json`, source hash `3191b73e7a7a1483`. Training refuses to start
if `gate8d/protocol.py` moves once a checkpoint exists (`assert_unchanged`).

**Three amendments were made, all before any weight was trained, all from KITTI-360 evidence
alone, each re-hashed:**

1. **`IMAGE_EVIDENCE_NATIVE_STRIDE = 5`** — the protocol pinned LiDAR sweep density but not
   image density. Added, not changed.
2. **`SKY_PIXEL_STRIDE` 8 → 4** — at stride 8, ceiling supervision was 1.3 %, no better than
   the broken baseline; at stride 4 it is 31 % on the probe anchor. Sampling density only;
   `SKY_MIN_PROB` and `SKY_MIN_VIEWS` untouched.
3. **`MIN_OCC_FRAMES = 1`** — see §4. Added after the Phase 3 audit **failed**.

Amendment 3 deserves scrutiny because it resolves a genuine contradiction inside the brief:
*"genuine cross-sweep occupied/free conflicts are unknown"* versus a `PRECEDENCE` list that
ranks *LiDAR endpoint* (1) above *LiDAR interior* (2). It was resolved toward precedence.
"Conflict → unknown" therefore remains live but **cross-source**: a pseudo-teacher may never
overwrite a LiDAR endpoint, and every attempt is counted
(`n_pseudo_refused_by_lidar_endpoint`).

Firewall: `sscbench_kitti360.audit.FileAudit` wrapped margin fitting, target building,
sample building, training and selection. **Violations: 0 at every stage.** The evaluator
refuses to run without the frozen manifest and exposes no `--checkpoint` or `--threshold`
(asserted by `tests/gate8d`).

## 4. Target-construction audit

`artifacts/gate8d/target_audit.json`, 240 anchors across all four drives.

**Native-sweep gain 4.83×** — 101 LiDAR sweeps per anchor against Gate 8C-1's 21, over the
*identical* horizon. `tests/gate8d::test_native_window_never_exceeds_the_stream_horizon`
asserts that dense sampling fills the window and never extends it.

### The acceptance gate failed on the first build, and the fix was structural

| gate (pre-registered) | limit | first build | after fix |
|---|---|---|---|
| pseudo-free collision rate | ≤ 2.0 % | **0.147 %** PASS | **0.147 %** PASS |
| occupied recall vs 8C-1 plain-column | ≥ 98 % | **50.9 %** FAIL | **100.00 %** PASS |

Collision = a voxel MoGe or the sky teacher called free that a *held-out later* LiDAR sweep
reports as an endpoint. At 0.147 % it is **lower than LiDAR's own free-carving** (0.210 %):
the pseudo-teachers are more conservative than the sensor.

The recall failure was diagnosed by cross-tabulation, and the metric was misleading:

| of 8C-1's occupied voxels, Gate 8D called | share |
|---|---|
| occupied | 50.1 % |
| **free** | **0.0 %** |
| unknown | 49.9 % |

Nothing was contradicted — surfaces were *disowned*. The cause is that "any occupied/free
disagreement → unknown" does not scale with sampling density: at 5× the sweeps a voxel is
5× more likely to be grazed by some ray from some viewpoint. Measured across corroboration
thresholds on KITTI-360:

| MIN_OCC_FRAMES | recall vs 8C-1 | occupied voxels | collision |
|---|---|---|---|
| **1** | **100.00 %** | **3 066 364** | 0.019 % |
| 2 | 95.84 % | 2 565 216 | 0.019 % |
| 3 | 91.20 % | 2 300 811 | 0.019 % |

Only 1 clears the pre-registered floor. It raises occupied supervision from 1.72 M to
3.07 M voxels — a strict superset of the baseline's surfaces.

### Supervision by height, versus the baselines (% of voxels supervised)

| z (m) | **Gate 8D** | 8C-1 raw LiDAR | 8C-1 plain column | rejected neighbourhood |
|---|---|---|---|---|
| +4.3 | 7.2 | 0.5 | **19.4** | 63.0 |
| +3.1 | 14.3 | 4.2 | **21.4** | 57.2 |
| +1.9 | 27.2 | 21.4 | **33.0** | 53.7 |
| +0.1 | **56.9** | 52.3 | 53.8 | 54.2 |
| −1.1 | **46.0** | 36.1 | 36.7 | 36.6 |
| −1.7 | **20.4** | 14.0 | 14.0 | 14.0 |

State fractions: occupied 6.09 %, free 25.87 % (17.84 % LiDAR + 8.03 % pseudo-teachers),
unknown 68.04 %.

**This table predicted the failure and was reported before training.** Gate 8D dominates
below +2 m — where the road-thickening error lives — and is *below the plain-column
baseline above +3 m*, because view-based sky rays only reach ceiling the camera actually
saw through, while a column rule fills every column unconditionally. The brief forbids
re-adding a height heuristic, so this was accepted as the experiment's risk.

### MoGe safety margin

Fitted as the `MOGE_MARGIN_QUANTILE = 0.95` quantile of |MoGe − LiDAR| per depth bin on the
**train drives**, validated on drive 0006 (`artifacts/gate8d/moge_margin.json`):

| depth bin (m) | 0–5 | 5–10 | 10–15 | 15–20 | 20–30 | 30–40 |
|---|---|---|---|---|---|---|
| margin (m) | 0.96 | 1.96 | 4.52 | 8.44 | 14.14 | 22.50 |
| val-drive coverage | 95.5 % | 95.5 % | 94.9 % | 94.3 % | 95.3 % | 95.1 % |

Coverage lands on the 95 % target on a held-out drive, so the margin generalises. It is
large at range, so MoGe carving saturates near 16–18 m: it supplies dense **near-field**
free space and nothing far away. That is a property of the frozen model, not a choice.

### Sky teacher

KITTI-360's vocabulary has no sky class, so Trident-H was run with the project's fixed
phrase list extended by the protocol's single `SKY_PROMPT = "sky"`. 8.77 % of pixels exceed
`SKY_MIN_PROB = 0.80`, concentrated in the top image bands (mean 0.449 in the top eighth,
0.001 in the bottom half) — a clean, confident segmentation. 3 589 frames cached per teacher.

## 5. Architecture and losses

`gate8d/net.py::SurfaceCompletionUNet` — the Gate 8C-1 network with **one** added output.

| | |
|---|---|
| trained parameters | **986 139** (Gate 8C-1: 986 114, **+25**) |
| added head | `Conv3d(24, 1, 1)` → truncated unsigned distance, **metres** |
| truncation | 1.0 m (`TUDF_TRUNCATION_M`) |
| loss weight | 0.25 (`w_surface`) |
| padding | `replicate` (inherited Gate 8C-1 fix) |
| occupancy path | unchanged — same residual, same lock, same 32 channels |

Distances are in metres, not voxel indices, because the stress protocol varies voxel size
from 0.15 m to 0.45 m; a voxel-indexed distance would mean a different physical quantity in
each variant. `tests/gate8d::test_distance_target_is_in_metres_not_voxels` asserts the EDT
is scaled by voxel size. The distance loss is masked to supervised voxels only
(`surface_distance_loss`), and occupancy remains the sole inference output.

Semantics: the existing open-vocabulary path was **improved, not replaced**. Observed
semantics can never be overwritten — `torch.where((semw > 0), map_probs, net_probs)`, with
a test asserting exactly that line. Supervision uses future Trident-H masked by the existing
two-half temporal-agreement rule; no human labels anywhere.

## 6. KITTI-360 source results

Training: 3 seeds, 6 000 steps, crop 128×128×32, batch 4, AdamW 1e-3 — one config per seed
differing only in `seed` (asserted by test). 15.3 min per seed, **0 firewall violations**.

Selection (`tools/gate8d/selection.py`, drive 0006, 100 samples, 8 stress variants):

| candidate | score | mean stress IoU | worst | τ | pred/GT | thickness |
|---|---|---|---|---|---|---|
| **seed0_best** | **0.0873** | 21.61 | 18.78 | **−0.4375** | 1.22 | 0.984 m |
| seed1_last | 0.0237 | 22.89 | 19.86 | −0.7500 | 1.69 | 0.969 m |
| seed0_last | 0.0236 | 22.95 | 20.07 | −0.6875 | 1.68 | 0.969 m |
| seed2_best | 0.0416 | 22.81 | 19.66 | −0.6875 | 1.56 | 0.977 m |
| seed2_last | −0.0207 | 23.24 | 20.18 | −0.8750 | 1.98 | 0.969 m |
| seed1_best | −0.1753 | 22.51 | 19.62 | −0.6875 | 2.99 | 0.977 m |

Source AUROC was deliberately **excluded** from selection: Gate 8C-1's neighbourhood variant
raised it 0.68 → 0.74 while Occ3D IoU fell 33.28 → 28.12.

## 7. Frozen checkpoint

`artifacts/gate8d/frozen_manifest.json` — selected `seed0_best`,
`artifacts/gate8d/checkpoints/g8d_seed0_best.pt`, sha256 `2e32cc02aba705ab…`,
occupancy threshold **−0.4375**, semantic threshold 0.0. The target firewall lifted only
after this file existed.

## 8. Target results (one evaluation, no adaptation)

`tools/gate8d/eval_target.py`, threshold −0.4375 from the manifest, no overrides possible.

### SemanticKITTI 08 — past-5 causal (163 anchors)

| method | SC IoU | precision | recall | pred/GT | pred > 2 m |
|---|---|---|---|---|---|
| **Gate 8D completion** | **10.96** | 12.64 | 45.27 | **3.58** | **35.07 %** |
| Gate 8D + pooling | 10.72 | 11.68 | 56.40 | 4.83 | 45.67 % |
| mapper only | 8.90 | 33.32 | 10.83 | 0.33 | 1.20 % |
| mapper + 0.4 m dilation | 15.41 | 23.58 | 30.80 | 1.31 | 4.79 % |
| Gate 8C-1 plain column *(prior)* | **22.61** | 32.67 | 42.33 | 1.30 | — |
| OccAny reproduced, raw *(163-frame matched subset, no pooling)* | **25.28** | 45.49 | 36.26 | 0.80 | — |
| OccAny reproduced, pooled *(same subset, majority pooling)* | 25.92 | 36.68 | 46.93 | 1.28 | — |
| OccAny **published** *(sequence 08)* | 25.91 | 36.79 | 46.70 | — | — |

### Occ3D-nuScenes val (400 anchors)

| method | protocol | SC IoU | precision | recall | pred/GT | pred > 2 m |
|---|---|---|---|---|---|---|
| **Gate 8D completion** | past-5 causal | **41.33** | 68.85 | 50.83 | 0.74 | 23.06 % |
| Gate 8D + pooling | past-5 causal | 45.22 | 55.01 | 71.77 | 1.30 | 37.30 % |
| **Gate 8D completion** | matched forward *(non-causal)* | **38.60** | 67.62 | 47.36 | 0.70 | 22.66 % |
| mapper only | past-5 | 11.52 | 65.06 | 12.28 | 0.19 | 2.41 % |
| mapper + dilation | past-5 | 24.25 | 57.03 | 29.67 | 0.52 | 6.41 % |
| Gate 8C-1 plain column *(prior)* | past-5 | 33.28 | 72.25 | 38.16 | 0.53 | — |
| OccAny reproduced, raw *(882-sample subset)* | matched forward | 20.64 | 42.55 | 28.61 | 0.67 | — |
| OccAny **published** | — | 23.55 | 36.09 | 40.39 | — | — |

Published and reproduced OccAny numbers are kept in separate rows and labelled with subset,
sampling and pooling, as required.

**Harness cross-check:** the mapper baselines reproduce prior gates exactly on SemanticKITTI
(8.90 / 15.41 against Gate 8C-1's reported 8.90 / 15.41), so the evaluation harness agrees
with earlier gates and the Gate 8D deltas are attributable to the model.

## 9. Failure analysis

The failure is localised and unambiguous. Predicted occupancy against ground truth, by
height, SemanticKITTI past-5:

| z (m) | ground truth | Gate 8D |
|---|---|---|
| **+4.3** | **0.00 %** | **98.11 %** |
| +3.5 | 0.06 % | 19.00 % |
| +2.7 | 0.54 % | 1.59 % |
| +1.9 | 2.63 % | 1.80 % |
| +1.1 | 8.39 % | 1.91 % |
| +0.3 | 12.35 % | 2.46 % |
| −0.5 | 17.97 % | 4.26 % |
| **−1.3** | **38.55 %** | **91.03 %** |

Two failures, both at the grid boundary, with a hollow middle:

1. **The ceiling slab returned, worse than ever** — 98.11 % predicted where ground truth is
   0.00 %. Gate 8D's sky rays supervise only 7.2 % of that layer against the plain-column
   rule's 19.4 %, and the rest is unconstrained. This is exactly the risk flagged before
   training, and it materialised.
2. **The floor floods** — 91.03 % against 38.55 %.
3. **The middle is under-predicted 4–6×**, so the model is not "predicting more everywhere";
   it is predicting almost nothing where the geometry actually is and flooding both
   boundaries.

The selected threshold (−0.4375, against Gate 8C-1's −0.125) compounds this: a more
negative threshold turns "no opinion" into "occupied" more readily, and an unsupervised
region is exactly where the network has no opinion. The threshold was chosen on KITTI-360
by the pre-registered rule; it transferred badly. **It was not re-tuned.**

### 9.1 How much of the 10.96 is the model, and how much is the threshold?

The headline is worse than the model. A threshold sweep on SemanticKITTI -- **diagnostic
only, nothing was retuned, the frozen threshold stands** -- decomposes it (41 anchors):

| tau | SC IoU | precision | recall | pred/GT | IoU below +2 m |
|---|---|---|---|---|---|
| **-0.4375 (frozen)** | **10.04** | 11.31 | 47.16 | 4.17 | 26.28 |
| -0.250 | 12.74 | 15.87 | 39.27 | 2.47 | 24.82 |
| -0.125 *(Gate 8C-1's tau)* | 17.04 | 26.49 | 32.31 | 1.22 | 22.10 |
| **0.000 (best possible)** | **20.34** | 36.62 | 31.40 | 0.86 | 21.72 |

The gap to Gate 8C-1's 22.61 splits roughly **9.4 points of threshold transfer and 2.3
points of genuine model regression**. The ranking is largely intact; the operating point is
not. Quoting 20.34 would be an oracle number chosen by looking at the target, so it is
reported only to size the failure -- **the result remains 10.96**, and Gate 8D is worse than
the baseline even at its best possible threshold.

### 9.2 Why the source selection could not catch this

The source selection score is computed **only on supervised voxels** (`gt_valid`), and Gate
8D's target leaves 68 % of the volume UNKNOWN -- including the ceiling. The source metric is
therefore *structurally blind to the region where a negative threshold does its damage*: it
never scores the sky, so it never penalises flooding it.

Worse, the denser free-space supervision shifted the score distribution and moved the
source-optimal threshold from -0.125 to -0.4375 -- further into the regime that floods
unsupervised space. Gate 8C-1's threshold transferred by luck of calibration, not by design.

**No source-only selection rule can catch this failure mode, because the failure lives where
the source target has no opinion.** The preregistered score deliberately excluded source
AUROC to avoid the Gate 8C-1 neighbourhood trap; it fell into a different one of the same
family. Any future gate should either (a) score the *unsupervised* volume with an explicit
prior-plausibility term, or (b) fix the operating point by a calibration that does not
depend on the supervised subset -- for example matching predicted density to a source-side
prevalence estimate rather than maximising IoU on supervised voxels.

Occ3D tolerates the same behaviour because its ground truth at the top layer is genuinely
39.32 % occupied, so flooding there costs far less — which is why one model can improve
Occ3D by 8 points while halving SemanticKITTI.

**The central hypothesis is not supported as stated.** Denser privileged evidence did
substantially improve supervision below +2 m and did improve Occ3D. It did not teach the
model to distinguish occupied, free and unknown *at the grid boundary*, because view-based
evidence cannot reach the region a camera never observed. Supervision density was necessary
but not sufficient; the unsupervised-boundary failure mode identified in
`reports/ROOT_CAUSE.md` survives.

## 10. Cost

| | |
|---|---|
| trained parameters | 986 139 |
| total inference parameters | 1 713 984 844 (frozen stack 1 712 998 705) |
| completion forward, full 256×256×32 grid | **45.0 ms** median |
| peak GPU during that forward | 4.87 GiB |
| training | 15.3 min per seed, 3 seeds in parallel |

Per-frame system cost is dominated by the frozen Trident-H teacher (~1 904 ms/frame,
measured in `artifacts/gate8c1/cost_benchmark_semantickitti.json`), unchanged by Gate 8D.

## 11. Tests

| suite | result |
|---|---|
| gate8 | 25 passed |
| gate8a | 27 passed |
| gate8b | 17 passed |
| gate8c0 | 21 passed |
| **gate8c1** | **2 failed**, 28 passed |
| gate8d | 22 passed |

**The two Gate 8C-1 failures are real and are mine.** They were introduced earlier in this
session by the ceiling/padding work, *before* Gate 8D began:

* `test_gate8c0_artifacts_remain_unchanged` — `gate8/net.py` changed when
  `padding_mode`/`pad_z` were added. Defaults keep behaviour bit-identical, but the file
  hash guarantee is broken and the test is right to say so.
* `test_rebuilt_target_agrees_with_its_own_sweep_by_construction` — the open-sky rule added
  to `gate8c1/rawtarget.py` can convert an endpoint voxel that was already in
  occupied/free conflict (hence UNKNOWN) to FREE when it sits above the column top. That is
  a genuine, if narrow, inconsistency in the Gate 8C-1 target builder that I introduced.

Neither affects Gate 8D, which has its own builder and its own network, but both should be
fixed before Gate 8C-1 is cited again.

## 12. Reproduction

```bash
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST="12.0a" HF_HUB_OFFLINE=1
PY=/home/minh/anaconda3/envs/cu128/bin/python

$PY tools/gate8d/freeze_protocol.py
$PY tools/gate8d/fit_moge_margin.py --frames 200 --device cuda:1
$PY tools/gate8d/cache_future_teachers.py --teacher moge --drive <drive> --device cuda:1
#   sky teacher runs in the Trident env:
#   PYTHONPATH=$REPO:$TRIDENT /home/minh/workspace/third_party/trident_env/bin/python \
#       tools/gate8d/cache_future_teachers.py --teacher sky --drive <drive> --device cuda:0
$PY tools/gate8d/build_targets.py --drive <drive> --device cuda:1
$PY tools/gate8d/audit_targets.py --n 60 --device cuda:1        # exits 2 if not accepted
$PY tools/gate8d/build_samples.py --drive <drive> --device cuda:1
$PY tools/gate8d/train.py --config configs/gate8d/seed0.yaml --device cuda:0
$PY tools/gate8d/selection.py --limit 100 --device cuda:1
$PY tools/gate8d/freeze_manifest.py
$PY tools/gate8d/eval_target.py --dataset semantickitti --protocol past5 --device cuda:1
$PY -m pytest tests/gate8d -q
```

Environment: conda `cu128`, torch 2.7.1+cu128, RTX PRO 6000 Blackwell (sm_120).
Trident-H requires `/home/minh/workspace/third_party/trident_env` (mmseg 1.2.2).
Code commit recorded in `artifacts/gate8d/protocol.json` and `frozen_manifest.json`.

**Artifacts:** `artifacts/gate8d/{protocol,moge_margin,target_audit,selection,frozen_manifest}.json`,
`eval_{semantickitti,occ3d}_{past5,occany_fwd}.json`, `checkpoints/g8d_seed{0,1,2}_{best,last}.pt`.
**Data:** `/media/SSD1/MINH_DATASETS/lingbot_gate8d/{targets,samples,moge_future,sky_future}`.
Gate 8C-1 data and checkpoints untouched.

## 13. Limitations and deviations

1. **Deviation — three protocol amendments** (§3), all pre-training, all KITTI-360-driven,
   each re-hashed. The `MIN_OCC_FRAMES` one resolves a contradiction inside the brief and is
   the one to scrutinise.
2. **Deviation — the val drive is denser than Gate 8C-1's.** Gate 8D built anchors at stride
   1 on drive 0006 (1 768) where Gate 8C-1 used stride 3 (590). More source validation, no
   leakage, but source numbers are not anchor-for-anchor comparable with 8C-1.
3. **Not computed — semantic target metrics.** SSC mIoU, classwise IoU, open-vocabulary
   naming accuracy and observed-vs-completed naming accuracy on the *targets* were not run.
   Teacher agreement and completed-voxel accuracy were computed on KITTI-360 inside
   selection. Two success criteria therefore read "not established" rather than pass or
   fail, and the geometry/semantic aggregation reconciliation the brief asks for was not
   performed. Given the geometry verdict is already FAIL, I stopped rather than spend
   several more GPU-hours; this is a real gap in the deliverable and is not a pass.
4. **Not computed — paired per-frame confidence intervals, AP/AUROC/ECE on targets, and
   per-semantic-region error.** Same reason.
5. **Occ3D evaluated on 400 anchors**, not the full split, for time.
6. **Two Gate 8C-1 tests fail** (§11), from earlier work in this session, not from Gate 8D.
7. The ceiling regression was **predicted from the Phase 3 audit and reported before
   training**; it was not discovered after the fact and no post-hoc change was made.
