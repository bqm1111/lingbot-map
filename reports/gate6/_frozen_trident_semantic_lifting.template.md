# Gate 6 — Frozen Trident-H semantic lifting and multi-dataset coverage decomposition

**Date:** 2026-09-02 · **Status:** complete · **Nothing was trained, fitted, tuned or
selected on any of these three benchmarks.** No optimizer was constructed, no backward pass
was run, and no parameter of LingBot-Map, MoGe-2 or Trident was touched.

---

## 1. Diagnosis

# `FROZEN_TRIDENT_SEMANTICS_TRANSFER`
# `COVERAGE_DOMINATES`

All three benchmarks pass all four predeclared criteria (§6, §13, §15). On all three, the
B-D coverage-miss fraction exceeds the naming-error fraction, so the orthogonal bottleneck
diagnosis is `COVERAGE_DOMINATES`.

**Read `COVERAGE_DOMINATES` with its scope attached.** It is a statement about the
population of **valid ground-truth occupied voxels, pooled over the whole benchmark**, and
about nothing else. The three bins partition that population only, so the decomposition
**excludes false-positive occupied predictions entirely** — a voxel the pipeline invents
where the ground truth is empty appears in no bin. B-D already predicts occupancy at a
pooled precision of 0.25 / 0.56 / 0.36 (§7), so the diagnosis emphatically **does not**
establish that expanding occupancy would raise IoU: every expansion that recovers a missed
voxel also adds false positives, and the decomposition is blind to that side of the ledger.
It says where the *recall* loss lives, not that recall is the cheaper thing to buy. Whether
the missed occupancy is spatially reachable at an acceptable precision cost is a separate
question this gate does not answer.

The rules were written into
`configs/gate6/trident_semantic_precommit.yaml` and SHA-256-pinned before any Gate-6
target file was opened; they have not been rewritten.

---

## 2. Plain-language conclusion

**A frozen 2D open-vocabulary teacher can label the frozen reconstruction, and the thing
stopping this pipeline is not naming — it is that most of the scene was never
reconstructed.**

Lifting Trident-H through the Gate-5.2 geometry, with nothing trained anywhere in the
chain, produces a semantic occupancy prediction that beats **all 100** vocabulary
permutations on **all three** benchmarks, on both the full SSC mIoU and the TP-conditioned
balanced recall. Where the prediction and the ground truth are both occupied, the teacher
names the voxel correctly **56 %** of the time on SemanticKITTI (19 classes), **44 %** on
Occ3D-nuScenes (17 classes) and **60 %** on SSCBench-KITTI-360 (18 classes). Uniform chance
at these class counts is 5.3 % / 5.9 % / 5.6 %; the permuted-vocabulary controls land at
5.1 % / 6.2 % / 5.7 % balanced recall, i.e. right where chance predicts.

The full SSC mIoU is nevertheless small — 0.0511, 0.0480, 0.0266 — and the decomposition
says exactly why. Of every valid ground-truth occupied voxel:

| | coverage miss | naming error | correct |
|---|---:|---:|---:|
| SemanticKITTI 08 | **69.8 %** | 13.3 % | 16.9 % |
| Occ3D-nuScenes | **75.0 %** | 13.9 % | 11.1 % |
| SSCBench-KITTI-360 | **82.7 %** | 7.0 % | 10.3 % |

Between five and eight of every ten occupied voxels are simply **absent** from the
reconstruction; only one to one-and-a-half of every ten are present and misnamed. The
ratio is 5:1, 5.4:1 and 11.8:1. That is the answer the gate was built to get: **if this
pipeline is to improve, the thing to work on is completion, not a better semantic
teacher.**

Two qualifications belong with that sentence. First, the partition counts only
ground-truth **occupied** voxels, so it says nothing about the false positives an expansion
would add (§1, §9) — it locates the recall loss without proving recall is the cheaper thing
to buy. Second, it is a **global** statement: in SemanticKITTI's 0–10 m band the ordering
reverses and naming is the larger term (§12). Whether the missed occupancy is close enough
to existing support for a local prior to recover it — and at what precision cost — is not
answered here and is the subject of the follow-up gate.

Three findings sharpen this.

* **The dilation-only voxels are labelled almost as well as the reconstructed ones.**
  Propagating the probability vector of the nearest raw voxel across the fixed 0.4 m
  dilation gives 53.4 % / 44.0 % / 58.8 % accuracy against 63.3 % / 45.0 % / 62.5 % on the
  voxels the reconstruction actually produced (§10). Semantic information survives
  short-range geometric extrapolation. A completion prior would inherit usable labels, not
  noise — which is precisely the assumption the recommended next experiment rests on.

* **The naming errors are not diffuse; they concentrate on classes defined by annotation
  convention rather than by appearance.** `terrain` collapses on all three benchmarks —
  TP-recall 0.0001, 0.019, 0.003 — and it does not scatter: it is absorbed by exactly one
  neighbour each time (`other-ground` 74.7 % on SemanticKITTI, `other_flat` 56.5 % on
  Occ3D, `vegetation` 48.8 % on KITTI-360). The same happens to `manmade` (→ `other_flat`
  27.5 %) and `others` on Occ3D, and to `other-structure` on KITTI-360. Classes with a
  clear visual referent transfer well in the same run: `bus` 0.806, `vegetation` 0.844,
  `road` 0.881, `car` 0.716. **The failure is a vocabulary-boundary failure, not a
  perception failure** (§11).

* **Coverage failure is overwhelmingly a range effect.** Inside 10 m the decomposition is
  nearly balanced (SemanticKITTI: 25.6 % miss, 26.4 % naming, 47.9 % correct); beyond 30 m
  it is not a contest (82.0 % / 8.8 % / 9.2 %, and 98.5 % / 0.6 % / 0.9 % on KITTI-360).
  A five-frame, single-camera, visible-surface reconstruction cannot see the far half of
  an accumulated-LiDAR target (§12).

Everything in §17 about what this does *not* show still holds — in particular, "frozen"
refers to our adaptation, not to the pretraining of CLIP, DINO and SAM, and the class
names of each benchmark are supplied to the teacher at inference.

---

## 3. The frozen method

Every element below was fixed before Gate 6 and is unchanged here.

| stage | what runs | frozen from |
|---|---|---|
| geometry | LingBot-Map, five-frame single-camera clip, anchor last | Gate 0 |
| metric scale | **G51-B**: full image → frozen MoGe-2 given the calibrated horizontal FOV → one scalar per clip → applied identically to depth and pose **translation**, rotations untouched | Gate 5.1 / 5.2 |
| gates | LingBot confidence ≥ 1.5, metric depth in (1, 60) m | Gate 0–4 |
| occupancy | **B-R** raw; **B-D** = B-R + `dilate_r2` (radius 2 voxels at 0.2 m = 0.4 m) | Gate 3.1 / 4 |
| grid | SemanticKITTI and KITTI-360 native 256×256×32 @ 0.2 m; Occ3D on the frozen canonical 400×400×32 @ 0.2 m, reduced to the official 200×200×16 @ 0.4 m by "any subvoxel occupied" | Gate 4 |

No C3, no V3, no learned corrector, no oracle scale, no scale distillation. MoGe stays at
inference. **B-D is the predeclared primary condition** and is the same choice on all three
benchmarks.

What Gate 6 adds, and only this: each fused LingBot point carries the pixel it came from,
so it can also carry the teacher's probability vector at that pixel. A raw occupied voxel
takes the **arithmetic mean** of every contributing vector over all points and all five
frames — no confidence weighting, no class prior, no frame-count normalisation — and is
assigned the argmax. A dilation-only voxel takes the vector of its **nearest** raw source
inside the same fixed neighbourhood, averaging exact-distance ties. Outside the frozen
occupancy the prediction is empty, always.

`gate6/lifting.py:points_with_pixels` is asserted **bit-identical** to the frozen
`voxel_gate.c3.c3_points` in `tests/gate6`, so the pixel bookkeeping cannot have perturbed
the geometry. The three binary reproductions in §6 confirm it end to end.

---

## 4. The teacher: version, weights, supervision

{{TABLE_PROVENANCE}}

Trident is the **predeclared primary** teacher; it was not selected on Gate-6 performance,
and the technical-fallback rule was never triggered (§22). The official configuration used
is `configs/cfg_city_scapes.py` inheriting `configs/base_config.py` — the only official
Trident configuration for automotive imagery — adopted unchanged for all three automotive
benchmarks so the teacher runs at a single operating point: CLIP ViT-H/14
`laion2b_s32b_b79k`, **DINO v1 ViT-B/16** as the spatial encoder (*not* DINOv2 — see
below), SAM ViT-H with refinement on, `cos_fac = 3.0`,
`refine_neg_cos = False`, `coarse_thresh = 0.10`, slide 224/336, `sam_iou_thresh = 0.80`,
`logit_scale = 40`, and the official `openai_imagenet_template` 80-template prompt
ensemble. One deterministic phrase per class, no synonyms, no alternatives evaluated.

**Provenance honesty.** The brief specified "DINOv2 spatial features as specified by the
official implementation". The official implementation specifies **DINO v1 ViT-B/16**:
`trident.py:47` calls `torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')`
unconditionally, and the `vfm_model` argument only selects which attention hook is
registered. DINOv2 is not reachable through the official code. Following the controlling
instruction ("as specified by the official implementation"), DINO v1 is what ran. This is
deviation **D2**.

### Dense score extraction

The adapter must expose the per-class scores *without changing the official hard
prediction*. With SAM refinement enabled, the official pipeline ends at

```
refined_masks, scores, refined_logits, boxes = sam_refinement(...)
seg_pred, seg_logits = refined_masks, refined_logits
```

Two facts had to be established by reading the official code, and both are asserted in the
tests rather than assumed:

1. The label is taken as `argmax(0)` of the dense volume built by `map_refinement_coarse`,
   **except** on pixels no SAM region claimed, where the official code falls back to the
   coarse CLIP prediction (`refined_logits.sum(0) == 0 → segmentations`). The dense tensor
   that reproduces the official label everywhere is therefore
   `where(sum == 0, coarse_softmax, refined_logits)`.
2. `map_failed_regions` runs *afterwards* and rewrites the returned scores, but provably
   cannot move the label: it maxes `refined_logits` against `failed_logit` and only then
   tests `failed_logit.max(0) > refined_logits.max(0)`, which is False by construction.
   **The tensor Trident returns is not the tensor its own label was computed from.** The
   adapter reads the scores at the point the label is taken and reproduces this behaviour
   rather than repairing it.

Probabilities: the coarse volume already carries the official temperature
(`logit_scale = 40`, then softmax over classes). The SAM-refined volume is
`sigmoid(sam_logit) × coarse_prob`, non-negative and unnormalised, so it is divided by its
per-pixel sum — strictly monotone per pixel, so the official argmax is preserved exactly.
No second softmax: the scores live in [0, 1] and re-softmaxing them would flatten the
signal to near-uniform.

### Parity with official Trident

{{TABLE_PARITY}}

Zero disagreeing pixels in 82.4 million, over 102 deterministically chosen frames spanning
all three benchmarks — so the "documented floating-point ties" exception the brief allows
was never needed. The recorder that observes the official `sam_refinement` call was also
removed on 12 frames and the official label re-computed: bit-identical every time, so the
parity result cannot be an artefact of the instrumentation.

---

## 5. Datasets, clips and frames

{{TABLE_COUNTS}}

No manifest was replaced, reduced or rebuilt: the SemanticKITTI manifest is the frozen
Gate-3.1/5.1 one (163 clips of sequence 08), the Occ3D-nuScenes manifest is the frozen
Gate-4 one (1,182 clips over the 150 official validation scenes, CAM_FRONT, five-frame),
and the KITTI-360 manifest is the frozen Gate-5.2 one (1,753 eligible clips of
`2013_05_28_drive_0006_sync`, with the verified `pose_frames[i+1]` frame mapping). Every
clip in every manifest produced a prediction: **zero** were skipped for a missing scale, a
missing teacher cache or a missing geometry cache.

The teacher ran **once per unique rectified frame** (8,502 in total), not once per clip, so
two clips sharing a frame see identical semantics by construction.

---

## 6. Binary sanity checks

Before any semantic conclusion, the frozen occupancy had to come back out of entirely new
Gate-6 code.

{{TABLE_SANITY}}

All four reproduce to four decimal places, `|Δ| ≤ 0.00003` against a tolerance of 0.0005 —
a factor of 17 inside it.

Reaching this exposed one real discrepancy worth recording. Gate 5.1 and Gate 5.2 report
the **mean of per-clip** binary IoU; Gate 6's semantic metrics are pooled over voxels
(micro), which is the standard SSC convention and the only aggregation under which a
bootstrap over units is meaningful. Computed the pooled way, B-D on KITTI-360 is 0.1321,
not 0.1342. The geometry was never in question: a direct diff showed the Gate-6 raw
occupancy volume is **bit-identical** to the Gate-5.2 one, voxel for voxel, on every clip
tested. Both statistics are now reported (`binary_iou_mean_per_clip` and
`binary_iou_pooled`), and the reproduction is checked against the macro one, which is the
statistic the pinned numbers were produced with.

---

## 7. Official SSC metrics

{{TABLE_MAIN}}

Absolute values are **not comparable across the three rows**: the grids, evaluation masks,
class counts and target-construction rules all differ (§17).

---

## 8. Semantic quality conditional on occupancy

Restricted to voxels where the prediction **and** the ground truth are occupied — the
measurement of whether the teacher's semantics survived lifting and temporal fusion,
independent of how much geometry was reconstructed.

{{TABLE_SUPPORT}}

Read against the permuted-vocabulary controls in §13, these numbers are the core positive
result of the gate. The B-D balanced recalls are 0.355 / 0.380 / 0.284; the permutation
95th percentiles are 0.090 / 0.111 / 0.098 and not one of the 300 permutations reaches the
real value.

---

## 9. Coverage versus naming

Every valid ground-truth occupied voxel falls into exactly one of three bins. The three
counts sum to the number of such voxels by construction, and the identity is asserted in
the tests rather than assumed.

{{TABLE_DECOMPOSITION}}

On all three benchmarks the coverage-miss fraction exceeds the naming-error fraction, which
fires `COVERAGE_DOMINATES` on 3 of 3 — the criterion required only 2. Dilation is what
changes the balance: it converts coverage misses into a mixture of correct labels and
naming errors, moving the ratio from 33:1 / 20:1 / 66:1 at B-R to 5:1 / 5:1 / 12:1 at B-D.
Even after that conversion, coverage still dominates by five to one at its narrowest.

**What this partition does not contain.** The denominator is the set of valid ground-truth
**occupied** voxels. Voxels the prediction occupies where the ground truth is empty are
false positives and fall outside all three bins by construction. The partition therefore
measures the composition of the recall loss and is silent on precision. B-D's pooled
occupancy precision is 0.25 / 0.56 / 0.36, so a correction that converted coverage misses
into correct voxels at that precision would not obviously improve IoU at all. Nothing here
proves that expanding occupancy is the profitable move; it establishes only that if the
pipeline is to improve, most of the ground truth it is failing on is ground truth it never
reconstructed rather than ground truth it misnamed.

---

## 10. Reconstruction support versus dilation-only

The first category is called **reconstruction support**, not visibility: it says the raw
reconstruction placed a voxel there, which is not the same claim as the voxel being
visible.

{{TABLE_SUPPORT}}

The dilation-only voxels — which outnumber the reconstruction-supported ones roughly 2:1
to 3.5:1 — are labelled at 53.4 % / 44.0 % / 58.8 % against 63.3 % / 45.0 % / 62.5 % for
the supported ones. On Occ3D the gap is under one point. Propagating a probability vector
0.4 m through space costs remarkably little accuracy.

---

## 11. Per-class results

### SemanticKITTI 08 (B-D)

{{TABLE_PERCLASS_SEMANTICKITTI}}

Most frequent confusions:

{{TABLE_CONFUSION_SEMANTICKITTI}}

### Occ3D-nuScenes (B-D)

{{TABLE_PERCLASS_OCC3D}}

{{TABLE_CONFUSION_OCC3D}}

### SSCBench-KITTI-360 (B-D)

{{TABLE_PERCLASS_KITTI360}}

{{TABLE_CONFUSION_KITTI360}}

**The pattern is consistent and it is a vocabulary-boundary pattern.** Classes with a
direct visual referent are recovered well — `road` 0.881, `vegetation` 0.808/0.775/0.844,
`bus` 0.806, `car` 0.716/0.556/0.561, `truck` 0.671/0.545/0.663, `building`
0.538/0.571. Classes that exist because a LiDAR annotation guideline needed them collapse,
and each collapses into one specific neighbour rather than scattering:

* `terrain` → `other-ground` 74.7 % (SemanticKITTI), → `other_flat` 56.5 % (Occ3D),
  → `vegetation` 48.8 % (KITTI-360). TP-recall 0.0001 / 0.019 / 0.003.
* `manmade` → `other_flat` 27.5 %, `others` 17.9 % (Occ3D). TP-recall 0.106.
* `other-structure` → `vegetation` 31.8 %, `building` 21.0 % (KITTI-360).
* `parking` → `road` 36.9 % (SemanticKITTI), → `vegetation` 27.6 % (KITTI-360).
* `sidewalk` ↔ `road` / `other_flat`: a near coin-flip on a genuine boundary
  (SemanticKITTI: `road` 35.5 % vs `sidewalk` 34.1 %).

This matters for the next experiment: it says the residual naming error is largely a
*mapping* problem between an open vocabulary and a fixed benchmark ontology, and would not
obviously be fixed by a stronger 2D segmenter.

---

## 12. Distance and height diagnostics

{{TABLE_DISTANCE}}

{{TABLE_HEIGHT}}

Coverage miss rises monotonically with range on all three benchmarks while naming error
falls, because the far field contains almost no reconstructed voxels to misname.

**The global diagnosis reverses in the near field of SemanticKITTI.** In the 0–10 m band
at B-D, naming error is **larger** than coverage miss — 0.264 against 0.256 — so at short
range on that benchmark the binding constraint is naming, not coverage. This is the one
place in the gate where `COVERAGE_DOMINATES` does not hold, and it is stated rather than
averaged away.

{{TABLE_NEAR_RANGE}}

Height is far less discriminative than range, which is consistent with the miss being a
frustum-and-range effect rather than a ground-plane effect.

---

## 13. Vocabulary-permutation control

100 deterministic permutations per benchmark, seed 0, applied **only** to the
class-to-text assignment. Predicted occupancy, geometry, image features and the fused
probability vectors are untouched and nothing is recomputed — a permutation changes only
which class name a teacher channel answers to. The control is therefore exact and free,
and the identity permutation is asserted in the tests to reproduce the evaluator's own
numbers bit-for-bit.

{{TABLE_PERMUTATION}}

**Not one of the 600 permuted evaluations (100 × 2 metrics × 3 benchmarks) reaches the
real vocabulary's value.** The real result sits at the 100th percentile everywhere.

---

## 14. Bootstrap confidence intervals

10,000 resamples, seed 0, paired, over the benchmark's own unit: 150 official scenes on
Occ3D-nuScenes, 8 contiguous blocks of 20 clips on SemanticKITTI sequence 08, and the 87
Gate-5.2 contiguous blocks of 20 clips on KITTI-360 sequence 06. **The blocks come from a
single drive and are not independent scenes; they are not described as such.**

{{TABLE_BOOTSTRAP}}

{{BOOTSTRAP_ZERO_NOTE}} Dilation buys a large, unambiguous gain in full SSC mIoU and a
large reduction in coverage miss, at the cost of a smaller but real rise in naming error
and — on two of three benchmarks — a small drop in TP-conditioned accuracy. On
Occ3D-nuScenes that drop is **not distinguishable from zero**: the paired interval spans
it, so on that benchmark dilation should be read as buying coverage at no measurable
conditional-accuracy cost, not at a small one. SemanticKITTI's 8 blocks are few; its
intervals should be read as the weakest of the three.

---

## 15. Leakage audit

Semantic prediction was a strictly target-free phase, and this was **observed**, not
argued from the code.

{{TABLE_AUDIT}}

The auditor wraps `builtins.open`, `numpy.load`, `numpy.fromfile` **and** `cv2.imread`
(which the teacher uses and which bypasses `builtins.open` entirely), and raises on any
path matching `.label`, `.invalid`, `_1_1.npy`, `_1_2.npy`, `_1_8.npy`, `labels.npz`,
`/gts/`, `/voxels/`, `/labels/`, `velodyne`, `lidar_surface`, `oracle`, `scale_targets` or
`visible_ceiling`. Across the three runs it observed **37,192 file opens on 11,613 unique
paths and rejected nothing** — and it observed a non-zero number, which the tests check,
because an auditor that sees nothing proves nothing.

Ordering: every prediction was written to an immutable file and the per-file plus rollup
SHA-256 manifest emitted **before** the evaluator was allowed to run. The evaluator refuses
to start without that manifest, re-hashes every prediction file it scores (0 mismatches),
and records the rollup hash in its own summary. The tests additionally assert that each
manifest's mtime precedes its summary's.

Tamper test: randomising a target in a scratch tree — proved to be a real change by
loading it back and comparing — leaves the deployable prediction files bit-identical to
their pinned hashes. Nothing was ever written into a released dataset directory.

The static side is checked too: no module on the prediction path imports `gate6.targets`,
and `gate6/pipelines.py` may reach `prompted_lingbot.occ_datasets` only for the
**calibration** it exposes (`SemanticKittiOccSpec.cam_to_velo`) — the test asserts that
`.target(` and `load_semantickitti_target` appear nowhere in it.

### Float16 caching

The teacher cache stores float16. The brief permits that only once it is shown not to
matter downstream, so the teacher was re-run in **float32** over whole clips and the fused
voxel labels compared.

{{TABLE_FP16}}

Two label changes in 92,510 raw voxels and four in 482,908 dilated ones, with predicted
occupancy bit-identical — occupancy cannot depend on the teacher at all, and the test
asserts it.

### Decision-rule audit

{{TABLE_DECISION}}

Three of three benchmarks pass → `FROZEN_TRIDENT_SEMANTICS_TRANSFER`. B-D coverage miss
exceeds naming error on three of three (the rule required two) →`COVERAGE_DOMINATES`. The
branches do not overlap: `PARTIAL` and `FAIL` each require at least one dataset to fail,
and none does; `NAMING_DOMINATES` and `MIXED_BOTTLENECK` each require at least two
datasets where naming exceeds coverage, and there are none.

---

## 16. Optional OccAny-aligned comparator

{{OCCANY_SECTION}}

---

## 17. Interpretation limitations

Stated explicitly, because several of these are easy to overclaim.

* This is **frozen inference-only semantic lifting**. No target-domain semantic or
  occupancy training occurred, on any of the three benchmarks.
* **Trident is itself a composition of three pretrained foundation models** — CLIP
  (LAION-2B image–text pairs), DINO (ImageNet-1k self-supervised) and SAM (SA-1B
  class-agnostic masks), 1.71 billion frozen parameters in total. "Training-free" refers to
  **our** semantic adaptation and downstream pipeline, **not** to the original training of
  those models. None of them saw dense semantic segmentation labels, but CLIP saw a very
  large quantity of web image–text supervision.
* **The dataset class names are supplied to the teacher at inference.** This is
  open-vocabulary transfer with the target ontology given, not category discovery.
* **MoGe-2 supplies pretrained metric knowledge**, and **camera intrinsics are used** — the
  calibrated horizontal FOV reaches MoGe and the calibrated intrinsics reach the
  unprojection. The method is calibration-aware, not RGB-only.
* Validation LiDAR, occupancy targets and semantic targets are **evaluation-only**.
* **SemanticKITTI and SSCBench-KITTI-360 are related datasets** (same city, same sensor
  family, overlapping capture campaign). Two of the three benchmarks are not independent
  evidence of open-domain generalisation, and Occ3D-nuScenes is the only genuinely
  different sensor suite and continent.
* **This does not prove unrestricted open-domain generalisation.** Three automotive
  benchmarks with a driving-scene teacher configuration is not "any scene".
* **Absolute results across the three benchmarks are not comparable.** Different grids
  (0.2 m vs 0.4 m), different evaluation masks (camera mask plus a rear-half cut on Occ3D;
  `.invalid`-derived on the KITTI family), different class counts and different
  target-construction rules.
* **No claim of superiority over OccAny is made**, and §16 explains why one could not be
  made from this experiment even in principle.
* **Strong conditional semantics with a low full mIoU is exactly the signature of a
  coverage bottleneck**, and that is how §2 reads it — not as evidence that the semantics
  are good enough in absolute terms.
* The prior LiDAR scale oracle used in Gates 4–5.2 removes **scalar scale error only**. It
  is not an oracle for depth shape, poses, fusion or visibility, and nothing here changes
  that.
* The permutation control tests whether the **channel-to-class assignment** carries
  information. It is not a test of whether the spatial partition itself is meaningful; a
  teacher that segmented the image into correct regions but named them all identically
  would fail this control, and one that named regions correctly but segmented poorly could
  still pass it.

---

## 18. Recommended next experiment

**Exactly one:** train a small voxel completion-and-semantic prior on cached geometry and
cached teacher probabilities, keeping LingBot-Map, MoGe-2 and Trident frozen.

The diagnosis is `FROZEN_TRIDENT_SEMANTICS_TRANSFER` + `COVERAGE_DOMINATES`, which selects
this branch by the predeclared rule. Concretely:

* **Inputs**: the frozen per-voxel geometry features already used by Gate 3.1 (occupancy,
  point count, distinct contributing frames, mean confidence, mean point depth,
  correction-region flag) **plus** the fused per-voxel teacher probability vector. Both are
  already cached by this gate; no new inference is needed.
* **Output**: an occupancy-completion decision plus a semantic label, over a correction
  region wider than the current 0.4 m — the point is to add voxels the reconstruction never
  produced, which is what §9 says the loss is made of.
* **Train on one source dataset plus one diverse source dataset**, and **validate
  leave-one-dataset-out.** This is not optional. Gate 5.2 showed the Gate-2 clip head
  *anti-transferred* — Spearman 0.980 with the oracle and a 29.7 % gain error, significantly
  worse than a constant. A completion prior has exactly the same failure mode available to
  it, and only leave-one-dataset-out validation exposes it.
* **Do not train LingBot, MoGe or Trident**, and do not revisit scale distillation.

Two things from this gate should be carried into that design. First, §10 says propagated
labels stay accurate over short distances, so a completion prior can be expected to inherit
usable semantics rather than having to re-derive them. Second, §11 says the residual naming
error is concentrated in ontology-boundary classes (`terrain`, `manmade`, `others`,
`other-*`); a completion prior that also learns a small ontology-mapping correction on top
of the frozen probability vectors would target the *measured* error, and it is cheap to add
as a second head. If the naming-error fraction ever overtakes coverage miss on two of three
benchmarks, that branch reverses and fusion/class ambiguity is diagnosed first.

---

## 19. Files created and modified

**Created — modules** (`gate6/`): `__init__.py`, `vocab.py`, `trident_adapter.py`,
`frames.py`, `grids.py`, `lifting.py`, `pipelines.py`, `targets.py`, `metrics.py`,
`permute.py`, `stats.py`, `audit.py`.

**Created — tools** (`tools/gate6/`): `stage0_audit.py`, `teacher_provenance.py`,
`precommit.py`, `verify_trident_parity.py`, `cache_semantics.py`, `run_cache_all.sh`,
`predict.py`, `evaluate.py`, `analyze.py`, `figures.py`, `fp16_fidelity.py`,
`report_tables.py`, `occany_comparator.py`, `run_occany.sh`.

**Created — configuration, tests, report**: `configs/gate6/trident_semantic_precommit.yaml`
(SHA-256-pinned), `tests/gate6/test_gate6.py`, and this report plus its template.

**Created — artifacts** (`artifacts/gate6/`, {{ART_SIZE}}): the stage-0 audit, teacher
provenance, precommit pin, parity report, per-dataset prediction manifests, per-clip count
blocks, summaries, analyses, the combined diagnosis, the fp16 fidelity report, three
qualitative figures with their selection records, and the caching logs.

**Modified: nothing.** No existing module, configuration, checkpoint, manifest, artifact or
report was changed. The two pre-existing dirty files, `lingbot_map/models/gct_stream.py`
and `research/sem_bypass/model.py`, were recorded by SHA-256 in the stage-0 audit and are
untouched. `git status --short` shows exactly the same two modified files as before the
gate.

**Third-party working trees**: the Trident checkout is byte-identical to upstream
(`git diff` empty; only `__pycache__` is untracked) — the one construction defect it has is
repaired *post-construction* by the adapter, never by editing the file. The OccAny/
Grounded-SAM-2 checkout has one build-configuration line temporarily added and restored;
see §22.

---

## 20. Reproduction commands

```bash
# environment (Trident needs mmcv/mmsegmentation, which the main env does not have)
python -m venv --system-site-packages third_party/trident_env
CUDA_VISIBLE_DEVICES="" MMCV_WITH_OPS=1 FORCE_CUDA=0 \
  third_party/trident_env/bin/pip install --no-build-isolation --no-deps mmcv==2.1.0
third_party/trident_env/bin/pip install --no-deps mmsegmentation==1.2.2 prettytable openpyxl

# stage 0, provenance, precommit  (must precede everything else)
python tools/gate6/stage0_audit.py
PYTHONPATH=$PWD:$TRIDENT trident_env/bin/python tools/gate6/teacher_provenance.py
python tools/gate6/precommit.py                       # writes + SHA-256-pins the config

# adapter parity against official Trident (no target is opened)
cd $TRIDENT && PYTHONPATH=$REPO:$TRIDENT trident_env/bin/python \
  $REPO/tools/gate6/verify_trident_parity.py --n 102

# teacher cache: one run per unique frame, 4 GPUs
bash tools/gate6/run_cache_all.sh

# prediction (audited + hashed), then evaluation (targets opened only now)
for DS in semantickitti occ3d kitti360; do
  python tools/gate6/predict.py  --dataset $DS
  python tools/gate6/evaluate.py --dataset $DS
  python tools/gate6/analyze.py  --dataset $DS
  python tools/gate6/figures.py  --dataset $DS
done
python tools/gate6/analyze.py --combine

# float16 fidelity, comparator, report
cd $TRIDENT && PYTHONPATH=$REPO:$TRIDENT trident_env/bin/python \
  $REPO/tools/gate6/fp16_fidelity.py --dataset kitti360 --n 5
bash tools/gate6/run_occany.sh && python tools/gate6/occany_comparator.py --stage predict
python -m pytest tests/gate6 -q
python tools/gate6/report_tables.py --write
```

---

## 21. Runtime, VRAM and disk

| stage | wall time | peak VRAM | notes |
|---|---:|---:|---|
| Trident cache — SSCBench-KITTI-360 (1,777 frames) | 14.1 min | 10.02 GiB | 4 GPUs, 1.75 s/frame each |
| Trident cache — SemanticKITTI (815 frames) | 6.9 min | 10.03 GiB | 4 GPUs, 1.90 s/frame each |
| Trident cache — Occ3D-nuScenes (5,910 frames) | 51.1 min | 10.54 GiB | 4 GPUs, 1.95 s/frame each |
| adapter parity (102 frames + 12 pristine re-runs) | 3.9 min | 10.1 GiB | |
| prediction — SemanticKITTI / Occ3D / KITTI-360 | 1.3 / 3.1 / 11.5 min | 0.20 / 0.25 / 0.14 GiB | audited; CPU-bound |
| evaluation — SemanticKITTI / Occ3D / KITTI-360 | 0.6 / 0.4 / 2.4 min | — | CPU |
| analysis (100 permutations + 4 × 10,000 bootstraps × 3) | < 3 min | — | CPU, closed form |
| float16 fidelity (5 clips re-run in float32) | 1.1 min | 10.0 GiB | |
| comparator — Grounded-SAM-2 cache (815 frames) | 2.1 min | 6.8 GiB | 4 GPUs; 0 frames with no detection |
| comparator — lift + score (163 clips) | 0.6 min | 0.2 GiB | |
| **total** | **≈ 105 min** | **10.54 GiB** | 4 × RTX PRO 6000 Blackwell |

Disk: **37 GB** teacher probability cache (float16, on the LingBot lattice — storing it at
native resolution would have cost ≈ 250 GB), **1.5 GB** predictions, **37 MB** comparator
cache, **{{ART_SIZE}}** artifacts in the repository. Model weights: 3.94 GB (CLIP ViT-H/14),
2.56 GB (SAM ViT-H), 0.33 GB (DINO ViT-B/16), plus 0.94 GB (GroundingDINO SwinB) and
0.90 GB (SAM 2.1 Hiera-L) for the comparator. The root filesystem has 14 GB free, so every
bulk cache lives on `/media/SSD1`, as in Gates 4–5.2.

---

## 22. Deviations and blockers

**No stop condition fired.** Trident-H was reproduced, every required manifest and target
was available, the frozen binary occupancy reproduced exactly, the official mappings and
masks were verified, no prediction touched a target, no training occurred, and no large or
unlicensed download was needed.

**D1 — Trident's ViT-H + SAM-refinement construction defect.** `trident.py:85` reads
`if sam_model_type != 'vit_h' or not sam_refinement: self.sam = ...` and the `else` branch
is missing, so the predeclared large configuration raises `AttributeError` before it can
run. The inline comment cites segment-anything issue #540 (ViT-H overflows in fp16), so the
intent — build ViT-H in float32 — is unambiguous. **The upstream file was not edited.** The
model is built through the official code path with `sam_refinement=False` and the float32
SAM-H installed post-construction, which is the state the missing branch would have
produced. The Trident working tree is byte-identical to upstream.

**D2 — DINO v1, not DINOv2.** The brief said DINOv2; the official implementation
hard-codes DINO ViT-B/16 and cannot reach DINOv2. Following "as specified by the official
implementation", DINO v1 ran. Detailed in §4.

**D3 — the Cityscapes configuration applied to all three benchmarks.** Trident ships no
config for KITTI, nuScenes or KITTI-360. `cfg_city_scapes.py` is its only automotive
config; it was adopted unchanged for all three so the teacher runs at one operating point,
and it was pinned in the precommit before any target was opened. Nothing in it was tuned.

**D4 — float16 teacher cache, stored on the LingBot lattice.** Quantified in §15: two label
changes in 92,510 voxels, occupancy bit-identical. Storing at native resolution would have
required ≈ 250 GB.

**D5 — macro versus pooled binary IoU.** Gates 5.1/5.2 report the mean of per-clip IoU;
Gate 6's semantic metrics are pooled over voxels. Both are reported and the geometry was
verified bit-identical (§6).

**D6 — Occ3D canonical-to-native semantic reduction.** The frozen occupancy rule reduces
the canonical 0.2 m grid to the official 0.4 m one by "any subvoxel occupied". That rule
says nothing about labels, so one had to be declared: an occupied native voxel takes the
**mean probability vector of its occupied subvoxels**, which is the natural extension of
both the frozen occupancy rule and the uniform voxel-fusion rule. Declared in the precommit
before any target was opened.

**D7 — SemanticKITTI bootstrap blocks.** The brief specifies contiguous blocks of 20 clips;
sequence 08 has 163 clips, giving only **8** blocks (the remainder appended to the last, as
in Gate 5.2). Its intervals are correspondingly the widest and should be read as the
weakest of the three.

{{OCCANY_DEVIATIONS}}

---

*Gate 6 complete. Diagnosis `FROZEN_TRIDENT_SEMANTICS_TRANSFER` + `COVERAGE_DOMINATES`.
Report generated by `tools/gate6/report_tables.py`; every number above is substituted from
`artifacts/gate6/` rather than transcribed.*
