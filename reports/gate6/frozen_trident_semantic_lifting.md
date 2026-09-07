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

| component | identity | parameters | licence | supervision | SHA-256 |
|---|---|---:|---|---|---|
| repository | Trident `7feea10222bf` (2025-11-22) | — | Apache-2.0 (LICENSE in repository root) | training-free framework | `trident.py` 0924813e951f |
| semantic encoder | OpenCLIP ViT-H-14 `laion2b_s32b_b79k` | 986,109,441 | MIT (LAION OpenCLIP release) | LAION-2B image-text pairs; contrastive, no dense segmentation labels | 9a78ef8e8c73 |
| spatial encoder | **DINO v1** dino_vitb16 (ViT-B/16, patch 16) — *not DINOv2* | 85,798,656 | Apache-2.0 | ImageNet-1k, self-supervised (no labels) | torch.hub `facebookresearch/dino:main` |
| mask refinement | SAM vit_h | 641,090,608 | Apache-2.0 | SA-1B, class-agnostic masks (no semantic class labels) | a7bf3b02f3eb |
| **total** | | **1,712,998,705** | | all frozen | |

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

| quantity | value |
|---|---:|
| frames tested | 102 across 3 benchmarks |
| pixels compared | 82,382,952 |
| pixels where argmax ≠ official label | **0** |
| frames with any disagreement | 0 |
| frames re-run with the recorder removed | 12 |
| those bit-identical to the recorded run | all |
| max \|Σ probabilities − 1\| | 3.58e-07 |

Zero disagreeing pixels in 82.4 million, over 102 deterministically chosen frames spanning
all three benchmarks — so the "documented floating-point ties" exception the brief allows
was never needed. The recorder that observes the official `sam_refinement` call was also
removed on 12 frames and the official label re-computed: bit-identical every time, so the
parity result cannot be an artefact of the instrumentation.

---

## 5. Datasets, clips and frames

| benchmark | clips | unique RGB frames | classes | prediction grid | prediction rollup SHA-256 |
|---|---:|---:|---:|---|---|
| SemanticKITTI 08 | 163 | 815 | 19 | semantickitti (256, 256, 32) | `f8f8a66e9d66aa9d…` |
| Occ3D-nuScenes | 1182 | 5910 | 17 | occ3d_nuscenes (200, 200, 16) | `32fbe8c6b6677ec6…` |
| SSCBench-KITTI-360 | 1753 | 1777 | 18 | sscbench_kitti360 (256, 256, 32) | `2d6effe3b0e6b351…` |

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

| benchmark | condition | frozen value (mean-per-clip) | Gate 6 (mean-per-clip) | \|Δ\| | within 0.0005 | Gate 6 (pooled, for reference) |
|---|---|---:|---:|---:|---|---:|
| SemanticKITTI 08 | B-D | 0.1584 | 0.1584 | 0.00000 | yes | 0.1600 |
| Occ3D-nuScenes | B-D | 0.2070 | 0.2070 | 0.00003 | yes | 0.2094 |
| SSCBench-KITTI-360 | B-D | 0.1342 | 0.1342 | 0.00003 | yes | 0.1321 |
| SSCBench-KITTI-360 | B-R | 0.0376 | 0.0376 | 0.00003 | yes | 0.0363 |

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

| benchmark | clips | cond | binary IoU (pooled) | prec (pooled) | rec (pooled) | SSC mIoU (pooled) | TP-cond. acc | TP-bal. recall | coverage miss | naming error | correct |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SemanticKITTI 08 | 163 | B-R | 0.0670 | 0.3492 | 0.0766 | 0.0281 | 0.6326 | 0.4388 | 0.9234 | 0.0281 | 0.0484 |
|  |  | **B-D** | **0.1600** | 0.2537 | 0.3022 | **0.0511** | **0.5592** | 0.3551 | **0.6978** | **0.1332** | 0.1690 |
| Occ3D-nuScenes | 1182 | B-R | 0.0793 | 0.6289 | 0.0832 | 0.0245 | 0.4483 | 0.4168 | 0.9168 | 0.0459 | 0.0373 |
|  |  | **B-D** | **0.2094** | 0.5641 | 0.2499 | **0.0480** | **0.4431** | 0.3795 | **0.7501** | **0.1392** | 0.1107 |
| SSCBench-KITTI-360 | 1753 | B-R | 0.0363 | 0.3747 | 0.0386 | 0.0088 | 0.6245 | 0.2789 | 0.9614 | 0.0145 | 0.0241 |
|  |  | **B-D** | **0.1321** | 0.3587 | 0.1729 | **0.0266** | **0.5959** | 0.2839 | **0.8271** | **0.0699** | 0.1030 |

Absolute values are **not comparable across the three rows**: the grids, evaluation masks,
class counts and target-construction rules all differ (§17).

---

## 8. Semantic quality conditional on occupancy

Restricted to voxels where the prediction **and** the ground truth are occupied — the
measurement of whether the teacher's semantics survived lifting and temporal fusion,
independent of how much geometry was reconstructed.

| benchmark | population | voxels | top-1 accuracy | balanced recall |
|---|---|---:|---:|---:|
| SemanticKITTI 08 | reconstruction support | 1,436,514 | 0.6326 | 0.4388 |
|  | dilation-only | 4,233,239 | 0.5343 | 0.3295 |
|  | all co-occupied | 5,669,753 | 0.5592 | 0.3551 |
| Occ3D-nuScenes | reconstruction support | 995,434 | 0.4497 | 0.4193 |
|  | dilation-only | 1,992,460 | 0.4398 | 0.3609 |
|  | all co-occupied | 2,987,894 | 0.4431 | 0.3795 |
| SSCBench-KITTI-360 | reconstruction support | 5,319,813 | 0.6245 | 0.2789 |
|  | dilation-only | 18,513,975 | 0.5876 | 0.2837 |
|  | all co-occupied | 23,833,788 | 0.5959 | 0.2839 |

Read against the permuted-vocabulary controls in §13, these numbers are the core positive
result of the gate. The B-D balanced recalls are 0.355 / 0.380 / 0.284; the permutation
95th percentiles are 0.090 / 0.111 / 0.098 and not one of the 300 permutations reaches the
real value.

---

## 9. Coverage versus naming

Every valid ground-truth occupied voxel falls into exactly one of three bins. The three
counts sum to the number of such voxels by construction, and the identity is asserted in
the tests rather than assumed.

| benchmark | cond | valid GT occupied voxels | coverage miss | naming error | correct | miss : naming |
|---|---|---:|---:|---:|---:|---:|
| SemanticKITTI 08 | B-R | 18,762,930 | 0.9234 | 0.0281 | 0.0484 | 32.8 : 1 |
|  | **B-D** | 18,762,930 | **0.6978** | **0.1332** | 0.1690 | 5.2 : 1 |
| Occ3D-nuScenes | B-R | 11,957,590 | 0.9168 | 0.0459 | 0.0373 | 20.0 : 1 |
|  | **B-D** | 11,957,590 | **0.7501** | **0.1392** | 0.1107 | 5.4 : 1 |
| SSCBench-KITTI-360 | B-R | 137,841,471 | 0.9614 | 0.0145 | 0.0241 | 66.3 : 1 |
|  | **B-D** | 137,841,471 | **0.8271** | **0.0699** | 0.1030 | 11.8 : 1 |

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

| benchmark | population | voxels | top-1 accuracy | balanced recall |
|---|---|---:|---:|---:|
| SemanticKITTI 08 | reconstruction support | 1,436,514 | 0.6326 | 0.4388 |
|  | dilation-only | 4,233,239 | 0.5343 | 0.3295 |
|  | all co-occupied | 5,669,753 | 0.5592 | 0.3551 |
| Occ3D-nuScenes | reconstruction support | 995,434 | 0.4497 | 0.4193 |
|  | dilation-only | 1,992,460 | 0.4398 | 0.3609 |
|  | all co-occupied | 2,987,894 | 0.4431 | 0.3795 |
| SSCBench-KITTI-360 | reconstruction support | 5,319,813 | 0.6245 | 0.2789 |
|  | dilation-only | 18,513,975 | 0.5876 | 0.2837 |
|  | all co-occupied | 23,833,788 | 0.5959 | 0.2839 |

The dilation-only voxels — which outnumber the reconstruction-supported ones roughly 2:1
to 3.5:1 — are labelled at 53.4 % / 44.0 % / 58.8 % against 63.3 % / 45.0 % / 62.5 % for
the supported ones. On Occ3D the gap is under one point. Propagating a probability vector
0.4 m through space costs remarkably little accuracy.

---

## 11. Per-class results

### SemanticKITTI 08 (B-D)

| class | SSC IoU | TP-cond. recall | GT occupied voxels | co-occupied voxels |
|---|---:|---:|---:|---:|
| car | 0.0991 | 0.5564 | 790,737 | 307,499 |
| bicycle | 0.0402 | 0.3164 | 15,464 | 5,193 |
| motorcycle | 0.0590 | 0.2813 | 7,383 | 2,773 |
| truck | 0.0595 | 0.5446 | 14,404 | 5,094 |
| other-vehicle | 0.0345 | 0.3421 | 41,239 | 13,711 |
| person | 0.0258 | 0.1537 | 47,687 | 14,402 |
| bicyclist | 0.0308 | 0.1880 | 79,503 | 19,267 |
| motorcyclist | 0.0470 | 0.2226 | 974 | 310 |
| road | 0.2727 | 0.8809 | 2,221,712 | 1,459,315 |
| parking | 0.0126 | 0.1347 | 163,618 | 76,823 |
| sidewalk | 0.0931 | 0.3407 | 1,478,528 | 612,231 |
| other-ground | 0.0007 | 0.3186 | 19,281 | 6,024 |
| building | 0.0420 | 0.5381 | 2,519,530 | 428,777 |
| fence | 0.0289 | 0.1852 | 212,542 | 51,404 |
| vegetation | 0.0813 | 0.8081 | 7,875,561 | 1,507,377 |
| trunk | 0.0179 | 0.2765 | 176,892 | 41,343 |
| terrain | 0.0000 | 0.0001 | 3,027,032 | 1,097,323 |
| pole | 0.0125 | 0.3709 | 56,021 | 16,809 |
| traffic-sign | 0.0135 | 0.2881 | 14,822 | 4,078 |

Most frequent confusions:

| ground-truth class | co-occupied voxels | most frequent predictions |
|---|---:|---|
| vegetation | 1,507,377 | vegetation 80.8%, other-ground 4.6%, car 3.3% |
| road | 1,459,315 | road 88.1%, parking 3.3%, other-ground 3.0% |
| terrain | 1,097,323 | other-ground 74.7%, vegetation 11.6%, road 5.2% |
| sidewalk | 612,231 | road 35.5%, sidewalk 34.1%, other-ground 15.6% |
| building | 428,777 | building 53.8%, vegetation 29.7%, other-ground 5.5% |
| car | 307,499 | car 55.6%, parking 18.9%, road 16.7% |
| parking | 76,823 | road 36.9%, sidewalk 22.3%, other-ground 19.9% |
| fence | 51,404 | other-ground 19.6%, sidewalk 19.2%, fence 18.5% |

### Occ3D-nuScenes (B-D)

| class | SSC IoU | TP-cond. recall | GT occupied voxels | co-occupied voxels |
|---|---:|---:|---:|---:|
| others | 0.0005 | 0.0840 | 20,611 | 3,966 |
| barrier | 0.0188 | 0.4653 | 38,980 | 9,222 |
| bicycle | 0.0175 | 0.1799 | 3,775 | 917 |
| bus | 0.0737 | 0.8060 | 70,640 | 17,783 |
| car | 0.0943 | 0.7159 | 328,038 | 98,901 |
| construction_vehicle | 0.0287 | 0.4767 | 14,465 | 1,741 |
| motorcycle | 0.0498 | 0.3591 | 5,817 | 1,395 |
| pedestrian | 0.0113 | 0.2279 | 40,056 | 10,264 |
| traffic_cone | 0.0314 | 0.2824 | 8,199 | 2,341 |
| trailer | 0.0232 | 0.2284 | 38,423 | 6,139 |
| truck | 0.0819 | 0.6711 | 137,482 | 32,397 |
| driveable_surface | 0.1894 | 0.5383 | 3,853,177 | 1,397,391 |
| other_flat | 0.0080 | 0.1764 | 128,192 | 47,383 |
| sidewalk | 0.0808 | 0.3391 | 1,137,665 | 320,890 |
| terrain | 0.0047 | 0.0194 | 1,393,593 | 341,993 |
| manmade | 0.0140 | 0.1060 | 2,245,615 | 311,486 |
| vegetation | 0.0878 | 0.7753 | 2,492,862 | 383,685 |

| ground-truth class | co-occupied voxels | most frequent predictions |
|---|---:|---|
| driveable_surface | 1,397,391 | driveable_surface 53.8%, other_flat 17.2%, others 11.5% |
| vegetation | 383,685 | vegetation 77.5%, others 8.2%, other_flat 7.0% |
| terrain | 341,993 | other_flat 56.5%, vegetation 11.6%, sidewalk 10.8% |
| sidewalk | 320,890 | sidewalk 33.9%, other_flat 28.1%, driveable_surface 12.5% |
| manmade | 311,486 | other_flat 27.5%, others 17.9%, vegetation 13.6% |
| car | 98,901 | car 71.6%, driveable_surface 7.5%, truck 5.1% |
| other_flat | 47,383 | driveable_surface 38.0%, sidewalk 20.1%, other_flat 17.6% |
| truck | 32,397 | truck 67.1%, car 9.2%, driveable_surface 5.6% |

### SSCBench-KITTI-360 (B-D)

| class | SSC IoU | TP-cond. recall | GT occupied voxels | co-occupied voxels |
|---|---:|---:|---:|---:|
| car | 0.0553 | 0.5605 | 3,020,969 | 567,150 |
| bicycle | 0.0099 | 0.2117 | 3,404 | 718 |
| motorcycle | 0.0067 | 0.0618 | 15,711 | 2,831 |
| truck | 0.0421 | 0.6629 | 150,807 | 28,745 |
| other-vehicle | 0.0123 | 0.1160 | 275,795 | 68,795 |
| person | 0.0139 | 0.2987 | 29,040 | 4,868 |
| road | 0.0494 | 0.4750 | 17,467,826 | 2,151,871 |
| parking | 0.0098 | 0.0647 | 4,266,993 | 803,539 |
| sidewalk | 0.0262 | 0.1487 | 6,076,908 | 1,309,624 |
| other-ground | 0.0072 | 0.0674 | 2,375,824 | 465,463 |
| building | 0.0761 | 0.5708 | 26,930,085 | 4,887,285 |
| fence | 0.0191 | 0.2241 | 1,414,874 | 281,614 |
| vegetation | 0.1096 | 0.8443 | 63,539,390 | 11,134,725 |
| terrain | 0.0006 | 0.0034 | 5,092,811 | 895,718 |
| pole | 0.0070 | 0.3034 | 175,378 | 25,675 |
| traffic-sign | 0.0044 | 0.1781 | 47,528 | 5,172 |
| other-structure | 0.0236 | 0.2502 | 6,591,270 | 1,139,239 |
| other-object | 0.0052 | 0.0694 | 366,858 | 60,756 |

| ground-truth class | co-occupied voxels | most frequent predictions |
|---|---:|---|
| vegetation | 11,134,725 | vegetation 84.4%, building 5.5%, other-structure 2.5% |
| building | 4,887,285 | building 57.1%, other-structure 30.8%, vegetation 7.1% |
| road | 2,151,871 | road 47.5%, vegetation 16.7%, car 13.0% |
| sidewalk | 1,309,624 | vegetation 31.0%, sidewalk 14.9%, road 13.3% |
| other-structure | 1,139,239 | vegetation 31.8%, other-structure 25.0%, building 21.0% |
| terrain | 895,718 | vegetation 48.8%, other-ground 15.0%, building 10.4% |
| parking | 803,539 | vegetation 27.6%, building 12.3%, car 11.3% |
| car | 567,150 | car 56.0%, vegetation 12.4%, other-vehicle 8.3% |

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

| benchmark | band | coverage miss | naming error | correct | semantic mIoU |
|---|---|---:|---:|---:|---:|
| SemanticKITTI 08 | 0-10m | 0.256 | 0.264 | 0.479 | 0.0838 |
|  | 10-20m | 0.515 | 0.211 | 0.274 | 0.0579 |
|  | 20-30m | 0.688 | 0.157 | 0.155 | 0.0412 |
|  | 30-40m | 0.820 | 0.088 | 0.092 | 0.0260 |
| Occ3D-nuScenes | 0-10m | 0.488 | 0.301 | 0.211 | 0.0661 |
|  | 10-20m | 0.640 | 0.193 | 0.167 | 0.0596 |
|  | 20-30m | 0.777 | 0.122 | 0.101 | 0.0480 |
|  | 30-40m | 0.898 | 0.060 | 0.042 | 0.0223 |
| SSCBench-KITTI-360 | 0-10m | 0.394 | 0.232 | 0.374 | 0.0601 |
|  | 10-20m | 0.712 | 0.123 | 0.165 | 0.0258 |
|  | 20-30m | 0.918 | 0.033 | 0.049 | 0.0084 |
|  | 30-40m | 0.985 | 0.006 | 0.009 | 0.0017 |

| benchmark | height band (m) | coverage miss | naming error | correct |
|---|---|---:|---:|---:|
| SemanticKITTI 08 | [-2, 0) | 0.623 | 0.186 | 0.190 |
|  | [0, 1) | 0.834 | 0.033 | 0.132 |
|  | [1, 2) | 0.860 | 0.023 | 0.117 |
|  | [2, 6) | 0.852 | 0.021 | 0.127 |
| Occ3D-nuScenes | [-2, 0) | 0.675 | 0.172 | 0.153 |
|  | [0, 1) | 0.681 | 0.194 | 0.125 |
|  | [1, 2) | 0.847 | 0.090 | 0.063 |
|  | [2, 6) | 0.866 | 0.057 | 0.077 |
| SSCBench-KITTI-360 | [-2, 0) | 0.810 | 0.103 | 0.088 |
|  | [0, 1) | 0.841 | 0.046 | 0.113 |
|  | [1, 2) | 0.839 | 0.044 | 0.118 |
|  | [2, 6) | 0.844 | 0.037 | 0.118 |

Coverage miss rises monotonically with range on all three benchmarks while naming error
falls, because the far field contains almost no reconstructed voxels to misname.

**The global diagnosis reverses in the near field of SemanticKITTI.** In the 0–10 m band
at B-D, naming error is **larger** than coverage miss — 0.264 against 0.256 — so at short
range on that benchmark the binding constraint is naming, not coverage. This is the one
place in the gate where `COVERAGE_DOMINATES` does not hold, and it is stated rather than
averaged away.

| benchmark | 0–10 m valid GT occupied | coverage miss | naming error | correct | larger at 0–10 m |
|---|---:|---:|---:|---:|---|
| SemanticKITTI 08 | 1,674,882 | 0.2563 | 0.2643 | 0.4794 | **naming** |
| Occ3D-nuScenes | 1,333,347 | 0.4880 | 0.3007 | 0.2113 | coverage |
| SSCBench-KITTI-360 | 17,287,132 | 0.3935 | 0.2323 | 0.3742 | coverage |

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

| benchmark | metric | real | permutation mean | permutation p95 | permutations beating real | real percentile | passes |
|---|---|---:|---:|---:|---:|---:|---|
| SemanticKITTI 08 | full SSC mIoU | **0.0511** | 0.0044 | 0.0131 | 0 / 100 | 100 | yes |
|  | TP-conditioned balanced recall | **0.3551** | 0.0508 | 0.0901 | 0 / 100 | 100 | yes |
| Occ3D-nuScenes | full SSC mIoU | **0.0480** | 0.0059 | 0.0136 | 0 / 100 | 100 | yes |
|  | TP-conditioned balanced recall | **0.3795** | 0.0615 | 0.1112 | 0 / 100 | 100 | yes |
| SSCBench-KITTI-360 | full SSC mIoU | **0.0266** | 0.0037 | 0.0081 | 0 / 100 | 100 | yes |
|  | TP-conditioned balanced recall | **0.2839** | 0.0570 | 0.0983 | 0 / 100 | 100 | yes |

**Not one of the 600 permuted evaluations (100 × 2 metrics × 3 benchmarks) reaches the
real vocabulary's value.** The real result sits at the 100th percentile everywhere.

---

## 14. Bootstrap confidence intervals

10,000 resamples, seed 0, paired, over the benchmark's own unit: 150 official scenes on
Occ3D-nuScenes, 8 contiguous blocks of 20 clips on SemanticKITTI sequence 08, and the 87
Gate-5.2 contiguous blocks of 20 clips on KITTI-360 sequence 06. **The blocks come from a
single drive and are not independent scenes; they are not described as such.**

| benchmark | metric | B-R | B-D | B-D − B-R | 95% CI | excludes 0 |
|---|---|---:|---:|---:|---|---|
| SemanticKITTI 08 | full SSC mIoU | 0.0281 | 0.0511 | +0.0230 | [+0.0188, +0.0257] | yes |
|  | TP-conditioned accuracy | 0.6326 | 0.5592 | -0.0733 | [-0.0854, -0.0603] | yes |
|  | coverage-miss fraction | 0.9234 | 0.6978 | -0.2256 | [-0.2504, -0.2006] | yes |
|  | naming-error fraction | 0.0281 | 0.1332 | +0.1051 | [+0.0787, +0.1321] | yes |
| Occ3D-nuScenes | full SSC mIoU | 0.0245 | 0.0480 | +0.0235 | [+0.0213, +0.0256] | yes |
|  | TP-conditioned accuracy | 0.4483 | 0.4431 | -0.0053 | [-0.0180, +0.0079] | no |
|  | coverage-miss fraction | 0.9168 | 0.7501 | -0.1666 | [-0.1779, -0.1556] | yes |
|  | naming-error fraction | 0.0459 | 0.1392 | +0.0932 | [+0.0869, +0.0997] | yes |
| SSCBench-KITTI-360 | full SSC mIoU | 0.0088 | 0.0266 | +0.0178 | [+0.0166, +0.0188] | yes |
|  | TP-conditioned accuracy | 0.6245 | 0.5959 | -0.0286 | [-0.0332, -0.0236] | yes |
|  | coverage-miss fraction | 0.9614 | 0.8271 | -0.1343 | [-0.1396, -0.1292] | yes |
|  | naming-error fraction | 0.0145 | 0.0699 | +0.0554 | [+0.0516, +0.0592] | yes |

All reported differences except Occ3D-nuScenes TP-conditioned accuracy ([-0.0180, +0.0079]) exclude zero (11 of 12). Dilation buys a large, unambiguous gain in full SSC mIoU and a
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

| benchmark | file opens observed | unique paths | forbidden accesses | prediction pinned before targets |
|---|---:|---:|---:|---|
| SemanticKITTI 08 | 1,962 | 983 | **0** | yes |
| Occ3D-nuScenes | 14,189 | 7,096 | **0** | yes |
| SSCBench-KITTI-360 | 21,041 | 3,534 | **0** | yes |

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

| quantity | value |
|---|---:|
| clips re-run in float32 | 5 |
| max abs probability error | 2.44e-04 |
| raw voxels compared | 92,510 |
| raw label disagreements | 2 (2.16e-05) |
| dilated voxels compared | 482,908 |
| dilated label disagreements | 4 (8.28e-06) |
| predicted occupancy identical | yes |

Two label changes in 92,510 raw voxels and four in 482,908 dilated ones, with predicted
occupancy bit-identical — occupancy cannot depend on the teacher at all, and the test
asserts it.

### Decision-rule audit

| benchmark | 1. mIoU > perm p95 | 2. TP-bal > perm p95 | 3. target-independent | 4. mapping/eval tests | passes |
|---|---|---|---|---|---|
| SemanticKITTI 08 | yes | yes | yes | yes | **yes** |
| Occ3D-nuScenes | yes | yes | yes | yes | **yes** |
| SSCBench-KITTI-360 | yes | yes | yes | yes | **yes** |

Three of three benchmarks pass → `FROZEN_TRIDENT_SEMANTICS_TRANSFER`. B-D coverage miss
exceeds naming error on three of three (the rule required two) →`COVERAGE_DOMINATES`. The
branches do not overlap: `PARTIAL` and `FAIL` each require at least one dataset to fail,
and none does; `NAMING_DOMINATES` and `MIXED_BOTTLENECK` each require at least two
datasets where naming exceeds coverage, and there are none.

---

## 16. Optional OccAny-aligned comparator

Reproduced on **SemanticKITTI sequence 08** (163 clips), the
smallest of the three benchmarks. **Both readouts are lifted through the identical frozen
geometry, on the identical clips, and scored by the identical evaluator and mask**, so the
only thing that differs is the 2D semantic readout. This was run *after* the primary
predictions were SHA-256-pinned and was not used to select, tune or modify anything.

**Read the table with three limits attached, stated before the numbers rather than after
them.** This is a **one-dataset** comparison — SemanticKITTI sequence 08 only, one drive,
163 clips. **No paired confidence interval was computed for it**, so no
difference in the table is known to be distinguishable from resampling noise; every other
comparison in this report carries a 10,000-resample paired interval and this one does not.
And the two readouts were **not given the same vocabulary**: OccAny prompts with per-class
**synonym lists** while the Gate-6 protocol mandated a **single deterministic phrase per
class**, so the comparison confounds the choice of teacher with the choice of prompt. The
row below is a measurement under our protocol, not a ranking of the two teachers.

| 2D readout | binary IoU (mean-per-clip) | SSC mIoU (pooled) | TP-cond. acc | TP-bal. recall | coverage miss | naming error | frozen voxels named |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Trident-H** (primary, predeclared) | 0.1584 | 0.0511 | 0.5592 | 0.3551 | 0.6978 | 0.1332 | 100.0 % |
| OccAny-aligned Grounded-SAM-2 (comparator) | 0.1507 | 0.0554 | 0.7344 | 0.3881 | 0.7199 | 0.0744 | 93.5 % |

**The comparator is more accurate where it names, and names less.** Grounded-SAM-2 is
detection-driven: Grounding DINO proposes a median of
151 boxes per frame, SAM 2.1 turns them into
masks, and OccAny paints the highest-confidence masks first without overwriting. That
covers a median 95.8 % of pixels and
93.5 % of the frozen occupied voxels; the rest
are left unnamed and are scored as empty, which is why its binary IoU
(0.1507) sits below the frozen 0.1584.
Where it does commit to a class it is right
73.4 % of the time against Trident's
55.9 %, and its full SSC mIoU is correspondingly
higher (0.0554 vs 0.0511).

This result does **not** favour the predeclared primary, and it is reported as measured.
Three things must be said with it, none of which are excuses and all of which are
verifiable in the code:

* **The two readouts were not given the same vocabulary.** The Gate-6 protocol mandated
  *one deterministic phrase per class, no synonyms, no alternatives evaluated* — so Trident
  ran on `bicycle`, `vegetation`, `terrain`. OccAny's own prompts are per-class **synonym
  lists**: `bicycle; bike`, `vegetation; bush; shrub; foliage`, `terrain; grass; soil`,
  `motorcycle; motorbike; scooter`. Those synonyms attack exactly the ontology-boundary
  classes §11 identifies as the dominant naming failure. The gap therefore confounds
  "Grounded-SAM vs Trident" with "synonym lists vs single phrases", and this experiment
  cannot separate them.
* **The two are not doing the same job.** A detector-driven readout answers "what did I
  detect and where"; Trident answers "which of these classes is this pixel" for every
  pixel. Declining to name 6.5 % of the occupied voxels is not free — those voxels become
  coverage misses, which is why the comparator's coverage-miss fraction
  (0.7199) is *higher* than Trident's
  (0.6978) even though its naming error is
  lower.
* **It is one benchmark, and it carries no interval.** SemanticKITTI 08 only, one drive,
  8 bootstrap blocks, and **no paired confidence interval was computed for the
  comparator** — so none of the differences above is known to exceed resampling noise.
  Producing one would require re-running the comparator's per-clip count blocks through
  the Gate-6 bootstrap, which was not done.

**The bottleneck diagnosis is unchanged and is now teacher-independent.** For the
comparator too, coverage miss (0.7199) exceeds
naming error (0.0744) by roughly ten to one.
Swapping the frozen 2D teacher for a stronger-when-it-commits alternative moves the SSC
mIoU by +0.0042 and leaves seven of every ten occupied voxels
still missing. That is the most useful thing the comparator says, and it argues the
recommended next experiment (§18) even more strongly than the primary result alone does.

**No claim of superiority over OccAny, in either direction, is made.** OccAny's
reconstruction, grid, evaluation mask and protocol are not reproduced here — only its
semantic-mask procedure, transplanted onto our frozen occupancy. The row above compares two
*2D readouts under our protocol*, nothing more.

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

**Created — artifacts** (`artifacts/gate6/`, 7 MB): the stage-0 audit, teacher
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
cache, **7 MB** artifacts in the repository. Model weights: 3.94 GB (CLIP ViT-H/14),
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

**C1 — SAM 3 is gated; the Grounded-SAM-2 branch was used.** OccAny's newer
semantic path loads `facebook/sam3`, a **manually gated** Hugging Face repository. No
credentials are configured on this machine and the brief forbids bypassing authentication
or licensing, so that branch is unavailable. The brief names Grounded-SAM2 as an accepted
alternative and that is what ran.

**C2 — OccAny's `occany/model/model_sam2.py` cannot be imported as released.** Line 26 does
`from PIL import Image` (binding the *module*) while line 521 writes
`Union[np.ndarray, Image]` and line 537 `isinstance(image, Image)`, both of which require
the *class* — which is what upstream sam2 binds (`from PIL.Image import Image`). Under
Python 3.10 the module raises `TypeError` on import. Binding the class is the only reading
under which the file runs, so it is a repair with no methodological content. **The upstream
file was not edited**: everything `model_sam2` imports before line 26 is pre-warmed, a shim
of the `PIL` package whose `Image` is the class is installed for the duration of that single
import, and it is removed immediately afterwards so no other module ever sees it.

**C3 — OccAny's vendored dust3r requires Python ≥ 3.11.** `occany/datasets/__init__.py`
unconditionally re-exports dust3r, whose `datasets/base/batched_sampler.py:290` contains
`np.c_[sample_idxs, *idxs]` — a starred expression inside a subscript, a **SyntaxError**
before Python 3.11. The entire frozen Gate-1..6 stack runs on Python 3.10, so that package
cannot be imported here at all. Only a literal constant was needed from it
(`KITTI_CLASS_PROMPTS`), so it is parsed straight out of the source with
`ast.literal_eval` — byte-identical data, no import, no edit.

**C4 — GroundingDINO's CUDA extension had to be rebuilt for this GPU.** Its `setup.py`
hardcodes `-gencode` for sm_70/75/80/86 with no PTX and ignores `TORCH_CUDA_ARCH_LIST`, so
on these sm_120 Blackwell cards the kernel fails at runtime with
`no kernel image is available for execution on the device` — and it fails *silently*,
printing to stderr while returning garbage, because `ms_deform_attn.py:331` takes the CUDA
branch whenever the tensor is on CUDA and the PyTorch reference path is unreachable. Three
`-gencode` lines for `compute_120` were added to that vendored `setup.py`, the extension
rebuilt, and **the file restored to its original contents**; only the compiled `.so`
differs from a stock checkout. This is a build-configuration change, not a method change.

**C5 — full-extent resize instead of OccAny's principal-point crop.** OccAny crops around
the principal point before rescaling, a crop tied to its own reconstruction. Using it would
misalign the readout with the frozen LingBot lattice this gate is required to keep, so the
native image is resized full-extent to OccAny's inference resolution (long side 1216)
instead. This is also the fairer comparator setup: both teachers then see the same pixels.
Everything else — prompts, `box_threshold = 0.1`, `text_threshold = 0.0`, SAM 2.1 Hiera-L,
`infer_semantic` itself and its confidence-ordered non-overwriting overlap rule — is
OccAny's own code, called unchanged.

**C6 — hard labels, so the voxel rule is a majority vote.** The readout emits integer
labels, not scores. Writing each pixel as a one-hot vector makes the frozen fusion (mean,
then argmax) exactly a majority vote among contributing pixels, so the identical lifting
code is reused. An extra channel represents "the teacher detected nothing here", making
*unnamed* a legitimate outcome rather than an arbitrary fallback class; those voxels are
scored as empty and their rate is reported explicitly above.

---

*Gate 6 complete. Diagnosis `FROZEN_TRIDENT_SEMANTICS_TRANSFER` + `COVERAGE_DOMINATES`.
Report generated by `tools/gate6/report_tables.py`; every number above is substituted from
`artifacts/gate6/` rather than transcribed.*
