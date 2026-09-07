# Gate 7A — Completion reachability and oracle-envelope diagnosis

**Date:** 2026-09-02 · **Status:** complete · **Nothing was trained.** No optimizer was
constructed, no backward pass was run, no network was created, and no radius was tuned on
any target benchmark. LingBot-Map, MoGe-2 and Trident were not re-run and not modified.

> **Gate 7A is a target-dependent oracle analysis.** The oracle envelope uses the ground
> truth to decide which voxels are added. It is a *ceiling*, not a prediction, and nothing
> in this report may be consumed as one. Gate 6's target-free claim is untouched and its
> prediction rollup hashes are recorded before and after (§3).

---

## 1. Executive diagnosis

# `COVERAGE_IS_NOT_LOCALLY_REACHABLE`  (3 of 3)
# `MISSES_ARE_IN_FRUSTUM_BUT_NOT_IN_DEPTH`  (2 of 3; Occ3D-nuScenes is the exception)
# `DILATION_GAIN_IS_NOT_PORTABLE`  (mixed by dataset, and said so)

| benchmark | clips | B-D binary IoU | B-D SSC mIoU | misses within 0.4 m | misses within 4.0 m | median miss distance | oracle mIoU @4 m | dilation mIoU @0.4 m | in-frustum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SemanticKITTI 08 | 163 | 0.1600 | 0.0511 | 0.101 | 0.551 | 3.45 m | 0.0704 | 0.0443 | 0.917 |
| Occ3D-nuScenes | 1182 | 0.2094 | 0.0480 | 0.116 | 0.498 | 4.05 m | 0.0848 | 0.0531 | 0.634 |
| SSCBench-KITTI-360 | 1753 | 0.1321 | 0.0266 | 0.082 | 0.431 | 5.45 m | 0.0572 | 0.0311 | 0.965 |

**The missing occupancy Gate 6 identified is not recoverable by a small local voxel
prior.** Three measurements say so, and they agree on all three benchmarks:

1. **The misses are not near the support.** Only 8.2–11.6 % of B-D coverage misses lie
   within 0.4 m of a frozen occupied voxel, and the median miss sits 3.5–5.5 m away
   — 17 to 27 voxels on the KITTI-family grid. A local prior would not be *completing*
   these voxels, it would be inventing them (§4).
2. **Even a perfect local corrector buys little semantics.** The oracle — which is
   permitted to add ground-truth-occupied voxels for free and never adds a false positive —
   roughly doubles binary IoU at a 4 m radius, but lifts full SSC mIoU only from
   0.0511 / 0.0480 / 0.0266 to 0.0704 / 0.0848 / 0.0572. That is the *ceiling*, with the target handed to it
   and no precision charged for it (§5).
3. **Ordinary metric dilation — the deployable version of the same move — helps a little on 2 of 3 benchmarks and hurts on the other, and it never comes close to the oracle.** Its best mIoU gain anywhere is +0.0051 (Occ3D-nuScenes, at 0.4 m) / +0.0062 (SSCBench-KITTI-360, at 1.2 m), against -0.0068 on SemanticKITTI 08 at the same first step. **The gain does not port across benchmarks**, which is the strongest argument against adopting it as a fix (§6, §9).

**On two of three benchmarks the evidence is not missing from the images.**
91.7 % / 63.4 % / 96.5 % of the B-D coverage misses project into at least one of the five input images. Taken over **all** misses (not only the in-frustum ones), 57.6 % / 10.7 % / 37.5 % lie **behind** the predicted surface and 32.0 % / 38.7 % / 51.7 % project onto pixels where the frozen confidence-and-range gate left no valid predicted depth at all. On SemanticKITTI and SSCBench-KITTI-360 what fails is not the field of
view but the *depth* along rays the camera did see, and that distinction is what separates
"accept the five-frame ceiling" from "add ray-aligned evidence". **Occ3D-nuScenes is
different and is not averaged in:** its evaluation volume spans ±40 m in both horizontal
axes against a single front camera, so a third of its ground truth was never observable
from the five input frames at all. That share is a hard ceiling no completion method can
cross (§8).

**Propagated semantics degrade but do not collapse.** A teacher label carried 2–4 m
through empty space is still right 32 % / 23 % / 46 % of the time against a uniform chance
of 5–6 % (§7). They are not the thing that breaks — but they are not free either, and they
are why the oracle roughly doubles binary IoU while barely doubling mIoU.

**Do not read this as "coverage dominates, therefore train a local 3D CNN."** That is the
inference Gate 7A was built to test, and it does not survive contact with the distances.

---

## 2. The corrected Gate-6 reporting

Seven reporting issues in the Gate-6 report were fixed **in the generating source** — the
template and the table generator — and the report was regenerated from the unchanged
artifacts. No number was edited in the rendered Markdown, and no Gate-6 numerical
prediction, count block, summary or diagnosis was altered.

| # | correction | where the fix was made |
|---|---|---|
| 1 | §14 no longer claims that every bootstrap interval excludes zero. The statement is now generated from the artifacts and names the exception, the Occ3D-nuScenes TP-conditioned accuracy difference, with its interval; the surrounding prose now reads that drop as not distinguishable from zero on that benchmark. | `report_tables.t_bootstrap_zero_note` (new generator) + template §14 |
| 2 | The 'Official SSC metrics' table now reports **pooled** binary IoU, precision and recall alongside its pooled semantic metrics, with every column labelled by aggregation. The values are computed from the stored counts, not transcribed. | `report_tables.t_main` |
| 3 | The reproduction/sanity table keeps the **mean-per-clip** binary IoU the frozen gates were pinned with, labels it as such, and shows the pooled value beside it for reference. | `report_tables.t_sanity` |
| 4 | `COVERAGE_DOMINATES` is now explicitly scoped to the population of valid ground-truth **occupied** voxels. §1 and §9 state that the partition excludes false-positive occupied predictions entirely and therefore does not establish that expanding occupancy would raise IoU. | template §1, §2, §9 |
| 5 | §12 states the near-field reversal outright: in SemanticKITTI's 0–10 m band at B-D, naming error (0.264) exceeds coverage miss (0.256). A generated table shows the 0–10 m band of all three benchmarks. | `report_tables.t_near_range_note` (new generator) + template §12 |
| 6 | The Grounded-SAM-2 comparator now carries its limits **before** the table: one dataset, one drive, no paired confidence interval, and a vocabulary confound (OccAny synonym lists against Gate 6's single deterministic phrase per class). | `report_tables.t_occany_section` |
| 7 | Trident's spatial encoder is named as **DINO v1 ViT-B/16, not DINOv2**, in the provenance table and in the operating-point listing, not only in the deviation note. | `report_tables.t_provenance` + template §4 |

| item | value |
|---|---|
| report | `reports/gate6/frozen_trident_semantic_lifting.md` |
| SHA-256 before | `14fbf23e12968bb537332199bd89e65d7d71f14339077e4f359a11651bfa1bf2` |
| SHA-256 after | `3724fbdce4a8d554bc155efd158d5d9007830ab8294a2a94475d39181c625c5e` |
| lines before → after | 956 → 1016 |
| template SHA-256 before | `2fa98446d5af9ee55d57eec9eaf4e36a3761d261c2fb0348f8503a10383e69c6` |
| template SHA-256 after | `0a8b47db3fd1844286ae2d8ac12766630206a6120a007981b218c507d1558008` |
| generator SHA-256 before | `d31155d1063282798bc378c680e9eec637455298b70642ce408ce400252bc1fe` |
| generator SHA-256 after | `44c9a11fd0635d7f4a846e034cb608a8e3b2431092a24c490d7b1139c1e1e5a8` |
| Gate-6 numerical artifacts changed | none — all 44 byte-identical |

All 44 Gate-6 artifact files are byte-identical to their stage-0 record, so the
corrections provably changed prose and table *rendering* only. Two of them matter for what
follows: `COVERAGE_DOMINATES` is now explicitly scoped to ground-truth-**occupied** voxels
and stated not to prove that expanding occupancy raises IoU — which is exactly the question
this gate answers, and answers with a firm no for a local prior and a qualified,
benchmark-dependent no for ordinary dilation.

---

## 3. Frozen inputs and provenance

| item | value |
|---|---|
| repository commit | `9a648322e6dc` (branch `dev`) |
| Gate-7A precommit | `configs/gate7a/completion_reachability_precommit.yaml` |
| Gate-7A precommit SHA-256 | `2da49cc6b266b2fc22cddfa743f0cd54f37005c72c9da8cb1a281c0885f37a4e` |
| Gate-6 precommit SHA-256 | `97a025ecdd0cbe616a0128ef13295c5ada5b28796ecef556ac339e0bfa04ac4d` (unchanged) |
| SemanticKITTI 08 prediction rollup SHA-256 | `f8f8a66e9d66aa9d3dc641352e4417d8bad6087e164005da3ca14016be23abc5` (163 files) |
| Occ3D-nuScenes prediction rollup SHA-256 | `32fbe8c6b6677ec680597c3b85782dda60460275f4a8917b30fc6b4c5a634782` (1182 files) |
| SSCBench-KITTI-360 prediction rollup SHA-256 | `2d6effe3b0e6b351f7a347821f631e6bc0c5a9daf05fb004123231221c28bbbf` (1753 files) |
| pre-existing dirty file `lingbot_map/models/gct_stream.py` | `e1f457088894c156…` (23,445 bytes), preserved |
| pre-existing dirty file `research/sem_bypass/model.py` | `51ce29bf5a3a6abc…` (6,405 bytes), preserved |
| SemanticKITTI 08 clips verified against the pinned prediction | 163 / 163 (geometry bit-exact) |
| Occ3D-nuScenes clips verified against the pinned prediction | 1182 / 1182 (geometry bit-exact) |
| SSCBench-KITTI-360 clips verified against the pinned prediction | 1753 / 1753 (geometry bit-exact) |
| worst camera-chain round-trip error | 1.08e-12 px |

**The frozen state was recovered, not re-run.** Gate 6 stored the argmax channel of each
occupied voxel, not the fused probability vector, and Gate 7A needs the vectors to
propagate them. They were recomputed from the same three frozen inputs Gate 6 used — the
LingBot cache, the G51-B scale table and the Trident cache — and then checked against the
pinned prediction file. **Every clip of all three benchmarks reproduced the B-R and B-D
geometry bit-for-bit**: identical flat indices in identical order, identical support flags.

| benchmark | base | Gate 6 pooled binary IoU | Gate 7A | Gate 6 pooled SSC mIoU | Gate 7A | Gate 6 mean-per-clip binary IoU | Gate 7A | identical |
|---|---|---:|---:|---:|---:|---:|---:|---|
| SemanticKITTI 08 | B-R | 0.067000 | 0.067000 | 0.028087 | 0.028087 | 0.067469 | 0.067469 | yes |
| SemanticKITTI 08 | B-D | 0.159965 | 0.159965 | 0.051112 | 0.051112 | 0.158405 | 0.158405 | yes |
| Occ3D-nuScenes | B-R | 0.079349 | 0.079349 | 0.024496 | 0.024496 | 0.079924 | 0.079924 | yes |
| Occ3D-nuScenes | B-D | 0.209429 | 0.209429 | 0.047993 | 0.047993 | 0.206968 | 0.206968 | yes |
| SSCBench-KITTI-360 | B-R | 0.036259 | 0.036259 | 0.008789 | 0.008789 | 0.037629 | 0.037629 | yes |
| SSCBench-KITTI-360 | B-D | 0.132077 | 0.132077 | 0.026588 | 0.026588 | 0.134171 | 0.134171 | yes |

**The camera-frame transformations were verified, not assumed.** For every clip the frozen
reconstruction's own points were pushed back through the whole chain — grid → anchor camera
→ frame *f* → pixel — and compared with the pixel each point came from. The worst error
over all three benchmarks is 1.1e-12 px. A mistake in the anchor convention, in the
pose scaling or in the intrinsics would show up as a whole pixel.

| benchmark | clips | clips with an empty B-R | clips with an empty B-D |
|---|---:|---:|---:|
| SemanticKITTI 08 | 163 | 0 | 0 |
| Occ3D-nuScenes | 1182 | 1 | 1 |
| SSCBench-KITTI-360 | 1753 | 0 | 0 |

---

## 4. How far is the missing occupancy? (miss-distance CDF and quantiles)

Exact Euclidean distance transform on the benchmark's **native** evaluation grid — 0.2 m
for the KITTI family, 0.4 m for the official Occ3D grid — converted to metres once, at the
end. Voxel indices are never treated as metres, and the radius test is done on integer
squared lattice distances because `1.2 / 0.2` is `5.999999999999999` in binary floating
point and a naive comparison silently drops the boundary shell.

| benchmark | base | valid GT occupied | true positives | coverage misses | mean | p25 | p50 | p75 | p90 | p95 | p99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SemanticKITTI 08 | B-R | 18,762,930 | 1,436,514 | 17,326,416 | 4.22 | 0.65 | 2.45 | 6.35 | 11.15 | 14.20 | 19.75 |
|  | **B-D** | 18,762,930 | 5,669,753 | 13,093,177 | 4.94 | 1.25 | 3.45 | 7.35 | 11.85 | 14.75 | 20.05 |
| Occ3D-nuScenes | B-R | 11,957,590 | 995,434 | 10,962,156 | 5.72 | 0.90 | 3.25 | 8.30 | 15.20 | 19.60 | 27.30 |
|  | **B-D** | 11,957,590 | 2,987,894 | 8,969,696 | 6.35 | 1.30 | 4.05 | 9.35 | 16.05 | 20.20 | 27.60 |
| SSCBench-KITTI-360 | B-R | 137,841,471 | 5,319,813 | 132,521,658 | 7.64 | 1.25 | 4.35 | 12.35 | 20.20 | 24.25 | 30.40 |
|  | **B-D** | 137,841,471 | 23,833,788 | 114,007,683 | 8.28 | 1.60 | 5.45 | 13.35 | 20.65 | 24.45 | 30.30 |

![miss distance CDF](../../artifacts/gate7a/fig_miss_distance_cdf.png)

| benchmark | base | quantity | 0 m | 0.4 m | 0.8 m | 1.2 m | 2 m | 4 m |
|---|---|---|---:|---:|---:|---:|---:|---:|
| SemanticKITTI 08 | B-R | fraction of misses reachable | 0.000 | 0.184 | 0.287 | 0.360 | 0.459 | 0.625 |
|  |  | fraction of all GT occupied recoverable | 0.077 | 0.246 | 0.342 | 0.409 | 0.500 | 0.654 |
|  | B-D | fraction of misses reachable | 0.000 | 0.101 | 0.182 | 0.250 | 0.357 | 0.551 |
|  |  | fraction of all GT occupied recoverable | 0.302 | 0.373 | 0.429 | 0.477 | 0.551 | 0.686 |
| Occ3D-nuScenes | B-R | fraction of misses reachable | 0.000 | 0.136 | 0.238 | 0.310 | 0.399 | 0.555 |
|  |  | fraction of all GT occupied recoverable | 0.083 | 0.208 | 0.302 | 0.368 | 0.449 | 0.592 |
|  | B-D | fraction of misses reachable | 0.000 | 0.116 | 0.181 | 0.240 | 0.331 | 0.498 |
|  |  | fraction of all GT occupied recoverable | 0.250 | 0.337 | 0.386 | 0.430 | 0.498 | 0.624 |
| SSCBench-KITTI-360 | B-R | fraction of misses reachable | 0.000 | 0.094 | 0.178 | 0.248 | 0.343 | 0.482 |
|  |  | fraction of all GT occupied recoverable | 0.039 | 0.129 | 0.210 | 0.277 | 0.368 | 0.502 |
|  | B-D | fraction of misses reachable | 0.000 | 0.082 | 0.151 | 0.209 | 0.294 | 0.431 |
|  |  | fraction of all GT occupied recoverable | 0.173 | 0.240 | 0.298 | 0.346 | 0.416 | 0.529 |

![reachable miss fraction](../../artifacts/gate7a/fig_reachable_miss_fraction.png)

| benchmark | range band | B-D misses | ≤ 0.4 m | ≤ 0.8 m | ≤ 1.2 m | ≤ 2 m | ≤ 4 m |
|---|---|---:|---:|---:|---:|---:|---:|
| SemanticKITTI 08 | 0-10m | 429,282 | 0.363 | 0.576 | 0.708 | 0.837 | 0.952 |
|  | 10-20m | 2,205,889 | 0.195 | 0.344 | 0.453 | 0.596 | 0.793 |
|  | 20-30m | 3,205,082 | 0.114 | 0.212 | 0.298 | 0.437 | 0.653 |
|  | 30-40m | 3,237,267 | 0.078 | 0.144 | 0.207 | 0.316 | 0.555 |
|  | 40-60m | 4,015,657 | 0.029 | 0.058 | 0.087 | 0.142 | 0.290 |
| Occ3D-nuScenes | 0-10m | 650,680 | 0.259 | 0.357 | 0.444 | 0.574 | 0.793 |
|  | 10-20m | 2,160,142 | 0.176 | 0.263 | 0.340 | 0.452 | 0.646 |
|  | 20-30m | 2,677,704 | 0.121 | 0.193 | 0.258 | 0.355 | 0.530 |
|  | 30-40m | 2,633,006 | 0.060 | 0.106 | 0.151 | 0.226 | 0.380 |
|  | 40-60m | 848,164 | 0.016 | 0.032 | 0.050 | 0.083 | 0.165 |
| SSCBench-KITTI-360 | 0-10m | 6,802,622 | 0.405 | 0.626 | 0.758 | 0.893 | 0.984 |
|  | 10-20m | 25,218,757 | 0.187 | 0.356 | 0.492 | 0.666 | 0.872 |
|  | 20-30m | 30,227,457 | 0.051 | 0.110 | 0.174 | 0.288 | 0.521 |
|  | 30-40m | 25,849,583 | 0.010 | 0.023 | 0.038 | 0.071 | 0.166 |
|  | 40-60m | 25,909,264 | 0.001 | 0.002 | 0.003 | 0.006 | 0.017 |

**Reachability is strongly range-dependent, and averaging it away would hide the one place a local prior could work.** Inside 10 m, 84% / 57% / 89% of the B-D misses lie within 2 m of existing support; in the 40–60 m band the same figure is 14% / 8% / 1%. The near field is genuinely a completion problem — the reconstruction is there and full of holes. The far field is not: there is almost nothing to complete *from*. Any local corrector would therefore be a near-field-only device, and the near field is also where Gate 6 found naming, not coverage, to be the larger error on SemanticKITTI. The two findings point the same way: **the near field needs better labels, the far field needs more evidence, and neither wants a blind local occupancy prior.**

---

## 5. The oracle-local completion ceiling

`P_oracle(r) = P_base OR (GT_occupied AND valid AND distance_to_P_base <= r)`

It may add only true positives, it removes nothing, and **its false-positive count is
identical to the base's at every radius** — checked on every clip, not argued. It is the
most optimistic result any model confined to that local correction region could reach, and
it is not deployable.

| benchmark | base | radius | precision | recall | binary IoU | Δ IoU | SSC mIoU | Δ mIoU | TP recovered | FP added | added-volume ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SemanticKITTI 08 | B-R | 0 m | 0.3492 | 0.0766 | 0.0670 | +0.0000 | 0.0281 | +0.0000 | 0 | 0 | ×0.00 |
|  |  | 0.4 m | 0.6331 | 0.2462 | 0.2155 | +0.1485 | 0.0748 | +0.0467 | 3,183,478 | 0 | ×0.77 |
|  |  | 0.8 m | 0.7055 | 0.3419 | 0.2992 | +0.2322 | 0.0918 | +0.0637 | 4,979,052 | 0 | ×1.21 |
|  |  | 1.2 m | 0.7414 | 0.4092 | 0.3581 | +0.2911 | 0.1000 | +0.0719 | 6,241,636 | 0 | ×1.52 |
|  |  | 2 m | 0.7780 | 0.5002 | 0.4377 | +0.3707 | 0.1067 | +0.0787 | 7,948,226 | 0 | ×1.93 |
|  |  | 4 m | 0.8209 | 0.6539 | 0.5722 | +0.5052 | 0.1094 | +0.0813 | 10,832,346 | 0 | ×2.63 |
|  | B-D | 0 m | 0.2537 | 0.3022 | 0.1600 | +0.0000 | 0.0511 | +0.0000 | 0 | 0 | ×0.00 |
|  |  | **0.4 m** | 0.2953 | 0.3725 | **0.1972** | +0.0372 | **0.0581** | +0.0069 | 1,319,707 | 0 | ×0.06 |
|  |  | 0.8 m | 0.3256 | 0.4293 | 0.2273 | +0.0673 | 0.0626 | +0.0115 | 2,385,048 | 0 | ×0.11 |
|  |  | 1.2 m | 0.3492 | 0.4770 | 0.2525 | +0.0925 | 0.0655 | +0.0143 | 3,279,760 | 0 | ×0.15 |
|  |  | 2 m | 0.3827 | 0.5512 | 0.2918 | +0.1318 | 0.0682 | +0.0171 | 4,671,859 | 0 | ×0.21 |
|  |  | **4 m** | 0.4357 | 0.6865 | **0.3634** | +0.2034 | **0.0704** | +0.0193 | 7,210,901 | 0 | ×0.32 |
| Occ3D-nuScenes | B-R | 0 m | 0.6289 | 0.0832 | 0.0793 | +0.0000 | 0.0245 | +0.0000 | 0 | 0 | ×0.00 |
|  |  | 0.4 m | 0.8088 | 0.2078 | 0.1981 | +0.1187 | 0.0496 | +0.0251 | 1,489,526 | 0 | ×0.94 |
|  |  | 0.8 m | 0.8601 | 0.3019 | 0.2878 | +0.2084 | 0.0668 | +0.0423 | 2,614,398 | 0 | ×1.65 |
|  |  | 1.2 m | 0.8822 | 0.3678 | 0.3506 | +0.2712 | 0.0783 | +0.0538 | 3,402,324 | 0 | ×2.15 |
|  |  | 2 m | 0.9015 | 0.4495 | 0.4284 | +0.3491 | 0.0885 | +0.0641 | 4,378,953 | 0 | ×2.77 |
|  |  | 4 m | 0.9234 | 0.5920 | 0.5643 | +0.4850 | 0.0976 | +0.0731 | 6,083,891 | 0 | ×3.84 |
|  | B-D | 0 m | 0.5641 | 0.2499 | 0.2094 | +0.0000 | 0.0480 | +0.0000 | 0 | 0 | ×0.00 |
|  |  | **0.4 m** | 0.6358 | 0.3372 | **0.2826** | +0.0732 | **0.0608** | +0.0128 | 1,043,671 | 0 | ×0.20 |
|  |  | 0.8 m | 0.6663 | 0.3856 | 0.3232 | +0.1137 | 0.0675 | +0.0195 | 1,622,564 | 0 | ×0.31 |
|  |  | 1.2 m | 0.6902 | 0.4302 | 0.3606 | +0.1511 | 0.0729 | +0.0249 | 2,156,153 | 0 | ×0.41 |
|  |  | 2 m | 0.7206 | 0.4980 | 0.4174 | +0.2080 | 0.0786 | +0.0306 | 2,967,399 | 0 | ×0.56 |
|  |  | **4 m** | 0.7636 | 0.6238 | **0.5228** | +0.3134 | **0.0848** | +0.0369 | 4,470,661 | 0 | ×0.84 |
| SSCBench-KITTI-360 | B-R | 0 m | 0.3747 | 0.0386 | 0.0363 | +0.0000 | 0.0088 | +0.0000 | 0 | 0 | ×0.00 |
|  |  | 0.4 m | 0.6669 | 0.1289 | 0.1211 | +0.0849 | 0.0277 | +0.0189 | 12,451,811 | 0 | ×0.88 |
|  |  | 0.8 m | 0.7654 | 0.2101 | 0.1974 | +0.1611 | 0.0425 | +0.0337 | 23,640,020 | 0 | ×1.67 |
|  |  | 1.2 m | 0.8114 | 0.2771 | 0.2603 | +0.2241 | 0.0527 | +0.0439 | 32,876,073 | 0 | ×2.32 |
|  |  | 2 m | 0.8512 | 0.3683 | 0.3461 | +0.3098 | 0.0625 | +0.0537 | 45,453,382 | 0 | ×3.20 |
|  |  | 4 m | 0.8864 | 0.5024 | 0.4720 | +0.4357 | 0.0699 | +0.0611 | 63,931,719 | 0 | ×4.50 |
|  | B-D | 0 m | 0.3587 | 0.1729 | 0.1321 | +0.0000 | 0.0266 | +0.0000 | 0 | 0 | ×0.00 |
|  |  | **0.4 m** | 0.4375 | 0.2404 | **0.1836** | +0.0516 | **0.0358** | +0.0092 | 9,303,953 | 0 | ×0.14 |
|  |  | 0.8 m | 0.4906 | 0.2978 | 0.2274 | +0.0954 | 0.0425 | +0.0159 | 17,208,547 | 0 | ×0.26 |
|  |  | 1.2 m | 0.5282 | 0.3461 | 0.2644 | +0.1323 | 0.0470 | +0.0204 | 23,875,886 | 0 | ×0.36 |
|  |  | 2 m | 0.5738 | 0.4163 | 0.3180 | +0.1859 | 0.0520 | +0.0255 | 33,547,808 | 0 | ×0.50 |
|  |  | **4 m** | 0.6314 | 0.5295 | **0.4045** | +0.2724 | **0.0572** | +0.0306 | 49,151,939 | 0 | ×0.74 |

| benchmark | base | construction | recall monotone | IoU monotone | FP constant | r=0 reproduces base |
|---|---|---|---|---|---|---|
| SemanticKITTI 08 | B-R | oracle | yes | yes | yes | yes |
| SemanticKITTI 08 | B-R | morph | yes | no (expected for dilation) | — | yes |
| SemanticKITTI 08 | B-D | oracle | yes | yes | yes | yes |
| SemanticKITTI 08 | B-D | morph | yes | no (expected for dilation) | — | yes |
| Occ3D-nuScenes | B-R | oracle | yes | yes | yes | yes |
| Occ3D-nuScenes | B-R | morph | yes | no (expected for dilation) | — | yes |
| Occ3D-nuScenes | B-D | oracle | yes | yes | yes | yes |
| Occ3D-nuScenes | B-D | morph | yes | no (expected for dilation) | — | yes |
| SSCBench-KITTI-360 | B-R | oracle | yes | yes | yes | yes |
| SSCBench-KITTI-360 | B-R | morph | yes | yes | — | yes |
| SSCBench-KITTI-360 | B-D | oracle | yes | yes | yes | yes |
| SSCBench-KITTI-360 | B-D | morph | yes | yes | — | yes |

**Binary IoU and semantic mIoU respond very differently to the oracle, and the gap is the
point.** At 4 m the oracle roughly doubles to two-and-a-half-times binary IoU on all three
benchmarks, while full SSC mIoU rises by only +0.0193 / +0.0369 / +0.0306. The reason is in §7:
every recovered voxel is labelled by propagating the nearest frozen teacher vector, and
that propagation is right well under half the time at these distances. **Even perfect local
occupancy completion, with the target handed to it, leaves semantic occupancy roughly where
a doubling would leave it** — which is the single most important number in this report for
sizing any future training budget.

![binary IoU vs radius](../../artifacts/gate7a/fig_binary_iou_vs_radius.png)

![SSC mIoU vs radius](../../artifacts/gate7a/fig_ssc_miou_vs_radius.png)

---

## 6. Ordinary metric dilation: what the precision actually costs

`P_morph(r) = P_base OR (valid AND distance_to_P_base <= r)`

Deployable, non-learned, target-free — and the honest counterpart of the oracle. For base
B-D **these radii are incremental around the already-dilated B-D support**, not total
radius from B-R: 0.4 m here is roughly 0.8 m of total dilation around the raw
reconstruction.

| benchmark | base | radius | precision | recall | binary IoU | Δ IoU | SSC mIoU | Δ mIoU | TP recovered | FP added | added-volume ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SemanticKITTI 08 | B-R | 0 m | 0.3492 | 0.0766 | 0.0670 | +0.0000 | 0.0281 | +0.0000 | 0 | 0 | ×0.00 |
|  |  | 0.4 m | 0.2871 | 0.2462 | 0.1528 | +0.0858 | 0.0519 | +0.0238 | 3,183,478 | 8,794,434 | ×2.91 |
|  |  | 0.8 m | 0.2194 | 0.3419 | 0.1543 | +0.0873 | 0.0468 | +0.0187 | 4,979,052 | 20,143,416 | ×6.11 |
|  |  | 1.2 m | 0.1793 | 0.4092 | 0.1424 | +0.0754 | 0.0403 | +0.0122 | 6,241,636 | 32,460,831 | ×9.41 |
|  |  | 2 m | 0.1395 | 0.5002 | 0.1225 | +0.0555 | 0.0317 | +0.0036 | 7,948,226 | 55,188,342 | ×15.35 |
|  |  | 4 m | 0.1053 | 0.6539 | 0.0997 | +0.0327 | 0.0224 | -0.0057 | 10,832,346 | 101,612,858 | ×27.33 |
|  | B-D | 0 m | 0.2537 | 0.3022 | 0.1600 | +0.0000 | 0.0511 | +0.0000 | 0 | 0 | ×0.00 |
|  |  | **0.4 m** | 0.2018 | 0.3725 | **0.1506** | -0.0094 | **0.0443** | -0.0068 | 1,319,707 | 10,967,526 | ×0.55 |
|  |  | 0.8 m | 0.1700 | 0.4293 | 0.1387 | -0.0213 | 0.0386 | -0.0125 | 2,385,048 | 22,635,833 | ×1.12 |
|  |  | 1.2 m | 0.1488 | 0.4770 | 0.1280 | -0.0320 | 0.0339 | -0.0172 | 3,279,760 | 34,497,182 | ×1.69 |
|  |  | 2 m | 0.1249 | 0.5512 | 0.1134 | -0.0466 | 0.0278 | -0.0233 | 4,671,859 | 55,751,837 | ×2.70 |
|  |  | **4 m** | 0.1009 | 0.6865 | **0.0964** | -0.0636 | **0.0210** | -0.0301 | 7,210,901 | 98,152,681 | ×4.71 |
| Occ3D-nuScenes | B-R | 0 m | 0.6289 | 0.0832 | 0.0793 | +0.0000 | 0.0245 | +0.0000 | 0 | 0 | ×0.00 |
|  |  | 0.4 m | 0.6259 | 0.2078 | 0.1849 | +0.1055 | 0.0436 | +0.0191 | 1,489,526 | 898,102 | ×1.51 |
|  |  | 0.8 m | 0.5571 | 0.3019 | 0.2435 | +0.1641 | 0.0519 | +0.0274 | 2,614,398 | 2,282,546 | ×3.09 |
|  |  | 1.2 m | 0.4554 | 0.3678 | 0.2554 | +0.1761 | 0.0518 | +0.0273 | 3,402,324 | 4,671,843 | ×5.10 |
|  |  | 2 m | 0.3220 | 0.4495 | 0.2309 | +0.1515 | 0.0453 | +0.0208 | 4,378,953 | 10,730,952 | ×9.55 |
|  |  | 4 m | 0.2274 | 0.5920 | 0.1966 | +0.1173 | 0.0368 | +0.0123 | 6,083,891 | 23,463,937 | ×18.67 |
|  | B-D | 0 m | 0.5641 | 0.2499 | 0.2094 | +0.0000 | 0.0480 | +0.0000 | 0 | 0 | ×0.00 |
|  |  | **0.4 m** | 0.5075 | 0.3372 | **0.2540** | +0.0446 | **0.0531** | +0.0051 | 1,043,671 | 1,602,700 | ×0.50 |
|  |  | 0.8 m | 0.4313 | 0.3856 | 0.2556 | +0.0462 | 0.0517 | +0.0037 | 1,622,564 | 3,770,906 | ×1.02 |
|  |  | 1.2 m | 0.3577 | 0.4302 | 0.2427 | +0.0333 | 0.0481 | +0.0001 | 2,156,153 | 6,926,982 | ×1.71 |
|  |  | 2 m | 0.2767 | 0.4980 | 0.2164 | +0.0070 | 0.0420 | -0.0060 | 2,967,399 | 13,254,341 | ×3.06 |
|  |  | **4 m** | 0.2216 | 0.6238 | **0.1955** | -0.0139 | **0.0362** | -0.0118 | 4,470,661 | 23,886,634 | ×5.35 |
| SSCBench-KITTI-360 | B-R | 0 m | 0.3747 | 0.0386 | 0.0363 | +0.0000 | 0.0088 | +0.0000 | 0 | 0 | ×0.00 |
|  |  | 0.4 m | 0.3668 | 0.1289 | 0.1055 | +0.0692 | 0.0225 | +0.0138 | 12,451,811 | 21,807,883 | ×2.41 |
|  |  | 0.8 m | 0.3496 | 0.2101 | 0.1510 | +0.1148 | 0.0295 | +0.0207 | 23,640,020 | 45,006,091 | ×4.84 |
|  |  | 1.2 m | 0.3269 | 0.2771 | 0.1764 | +0.1402 | 0.0322 | +0.0234 | 32,876,073 | 69,773,890 | ×7.23 |
|  |  | 2 m | 0.2955 | 0.3683 | 0.1961 | +0.1599 | 0.0325 | +0.0238 | 45,453,382 | 112,151,061 | ×11.10 |
|  |  | 4 m | 0.2674 | 0.5024 | 0.2114 | +0.1752 | 0.0309 | +0.0221 | 63,931,719 | 180,829,136 | ×17.24 |
|  | B-D | 0 m | 0.3587 | 0.1729 | 0.1321 | +0.0000 | 0.0266 | +0.0000 | 0 | 0 | ×0.00 |
|  |  | **0.4 m** | 0.3416 | 0.2404 | **0.1643** | +0.0322 | **0.0311** | +0.0045 | 9,303,953 | 21,267,968 | ×0.46 |
|  |  | 0.8 m | 0.3207 | 0.2978 | 0.1826 | +0.0505 | 0.0326 | +0.0060 | 17,208,547 | 44,310,258 | ×0.93 |
|  |  | 1.2 m | 0.3034 | 0.3461 | 0.1928 | +0.0608 | 0.0327 | +0.0062 | 23,875,886 | 66,952,170 | ×1.37 |
|  |  | 2 m | 0.2829 | 0.4163 | 0.2026 | +0.0705 | 0.0321 | +0.0055 | 33,547,808 | 102,807,061 | ×2.05 |
|  |  | **4 m** | 0.2642 | 0.5295 | **0.2140** | +0.0819 | **0.0306** | +0.0041 | 49,151,939 | 160,668,976 | ×3.16 |

![precision-recall](../../artifacts/gate7a/fig_precision_recall.png)

**The deployable baseline behaves differently on every benchmark, and the spread is the finding.** This is what the Gate-6 coverage-versus-naming partition could not contain: it counts only ground-truth-occupied voxels, so the false positives an expansion adds are invisible to it. Here they are counted.

On SemanticKITTI 08 every radius lowers both metrics: binary IoU -0.0094 and mIoU -0.0068 at 0.4 m, falling to -0.0301 mIoU at 4 m. Every one of those intervals excludes zero (§9). Recall does rise; precision falls faster.

On Occ3D-nuScenes and SSCBench-KITTI-360 it *does* help, and that is reported as measured: Occ3D-nuScenes peaks at mIoU +0.0051 at 0.4 m, SSCBench-KITTI-360 peaks at mIoU +0.0062 at 1.2 m. The two behave differently even so — Occ3D-nuScenes is negative (-0.0118) by 4 m; SSCBench-KITTI-360 is still positive (+0.0041) by 4 m — so this is not one effect with three magnitudes, it is three different curves.

**Binary occupancy IoU and semantic mIoU disagree about dilation**, and the disagreement matters. Occ3D-nuScenes gains +0.0462 binary IoU at 0.8 m; SSCBench-KITTI-360 gains +0.0819 binary IoU at 4 m. Filling space around the reconstruction genuinely improves *where things are*; it improves *what they are* far less, because every added voxel inherits a propagated label that is right well under half the time (§7). Any future claim built on binary IoU alone would badly overstate what this move buys.

**In no case does dilation approach the oracle, and the comparison is made within each benchmark rather than across them.** Best deployable mIoU gain against that same benchmark's 4 m oracle gain: SemanticKITTI 08 +0.0000 of +0.0193 (0%); Occ3D-nuScenes +0.0051 of +0.0369 (14%); SSCBench-KITTI-360 +0.0062 of +0.0306 (20%). **Ordinary dilation does not capture most of the oracle gain on any benchmark**, which closes off the 'just dilate more' branch of the decision rule (§11).

**A consistency check that comes free with this table.** B-D is the frozen `dilate_r2`, a *Chebyshev* ball of two voxels; morphology at 0.4 m on B-R is the *Euclidean* ball of the same radius, which is a strict subset of it. Its recall must therefore land at or below B-D's, and it does on all three — SemanticKITTI 08 0.2462 vs 0.3022; Occ3D-nuScenes 0.2078 vs 0.2499; SSCBench-KITTI-360 0.1289 vs 0.1729. The distance transform, the radius test and the frozen dilation agree.

---

## 7. Semantic transport versus distance

Every added voxel copies the **complete fused probability vector** of its nearest frozen
occupied source; exact-distance ties are averaged over all tied sources, which is Gate 6's
own dilation convention extended to the larger radii. Ground truth never enters a class:
the oracle uses the target to decide *which* voxels exist, and the class still comes only
from the frozen teacher distribution.

The oracle-added and the morphology-added **true positives are the same set** at every
radius — morphology adds every valid voxel in range and the oracle adds exactly its
ground-truth-occupied subset — so one table serves both constructions. The test suite
asserts that identity rather than assuming it.

| benchmark | base | propagation distance | added true positives | top-1 accuracy | balanced recall | classes present | mean max prob | mean entropy (nats) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| SemanticKITTI 08 | B-R | (0, 0.4] m | 3,183,478 | 0.5466 | 0.3512 | 19 | 0.832 | 0.534 |
|  |  | (0.4, 0.8] m | 1,795,574 | 0.4883 | 0.2588 | 19 | 0.799 | 0.645 |
|  |  | (0.8, 1.2] m | 1,262,584 | 0.4648 | 0.2155 | 18 | 0.787 | 0.688 |
|  |  | (1.2, 2] m | 1,706,590 | 0.4311 | 0.1716 | 18 | 0.790 | 0.676 |
|  |  | (2, 4] m | 2,884,120 | 0.3488 | 0.1230 | 19 | 0.799 | 0.640 |
|  | B-D | (0, 0.4] m | 1,319,707 | 0.4760 | 0.2401 | 18 | 0.787 | 0.685 |
|  |  | (0.4, 0.8] m | 1,065,341 | 0.4530 | 0.1997 | 18 | 0.785 | 0.692 |
|  |  | (0.8, 1.2] m | 894,712 | 0.4328 | 0.1715 | 18 | 0.788 | 0.681 |
|  |  | (1.2, 2] m | 1,392,099 | 0.3970 | 0.1394 | 19 | 0.795 | 0.654 |
|  |  | (2, 4] m | 2,539,042 | 0.3196 | 0.1089 | 19 | 0.802 | 0.627 |
| Occ3D-nuScenes | B-R | (0, 0.4] m | 1,489,526 | 0.4562 | 0.3582 | 17 | 0.742 | 0.769 |
|  |  | (0.4, 0.8] m | 1,124,872 | 0.4266 | 0.3190 | 17 | 0.747 | 0.758 |
|  |  | (0.8, 1.2] m | 787,926 | 0.3599 | 0.3013 | 17 | 0.740 | 0.800 |
|  |  | (1.2, 2] m | 976,629 | 0.3095 | 0.2680 | 17 | 0.743 | 0.788 |
|  |  | (2, 4] m | 1,704,938 | 0.2537 | 0.2248 | 17 | 0.741 | 0.791 |
|  | B-D | (0, 0.4] m | 1,043,671 | 0.4205 | 0.3144 | 17 | 0.739 | 0.781 |
|  |  | (0.4, 0.8] m | 578,893 | 0.3378 | 0.2877 | 17 | 0.740 | 0.796 |
|  |  | (0.8, 1.2] m | 533,589 | 0.3122 | 0.2763 | 17 | 0.741 | 0.793 |
|  |  | (1.2, 2] m | 811,246 | 0.2817 | 0.2478 | 17 | 0.742 | 0.789 |
|  |  | (2, 4] m | 1,503,262 | 0.2323 | 0.2020 | 17 | 0.740 | 0.793 |
| SSCBench-KITTI-360 | B-R | (0, 0.4] m | 12,451,811 | 0.5916 | 0.2866 | 18 | 0.829 | 0.547 |
|  |  | (0.4, 0.8] m | 11,188,209 | 0.5761 | 0.2733 | 18 | 0.823 | 0.568 |
|  |  | (0.8, 1.2] m | 9,236,053 | 0.5760 | 0.2624 | 18 | 0.822 | 0.573 |
|  |  | (1.2, 2] m | 12,577,309 | 0.5528 | 0.2306 | 18 | 0.820 | 0.581 |
|  |  | (2, 4] m | 18,478,337 | 0.4795 | 0.1612 | 18 | 0.813 | 0.607 |
|  | B-D | (0, 0.4] m | 9,303,953 | 0.5761 | 0.2723 | 18 | 0.817 | 0.584 |
|  |  | (0.4, 0.8] m | 7,904,594 | 0.5745 | 0.2584 | 18 | 0.821 | 0.577 |
|  |  | (0.8, 1.2] m | 6,667,339 | 0.5581 | 0.2308 | 18 | 0.819 | 0.583 |
|  |  | (1.2, 2] m | 9,671,922 | 0.5171 | 0.1926 | 18 | 0.815 | 0.597 |
|  |  | (2, 4] m | 15,604,131 | 0.4569 | 0.1400 | 18 | 0.811 | 0.611 |

![semantic transport](../../artifacts/gate7a/fig_semantic_transport.png)

| benchmark | Gate-6 B-D TP-conditioned accuracy on the frozen support | Gate-7A top-1 accuracy on voxels propagated (0, 0.4] m | (0.4, 0.8] m | (2.0, 4.0] m |
|---|---:|---:|---:|---:|
| SemanticKITTI 08 | 0.5592 | 0.4760 | 0.4530 | 0.3196 |
| Occ3D-nuScenes | 0.4431 | 0.4205 | 0.3378 | 0.2323 |
| SSCBench-KITTI-360 | 0.5959 | 0.5761 | 0.5745 | 0.4569 |

**Semantics degrade with distance but do not collapse.** A label carried into the first 0.4 m shell is right 47.6 % / 42.0 % / 57.6 % of the time; carried 2–4 m it is still right 32.0 % / 23.2 % / 45.7 %, against a uniform chance of 5.3 % / 5.9 % / 5.6 % at these class counts — 6.1× / 3.9× / 8.2× chance. The decay is gradual and has no cliff, so a completion method would inherit usable labels rather than noise.

**But it is a real cost, and it is what caps the oracle's semantic ceiling.** Gate 6 measured top-1 accuracy on the frozen support at 0.5592 / 0.4431 / 0.5959. Propagation into the first shell already costs 8.3 / 2.3 / 2.0 points and the full 4 m reach costs 24.0 / 21.1 / 13.9 points. That is why §5's oracle multiplies binary IoU by two to two-and-a-half while barely doubling mIoU: **perfect local occupancy plus propagated semantics is still only about half-right on the voxels it recovers.** Semantics are not the binding constraint, but they are not free either, and a completion-only fix inherits this ceiling.

**The teacher's own confidence does not track the decay.** The mean maximum propagated probability is essentially flat across the intervals — it moves by at most 1.5 points on any benchmark while accuracy falls by 12 to 19 — and on SemanticKITTI 08 and Occ3D-nuScenes it *rises* with distance as accuracy falls. Confidence is therefore not a usable gate on how far a label may be carried; a later method that wants one will have to learn it rather than read it off the frozen teacher.

| benchmark | base | voxels propagated | mean tied sources | fraction with an exact tie |
|---|---|---:|---:|---:|
| SemanticKITTI 08 | B-R | 139,777,329 | 1.163 | 0.1296 |
| SemanticKITTI 08 | B-D | 130,756,345 | 1.083 | 0.0737 |
| Occ3D-nuScenes | B-R | 120,323,735 | 1.159 | 0.1270 |
| Occ3D-nuScenes | B-D | 119,522,683 | 1.112 | 0.0991 |
| SSCBench-KITTI-360 | B-R | 937,856,775 | 1.145 | 0.1159 |
| SSCBench-KITTI-360 | B-D | 914,996,324 | 1.090 | 0.0804 |

---

## 8. Frustum and range decomposition

The term is **in-frustum**, never "visible". A voxel is in-frustum when its centre has
positive depth in one of the five input cameras, projects inside the image, and lies in the
frozen metric depth range. Occlusion is not tested, so a voxel behind a wall counts as
in-frustum here.

| benchmark | base | coverage misses | in-frustum | outside all five | near surface | behind surface | in front of surface | no valid predicted depth |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SemanticKITTI 08 | B-R | 17,326,416 | 0.9355 | 0.0645 | 0.0088 | 0.5742 | 0.0794 | 0.2731 |
|  | B-D | 13,093,177 | 0.9165 | 0.0835 | 0.0000 | 0.5764 | 0.0201 | 0.3201 |
| Occ3D-nuScenes | B-R | 10,962,156 | 0.6951 | 0.3049 | 0.0040 | 0.1169 | 0.2159 | 0.3583 |
|  | B-D | 8,969,696 | 0.6337 | 0.3663 | 0.0000 | 0.1066 | 0.1401 | 0.3869 |
| SSCBench-KITTI-360 | B-R | 132,521,658 | 0.9694 | 0.0306 | 0.0069 | 0.3763 | 0.1151 | 0.4711 |
|  | B-D | 114,007,683 | 0.9654 | 0.0346 | 0.0000 | 0.3753 | 0.0735 | 0.5167 |

| benchmark | range band | B-D misses | in-frustum | behind predicted surface | no valid predicted depth | outside all five |
|---|---|---:|---:|---:|---:|---:|
| SemanticKITTI 08 | 0-10m | 429,282 | 0.782 | 0.612 | 0.071 | 0.218 |
|  | 10-20m | 2,205,889 | 0.794 | 0.658 | 0.098 | 0.206 |
|  | 20-30m | 3,205,082 | 0.853 | 0.645 | 0.190 | 0.147 |
|  | 30-40m | 3,237,267 | 0.977 | 0.606 | 0.356 | 0.023 |
|  | 40-60m | 4,015,657 | 1.000 | 0.450 | 0.543 | 0.000 |
| Occ3D-nuScenes | 0-10m | 650,680 | 0.586 | 0.030 | 0.164 | 0.414 |
|  | 10-20m | 2,160,142 | 0.561 | 0.079 | 0.230 | 0.439 |
|  | 20-30m | 2,677,704 | 0.637 | 0.114 | 0.400 | 0.363 |
|  | 30-40m | 2,633,006 | 0.722 | 0.132 | 0.547 | 0.278 |
|  | 40-60m | 848,164 | 0.568 | 0.134 | 0.420 | 0.432 |
| SSCBench-KITTI-360 | 0-10m | 6,802,622 | 0.912 | 0.414 | 0.129 | 0.088 |
|  | 10-20m | 25,218,757 | 0.931 | 0.432 | 0.335 | 0.069 |
|  | 20-30m | 30,227,457 | 0.949 | 0.406 | 0.494 | 0.051 |
|  | 30-40m | 25,849,583 | 0.997 | 0.354 | 0.635 | 0.003 |
|  | 40-60m | 25,909,264 | 1.000 | 0.296 | 0.704 | 0.000 |

![frustum by range](../../artifacts/gate7a/fig_frustum_by_range.png)

| benchmark | distance to B-D support | B-D misses | in-frustum fraction |
|---|---|---:|---:|
| SemanticKITTI 08 | <= 0.4 m | 1,319,707 | 0.983 |
|  | 0.4-0.8 m | 1,065,341 | 0.972 |
|  | 0.8-1.2 m | 795,093 | 0.958 |
|  | 1.2-2 m | 1,491,718 | 0.948 |
|  | 2-4 m | 2,539,042 | 0.913 |
|  | > 4 m | 5,882,276 | 0.880 |
| Occ3D-nuScenes | <= 0.4 m | 1,043,671 | 0.956 |
|  | 0.4-0.8 m | 578,893 | 0.910 |
|  | 0.8-1.2 m | 337,278 | 0.848 |
|  | 1.2-2 m | 1,007,557 | 0.839 |
|  | 2-4 m | 1,503,262 | 0.735 |
|  | > 4 m | 4,499,035 | 0.427 |
| SSCBench-KITTI-360 | <= 0.4 m | 9,303,953 | 0.986 |
|  | 0.4-0.8 m | 7,904,594 | 0.979 |
|  | 0.8-1.2 m | 6,050,674 | 0.973 |
|  | 1.2-2 m | 10,288,587 | 0.970 |
|  | 2-4 m | 15,604,131 | 0.958 |
|  | > 4 m | 64,855,744 | 0.961 |

**The `near_surface` class is empty for B-D by construction, and that is informative.** The
predeclared tolerance is one voxel diagonal — 0.35 m on the KITTI grid, 0.69 m on Occ3D —
and B-D is exactly a 0.4 m dilation of the reconstruction. Anything within a voxel diagonal
of the predicted surface has therefore *already been absorbed into B-D* and is not a miss.
B-R is reported alongside for that reason; there the class is non-empty but still tiny.

**Two different failures hide under one number, and they point in opposite directions.**

On SSCBench-KITTI-360 only 3.5 % of the misses fall outside all five frusta: essentially the whole ground truth was inside the camera, and the reconstruction still did not place a voxel there. That is a *depth* failure along rays the camera did see — 37.5 % sit behind the predicted surface and 51.7 % land on pixels the frozen confidence-and-range gate rejected outright. Both are addressable with image-aligned evidence; neither is addressable by hallucinating voxels next to existing ones.

On Occ3D-nuScenes the picture is different: 36.6 % of the misses are outside every input frustum. That benchmark's evaluation volume spans ±40 m in both horizontal axes while the input is a single front camera, so a large share of its ground truth was never observable from the five frames at all. **No completion method operating on this input can recover those voxels**, and any gain reported on that fraction would have to come from a scene prior rather than from evidence.

The range breakdown sharpens both. In-frustum fraction *rises* with range on the KITTI family — the far field is squarely in front of the camera and simply was not reconstructed — while the share with no valid predicted depth rises with it, reaching 54.3% / 42.0% / 70.4% of the misses in the 40–60 m band. The frozen depth gate (confidence >= 1.5, range 1-60 m) is doing a large part of the excluding, and it is a threshold, not a model - which is why §11 puts sweeping it first.

---

## 9. Per-dataset and pooled results

Every number above is **pooled over voxels** within its benchmark; the only mean-per-clip
quantity in this report is the one labelled as such in §3, kept because that is the
statistic the frozen reproduction gates were pinned with.

**Results are not pooled across the three benchmarks, and absolute scores are not compared
between them.** The grids differ (0.2 m against 0.4 m), the evaluation masks differ (a
camera mask plus a rear-half cut on Occ3D, `.invalid`-derived on the KITTI family), the
class counts differ (19 / 17 / 18) and the target-construction rules differ. What *is*
comparable across benchmarks is the shape of the fractions — reachable fraction, in-frustum
fraction, transport accuracy relative to chance — and those are what the cross-benchmark
statements in §1 and §11 rest on.

| benchmark | unit (n) | base | construction | radius | Δ binary IoU | 95% CI | Δ SSC mIoU | 95% CI |
|---|---|---|---|---:|---:|---|---:|---|
| SemanticKITTI 08 | contiguous block of 20 clips (8) | B-R | oracle | 0.4 m | +0.1485 | [+0.1329, +0.1642] | +0.0467 | [+0.0413, +0.0492] |
|  | contiguous block of 20 clips (8) | B-R | oracle | 4 m | +0.5052 | [+0.4796, +0.5311] | +0.0813 | [+0.0708, +0.0875] |
|  | contiguous block of 20 clips (8) | B-R | morph | 0.4 m | +0.0858 | [+0.0786, +0.0931] | +0.0238 | [+0.0198, +0.0258] |
|  | contiguous block of 20 clips (8) | B-R | morph | 4 m | +0.0327 | [+0.0257, +0.0429] | -0.0057 | [-0.0085, -0.0024] |
|  | contiguous block of 20 clips (8) | B-D | oracle | 0.4 m | +0.0372 | [+0.0358, +0.0387] | +0.0069 | [+0.0059, +0.0081] |
|  | contiguous block of 20 clips (8) | B-D | oracle | 4 m | +0.2034 | [+0.1913, +0.2186] | +0.0193 | [+0.0156, +0.0229] |
|  | contiguous block of 20 clips (8) | B-D | morph | 0.4 m | -0.0094 | [-0.0139, -0.0045] | -0.0068 | [-0.0075, -0.0056] |
|  | contiguous block of 20 clips (8) | B-D | morph | 4 m | -0.0636 | [-0.0751, -0.0509] | -0.0301 | [-0.0316, -0.0264] |
| Occ3D-nuScenes | scene (150) | B-R | oracle | 0.4 m | +0.1187 | [+0.1106, +0.1269] | +0.0251 | [+0.0227, +0.0274] |
|  | scene (150) | B-R | oracle | 4 m | +0.4850 | [+0.4660, +0.5042] | +0.0731 | [+0.0671, +0.0788] |
|  | scene (150) | B-R | morph | 0.4 m | +0.1055 | [+0.0987, +0.1123] | +0.0191 | [+0.0175, +0.0208] |
|  | scene (150) | B-R | morph | 4 m | +0.1173 | [+0.1085, +0.1262] | +0.0123 | [+0.0093, +0.0156] |
|  | scene (150) | B-D | oracle | 0.4 m | +0.0732 | [+0.0678, +0.0788] | +0.0128 | [+0.0117, +0.0139] |
|  | scene (150) | B-D | oracle | 4 m | +0.3134 | [+0.2992, +0.3281] | +0.0369 | [+0.0327, +0.0414] |
|  | scene (150) | B-D | morph | 0.4 m | +0.0446 | [+0.0393, +0.0502] | +0.0051 | [+0.0041, +0.0061] |
|  | scene (150) | B-D | morph | 4 m | -0.0139 | [-0.0264, -0.0012] | -0.0118 | [-0.0155, -0.0077] |
| SSCBench-KITTI-360 | contiguous block of 20 clips (87) | B-R | oracle | 0.4 m | +0.0849 | [+0.0816, +0.0882] | +0.0189 | [+0.0175, +0.0201] |
|  | contiguous block of 20 clips (87) | B-R | oracle | 4 m | +0.4357 | [+0.4248, +0.4466] | +0.0611 | [+0.0568, +0.0647] |
|  | contiguous block of 20 clips (87) | B-R | morph | 0.4 m | +0.0692 | [+0.0670, +0.0715] | +0.0138 | [+0.0128, +0.0145] |
|  | contiguous block of 20 clips (87) | B-R | morph | 4 m | +0.1752 | [+0.1705, +0.1803] | +0.0221 | [+0.0205, +0.0239] |
|  | contiguous block of 20 clips (87) | B-D | oracle | 0.4 m | +0.0516 | [+0.0502, +0.0530] | +0.0092 | [+0.0085, +0.0099] |
|  | contiguous block of 20 clips (87) | B-D | oracle | 4 m | +0.2724 | [+0.2657, +0.2795] | +0.0306 | [+0.0283, +0.0328] |
|  | contiguous block of 20 clips (87) | B-D | morph | 0.4 m | +0.0322 | [+0.0312, +0.0332] | +0.0045 | [+0.0041, +0.0050] |
|  | contiguous block of 20 clips (87) | B-D | morph | 4 m | +0.0819 | [+0.0766, +0.0875] | +0.0041 | [+0.0031, +0.0053] |

Paired bootstrap, 10,000 resamples, seed 0, over Gate 6's own resampling units: official
scenes on Occ3D-nuScenes, contiguous blocks of 20 clips on the two single-sequence
benchmarks. **The blocks come from a single drive and are not independent scenes; they are
not described as such.** SemanticKITTI's 8 blocks are few and its intervals are the weakest
of the three.

**The benchmarks agree on the diagnosis and disagree on one detail, and the disagreement is stated rather than averaged away.**

They agree on everything the diagnosis rests on: the reachable fraction at 0.4 m (0.101 / 0.116 / 0.082), the median miss distance (3.45 m / 4.05 m / 5.45 m), the sign and rough size of the oracle gain, and the fact that propagated semantics stay well above chance.

They disagree on **ordinary dilation at the smallest radius**: it raises mIoU on Occ3D-nuScenes and SSCBench-KITTI-360 and lowers it on SemanticKITTI 08. This is a real dataset difference, not noise — every one of those intervals excludes zero (§9). It tracks the base occupancy precision, which orders the same way (SemanticKITTI 08 0.2537 / Occ3D-nuScenes 0.5641 / SSCBench-KITTI-360 0.3587): the more precise the frozen occupancy already is, the more a blind expansion can afford. **A non-learned dilation step is therefore not a portable recommendation.**

They also disagree on **how much of the ground truth was observable at all**: 8.3 % / 36.6 % / 3.5 % of the misses lie outside all five frusta. That is a property of the benchmark's evaluation volume against a single front camera, not of the method, and it caps what any completion model could reach on each benchmark at a different level.

---

## 10. Limitations

* **The oracle is not a result.** It is an upper bound computed with the target in hand.
  Its semantic mIoU in particular pairs *target-chosen occupancy* with *teacher-propagated
  labels*, and is not achievable by any deployable method.
* **The oracle is optimistic in a second way.** It is the ceiling for a corrector that is
  perfect at exactly the predeclared radius. A real model would have to decide *which*
  voxels in that shell to add, and every wrong decision costs the precision the oracle is
  given for free.
* **In-frustum is not visible.** Occlusion is not tested. A large share of the in-frustum
  misses sit behind the predicted surface and would be genuinely invisible to any
  single-view method; this analysis cannot separate "occluded in reality" from "the
  predicted depth stopped short".
* **The residual test compares against one frame.** The predeclared frame is the anchor
  when the voxel is in-frustum there and otherwise the lowest-index frame in which it is.
  A voxel that is behind the predicted surface in that frame may be in front of it in
  another.
* **The morphology baseline is restricted to the valid mask.** This is metric-neutral —
  voxels outside the mask are scored by no benchmark — but it makes the added-volume ratio
  a within-mask quantity, not a physical volume.
* **The distance is to the nearest occupied voxel, not to the nearest *surface*.** Two
  benchmarks' targets are themselves derived from accumulated LiDAR and carry their own
  completion; the miss distances inherit that.
* **Quantiles are histogram-limited.** They are read off a 0.05 m histogram and reported at
  the bin's upper edge, so they are exact to 0.05 m and no better.
* **SemanticKITTI and SSCBench-KITTI-360 are related datasets** — same city, same sensor
  family. Two of the three benchmarks are not independent evidence.
* **Nothing here re-tests Gate 6's semantic conclusions.** The base labels are Gate 6's own
  pinned channels; this gate only adds voxels around them.

---

## 11. Recommendation for the next gate

The branch below is the brief's own decision logic, evaluated on the measured numbers. The three thresholds it needs were **not** predeclared in the pinned configuration — the precommit fixes the measurements, not the interpretation — so they are written out here, and every measurement lands far from its boundary:

| test | threshold | measured (mean over benchmarks) | verdict |
|---|---|---:|---|
| a large fraction of B-D misses lies near existing support | ≥ 50% within 0.4 m | 10.0% | **no** |
| ordinary dilation captures most of the oracle gain | ≥ 50% of the oracle's 4 m mIoU gain | 11.3% | **no** |
| propagated semantics collapse with distance | < 2× chance at 2–4 m | 6.1× chance | **no — they hold** |

→ **A local voxel prior is structurally insufficient. Do not train one.**

Roughly nine of every ten voxels the pipeline misses lie further from the B-D support than a second 0.4 m dilation would reach, the median is 3.5 m / 4.0 m / 5.5 m away, and a *perfect* corrector out to 4 m — one handed the ground truth and charged nothing for false positives — would still leave full SSC mIoU at 0.0704 / 0.0848 / 0.0572. A learned local prior would be inventing structure at that distance, not completing it. The deployable version of exactly that move, ordinary dilation, loses outright on SemanticKITTI and on the other two reaches only 14% and 20% of that benchmark's own oracle gain (§6) — and the sign of its effect is not even the same across benchmarks, so there is nothing portable to adopt.

**The next gate should add ray- and image-aligned evidence, not a local 3D prior.** The misses are overwhelmingly *inside* the input frusta — 92 % / 63 % / 97 % of them — and concentrated in two failure modes that both live along a camera ray: voxels behind the predicted surface, and voxels at pixels where the frozen confidence-and-range gate produced no depth at all. Concretely, and in descending order of expected return per unit of effort:

1. **Diagnose the depth gate before training anything.** 32 % / 39 % / 52 % of the B-D misses project onto pixels the frozen mask (confidence ≥ 1.5, depth in 1–60 m) discarded. That is a *free* experiment on cached data: sweep the confidence threshold and the range cap on the existing LingBot cache and measure the occupancy IoU envelope. If a large share of that class is recoverable by relaxing a threshold, no model is needed at all. This must be run before any training gate, and it does not touch a frozen weight.
2. **Then, if that is exhausted, a ray-space completion model** — one that predicts occupancy along the camera rays it can see, conditioned on the image and the frozen depth, rather than a 3D CNN over the voxel neighbourhood. The evidence for the missing voxels is in the pixels, and a local voxel prior cannot reach it.
3. **Accept, and state, a per-benchmark ceiling.** 8 % on SemanticKITTI 08, 37 % on Occ3D-nuScenes, 3 % on SSCBench-KITTI-360 of the misses are outside every input frustum. Those are unreachable from five frames of one camera by any method, and future results should quote the in-frustum ceiling alongside the raw score rather than appear to fail at something impossible.

**Do not run blind local hallucination**, and do not read Gate 6's `COVERAGE_DOMINATES` as licence for it.

**Two constraints carry forward to any model that is eventually trained.** First, occupancy is the binding constraint, not semantics: propagated teacher labels degrade but do not collapse over metres (§7), so a semantic head is not where the budget goes. The propagation loss is real even so — 12 to 19 points over 4 m — and it is what holds the oracle's mIoU gain down, so a completion-only fix inherits a ceiling it cannot raise. Second, **no fixed 17/18/19-class output head.** The three benchmarks have different ontologies and such a head would weaken exactly the open-vocabulary transfer claim Gate 6 established. Any later semantic learning must operate in a fixed-dimensional language-aligned feature space, or stay class-agnostic and propagate.

**Not recommended, explicitly:** scale distillation; a fixed-class semantic head; reviving C3 or V3; a 3D-CNN local completion prior; and any change to B-D, to Trident, or to the frozen occupancy.

---

## 12. Files created and modified

| path | role |
|---|---|
| `gate7a/config.py` | every predeclared choice; the precommit is generated from it |
| `gate7a/distance.py` | exact metric EDT, radius tests, offset shells, histograms |
| `gate7a/envelopes.py` | the two constructions and their count blocks |
| `gate7a/transport.py` | tie-averaged nearest-source semantic propagation |
| `gate7a/frustum.py` | in-frustum test, ray-depth residual, camera round-trip |
| `gate7a/pipeline.py` | recovers the frozen Gate-6 state and verifies it |
| `gate7a/stats.py` | vectorised paired bootstrap over Gate 6's units |
| `tools/gate7a/stage0.py` | hashes everything Gate 7A must not disturb |
| `tools/gate7a/precommit.py` | writes and pins the Gate-7A configuration |
| `tools/gate7a/reachability.py` | the per-clip analysis |
| `tools/gate7a/aggregate.py` | shards -> reported numbers |
| `tools/gate7a/figures.py` | the six figures |
| `tools/gate7a/report_tables.py` | this report's tables |
| `tools/gate7a/run_all.sh` | the full run, four shards per benchmark |
| `tests/gate7a/test_gate7a.py` | the Gate-7A test suite |
| `configs/gate7a/completion_reachability_precommit.yaml` | the pinned configuration |
| `artifacts/gate7a/` | count blocks, summaries, per-clip CSVs, figures, logs |
| `reports/gate7a/completion_reachability.md` | this report |
| `tools/gate6/report_tables.py` | **modified**: pooled binary IoU in the official-metrics table, both aggregations labelled, two generated correction notes, DINO v1 named explicitly, comparator caveats |
| `reports/gate6/_frozen_trident_semantic_lifting.template.md` | **modified**: the six prose corrections of §2 |
| `reports/gate6/frozen_trident_semantic_lifting.md` | **regenerated** from the artifacts |

**Not modified:** `lingbot_map/models/gct_stream.py` and `research/sem_bypass/model.py`
(the two pre-existing dirty files, hash-verified in §3), every Gate-6 prediction, cache,
manifest, mapping and evaluation mask, every Gate-6 numerical artifact, and every released
dataset directory. C3 and V3 remain retired. No weights or datasets were downloaded.

---

## 13. Reproduction commands

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
PY=/home/minh/anaconda3/envs/cu128/bin/python

# 0. hash everything Gate 7A must not disturb
$PY tools/gate7a/stage0.py

# 1. pin the configuration BEFORE the target-dependent analysis
$PY tools/gate7a/precommit.py

# 2. the analysis: three benchmarks, four shards each, one GPU per shard (~25 min)
tools/gate7a/run_all.sh

# 3. shards -> reported numbers (bootstrap included)
for DS in semantickitti occ3d kitti360; do $PY tools/gate7a/aggregate.py --dataset $DS; done

# 4. figures, the Gate-6 correction record, and this report
$PY tools/gate7a/figures.py
$PY tools/gate7a/record_gate6_correction.py
$PY tools/gate7a/report_tables.py --write

# 5. tests
$PY -m pytest tests/gate6 tests/gate7a -q

# the corrected Gate-6 report, regenerated from unchanged artifacts
$PY tools/gate6/report_tables.py --write
```

---

## 14. Runtime, peak memory and disk use

| benchmark | clips | wall time (4 shards in parallel) | summed GPU-time | s/clip | peak GPU per shard | pinned-channel mismatches |
|---|---:|---:|---:|---:|---:|---:|
| SemanticKITTI 08 | 163 | 1.3 min | 5.1 min | 1.87 | 0.22 GiB | 0 / 33,012,865 |
| Occ3D-nuScenes | 1182 | 2.5 min | 9.6 min | 0.49 | 0.25 GiB | 38 / 34,488,246 |
| SSCBench-KITTI-360 | 1753 | 9.8 min | 39.2 min | 1.34 | 0.16 GiB | 0 / 199,750,527 |
| **total** | | **13.6 min** | 53.9 min | | | artifacts 8.4 MB |

Four RTX PRO 6000 Blackwell GPUs, one shard per GPU. The analysis reads the 37 GB frozen
Trident cache and the 1.5 GB frozen prediction store on `/media/SSD1`; it writes nothing
outside `artifacts/gate7a/`, `configs/gate7a/` and `reports/gate7a/`. No new dataset or
checkpoint was downloaded.

---

## 15. Deviations and blockers

**D1 — one clip's numbers were computed before the precommit was pinned.**
While sizing the analysis, a single SemanticKITTI clip was pushed through a prototype of
the miss-distance code, which opened that clip's target. It is disclosed because the
ordering rule matters: nothing in the pinned configuration was chosen from it. The radii,
the range bands, the distance definition, the oracle and morphology formulas, the
propagation rule, the frustum test and the bootstrap units were all specified in the brief
and were transcribed, not selected. No threshold in this gate is data-derived.

**D2 — the Gate-6 predictions are not bit-reproducible, and the reason is measured.**
Rebuilding the frozen state reproduced B-R and B-D **geometry** bit-for-bit on every clip
of all three benchmarks. The *argmax channels* also agreed everywhere except
Occ3D-nuScenes, where 38 of 34,488,246 occupied voxels differed (1.1 per million), and in an earlier non-deterministic run the
disagreement was large enough to trip the equality check outright. The cause is `torch.Tensor.index_add_`, which accumulates the fused
probability mean with CUDA atomics: the summation order depends on how the GPU schedules
the adds, and on a voxel whose top two classes are separated by less than that accumulation
error the argmax flips. Gate 7A responds in three ways rather than papering over it — it
enables `torch.use_deterministic_algorithms` so its own run is reproducible, it takes the
**pinned Gate-6 channel** as the authoritative base label so this gate's base metrics are
*exactly* Gate 6's (§3), and it records the disagreement rate per shard. This is a property
of the Gate-6 artifacts worth knowing before anyone tries to reproduce them on other
hardware.

**D3 — the frustum analysis was run for B-R as well as B-D.** The brief asks for B-D. With
the predeclared tolerance of one voxel diagonal, the `near_surface` class is empty for B-D
*by construction*, because B-D is exactly a 0.4 m dilation and has already absorbed that
band. B-R is reported alongside so the residual classification is informative rather than
degenerate. This is an addition, not a substitution.

**D4 — miss-distance quantiles are histogram-limited.** They are read off a 0.05 m
histogram and reported at the bin's upper edge, so they are exact to 0.05 m. Storing every
distance would have cost roughly 25 GB for no gain at the precision reported.

**D5 — the semantic-transport table covers both constructions with one set of rows.** The
oracle-added and morphology-added *true positives* are provably the same set at every
radius, so a separate table would have been a copy. The identity is asserted in
`tests/gate7a`.

**D6 — decision thresholds were not predeclared.** The pinned configuration fixes the
measurements, not their interpretation, and the brief specified the branch logic in words
rather than numbers. The three thresholds are written out in §11 with the measured values
beside them; every measurement is far from its boundary, so no reasonable alternative
threshold changes the branch.

**No blocker was reached.** Every stop condition was checked and none fired: the Gate-6
prediction hashes match, B-R/B-D geometry reproduced exactly on 3,098
clips, the Occ3D native grid mapping is the unambiguous 0.4 m official grid, the camera
transformations round-trip to 1.1e-12 px, the frozen probability vectors were
recoverable from the existing caches, no foundation model was re-run, and no
target-dependent quantity entered anything outside the explicitly labelled oracle
analysis.
