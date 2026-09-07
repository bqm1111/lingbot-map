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

{{TABLE_HEADLINE}}

**The missing occupancy Gate 6 identified is not recoverable by a small local voxel
prior.** Three measurements say so, and they agree on all three benchmarks:

1. **The misses are not near the support.** Only {{NEAR_FRAC}} of B-D coverage misses lie
   within 0.4 m of a frozen occupied voxel, and the median miss sits {{MEDIAN_RANGE}} away
   — {{MEDIAN_VOXELS}} on the KITTI-family grid. A local prior would not be *completing*
   these voxels, it would be inventing them (§4).
2. **Even a perfect local corrector buys little semantics.** The oracle — which is
   permitted to add ground-truth-occupied voxels for free and never adds a false positive —
   roughly doubles binary IoU at a 4 m radius, but lifts full SSC mIoU only from
   {{MIOU_BASE}} to {{MIOU_ORACLE4}}. That is the *ceiling*, with the target handed to it
   and no precision charged for it (§5).
3. {{MORPH_HEADLINE}}

**On two of three benchmarks the evidence is not missing from the images.**
{{FRUSTUM_HEADLINE}} On SemanticKITTI and SSCBench-KITTI-360 what fails is not the field of
view but the *depth* along rays the camera did see, and that distinction is what separates
"accept the five-frame ceiling" from "add ray-aligned evidence". **Occ3D-nuScenes is
different and is not averaged in:** its evaluation volume spans ±40 m in both horizontal
axes against a single front camera, so a third of its ground truth was never observable
from the five input frames at all. That share is a hard ceiling no completion method can
cross (§8).

**Propagated semantics degrade but do not collapse.** A teacher label carried 2–4 m
through empty space is still right {{TRANSPORT_FAR}} of the time against a uniform chance
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

{{TABLE_CORRECTIONS}}

{{TABLE_G6_CORRECTION}}

All 44 Gate-6 artifact files are byte-identical to their stage-0 record, so the
corrections provably changed prose and table *rendering* only. Two of them matter for what
follows: `COVERAGE_DOMINATES` is now explicitly scoped to ground-truth-**occupied** voxels
and stated not to prove that expanding occupancy raises IoU — which is exactly the question
this gate answers, and answers with a firm no for a local prior and a qualified,
benchmark-dependent no for ordinary dilation.

---

## 3. Frozen inputs and provenance

{{TABLE_PROVENANCE}}

**The frozen state was recovered, not re-run.** Gate 6 stored the argmax channel of each
occupied voxel, not the fused probability vector, and Gate 7A needs the vectors to
propagate them. They were recomputed from the same three frozen inputs Gate 6 used — the
LingBot cache, the G51-B scale table and the Trident cache — and then checked against the
pinned prediction file. **Every clip of all three benchmarks reproduced the B-R and B-D
geometry bit-for-bit**: identical flat indices in identical order, identical support flags.

{{TABLE_BASE_REPRO}}

**The camera-frame transformations were verified, not assumed.** For every clip the frozen
reconstruction's own points were pushed back through the whole chain — grid → anchor camera
→ frame *f* → pixel — and compared with the pixel each point came from. The worst error
over all three benchmarks is {{ROUNDTRIP}} px. A mistake in the anchor convention, in the
pose scaling or in the intrinsics would show up as a whole pixel.

{{TABLE_EMPTY}}

---

## 4. How far is the missing occupancy? (miss-distance CDF and quantiles)

Exact Euclidean distance transform on the benchmark's **native** evaluation grid — 0.2 m
for the KITTI family, 0.4 m for the official Occ3D grid — converted to metres once, at the
end. Voxel indices are never treated as metres, and the radius test is done on integer
squared lattice distances because `1.2 / 0.2` is `5.999999999999999` in binary floating
point and a naive comparison silently drops the boundary shell.

{{TABLE_QUANTILES}}

![miss distance CDF](../../artifacts/gate7a/fig_miss_distance_cdf.png)

{{TABLE_REACHABLE}}

![reachable miss fraction](../../artifacts/gate7a/fig_reachable_miss_fraction.png)

{{TABLE_REACHABLE_BAND}}

{{REACHABLE_PROSE}}

---

## 5. The oracle-local completion ceiling

`P_oracle(r) = P_base OR (GT_occupied AND valid AND distance_to_P_base <= r)`

It may add only true positives, it removes nothing, and **its false-positive count is
identical to the base's at every radius** — checked on every clip, not argued. It is the
most optimistic result any model confined to that local correction region could reach, and
it is not deployable.

{{TABLE_ORACLE}}

{{TABLE_INVARIANTS}}

**Binary IoU and semantic mIoU respond very differently to the oracle, and the gap is the
point.** At 4 m the oracle roughly doubles to two-and-a-half-times binary IoU on all three
benchmarks, while full SSC mIoU rises by only {{ORACLE_MIOU_GAINS}}. The reason is in §7:
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

{{TABLE_MORPH}}

![precision-recall](../../artifacts/gate7a/fig_precision_recall.png)

{{MORPH_PROSE}}

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

{{TABLE_TRANSPORT}}

![semantic transport](../../artifacts/gate7a/fig_semantic_transport.png)

{{TABLE_TRANSPORT_REF}}

{{TRANSPORT_PROSE}}

{{TABLE_TIES}}

---

## 8. Frustum and range decomposition

The term is **in-frustum**, never "visible". A voxel is in-frustum when its centre has
positive depth in one of the five input cameras, projects inside the image, and lies in the
frozen metric depth range. Occlusion is not tested, so a voxel behind a wall counts as
in-frustum here.

{{TABLE_FRUSTUM}}

{{TABLE_FRUSTUM_RANGE}}

![frustum by range](../../artifacts/gate7a/fig_frustum_by_range.png)

{{TABLE_FRUSTUM_DIST}}

**The `near_surface` class is empty for B-D by construction, and that is informative.** The
predeclared tolerance is one voxel diagonal — 0.35 m on the KITTI grid, 0.69 m on Occ3D —
and B-D is exactly a 0.4 m dilation of the reconstruction. Anything within a voxel diagonal
of the predicted surface has therefore *already been absorbed into B-D* and is not a miss.
B-R is reported alongside for that reason; there the class is non-empty but still tiny.

{{FRUSTUM_PROSE}}

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

{{TABLE_BOOTSTRAP}}

Paired bootstrap, 10,000 resamples, seed 0, over Gate 6's own resampling units: official
scenes on Occ3D-nuScenes, contiguous blocks of 20 clips on the two single-sequence
benchmarks. **The blocks come from a single drive and are not independent scenes; they are
not described as such.** SemanticKITTI's 8 blocks are few and its intervals are the weakest
of the three.

{{MIXED_PROSE}}

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

{{RECOMMENDATION}}

---

## 12. Files created and modified

{{TABLE_FILES}}

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

{{TABLE_RUNTIME}}

Four RTX PRO 6000 Blackwell GPUs, one shard per GPU. The analysis reads the 37 GB frozen
Trident cache and the 1.5 GB frozen prediction store on `/media/SSD1`; it writes nothing
outside `artifacts/gate7a/`, `configs/gate7a/` and `reports/gate7a/`. No new dataset or
checkpoint was downloaded.

---

## 15. Deviations and blockers

{{DEVIATIONS}}
