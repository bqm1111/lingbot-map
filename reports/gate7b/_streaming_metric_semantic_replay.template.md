# Gate 7B — Native causal streaming replay, metric-gauge stability and complementary depth evidence

**Date:** 2026-09-02 · **Status:** complete · **Nothing was trained.** No optimizer was
constructed, no backward pass was run, no network was created. LingBot-Map, MoGe-2 and
Trident were not modified; Trident was not re-run at all.

> Every map in this report is **causal**: the map at timestamp *t* is built from frames
> with stream index ≤ *t* and from nothing else. The one forward-looking analysis (§8) is
> computed in a separate volume, written to a separate artifact, and can never reach a
> deployable output.

---

## 1. Executive diagnosis

{{DIAGNOSIS}}

{{TABLE_HEADLINE}}

{{EXEC_PROSE}}

---

## 2. The streaming interface, exactly

{{TABLE_INTERFACE}}

At timestep *t* the system receives **one RGB image**. Everything else it knows about the
past is carried by four things, and Gate 7B measures what each is worth:

1. **LingBot's native anchor context** — the five scale frames, pinned in the KV cache for
   the whole segment;
2. **the pose-reference window** — a sliding KV window of 64 blocks, so the model's own
   attention sees a *recent* past, not the whole one;
3. **trajectory and camera special tokens**, carried across frames;
4. **the persistent metric occupancy–semantic map**, which is the only component whose
   horizon is unbounded — and therefore the only one that can carry evidence from a
   hundred frames ago.

The five-frame clip protocol of Gates 6 and 7A remains as a frozen comparator only. It is
not the intended interface and is no longer treated as one.

**A limit of the frozen model that shapes everything below.** The 3D RoPE table covers
{{ROPE_MAX}} global frame indices, and a frame index past it does not raise — it silently
returns a wrongly-shaped slice. A non-keyframe consumes no global index, so one global
capacity rule (the smallest `keyframe_interval` that fits the segment) keeps every stream
inside the table: {{KEYFRAME_RULE}}. The rule depends only on stream length and the frozen
table size, never on a benchmark score. Note also that the published training range is
~320 frames, so the two KITTI streams are **extrapolating** beyond it; that is a property
of the frozen checkpoint, not a choice made here.

---

## 3. Direct-mode verification

{{TABLE_STREAM}}

The state is reset **once per segment** and nowhere else — one `clean_kv_cache()` call in
the whole replay path, asserted in the tests. Segment boundaries are genuine: one stream
for SemanticKITTI sequence 08, one for KITTI-360 drive 0006, and one per official
nuScenes validation scene. Frames are deduplicated first: the Gate-6 clip sets overlap
heavily, and replaying them as independent clips would have reset the model
{{N_CLIPS_TOTAL}} times and fed most frames to it several times over.

**Phase-0 findings worth recording before any result.**

* The Gate-5.1 dense MoGe caches for SemanticKITTI and Occ3D-nuScenes are the **G51-A**
  variant (MoGe's own inferred FOV), not the frozen **G51-B** gauge: they reproduce
  `scales_G51-A_*.csv` to 2.5 × 10⁻⁵ and miss `scales_G51-B_*.csv` by 4 % and 18 %. They
  were therefore *not* used. The calibrated-FOV variant was regenerated per stream frame
  and verified by reproducing the pinned G51-B clip scales to ≈ 10⁻⁵ (§4). The Gate-5.2
  KITTI-360 cache under `moge/B` *is* the calibrated variant and was reused untouched.
* One model instance per image geometry: the FlashInfer KV manager binds to the first
  frame shape it sees and is never rebuilt, so a 294 × 518 stream and a 154 × 518 stream
  cannot share a process. It fails loudly, which is how this was found.

---

## 4. Scale-policy comparison

The gauge lives in an explicit `ScaleState`, never inside already-fused metric voxels, and
the metric crop at every timestamp is **rematerialised** from canonical observations with
the scale in force at that timestamp — so a change of gauge moves old and new observations
together and cannot leave two copies of one wall. Its cost is in §11.

{{TABLE_SCALE}}

{{SCALE_PROSE}}

{{TABLE_THICKNESS}}

![scale versus time](../../artifacts/gate7b/fig_scale_vs_time.png)

![gauge jitter against thickness](../../artifacts/gate7b/fig_scale_jitter_thickness.png)

---

## 5. Temporal-horizon comparison

{{TABLE_HORIZON}}

![recall versus history](../../artifacts/gate7b/fig_horizon_recall.png)

![mIoU versus history](../../artifacts/gate7b/fig_horizon_miou.png)

{{HORIZON_PROSE}}

---

## 6. Depth-gate sweep

{{TABLE_S2}}

{{S2_PROSE}}

---

## 7. MoGe rescue and map-consistency-gated fusion

{{TABLE_S34}}

![precision-recall](../../artifacts/gate7b/fig_precision_recall.png)

![evidence sources](../../artifacts/gate7b/fig_evidence_sources.png)

{{S34_PROSE}}

---

## 8. Temporal recoverability and latency

Gate 7A could not separate genuine occlusion from depth failure. This can: for every B-D
coverage-miss voxel at an official timestamp, does the same frozen model, streamed, ever
place geometry there — from the causal past, from a later viewpoint, or never?

{{TABLE_RECOVERY}}

![recovery latency](../../artifacts/gate7b/fig_recovery_latency.png)

{{RECOVERY_PROSE}}

---

## 9. Semantic quality of newly recovered voxels

{{TABLE_SEMANTIC}}

{{SEMANTIC_PROSE}}

**The evaluation vectors are not a deployable representation.** These are the cached
per-benchmark Trident probability vectors, used because they make Gate 7B's numbers
directly comparable with Gate 6's. A deployable open-vocabulary map must carry
**fixed-dimensional language-aligned descriptors**, not permanent 17/18/19-class
probability vectors — otherwise the open-vocabulary transfer claim Gate 6 established is
lost the moment the ontology changes.

---

## 10. Precision and false-positive analysis

{{TABLE_PRECISION}}

{{PRECISION_PROSE}}

---

## 11. Runtime, memory and scaling

{{TABLE_RUNTIME}}

![cost versus window](../../artifacts/gate7b/fig_cost_vs_window.png)

**This system is not real-time, and the report does not call it that.** LingBot streams at
{{FPS}}, but the complete pipeline includes a Trident pass per frame that is orders of
magnitude slower and is run asynchronously from a cache here. The word used throughout is
**online**: causal, incremental, and never looking forward. A real-time claim would need
the whole measured chain, including the semantic teacher, and that chain does not support
one.

---

## 12. Cross-dataset consistency

{{TABLE_CROSS}}

{{CROSS_PROSE}}

---

## 13. Recommendation for the next gate

{{RECOMMENDATION}}

---

## 14. Files created and modified

{{TABLE_FILES}}

**Not modified:** `lingbot_map/models/gct_stream.py` and `research/sem_bypass/model.py`
(the two pre-existing dirty files, hash-verified), every Gate-6 and Gate-7A artifact,
manifest, prediction and report, and every released dataset directory. C3 and V3 remain
retired. Trident was not re-run.

---

## 15. Reproduction commands

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST="12.0a"
export HF_HUB_OFFLINE=1
PY=/home/minh/anaconda3/envs/cu128/bin/python

# 0. hash everything Gate 7B must not disturb, then pin the configuration
$PY tools/gate7b/stage0.py
$PY tools/gate7b/precommit.py

# 1. the two frozen-model passes: native streaming, and the calibrated-FOV MoGe cache
tools/gate7b/run_caches.sh

# 2. per-frame metric-gauge candidates (shared by all three policies)
for DS in semantickitti occ3d kitti360; do $PY tools/gate7b/scale_candidates.py --dataset $DS; done

# 3. the evaluation matrix: 12 predeclared configurations per benchmark, 4 GPUs
tools/gate7b/run_matrix.sh

# 4. map thickness and duplicate surfaces, and the forward-looking diagnostic
for DS in semantickitti occ3d kitti360; do
  $PY tools/gate7b/thickness.py     --dataset $DS --device cuda:0
  $PY tools/gate7b/recoverability.py --dataset $DS --device cuda:0
done

# 5. aggregate, plot and write the report
$PY tools/gate7b/aggregate.py
$PY tools/gate7b/figures.py
$PY tools/gate7b/report_tables.py --write

# 6. tests, including every Gate-6 and Gate-7A test
$PY -m pytest tests/gate6 tests/gate7a tests/gate7b -q
```

---

## 16. Deviations and blockers

{{DEVIATIONS}}
