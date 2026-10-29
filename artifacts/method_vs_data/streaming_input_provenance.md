# Streaming-validation phase, part 1: what the three source heads were actually trained on, and what the persistent-state mode changes

Scope: the input-side provenance needed before any streaming result can be read. It audits
the reconstruction-state policy, the metric-scale policy and the mapping contract of all
three training sources **from the code and the caches**, records two places where a
historical description does not survive that audit, and documents the state machine the new
persistent mode actually runs on.

Repository `/home/minh/workspace/lingbot-map_fork`, branch `feature-completion-study`,
`HEAD = db465c2`. Working tree at the start of the phase: two historical reports with
append-only addenda, the pre-existing unrelated modification to `lingbot_map/heads/dpt_head.py`
(65 lines, **not touched by this phase**), and the untracked artifacts of the previous phase.
StreamingSSC was not modified.

---

## 1. The question this part answers

The main KITTI-360 training was *reported* to use "independently scaled, locally reset
five-frame clips". Waymo and ONCE were reported to use five input frames, which does not by
itself say anything about whether their state was reset or how their scale was fixed — a
five-frame input is equally consistent with a persistent-state cache. The policies were
therefore read out of the code that wrote each cache, per source, rather than inferred.

The answer is that **all three are the same policy**, and the table below gives the lines
that establish it.

---

## 2. Per-source reconstruction and gauge policy, as implemented

| | **KITTI-360** (reference) | **Waymo** | **ONCE** (exploratory) |
|---|---|---|---|
| clip writer | `tools/baseline_reconstruction/targets/clips.py::infer_clip` | `tools/waymo_pilot/waymo_geometry.py::build_clip` | `tools/method_vs_data/once_arm/geometry.py::build_clip` |
| state reset | `model.clean_kv_cache()` **before every clip** | `model.clean_kv_cache()` **before every clip** | `model.clean_kv_cache()` **before every clip** |
| reset verified by | `assert frame_index_trace(model) == 0` *before* the pass | identical assertion | identical assertion |
| LingBot call | `replay.replay_segment(model, imgs, 1, dev)` | identical | identical |
| frames per clip | 5, offsets `(-4,-3,-2,-1,0)` | 5, same offsets (`waymo_pilot.config.CLIP_OFFSETS`) | 5, `input_clip_indices(t) = [t-4 … t]` |
| state retained across clips | **none** | **none** | **none** |
| MoGe-2 calls | **5 per clip**, one per frame | **5 per clip**, one per frame | **5 per clip**, one per frame |
| MoGe-2 revision | `39c4d5e957afe587e04eec59dc2bcc3be5ecd968` | asserted equal to that revision at build time | same |
| per-frame candidate | `gates.gate7b.scale.frame_candidate` | identical | identical |
| scale aggregation | `exp(nanmedian_q log_s[q])` over the clip's own 5 | identical | identical |
| scale entry point | `IncrementalMapper.scale_state.force(scale)`, once | identical | identical |
| applied to depth | `d_m = s · depth_canonical` | identical | identical |
| applied to pose translation | `T[:3,3] *= s`, the **same scalar** | identical | identical |
| mapper | `core.mapping.mapper.IncrementalMapper`, `sem_into=None`, `n_teacher=0` | identical (`build_local_geometry`, imported) | identical |
| pose re-basing | `inv(P_anchor) @ P_q`, anchor = slot 4 | identical | identical |
| network channels | `[S, 0, F]`, 3 channels | identical | identical |
| grid | `(256,256,32)` @ 0.2 m, anchor sensor frame | identical | identical |
| preprocessing | `load_and_preprocess_images(mode='crop', 518 px, patch 14)` | identical | identical |

Conclusion: **the reported KITTI-360 policy is in fact the policy of all three arms.** Waymo
reuses the KITTI-360 function by import (`tools/waymo_arm/cli/build_inputs.py` calls
`tools.baseline_reconstruction.current.geometry.build_local_geometry`), and ONCE reimplements
the same eight steps with the same frozen helpers. No source arm was trained on
persistent-state reconstruction, and none was trained on a startup-only scale. That is the
central asymmetry of this phase and §5 of the accuracy report returns to it.

### Frozen numeric constants (one set, shared by all three)

| group | values |
|---|---|
| mapper | voxel 0.2 m, conf threshold 1.5, depth window 1.0–60.0 m |
| rays | occupied band half-width 0.2 m, carve stride 4, carve near 1.0 m |
| log-odds | occupied +0.85, free −0.40, occupied-at > 0, clamp ±4.0, MoGe-only confirmations 2 |
| scale estimator | 3× MAD clip, ≥ 1000 jointly valid pixels, 5 anchor frames |
| LingBot | 518 px, patch 14, 5 scale frames, 64-frame sliding KV window, bf16 autocast |

### Checkpoints and seeds available

All nine exist and all nine are evaluated in this phase. The KITTI-360 arm is additionally
sha256-verified against `artifacts/baseline_reconstruction/code_freeze.json` at load time by
`load_frozen_geometry`.

| source | seed 0 | seed 1 | seed 2 | bytes | status |
|---|---|---|---|---|---|
| KITTI-360 | `7f5e853dd05bdbde…` | `2c4d79be4fa7766e…` | `6038b1f7d901a751…` | 3,887,295 | frozen baseline |
| Waymo | `e40fb93467e4d37c…` | `f06ff8a059bc41c2…` | `87cbad3cfc86aead…` | 3,887,359 | arm |
| ONCE | `03fc0d26fc8396d5…` | `f469f80d0cb66aba…` | `2057b05e2d230416…` | 3,887,713 | **exploratory** |

Each is `geo_seed{N}_step06000.pt`, the final step, selected by no criterion. Every head has
967,322 parameters and 3 input channels, asserted at load.

### Supervision window and sensor coverage (unchanged from `training_supervision_provenance.md`)

KITTI-360 and Waymo were supervised with ~10 s future-only targets (21 frames; Waymo 5 lasers
per frame); ONCE with ~2 s (5 frames, single 40-beam LiDAR). ONCE's original occupied-prevalence
gate **remains failed** (0.0131 against a predeclared band) and every ONCE row in this phase is
labelled exploratory. Because the ONCE target window differs from the other two, an ONCE-vs-
KITTI-360 difference is **not** an isolated dataset-identity experiment, and no table in this
phase describes it as one.

---

## 3. The state machine the persistent mode actually runs on

Read out of `lingbot_map/layers/flashinfer_cache.py` and `lingbot_map/aggregator/stream.py`,
because the duration a robot can run before something resets is a property of this machine and
not of the protocol.

| element | value | consequence |
|---|---|---|
| patch KV, scale frames | 5 frames, **never evicted** | the initialization frames stay in context for the whole stream |
| patch KV, recent window | **64 frames**, evicted beyond that | patch-level memory is a ~33 s window at 0.52 s cadence |
| special-token KV | 6 tokens/frame (camera + registers + scale) for **every** frame processed, packed | long-range memory survives eviction as special tokens, not as patches |
| special-token page budget | `max_total_frames = max_frame_num + 100 = 1124` | hard ceiling on frames per unbroken stream |
| RoPE table | `max_frame_num = 1024`, margin 8 | at keyframe interval 1, **1011 streamed frames** fit |
| keyframe rule | `gates.gate7b.replay.keyframe_interval_for` | a longer stream raises the interval, so non-keyframes consume no RoPE slot and the stream stays bounded indefinitely at reduced KV density |

**Supported duration, measured not assumed.** SemanticKITTI sequence 08 is one unbroken
815-frame stream spanning **422.9 s (7.05 min)** and it ran at keyframe interval **1** using
**815 of 1011** available RoPE slots, with the state reset exactly once. The ceiling at
interval 1 is therefore ~1011 frames ≈ **8.8 min** at this benchmark's 0.52 s cadence; past
that the capacity rule raises the interval rather than resetting. Each Occ3D scene is 39–41
samples, ~20 s, far inside every limit. No stream in this phase required a mid-stream reset,
and the reset log for every segment is in
`artifacts/method_vs_data/streaming_scale_validation/logs/build_maps_*_modeB.json`.

**Observed resets per pass: three `clean_kv_cache` calls, none mid-stream.** `replay_segment`
calls it, `inference_streaming` calls it again at entry, and `inference_streaming` calls it a
third time at exit. The instrumentation in `tools/method_vs_data/streaming/trace.py` records
the frame counter *at each call*: the values are `[0, 0, 815]` for SemanticKITTI and `[0, 0, n]`
for every scene — two before the first frame, one after the last. A reset strictly between
would have broken the mode, and none occurred.

---

## 4. Two historical descriptions that do not survive the audit

**(a) `rope_slots_used` is 0 for every pass ever recorded, and cannot evidence anything.**
`gates/gate7b/replay.replay_segment` reads the RoPE frame counter *after*
`inference_streaming` returns — but `inference_streaming` ends with `clean_kv_cache()` whenever
`output_device` is set, and that call zeroes `aggregator.total_frames_processed`. Every
`rope_slots_used` field in every clip cache and every gate log is therefore 0 regardless of
what the pass did.

`tools/baseline_reconstruction/reports/verification.py:93` states: *"clip rope_slots_used = 0
(five frames run as the batched scale block after clean_kv_cache)"*. The **conclusion** is
correct — with `S = 5` and `num_scale_frames = min(5, 5) = 5`, phase 2's loop `range(5, 5)` is
empty, so the five frames really are one bidirectional block — but the **evidence offered for
it is void**, because a 815-frame stream also records 0. The claim now rests on the loop bound,
and this phase measures slot usage from the counter's *peak* instead
(`trace.StateTrace.peak_frames`), which returns 815 for SemanticKITTI and 39–41 per Occ3D
scene, matching `rope_slots()` exactly. No result in this phase reads the broken field. The
frozen gate code was **not** modified.

**(b) the clip-reset reference is not the more causal of the two modes.** `inference_streaming`
processes the first `num_scale_frames` frames **as one block with bidirectional attention
among them**. In the clip-reset reference every clip has exactly five frames and
`num_scale_frames = 5`, so **all five frames of every reference clip attend to each other** —
the anchor's four predecessors are refined using the anchor, and each predecessor is refined
using its own successors. In the persistent mode only stream positions 0–4 sit in such a block;
from position 5 on, each frame is a strictly causal single-frame forward.

Neither mode ever uses a frame **later than the anchor**, which is the causality requirement
that matters and which the pilot verifies directly. But the common intuition that the
clip-reset reference is the stricter protocol is the wrong way round at the per-frame level,
and §2 of the accuracy report states this where the comparison is made.

---

## 5. Pilot validation of the new mode (all six checks pass on both benchmarks)

`tools/method_vs_data/streaming/pilot.py`, logged to
`artifacts/method_vs_data/streaming_scale_validation/logs/pilot.json`. SemanticKITTI segment
08; Occ3D `scene-0278`, chosen *before* the pilot ran as the scene whose median five-frame
window baseline is closest to the median over all 150 validation scenes (11.76 m vs 11.87 m) —
a declared, ground-truth-free representativeness criterion. The alphabetically first scene,
`scene-0003`, is near-stationary at its start (0.01 m windows) and cannot resolve the units
probe at all, which is why the choice is explicit.

| check | what it proves | result |
|---|---|---|
| **C1** state & resets | resets only at the stream boundary; the frame counter reaches the predicted slot count | PASS — counters `[0, 0, 52]` / `[0, 0, 34]`, zero mid-stream resets, peak = predicted |
| **C2/C7** causality & replay | the stream truncated at *N* and at *N+M* gives **bit-identical** depth, confidence, pose and intrinsics for every frame < *N* | PASS — max abs difference 0.0 on all four fields |
| **C3** startup-only gauge | exactly 5 MoGe reads per segment, scale frozen at position 4, one scalar across all anchors, `force()` leaves no path for a later update | PASS — 5 reads, 1 distinct scale |
| **C4** metric units | multiplying **only** the pose translation by α and measuring cross-frame surface disagreement over the window's full baseline | PASS — minimum exactly at α = 1 on both; SemanticKITTI 0.476 m at α=1 vs 0.993 m at α=0.5 and 1.946 m at α=2.0 |
| **C5** map window | the mapper integrates exactly the five frames *t−4…t*, matching the reference manifest anchor by anchor | PASS — 5 integrations, windows identical |
| **C6** completion inputs | `[S, 0, F]`, frozen dims/voxel size, and the end-to-end form: mode-A maps reproduce the **stored canonical prediction bit for bit** | PASS — 10/10 bit-identical, occupancy *and* hard-map support |

C6 deserves emphasis: running the frozen KITTI-360 seed-0 head over the mode-A maps this phase
built lands on the stored canonical volumes **exactly**, for occupancy and for the observed-support
mask. If the grid, the channel meaning, the coordinate convention, the pooling or the read-out had
drifted anywhere, that could not happen. Mode A is the historical procedure, not a
reimplementation of it, and every A-vs-B difference is therefore a property of mode B.

The C4 note on the α < 1 branch: shrinking the translation collapses the two views onto each
other, so the median disagreement saturates rather than growing without bound; the discriminator
is the local well around α = 1 together with the strictly monotone upper branch, both of which
hold on both benchmarks.

---

## 6. Stream manifests, written before the main run

`artifacts/method_vs_data/streaming_scale_validation/manifests/stream_manifest_{semantickitti,occ3d}.json`.

| | SemanticKITTI | Occ3D-nuScenes |
|---|---|---|
| segments (independent streams) | **1** (sequence 08) | **150** (official validation scenes) |
| frames processed | 815 | 6,019 |
| duration | 422.9 s | 19–21 s per scene |
| frame gap (median / max) | 0.520 s / 0.522 s | 0.500 s / 0.651 s |
| scored anchors | **801 / 801** canonical | **4,219 / 4,219** canonical |
| exclusions | **none** | **none** |
| mode-A resets | 801 (one per anchor) | 4,219 |
| mode-B resets | **1** | **150** |
| keyframe interval | 1 | 1 |
| RoPE slots used | 815 of 1011 | 39–41 of 1011 |

**No anchor is excluded by the initialization requirement**, and that is a property of the
canonical population rather than a lucky coincidence: a causal window needs four earlier
frames, so every canonical anchor already sits at stream position ≥ 4, and the mode-B gauge is
complete at position 4. The manifest asserts this per segment. Both modes are therefore scored
on the **identical, full** canonical populations, and no paired-subset recomputation is needed.

MoGe-2 calls, mode B: **5 per segment** — 5 in total for SemanticKITTI's 7-minute stream, 750
across Occ3D. Mode A calls it 5 times per anchor: 4,005 and 21,095 respectively.

---

## 7. Startup scales measured, and one consistency check worth recording

SemanticKITTI's single startup scale is **26.980392156862735** m per canonical unit. The
clip-reset scale of anchor `08_000020` — whose own five-frame clip *is* stream positions 0–4,
the same five images as the initialization window — is **26.980392156862735**. Identical to
every digit, which is the expected result and confirms the two gauges are the same estimator
differing only in which frames they are given.

Per-segment startup scales, candidate log-scales, valid-pixel counts, reset positions, RoPE
slot usage and per-anchor map coverage are in
`logs/build_maps_{semantickitti,occ3d}_modeB.json`.

---

## 8. What this part does NOT establish

* It does not show that persistent state or startup-only scale **works**. That is the accuracy
  report's question.
* It does not isolate the effect of calibration frequency. Mode B changes the reconstruction
  state and the calibration policy **together**, by construction.
* It does not establish that a stream longer than ~8.8 min at keyframe interval 1 behaves the
  same way. Nothing longer was run, and the capacity rule's behaviour past that point is read
  from the code, not measured.
* The ONCE arm's supervision window differs from the other two, so no ONCE comparison here or
  later is a clean dataset-identity experiment.

## 9. Reproduction

```bash
export CUDA_HOME=/usr/local/cuda-12.8 FLASHINFER_CUDA_ARCH_LIST="12.0a"
export PATH=/home/minh/anaconda3/envs/cu128/bin:/usr/local/cuda-12.8/bin:$PATH
cd /home/minh/workspace/lingbot-map_fork

# manifests (written before the main run)
PYTHONPATH=$PWD python -c "from tools.method_vs_data.streaming import streams as S; \
  [S.write_manifest(d) for d in ('semantickitti','occ3d')]"

# pilot validation
PYTHONPATH=$PWD python tools/method_vs_data/streaming/pilot.py --device cuda:1

# shared map caches
PYTHONPATH=$PWD python tools/method_vs_data/streaming/build_maps.py \
    --dataset semantickitti --mode A --device cuda:3
PYTHONPATH=$PWD python tools/method_vs_data/streaming/build_maps.py \
    --dataset semantickitti --mode B --device cuda:1
PYTHONPATH=$PWD python tools/method_vs_data/streaming/build_maps.py \
    --dataset occ3d --mode A --device cuda:2
PYTHONPATH=$PWD python tools/method_vs_data/streaming/build_maps.py \
    --dataset occ3d --mode B --device cuda:1
```

Map caches: `/media/SSD1/MINH_DATASETS/lingbot_streaming_validation/maps/<dataset>/mode<A|B>/`,
one compressed npz per anchor holding packed `(S, F)` plus the gauge, the five frame
positions/keys, the anchor transform and the map coverage.
