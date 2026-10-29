# Continuous-runtime cost of the intended robot operating mode

**Measured 2026-10-28.** Repository `/home/minh/workspace/lingbot-map_fork`, branch
`feature-completion-study`, HEAD `db465c2ce2a692817d9ce4a141df7c24c4fa2d13`. Geometry only;
nothing semantic runs. Nothing was trained or re-predicted: the existing checkpoints ran in
their existing configurations.

Every number is read from a JSON written by the tool that measured it, under
`artifacts/method_vs_data/streaming_scale_validation/runtime/`. This file is the reading of
those records.

**This replaces no earlier measurement.** The previous phase measured the clip-reset mode and
OccAny; those files are preserved in `artifacts/method_vs_data/runtime/` and
`artifacts/method_vs_data/runtime_comparison.md`. What is new here is the **persistent-state,
startup-only-scale** mode measured directly, end to end, in the same session and on the same
idle GPU as its own clip-reset reference.

---

## 1. Measurement conditions

| | |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, index 3, 94.97 GiB, sm 12.0 |
| CPU | Intel Xeon w7-3565X, 64 threads |
| Software | torch 2.7.1+cu128, cuDNN 9.7.1, driver 580.95.05, Linux 6.8.0-90 |
| Batch | one frame per state update, one anchor per prediction |
| LingBotMap | `checkpoints/lingbot-map/204754b/lingbot-map.pt`, 518 px processed lattice, bf16 autocast, 64-block sliding patch window |
| MoGe-2 | `Ruicheng/moge-2-vitl` @ `39c4d5e957afe587e04eec59dc2bcc3be5ecd968` |
| Completion head | KITTI-360 seed 0, `geo_seed0_step06000.pt`, 967,322 trained parameters |
| Output grid | SemanticKITTI 256×256×32 @ 0.2 m; Occ3D predicted at 400×400×32 and read out to the canonical 200×200×16 |
| Resident models | LingBotMap GCTStream + MoGe-2 ViT-L + CompletionUNet, all three loaded for the whole run |
| Isolation | GPU 3 at **0 % utilisation, 615 MiB** at the start of the measurement — `runtime/isolation_record.txt` |

**On isolation.** An earlier attempt at these numbers was contaminated: an orphaned duplicate of
the benchmark chain was timing on the same GPU concurrently, inflating the sequence-average cost
from 124.5 ms to 211 ms. Every file from that attempt was renamed `DISCARDED_contended_*.json`
and is kept, unused, for the record. The measurement reported here waited for the duplicate to
exit on its own and reserved GPU 3 alone; the tight dispersion is the evidence that it worked
(per-input update std **0.3 ms** across 9 runs, versus 5.5 ms on startup and 1.7 ms on the
sequence average). The two remaining GPU-0/GPU-1 jobs in the isolation record belong to other
users and never touched GPU 3.

**Timing boundary.** Start: decoded RGB frames and ordinary calibration metadata in host memory.
End: the completed occupancy grid on the evaluation grid, as a host array available to the
evaluator. Inside the clock: image preprocessing to the 518 px lattice, host→device transfer,
the LingBotMap forward (bidirectional 5-frame startup block, then one causal forward per frame),
the MoGe-2 forwards and the metric-scale estimate that the mode actually requires, pose decode,
the incremental hard mapper, the hard-map query onto the prediction grid, the CompletionUNet
forward with its occupancy decision, and the read-out onto the evaluation grid. Excluded, and
stated: image decode, ground-truth loading, scoring, visualisation, prediction serialisation,
and model construction / checkpoint loading. **Model loading is reported separately** (§5).

**Nothing cached.** Depth, point maps, poses, scale and hard maps are all computed inside the
timed region, in both modes. The only non-image read is the calibrated horizontal FOV, one
scalar per frame, a deterministic function of ordinary calibration. The accuracy caches used in
`streaming_accuracy_comparison.md` were not read by any timed path.

**Workload.** A fixed pilot selected the workload before the main measurement.
SemanticKITTI: **3 streams × 200 processed frames × 3 repetitions = 9 runs**, the three evenly
spaced 200-frame windows of sequence 08 (815 stride-5 frames in total). Each 200-frame stream is a recorded
**103.4 s** of driving (measured frame cadence 0.519 s median). Occ3D: **3 streams × 40 frames ×
3 repetitions**, the median-length validation scene and its two neighbours in the length
ranking. **The 200-frame target is not attainable on Occ3D at all** — nuScenes validation scenes
are ~40 keyframe samples (~20 s), which is the dataset's limit, not a reduction I chose. Every
amortized Occ3D cost below is therefore a 40-frame average and must be read as such. Each
repetition starts from clean application state after a separate software/kernel warm-up; CUDA is
synchronised at every reported boundary.

---

## 2. Persistent state with startup-only scale — the headline costs

SemanticKITTI, 9 runs, KITTI-360 seed 0.

| quantity | mean | median | p95 | min–max | std |
|---|---|---|---|---|---|
| **startup computation** (5 MoGe-2 forwards + the bidirectional 5-frame LingBot block + the scale estimate) | **305.2 ms** | 306.2 | 311.8 | 296.4–313.2 | 5.5 |
| **first-prediction compute** (startup + the first anchor's map and completion) | **367.2 ms** | 367.6 | 373.4 | 358.2–374.6 | 5.3 |
| frame-acquisition waiting before the first prediction, *reported separately* | **2078.1 ms** | 2078.7 | 2079.4 | 2076.2–2079.4 | 1.4 |
| **ordinary input update** (one new frame into persistent state) | **60.5 ms** | 60.6 | 60.9 | 60.1–60.9 | 0.3 |
| **ordinary prediction step** (map + completion, on top of the update) | **61.6 ms** | 61.6 | 61.7 | 61.3–61.7 | 0.1 |
| **sequence-average cost per completed occupancy grid** | **124.5 ms** | 123.8 | 126.8 | 122.9–126.9 | 1.7 |
| measured predictions per second | **8.03 /s** | 8.08 | 8.14 | 7.88–8.14 | 0.11 |
| total sequence computation, 200 frames / 196 grids | **23.98 s** | 24.10 | 24.28 | 23.50–24.29 | 0.32 |

The 2,078 ms of frame-acquisition waiting is **not compute**: it is the four 0.519 s frame
intervals the robot must wait through before five frames exist. It is reported separately
exactly because it is a property of the sensor cadence, not of this pipeline, and it would be
paid by any method that needs five frames.

**Output cadence is disclosed, not hidden.** On SemanticKITTI 196 of the 200 processed frames
are canonical anchors and emit a grid (0.98 grids per processed frame); on Occ3D the ratio is
0.70. The **sequence-average cost includes startup and every intervening state update whether or
not it emitted a grid** — that is what makes 124.5 ms larger than the 61.6 ms prediction step.
Per-input-update cost (60.5 ms) and per-prediction cost (61.6 ms) are reported separately above,
as required.

**Real-time margin.** The measured frame cadence is 0.519 s median (max 0.522 s). The ordinary
update takes 60.5 ms of that, an **8.6×** margin; the full sequence-average cost per grid takes
124.5 ms, a **4.2×** margin. On Occ3D (0.500 s median cadence, max 0.601 s) the sequence-average
236.7 ms leaves a **2.1×** margin.

### Occ3D / nuScenes, same mode

| quantity | mean | median | p95 | std |
|---|---|---|---|---|
| startup computation | **350.5 ms** | 351.2 | 359.2 | 5.9 |
| first-prediction compute | **490.3 ms** | 490.9 | 499.3 | 6.5 |
| frame-acquisition waiting, separate | 2000.2 ms | 2000.0 | 2001.2 | 0.7 |
| ordinary input update | **67.7 ms** | 67.6 | 68.6 | 0.6 |
| ordinary prediction step | **139.5 ms** | 139.7 | 140.3 | 0.7 |
| sequence-average per completed grid | **236.7 ms** | 236.3 | 238.2 | 1.2 |
| predictions per second | **4.22 /s** | 4.23 | 4.25 | 0.02 |
| total sequence computation, 40 frames / 28 grids | 6.63 s | 6.62 | 6.67 | 0.03 |

Occ3D's prediction step is 2.3× SemanticKITTI's because the completion runs on a 400×400×32
grid (6.25× the voxels of 256×256×32) before the read-out to 200×200×16.

---

## 3. The two operating modes, measured side by side

Both rows are **direct measurements in their own mode** on the same GPU in the same session.
No cost was estimated by subtracting a stage from the other mode.

| operating mode | SemanticKITTI, per completed grid | Occ3D, per completed grid | MoGe forwards per grid | state resets per grid |
|---|---|---|---|---|
| clip-reset reference (mode A) | **368.6 ms** (med 368.2, p95 370.2, std 1.0) → 2.71 /s | **492.7 ms** (med 491.3, p95 497.7) → 2.03 /s | 5 | 1 |
| persistent state, startup-only scale (mode B) | **124.5 ms** sequence average → 8.03 /s | **236.7 ms** → 4.22 /s | **5 per stream** (0.026 per grid) | **1 per stream** |

**SemanticKITTI: 2.96× cheaper per completed grid. Occ3D: 2.08×.**

Both modes are retained as operating modes. The clip-reset numbers are *this session's*
re-measurement; they agree with the previous phase's 373.9 ms to within 1.4 %, which is the
independent check that the measurement conditions are the same.

### Where the saving comes from — stage profile

A separate profiling pass (1 repetition, `--mode stages`), so the instrumentation cannot distort
the primary sequence measurement above. Totals over one 200-frame stream, averaged over the 3
streams; frame stages normalised per processed frame, anchor stages per emitted grid.

| stage | clip-reset, per grid | streaming, per processed frame | streaming, per grid |
|---|---|---|---|
| preprocess (518 px lattice) | 21.36 ms | 14.80 ms | — |
| host→device + LingBot causal forward | — | **44.27 ms** | — |
| pose decode | — | 0.29 ms | — |
| LingBot bidirectional 5-frame block | **62.65 ms** | *startup only, 62.15 ms per stream* | — |
| **MoGe-2 (5 forwards)** | **215.01 ms** | *startup only, 214.89 ms per stream* | — |
| metric-scale estimate | 7.69 ms | *startup only, 7.76 ms per stream* | — |
| incremental hard mapper | 15.67 ms | — | 14.56 ms |
| hard-map query onto the prediction grid | 3.10 ms | — | 3.03 ms |
| **CompletionUNet + occupancy decision** | 43.42 ms | — | **42.73 ms** |
| read-out onto the evaluation grid | 0.23 ms | — | 0.23 ms |
| **sum** | **369.1 ms** (e2e: 368.6) | **59.4 ms** (e2e: 60.5) | **60.6 ms** (e2e: 61.6) |

**MoGe-2 is 58 % of the clip-reset cost per grid** (215.0 of 368.6 ms). Together with the
bidirectional 5-frame block and the scale estimate it is 285.4 ms, or 77 %. In the persistent
mode that whole block is paid **once per stream** (285 ms, amortising to 1.5 ms per grid over
196 grids) and is replaced by one 44.3 ms causal LingBot forward per frame.

This is a measured property of the mode, not an arithmetic estimate. **No number in this report
was obtained by subtracting MoGe's time from the clip-reset measurement**, and no streaming
latency is paired with clip-reset accuracy anywhere (§6).

**The startup-only calibration cost is reported as having no recurring refresh component only
because the measured primary mode genuinely makes none.** The state trace of every run records
`clean_kv_cache_calls: 0` and `mid_stream_resets: []`, with 5 MoGe forwards at startup and none
afterwards. `moge_calls` in the configuration record reads "5 at startup, none afterwards".

---

## 4. Memory

Peak torch memory **including the resident weights of all three models**.

| | startup allocated | startup reserved | sustained allocated | sustained reserved |
|---|---|---|---|---|
| SemanticKITTI, streaming | 11.089 GiB | 13.145 GiB | **11.828 GiB** | **13.145 GiB** |
| SemanticKITTI, clip-reset | — | — | 11.780 GiB | 13.119 GiB |
| Occ3D, streaming | 14.505 GiB | 20.555 GiB | **17.636 GiB** | **20.555 GiB** |
| Occ3D, clip-reset | — | — | 17.633 GiB | 20.551 GiB |

Persistent state costs **+0.74 GiB allocated over 200 frames** on SemanticKITTI (+3.13 GiB over
40 frames on Occ3D, whose grids are larger); reserved memory does not grow, so the allocator
absorbs it. Sustained streaming and clip-reset peaks are within 0.05 GiB of each other — the KV
cache is not the dominant term, the resident weights and the prediction grid are. Growth is
bounded by design, not by luck: the patch KV cache is a 64-block sliding window, and only the
6-per-frame special-token KV accumulates, up to `max_total_frames = 1124`. **I measured 200
frames and report 200 frames**; the behaviour at the 1,011-slot ceiling is a code-level bound
documented in `streaming_input_provenance.md`, not something these runs exercised.

---

## 5. Model loading, parameters, and head cost consistency

**Model loading, reported separately and excluded from steady-state inference:** LingBotMap
10.82 s, MoGe-2 4.22 s, CompletionUNet 0.016 s — **15.06 s total**, paid once at robot start-up.

**Inference parameters:** 1,485,120,083 resident (LingBotMap 1,157,943,540 + MoGe-2 326,209,221
+ CompletionUNet 967,322). **Task-specific trained parameters: 967,322** — the completion head is
0.065 % of the resident parameter count.

**Head cost consistency (§6 of the brief, in place of repeating the full study per source/seed).**
All three source heads are the same `CompletionUNet` architecture: identical 3-channel input,
identical 967,322 parameters, verified by assertion at load. The CompletionUNet forward plus the
occupancy decision, on identical hard-map inputs, 20 anchors × 5 repetitions
(`runtime/../runtime/unet_arm_consistency.json`):

| head | mean | median | p95 | std |
|---|---|---|---|---|
| KITTI-360 seed 0 | 42.864 ms | 42.806 | 43.167 | 0.197 |
| Waymo seed 0 | 42.811 ms | 42.781 | 42.996 | 0.113 |
| ONCE seed 0 | 42.861 ms | 42.831 | 42.983 | 0.109 |

Spread across sources **0.05 ms, 0.12 %** — within run-to-run noise. Everything upstream of the
head is arm-independent *by construction*, not by measurement: one shared map cache per
benchmark and mode feeds all heads, and the hard map is bit-identical across arms. So the full
timing study for the KITTI-360 head transfers to the Waymo and ONCE heads, and to seeds 1 and 2,
with no further measurement.

**The timed configuration is the scored configuration — bounded, not asserted.** The timed path
calls MoGe-2 live; the accuracy path read the frozen MoGe frame cache. Re-running 8 anchors
through both (`verify_stream_predictions_semantickitti.json`): the startup scale is
26.97912907370954 live against 26.980392156862735 cached, a relative difference of **4.7e-5**,
and the occupancy grids are **not bit-identical** — 413 to 1,166 of 2,097,152 voxels disagree per
anchor (0.02–0.06 %), minimum agreement IoU **99.70 %**. That bounds the discrepancy between
what was timed and what was scored; it does not claim bit-equality.

---

## 6. OccAny comparison — only where it is compatible

Both OccAny presets were rerun on the same idle GPU 3, in this measurement session, at their
released configuration and the same output boundary: `occany_K_1FRAME_GEOM_e2e_rerun.json` and
`occany_K_5FRAMES_GEOM_e2e_rerun.json`, 150 anchors × 3 passes each.

| system | temporal input | latency per completed grid | SC IoU raw | SC IoU pooled |
|---|---|---|---|---|
| ours, persistent state, startup-only scale | 5 causal frames, state since the stream start | **124.5 ms** (sequence average) | 24.87 | 25.23 |
| ours, clip-reset | 5 causal frames | **368.6 ms** | 26.40 | 26.34 |
| OccAny, 1 causal frame | 1 frame, no future | **1346.6 ms** (med 1346.0, p95 1374.2, std 16.4, n=150) | 20.32 | 24.06 |
| OccAny, 5 forward frames | t … t+40, i.e. **4.0 s of future** | **1675.9 ms** (med 1676.5, p95 1706.2, std 17.8, n=150) | 25.21 | 25.93 |

**Against the only causal OccAny configuration, our persistent-state pipeline is 10.8× faster
per completed grid and more accurate: +4.55 raw and +1.16 pooled, both significant.** That is
the comparison where both systems see only the past, and it is the one that matters for a robot.

**Against OccAny's forward-frame configuration our persistent mode is 13.5× faster, at 0.34 raw
IoU below it and 0.70 pooled below it — neither difference significant.** At the clip-reset
operating point the ratio is 4.55× and we are ahead on accuracy (+1.19 raw, significant). The
4.55× reproduces the previous phase's 4.56× on independent runs, and the rerun latencies land
**1.7 % below** the previous phase's 1704.4 ms (5 forward) and **2.0 % above** its 1320.0 ms
(1 causal) — the independent check that both sessions measured the same thing.

Each accuracy figure above comes from **the same operating mode as the latency beside it**. The
accuracy-versus-latency figure is built the same way and carries that statement in its own
record.

**What this comparison does not say.** OccAny's forward configuration consumes four seconds of
future frames. Its temporal inputs are **not matched to our causal stream**, and no shared
evaluation mask makes them matched. Its waiting time for those future frames is kept separate
from its compute and is not folded into the 1675.9 ms. Three asymmetries from the previous phase
still hold and still run against us, so every ratio above is the conservative one: our
preprocessing is inside our clock while OccAny's runs in DataLoader workers outside theirs
(measured separately at 30.7 ms); their script also voxelises a reconstruction-only grid the
compared prediction never uses (254.9 ms), and the ratios are quoted *including* that overcount
in their favour; and we read one calibrated-FOV scalar per frame from cache. Parameter counts:
1.485 G resident for us against 0.651 G for them — **we carry 2.28× more inference parameters and
are still 13.5× faster per grid in the persistent mode**.

Peak memory, for completeness: OccAny 10.42 GiB allocated / 11.33 GiB reserved (1 causal) and
12.67 / 14.47 GiB (5 forward), against our 11.83 / 13.15 GiB sustained. The two systems are in
the same memory class; the speed difference is not bought with memory.

---

## 7. What this section establishes, and what it does not

**Established.**
- The persistent-state, startup-only-scale mode was executed frame by frame with live MoGe,
  nothing cached, and measured directly: startup **305 ms**, ordinary update **60.5 ms**,
  sequence-average **124.5 ms** per completed grid, **8.03 grids/s**, on 103-second
  SemanticKITTI streams.
- It is **2.96× cheaper per grid than the clip-reset reference** on SemanticKITTI and 2.08× on
  Occ3D, measured in both modes directly, because the five MoGe forwards and the bidirectional
  startup block move from per-grid to per-stream.
- Memory is bounded over the measured duration: +0.74 GiB allocated over 200 frames, no growth
  in reserved memory, sustained peak within 0.05 GiB of clip-reset.
- The timing transfers across all nine heads: identical architecture, 0.12 % cost spread, and an
  arm-independent upstream by construction.

**Not established.**
- Nothing here measures the pipeline past 200 processed frames (103 s) on SemanticKITTI or 40
  frames (20 s) on Occ3D. The 8.8-minute RoPE ceiling is a code bound, not a measured duration.
- The 2.96× saving is a cost statement only. It is paired with a **lower** accuracy (−1.53 raw
  for this head and seed), and §6 pairs each latency with the accuracy of its own mode precisely
  so that the trade-off is visible rather than concealed.
- Occ3D's amortized costs rest on 40-frame scenes. They are not comparable to SemanticKITTI's
  200-frame amortization, and the dataset admits no longer stream.
- The OccAny comparison is a speed comparison at a matched output boundary on matched hardware.
  Against their **causal** preset it is also a temporally matched accuracy comparison, and we win
  both. Against their **forward** preset it is **not** temporally matched — the four-second
  lookahead is a different problem setting, and the near-tie in accuracy there should be read
  with that in mind, not as a like-for-like result.

## 8. Reproduction

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/home/minh/anaconda3/envs/cu128/bin:/usr/local/cuda-12.8/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST="12.0a"
export PYTHONPATH=/home/minh/workspace/lingbot-map_fork

T=tools/method_vs_data/streaming/runtime_stream.py
python $T --dataset semantickitti --device cuda:3 --operating-mode stream    --mode e2e
python $T --dataset semantickitti --device cuda:3 --operating-mode clipreset --mode e2e
python $T --dataset semantickitti --device cuda:3 --operating-mode stream    --mode stages --reps 1
python $T --dataset semantickitti --device cuda:3 --operating-mode clipreset --mode stages --reps 1
python $T --dataset semantickitti --device cuda:3 --verify-predictions 60
python $T --dataset occ3d --device cuda:3 --operating-mode stream    --mode e2e
python $T --dataset occ3d --device cuda:3 --operating-mode clipreset --mode e2e

SCRATCH=/media/SSD1/occany_causal CUDA_VISIBLE_DEVICES=3 \
  /home/minh/workspace/third_party/occany_env/bin/python \
  tools/method_vs_data/runtime/bench_occany.py --preset K_5FRAMES_GEOM \
  --anchors artifacts/method_vs_data/runtime/anchors_semantickitti.json \
  --mode e2e --out artifacts/method_vs_data/streaming_scale_validation/runtime/occany_K_5FRAMES_GEOM_e2e_rerun.json

python tools/method_vs_data/streaming/figures.py --which latency
```

Run the timing on an idle GPU and check `runtime/isolation_record.txt` afterwards. A contended
GPU inflates the sequence average by roughly 70 % — the discarded attempt is kept as
`DISCARDED_contended_*.json` to show what that looks like.
