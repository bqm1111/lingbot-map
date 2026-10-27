# Computational cost for a real-world (robot) deployment: ours vs OccAny

Written 2026-10-20. Companion to [compute_cost_vs_occany.md](compute_cost_vs_occany.md), which costs the *benchmark*
runs. This report asks a different question: one new camera frame arrives, so what does it cost to produce an updated
occupancy (and semantic) estimate? Nothing was re-run, because all four GPUs were busy with other jobs. Every number
is read from an existing log or report, cited inline. The hardware is the desktop RTX PRO 6000 Blackwell Max-Q, **not**
robot hardware.

---

## 1. First: when can MoGe run "only a few times"?

Two LingBot configurations exist, and they treat scale differently.

| configuration | LingBot canonical scale | MoGe needed | evaluated accuracy |
|---|---|---|---|
| **L — 5-frame local reset** (the evaluated system, `CURRENT_SYSTEM.md` § C) | re-initialised **every clip** | a fresh metric scale **per clip** (median over the clip's 5 frames) | all paper numbers |
| **S — one continuous stream, frozen gauge G-A** (`core/mapping/scale.py::ScaleState`) | one scale for the whole stream | **5 frames at start-up, then never** | measured **worse** than L |

Freezing the scale after the first five frames is only valid in configuration S. That is the configuration the
project measured and rejected:

- Gate 7B (`reports/gate7b/streaming_metric_semantic_replay.md` § 12): **S1|G-A vs S0 (clips):
  −0.062 binary IoU on SemanticKITTI 08 and −0.094 on Occ3D**, with both intervals excluding zero.
- Local-registration audit (`reports/local_registration_audit.md`): continuous streaming drifts. The median
  reprojection error is **128 px** under S and **47 px** under L, and the audit froze L as the geometry representation.

G-A itself is a good gauge *within* a stream: frozen after 5 frames, it sits 1–7 % from the whole-sequence median
(Gate 7B § 4). The weakness is the continuous stream's pose drift, not the frozen scale.

**MoGe-2 is also cheap**: 0.06 s per frame (`artifacts/gate7b/logs/moge_semantickitti.log`). That is **≈ 2 % of the
semantic per-frame cost**, or ≈ 17 % of geometry-only. Removing it changes little.

There are three honest options for the robot experiment:

| option | what it is | MoGe calls | accuracy status |
|---|---|---|---|
| **A** | deploy L as evaluated, MoGe once per new frame (cached, so each frame is run once) | every frame, 0.06 s | **validated** |
| **B** | L, with MoGe on the first clip only, then each new clip's scale carried over by matching depth on the 4 frames it shares with the previous clip | 5 at start-up | **untested**: scale becomes a random walk, and its drift must be measured |
| **C** | S with G-A | 5 at start-up | **measured worse** (above); RoPE-limited (§ 4) |

If the paper says "MoGe runs only a few times", it must be option B with a measured drift, or option C with its
accuracy loss stated. Option A is the only one whose accuracy is already established.

---

## 2. Per-frame cost (SemanticKITTI-like camera, 1226×370, one B3 seed)

Seconds per new frame, serial on one GPU.

| stage | A (validated) | B (scale carried over) | C (stream, G-A) | source |
|---|---:|---:|---:|---|
| LingBot | 0.19 (5-frame clip) | 0.19 | 0.046 (1 frame, 21.6 FPS) | `occany_comparison/logs/clips_semantickitti.log`; Gate 7B § 3 |
| MoGe-2 | 0.06 | ≈ 0 (one-off 0.3 s) | ≈ 0 (one-off 0.3 s) | `gate7b/logs/moge_semantickitti.log` |
| hard map | 0.04 (5-frame map) | 0.04 | 0.18 median / 0.40 p95 (persistent map) | `final_zero_shot/logs/efficiency.json`; Gate 7B § 11 |
| CompletionUNet | 0.05 | 0.05 | 0.05 | `efficiency.json` |
| **geometry output** | **≈ 0.35 s (≈ 2.9 Hz)** | **≈ 0.29 s (≈ 3.4 Hz)** | **≈ 0.28 s (≈ 3.6 Hz)** | |
| Trident-H tokens | 1.69 | 1.69 | 1.69 | `final_zero_shot/logs/tokens_sk_s*.log` |
| lift + B0 + B3 (1 seed) | 1.39 | 1.39 | 1.39 | `efficiency.json` |
| **semantic output** | **≈ 3.4 s (≈ 0.29 Hz)** | **≈ 3.4 s** | **≈ 3.4 s** | |

**Pipelining.** Trident depends only on the image, so it can run on a second GPU concurrently with LingBot → map →
UNet. Semantic *throughput* is then bounded by the slowest branch, ≈ 1.7 s (≈ 0.6 Hz), but *latency* stays ≈ 3.4 s. A
practical robot design runs geometry at ≈ 3 Hz and refreshes semantics at ≈ 0.3–0.6 Hz. Newly completed voxels then
carry labels up to one semantic period old.

On an Occ3D-like camera (1600×900), Trident takes 2.09 s and the LingBot clip 0.29 s
(`final_zero_shot/logs/tokens_occ3d_*.log`, `occany_comparison/logs/clips_occ3d_*.log`).

**OccAny on the same frame** (`/media/SSD1/occany_causal/logs/`, GPU-seconds = throughput, not latency):

| OccAny setting | per output | rate | note |
|---|---:|---:|---|
| 1-frame geometry | 1.7 GPU-s | ≈ 0.6 Hz | causal |
| 5-frame semantic, causal window | 12.2 GPU-s + 0.43 GroundingDINO | ≈ 0.08 Hz | causal |
| 5-frame semantic, forward window (published) | 12.4 GPU-s + 0.43 | — | **not deployable online** (§ 3.1) |

| ratio (ours option A / OccAny) | geometry | semantic |
|---|---:|---:|
| per-output GPU time | ≈ 0.35 vs 1.7 → **≈ 5× cheaper** | ≈ 3.4 vs 12.6 → **≈ 3.7× cheaper** |

Neither system is real-time at a 10 Hz camera rate with semantics. Gate 7B already concluded that ours should be called
**online** (causal, incremental), not real-time, and that holds here.

---

## 3. Caveats on real-world time complexity

### 3.1 OccAny

1. **Its published setting looks into the future.** The KITTI `5frames` setting uses views at offsets 0, 2, 4, 6, 8
   *after* the target. Online, the estimate for time *t* exists only once frame *t*+8 has arrived (0.8 s at 10 Hz),
   plus 12+ s of compute. Only its causal and 1-frame settings are fair real-world baselines. The causal 5-frame
   setting is the matched one (pooled SemanticKITTI SC IoU tie, `reports/paper_numbers_occany_comparison.md`).
2. **It reuses nothing across outputs.** Each output re-runs Must3R on its window, renders ≈ 10 novel views per input
   view, and runs SAM2 distillation and propagation over them. Its cost per output is proportional to frames ×
   views-per-interval, whatever the overlap with the previous output. Semantics are ≈ 85 % of the cost (1-frame
   geometry 3.4 s vs semantic 23.1 s per-worker wall).
3. **Its times are pessimistic.** Every OccAny run logged the slow PyTorch RoPE2D fallback, and two workers shared one
   GPU, so single-output latency lies between 12.4 and 24.7 s. A tuned deployment (compiled curope, fewer rendered
   views via `-vpi`, larger `--batch_gen_view`) would be faster. How much faster, and at what accuracy cost, is
   unmeasured.
4. **GroundingDINO boxes** are precomputed offline in the benchmark. On a robot they are on the critical path
   (0.43 GPU-s per frame).

### 3.2 Ours

1. **Constant per frame in the evaluated configuration, but with redundant work.** L re-runs LingBot on all 5 frames
   for every output, so each frame passes through the 1.16 B model 5 times. That is ≈ 4× the per-frame LingBot cost of
   streaming (0.19 vs 0.046 s), the price of the drift fix. Neither the map nor the evidence grows with drive length,
   because the hard map is rebuilt from the 5-frame window each time.
2. **Continuous streaming does not scale indefinitely.** The KV cache is bounded (sliding window 64 plus 5 scale
   frames), but the RoPE table covers 1,024 frame indices. Past that, inference crashes, and beyond ~320 frames it
   already extrapolates past training (`CLAUDE.md`, *RoPE limit*). Option C therefore needs periodic resets anyway,
   and each reset starts a new canonical scale, which means MoGe again. "MoGe a few times" really means a few times
   *per reset*. A persistent map also grows with distance driven (Gate 7B: 226 MiB dense evidence, 0.18 s median map
   update vs 0.04 s for a 5-frame map).
3. **The semantic stages are scene-dependent, not constant.** B3 costs ≈ 2.8 µs per predicted-occupied voxel (369,670
   queries → 1.04 s on sample `08_000635`), and B0's EDT scales with grid size. Cluttered scenes cost more. The table
   above is one sample plus log means, and a deployment report should give p95 per-frame latency.
4. **Trident-H is the bottleneck, and it scales with image resolution** (1.69 s at 1226×370, 2.09 s at 1600×900). The
   logged time includes SAM ViT-H refinement, which `efficiency.json` says the token feature does not need. That claim
   is unverified, and removing SAM would also cut 641 M resident parameters.
5. **Memory on robot hardware.** About 2.84 B parameters must stay resident (LingBot 1.16 B in bf16, MoGe 0.33 B,
   Trident image side 1.36 B with SAM ViT-H in fp32), against ≈ 1.11 B for OccAny. Measured peaks: the per-sample stage
   5.5 GB, the LingBot stream 9–13 GiB (Gate 7B). The sum may not fit an embedded GPU (for example a Jetson Orin)
   without dropping Trident or quantising. On such hardware our parameter disadvantage matters more than our time
   advantage.
6. **The scale initialisation is a single point of failure in options B and C.** If the first five frames have fewer
   than 1,000 valid MoGe pixels in 1–60 m (a robot starting against a wall), the frozen scale is wrong for the rest of
   the run. Because MoGe is cheap, a periodic re-check (for example one frame every few seconds, compared with the
   frozen scale) is nearly free insurance.

### 3.3 Both

- **Not robot hardware.** These are desktop Blackwell GPUs shared with other users' jobs, and batch throughput rather
  than an isolated per-frame benchmark.
- **Not counted:** model load at start-up, image capture, preprocessing and transfer, and writing outputs. Some logged
  stages include disk I/O for cached tokens and NPZ outputs, which a live system would not have.
- **For the robot experiment**, run a per-frame latency benchmark on the robot's own GPU: median and p95 per stage,
  `torch.cuda.synchronize` around each stage, peak memory, and OccAny with compiled curope. Run option B's scale drift
  against the option-A per-clip scale on the same drive before choosing B.
