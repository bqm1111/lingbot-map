# Complete inference cost: our causal pipeline against the released OccAny checkpoint

**Measured 2026-10-28.** Repository `/home/minh/workspace/lingbot-map_fork`, branch
`feature-completion-study`, HEAD `db465c2ce2a692817d9ce4a141df7c24c4fa2d13`. Geometry only;
nothing semantic runs on either side. Nothing was trained, retrained or re-predicted: both
systems ran their existing checkpoints in their existing configurations.

Every number below is read from a JSON written by the tool that measured it. The generated
tables are `artifacts/method_vs_data/runtime/tables.md`; this file is the reading of them.

---

## 1. What was timed, and what was not

**Primary boundary.** Start: the decoded RGB frames and ordinary calibration metadata are in
host memory. End: the completed occupancy grid for the evaluation grid is a host array
available to the evaluator. Both sides use that boundary, batch size one evaluation anchor,
and the same physical GPU.

**Inside the clock for us**, all of it predicted, none of it read from cache: image
preprocessing to the 518 px processed lattice, the LingBotMap streaming forward over the clip
with its KV cache cleared, five MoGe-2 forwards, the metric-scale estimate from those five
candidates, the incremental hard mapper's five steps, the hard-map query onto the prediction
grid, the CompletionUNet forward and the occupancy decision, and the read-out onto the
evaluation grid.

**Inside the clock for OccAny**: the encoder / decoder / raymap-encoder / generation-decoder
forward that produces its reconstruction and its 150 generated novel views, the confidence
thresholding at its released thresholds, the transform into the voxel frame, and the
voxelisation of the combined reconstruction + generation point set to a host array.

**Excluded for both, and stated**: ground-truth voxel labels and LiDAR, scoring, visualisation,
serialisation of the prediction file, and model construction / checkpoint loading.

Image **decode** is outside the stated boundary. On our side it cannot be separated from the
timed preprocessing stage, because the frozen preprocessing call opens the files itself; the
timed 20.93 ms therefore *includes* a warm-page-cache decode. The separately reported 43.2 ms
is a **cold-cache** read taken once per anchor before the warm-up, so it is not subtractable
from the 20.93 ms and no subtraction is performed. On OccAny's side the two are separable and
were measured separately: decode 46.7 ms, required preprocessing **30.7 ms**. Our timed
preprocessing including a warm decode (20.9 ms) is smaller than their decode-free preprocessing
(30.7 ms), so this asymmetry does not favour us.

**Three asymmetries, disclosed rather than corrected.**

1. *Preprocessing placement.* Our preprocessing is inside the clock (20.9 ms). OccAny's happens
   in `DataLoader` worker processes, exactly as in the run that produced its accuracy, so it is
   outside their clock. It was measured separately on the main thread at **30.7 ms** per anchor
   for the five frames (`occany_preprocess.json`), excluding decode and excluding the
   ground-truth and SAM/GroundingDINO work in their `__getitem__` that a geometry prediction
   does not need. Adding their 30.7 ms to their total would make them slower, not faster, so
   every ratio below is the conservative one.
2. *An overcount against OccAny.* Their script also voxelises a reconstruction-only grid that
   the compared prediction never uses. That leg was measured separately at **254.9 ms**; the
   ratio is reported with and without it.
3. *Calibrated FOV.* We read one scalar per frame, the calibrated horizontal FOV, from the
   frame cache. It is a deterministic function of the camera intrinsics and the processed
   width — calibration metadata, not a prediction — and it is listed here because it is the
   only thing the timed path reads that is not an image.

**Information availability is not matched and is not treated as if it were.** Our five frames
are `t-4 … t`, all causal. OccAny's five-frame configuration reads stream offsets 0/2/4/6/8,
i.e. native frames `t … t+40` at 10 Hz: **4.0 s of future**. Its one-frame configuration is the
only genuinely causal OccAny mode. Future-frame waiting time is reported in §6 and is never
added into the latency figures.

## 2. Configurations

| | ours | OccAny |
|---|---|---|
| checkpoint | LingBotMap `204754b`, MoGe-2 `Ruicheng/moge-2-vitl` rev `39c4d5e9…`, CompletionUNet seed 0 step 6000 | `checkpoints/occany.pth`, 3,745,124,158 B, released |
| preset | the deployed causal pipeline | `sh/exp_lists/occany.sh` EXP_ID 0 and EXP_ID 1, verbatim |
| novel-view settings | none — no view synthesis in this pipeline | `-rot 60 -vpi 10 -fwd 3 -seed_trans 2`, `--batch_gen_view 12` (5-frame); `-vpi 50 -fwd 1`, `--batch_gen_view 16` (1-frame) |
| confidence thresholds | occupancy log-odds > 0 | `--recon_conf_thres 2.5 --gen_conf_thres 2.5` (5-frame), `2.0 / 2.0` (1-frame) |
| input resolution | 518 px processed lattice | 512 × 160 |
| output grid | 256 × 256 × 32 @ 0.2 m (SemanticKITTI) | the same grid |
| precision | bf16 autocast, fp32 residual | as released (`allow_tf32` on) |
| generated views | 0 | **150** at 160 × 512 — 12.3 M generated points |
| batch | one evaluation anchor | one evaluation anchor (`batch_size=1`) |

Nothing was substituted and no rendering setting was reduced: the timed OccAny run is the same
preset that produced the predictions our accuracy comparison scores.

**One caveat on the one-frame row.** Our one-frame arm is the five-frame-trained
CompletionUNet given a single frame (`one_frame_ablation.md` §1, arm A1), not a model optimised
for single-frame inference. Its latency is a faithful measurement of that configuration; its
accuracy is the accuracy of that configuration; the pair is consistent, but neither is a claim
about a single-frame system someone would design on purpose.

**Anchors.** 50 fixed anchors of SemanticKITTI 08, drawn once with seed 0 by taking one uniform
draw from each of 50 contiguous strata of the 801-anchor canonical list, written to
`runtime/anchors_semantickitti.json` and read by **both** benchmarks. Three warm-up anchors,
then **three timed passes** over the same 50, i.e. 150 timed predictions per configuration. The
pilot (3 anchors) showed the full workload costs a few minutes, so no sample-size reduction was
needed.

**Machine.** One NVIDIA RTX PRO 6000 Blackwell Max-Q (sm 12.0, driver 580.95.05), Intel Xeon
w7-3565X, 64 threads, Linux 6.8.0-90, torch 2.7.1+cu128, cuDNN 90701, Python 3.10.19 in both
interpreters. Every timed configuration ran on **the same GPU (index 3)** with that GPU to
itself; our own concurrent scoring job was suspended for the whole timing window. GPUs 0 and 1
carried other users' jobs throughout, which is outside our control on this shared machine;
load average during the window was 5.2.

## 3. The measured comparison

Per completed occupancy prediction. Latency is **not** divided by the number of input frames.

| configuration | benchmark | real frames | generated views | window | mean | median | p95 | predictions/s | peak allocated | peak reserved |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| **Ours — 5 causal frames** | SemanticKITTI | 5 | 0 | causal | **373.9 ms** | 373.1 ms | 382.8 ms | **2.67** | 11.78 GiB | 12.78 GiB |
| Ours — 1 causal frame | SemanticKITTI | 1 | 0 | causal | 164.4 ms | 163.2 ms | 168.5 ms | 6.08 | 11.78 GiB | 12.78 GiB |
| Ours — 5 causal frames | Occ3D | 5 | 0 | causal | 509.6 ms | 507.9 ms | 533.3 ms | 1.96 | 17.63 GiB | 19.65 GiB |
| OccAny — 1 causal frame | SemanticKITTI | 1 | 150 | causal | 1320.0 ms | 1316.0 ms | 1363.6 ms | 0.76 | 10.41 GiB | 12.08 GiB |
| **OccAny — 5 forward frames** | SemanticKITTI | 5 | 150 | **t…t+40** | **1704.4 ms** | 1700.7 ms | 1762.8 ms | **0.59** | 12.66 GiB | 13.91 GiB |

Per-pass means were 369.4 / 374.4 / 377.9 ms (ours, 5 frames) and 1694.3 / 1711.0 / 1707.9 ms
(OccAny, 5 frames): the three passes agree to about 2 %.

| comparison | ratio |
|---|---:|
| OccAny 5 forward ÷ ours 5 causal | **4.56 ×** |
| the same, after removing OccAny's unnecessary reconstruction-only voxelisation | **3.78 ×** |
| OccAny 1 causal ÷ ours 1 causal | 8.03 × |
| OccAny 1 causal ÷ ours 5 causal | 3.53 × |

**Our complete pipeline produces a completed occupancy grid 4.56 × faster than the OccAny
configuration it is compared against — 3.78 × on the most generous accounting for them — and
8.03 × faster in the one-frame-to-one-frame comparison.** Peak memory is comparable: 11.78 GiB
against 12.66 GiB, both well inside one GPU.

## 4. Where the time goes

| stage | SemanticKITTI 5f | share | Occ3D 5f |
|---|---:|---:|---:|
| preprocess | 20.93 ms | 5.5 % | 22.19 ms |
| LingBotMap streaming forward | 73.20 ms | 19.4 % | 117.53 ms |
| **MoGe-2 (5 forwards)** | **212.05 ms** | **56.1 %** | 215.67 ms |
| metric scale | 8.28 ms | 2.2 % | 12.88 ms |
| hard mapper (5 steps) | 16.81 ms | 4.4 % | 18.32 ms |
| hard-map query | 3.10 ms | 0.8 % | 7.46 ms |
| **CompletionUNet** | **43.38 ms** | **11.5 %** | 115.84 ms |
| grid read-out | 0.26 ms | 0.1 % | 0.20 ms |
| sum of stages | 378.01 ms | 100 % | 510.09 ms |

The sum of the stages (378.0 ms) matches the stages-mode end-to-end (378.2 ms) and exceeds the
instrumentation-free end-to-end (373.9 ms) by 1.1 %, which is the cost of the per-stage
synchronisations. The reported latency is the instrumentation-free one.

| OccAny segment | 5 forward frames | 1 causal frame |
|---|---:|---:|
| network forward (reconstruction + 150 generated views) | 1115.98 ms | 796.15 ms |
| output assembly | 20.45 ms | 3.99 ms |
| reconstruction-only voxelisation — **not needed for the compared prediction** | 254.85 ms | 250.75 ms |
| combined voxelisation (12.3 M points) | 270.99 ms | 260.40 ms |
| end-to-end minus the unnecessary leg | 1412.76 ms | 1065.65 ms |

**The cost difference is architectural, not a matter of model size.** OccAny spends 1.1 s
generating 150 novel views and a further 0.27 s voxelising the 12.3 M points they produce. Our
pipeline synthesises no views; its single largest cost is the five MoGe-2 forwards that supply
the metric gauge.

**The CompletionUNet is 11.5 % of our pipeline.** Its 967,322 parameters are a small *trainable*
budget, not a small *inference* budget, and the two must not be conflated.

## 5. Parameters

| module | total | trainable at inference | optimised by this work |
|---|---:|---:|---|
| ours: LingBotMap | 1,157,943,540 | 0 | no — frozen pretrained dependency |
| ours: MoGe-2 ViT-L | 326,209,221 | 0 | no — frozen pretrained dependency |
| ours: CompletionUNet | 967,322 | 0 | **yes** |
| **ours: total inference parameters** | **1,485,120,083** | 0 | 967,322 of them |
| OccAny: encoder + decoder + raymap encoder + gen decoder | 651,129,550 | 651,129,550 | released checkpoint |

**Our pipeline carries 2.28 × more inference parameters than OccAny and is still 4.56 × faster.**
The frozen dependencies are counted, not treated as free. The honest statement of the training
budget is separate and is given in §7.

## 6. Future-frame waiting time, reported separately

OccAny's five-frame configuration consumes native frames `t … t+40` at 10 Hz. Its last required
input arrives **4.0 s** after the anchor instant; no amount of compute removes that. Our five
causal frames end at the anchor, so the wait is **0 s**.

| | future-frame wait | compute | total from the anchor instant |
|---|---:|---:|---:|
| ours, 5 causal frames | 0 s | 0.374 s | 0.374 s |
| OccAny, 1 causal frame | 0 s | 1.320 s | 1.320 s |
| OccAny, 5 forward frames | 4.0 s | 1.704 s | 5.704 s |

These two columns are different quantities and are kept apart. **The five-frame comparison is
not temporally matched**, and no claim here treats it as such.

## 7. Training budget, stated separately from inference cost

The CompletionUNet was optimised for **6,000 steps at batch 4 on 128 × 128 × 32 crops**, three
seeds, about 7.7 minutes per seed on one GPU for the ONCE arm. OccAny's released checkpoint ran
**≈ 70,300 optimizer steps at effective batch 64 over 45,000 samples/epoch × 100 epochs on 16
GPUs** (`occany_training_audit.md`), itself warm-started from a prior OccAny run and initialised
from MUSt3R-512.

**This is a comparison of task-specific optimisation budgets only.** It excludes the pretraining
of LingBotMap and MoGe-2 on our side and of MUSt3R on theirs, and an optimizer-step count is not
total compute and not unique-data efficiency. Nothing here licenses a claim about total training
cost.

## 8. Verification: is this runtime paired with that accuracy?

The timed pipeline recomputes LingBotMap, MoGe-2, the scale and the hard map rather than reading
the cached clip, so bit-identity is not expected. What matters is that it reproduces the
predictions the accuracy numbers were computed from. On 10 fixed anchors:

- voxel agreement between the timed and cached occupancy sets: mean **99.868 %**, minimum 99.662 %;
- fraction of voxels that disagree: **0.0241 %**;
- SC IoU against ground truth, timed minus cached: mean **+0.0024**, largest absolute difference
  on any single anchor **0.0502**.

A 0.002-point mean difference on a 26.40 figure is not material, so the runtime in §3 is paired
with the accuracy in `waymo_canonical_rescore.md` and `one_frame_ablation.md` without
qualification.

## 9. Consistency checks

**Does the trained source change the cost?** No. The CompletionUNet forward was timed with each
arm's seed-0 checkpoint on identical hard-map inputs:

| arm | parameters | mean | median |
|---|---:|---:|---:|
| KITTI-360 | 967,322 | 42.864 ms | 42.806 ms |
| Waymo | 967,322 | 42.811 ms | 42.781 ms |
| ONCE | 967,322 | 42.861 ms | 42.831 ms |

The three agree within 0.05 ms (0.12 %), and every other stage is arm-independent by
construction — the hard map is bit-identical across the three prediction sets. Re-running the
full benchmark per source and per seed would therefore measure nothing new, which is why it was
not done.

**Does the benchmark change the cost?** Yes, and by the expected amount. On Occ3D the prediction
grid is 400 × 400 × 32 instead of 256 × 256 × 32 — 2.44 × the voxels — and the CompletionUNet
stage rises from 43.4 ms to 115.8 ms (2.67 ×), with the end-to-end going 373.9 → 509.6 ms. No
OccAny nuScenes timing was taken, so no cross-method ratio is claimed on Occ3D.

## 10. Reproduction

```bash
cd /home/minh/workspace/lingbot-map_fork
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/home/minh/anaconda3/envs/cu128/bin:/usr/local/cuda-12.8/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST="12.0a"          # needed by the 1-frame path's JIT kernel
P=/home/minh/anaconda3/envs/cu128/bin/python

# the fixed anchor list both benchmarks read (already written; regenerating is a no-op)
PYTHONPATH=$PWD $P -c "import sys;sys.path.insert(0,'.');\
from tools.method_vs_data.runtime import config as RC;\
from tools.zero_shot_eval import samples as ZS;\
[RC.write_anchors(d,[s.sample_id for s in ZS.evaluated_samples(d)]) for d in ('semantickitti','occ3d')]"

# ours
for F in 5 1; do for M in e2e stages; do
  PYTHONPATH=$PWD $P tools/method_vs_data/runtime/bench_ours.py \
      --dataset semantickitti --frames $F --mode $M --device cuda:3
done; done
PYTHONPATH=$PWD $P tools/method_vs_data/runtime/bench_ours.py \
    --dataset occ3d --frames 5 --mode e2e --device cuda:3
PYTHONPATH=$PWD $P tools/method_vs_data/runtime/bench_ours.py \
    --dataset semantickitti --frames 5 --device cuda:3 --verify 10
PYTHONPATH=$PWD $P tools/method_vs_data/runtime/bench_unet_arms.py --device cuda:3

# OccAny, in its own environment, on the same GPU
S=/media/SSD1/occany_causal
R=$PWD/artifacts/method_vs_data/runtime
cd $S/occany_shadow
for PRE in K_5FRAMES_GEOM K_1FRAME_GEOM; do for M in e2e stages; do
  PROJECT=$S/occany_shadow SCRATCH=$S \
  KITTI_ROOT=/media/welf/SSD2/MINH_DATASETS/kitti \
  NUSCENES_ROOT=/media/SSD1/MINH_DATASETS/occany_nuscenes_root \
  CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/minh/workspace/third_party/occany_env/bin/python \
      $R/../../../tools/method_vs_data/runtime/bench_occany.py \
      --preset $PRE --anchors $R/anchors_semantickitti.json --mode $M \
      --out $R/occany_${PRE}_${M}.json
done; done

# tables and the accuracy-versus-latency figure
cd /home/minh/workspace/lingbot-map_fork
$P tools/method_vs_data/runtime/tables.py
PYTHONPATH=$PWD $P tools/method_vs_data/phase_figures.py --which latency
```

Model identity: `artifacts/method_vs_data/runtime/checkpoint_manifest.json` carries the
sha256 and byte size of every weight file behind both sides — LingBotMap
`ee665103348e07e6…`, CompletionUNet seed 0 `7f5e853dd05bdbde…` (the digest
`one_frame_ablation.md` verifies against `code_freeze.json`), OccAny `05155377a79cff43…`,
MUSt3R-512 `96b0ae0fa3bc6e69…` — plus the pinned MoGe-2 revision
`39c4d5e957afe587e04eec59dc2bcc3be5ecd968`.

Artifacts: `artifacts/method_vs_data/runtime/` — `anchors_semantickitti.json`,
`anchors_occ3d.json`, `ours_*.json`, `occany_*.json`, `unet_arm_consistency.json`,
`verify_ours_semantickitti_5f.json`, `tables.md`. Figure:
`artifacts/method_vs_data/figures/phase_runtime_hidden/accuracy_vs_latency.png`, with its
plotted values in `figure_data.json`.

OccAny's own file was never modified. The instrumentation is monkeypatching applied before
`runpy` executes their script; the patched symbols are listed in the docstring of
`tools/method_vs_data/runtime/bench_occany.py`.

## 11. What is NOT claimed

- **Not** that a 967,322-parameter trainable head makes inference cheap. Our pipeline carries
  1.485 B inference parameters, 2.28 × OccAny's. The speed comes from not synthesising views.
- **Not** that the five-frame comparison is temporally matched. OccAny's five-frame mode reads
  4.0 s of future; the only causal OccAny comparator is its one-frame mode.
- **Not** a total-training-compute comparison. §7 compares task-specific optimizer budgets and
  explicitly excludes frozen-backbone pretraining on both sides.
- **Not** a claim about Occ3D speed-up: OccAny was timed on SemanticKITTI only.
- **Not** a throughput claim at batch sizes above one, nor under any streaming reuse this
  implementation does not perform. The pipeline was benchmarked as it exists; it re-runs the
  whole clip per anchor, and that cost is in the numbers.
- **Not** a claim that these ratios hold on other hardware.
