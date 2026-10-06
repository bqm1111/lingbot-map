# B4 training-data audit (2026-09-15)

**Scope.** Read-only. CPU only. No training, teacher rerun, download or model execution. SemanticKITTI and Occ3D were not accessed, and no human semantic label file was opened.

**Unchanged.** The B4 SOURCE FAIL, its checkpoints, the declarations and all existing artifacts.

**Teacher classes.** Every "class" below is the Trident-H pseudo-label: argmax cos(f, text), 18 KITTI-360 phrases. These are pseudo-label frequencies, not GT frequencies, and confidence is not correctness.

## Four answers

### 1. How many distinct training examples and reliable delayed targets did B4 actually use?

**Anchors.** 1,358 causal anchors: 0003 193, 0007 569, 0010 596.
- Every anchor was drawn by every seed: 24,000 anchor draws per seed, 4–34 draws per anchor, median 17–18.
- In independent units this is **69 temporal blocks** (20 consecutive anchors ≈ 10 s) and **227 distinct 51.2 m world cells**.
- The trajectory is 9.5 km.

**Masked-current targets.**
- About 58.4 M voxel-draws per seed.
- 19.9–20.0 M distinct (anchor, voxel) targets per seed, which is 73.6 % of the 27.14 M evidence voxels.

**Delayed targets.**
- All 4,536,663 stored non-evidence tier A+B targets passed the pilot reliability gate. No further filter was applied.
- About 28.4 M voxel-draws per seed.
- 4.11 M distinct per seed, which is 90.5 % of the stored targets; the union over seeds is 96.6 %.

**Replay check.** The sampler replay reproduces each seed's `draw_digest` and every logged per-step `n_masked` / `n_delayed` exactly, for all 3 × 6,000 steps.

**Independence.** Adjacent anchors overlap heavily, so voxel counts are not independent samples.

### 2. Which limitations are evidenced, and which remain hypotheses?

**Evidenced.**

*No silent data loss.* The sampler reached everything that was cached:
- no drive exclusion or missing cache;
- no step with an empty loss term;
- caps bind rarely: masked cap in 3 % of draws, delayed cap in 0.2 %;
- 5.4 % of draws had no hidden target inside the crop.

*Few independent units.*
- 69 temporal blocks in total.
- 0003 is a 1.3 km, 193-anchor drive.

*Rare pseudo-classes are thin and unreliable.*

| class | teacher share of sampled masked targets | temporal blocks with ≥ 100 masked voxel-draws | delayed two-view top-1 agreement |
|---|---:|---:|---:|
| motorcycle | 0.01 % | 8 of 69 | 0.25 |
| person | 0.04 % | 28 of 69 (19 for delayed) | 0.21 |
| bicycle | 0.09 % | 23 of 69 | 0.37 |
| terrain | — | — | 0.37 |
| other object | — | — | 0.38 |
| pole | — | — | 0.41 |
| vegetation (reference) | 57.4 % | — | 0.84 |
| road (reference) | — | — | 0.80 |

*Train/dev teacher-class mismatch.*

| class | share of training sampled masked targets | share of 0006 masked-diagnostic hidden voxels |
|---|---:|---:|
| vegetation | 57.4 % | 43.9 % |
| other structure | 3.0 % | 11.7 % |
| sidewalk | 0.6 % | 2.4 % |

Sidewalk and other structure are among the classes B4 lost most.

*Optimisation reached a flat region under the cosine schedule.*
- Masked teacher agreement was ≈ 0.86 from step ~3,000 on.
- Over the last 2,000 steps, masked loss moved 0.517 → 0.498 and delayed loss 0.428 → 0.419, while the learning rate annealed toward 0. This cannot be separated from annealing.

**Hypotheses (not established).**
- Too little independent data caused the failure.
- More drives would fix the candidate-supported loss (−13.1). That loss is about using nearby evidence, which is not obviously a count problem.
- More drives would supply reliable rare-class supervision.
- More optimisation would help.

**Unmatched comparison.** The 86 % training agreement and the 70.7 % 0006 agreement are not comparable. They differ in drives, crop vs full grid, subsampling, running pre-update weights vs final checkpoint, and class mix: a constant majority prediction scores 57 % vs 44 %. No overfitting conclusion is drawn.

**Disclosed discrepancy.** The source report says training agreement rose from **0.67**. No log window reproduces that start value: the first 100 steps average 0.59–0.60 and the first 500 steps 0.745. The end value 0.86 is reproduced. The frozen report was not edited.

### 3. What exact additional KITTI-360 data could address the observed coverage limitations, and at what cost?

**The pool.** The four unused SSCBench-KITTI-360 training drives 0000, 0002, 0004 and 0005:
- 8,314 eligible anchors by the frozen stream and anchor rule (verified to reproduce B4's anchors on 0003/0007/0010);
- after excluding every anchor within 114.4 m (2 × grid reach) of any 0006 pose: **7,708 anchors in 392 temporal blocks**, 5.7× the current 69;
- ≈ 34 km of trajectory;
- ≤ 0.6 % of anchors within 57 m of current training anchors, so geographically new.

**Overlap with 0006.**
- 0005: 35 % of its anchors lie within 114 m of 0006, and 0005 was recorded just before it.
- 0004: 7.6 %.

**What already exists.**
- Raw images and LiDAR on `/media/welf`.
- Poses on SSD1.
- Gate 8C-1 samples, which the B4 pipeline cannot use.

**What is missing.**
- 1024-d Trident-H tokens
- local-reset LingBot clips
- hard-map inputs
- Gate 15 evidence
- delayed targets
- a B3 train cache

**Estimated preparation cost** (from logged per-anchor rates):
- 5.3–6.0 single-GPU hours, dominated by tokens (8,350 frames × 1.65–1.84 s);
- **≈ 355–475 GB** of storage, mostly evidence at 34–49 MB per anchor.

**Blockers.**
- SSD1 has 54 GB free; `/media/welf` has 4.0 TB free, with unmeasured throughput.
- The B4 and B3 trainers load every anchor onto the GPU. The current pool already occupies ≈ 56 GB of evidence features; the expanded pool would need an estimated 315–430 GB, more than the 98 GB GPU and the 125 GB host RAM. An on-demand loader is required.

**Training cost.** About 20 GPU-h for the proposed arms, plus unmeasured I/O.

### 4. Single recommendation

**Propose a data-scaling extension**, specified in `data_scaling_proposal.md`, as an optional new protocol version. It is not a launch and not a result.

**Why.**
- The audit found a concrete, accessible, geographically verified KITTI-360-only pool that multiplies independent temporal blocks 5.7×.
- It uses the unchanged pipeline, at a bounded cost.
- The current data is demonstrably small in independent units, and thin for rare pseudo-classes.

**The case is weak on expected benefit.**
- Nothing here shows that data quantity caused the SOURCE FAIL.
- Rare-class delayed pseudo-labels are low-agreement, so scaling also scales their noise.
- The earlier 7-drive scaling of the Gate 8C-1 model gained +0.1 pp.
- S1 requires +2.58 mIoU over B4's current result.

A reasonable reader could decline the extension on these grounds.

---

## Detail

### A. Sources and entry points (verified by reading, not by filename)

| item | path |
|---|---|
| B4 trainer / sampler | `tools/semantic_field/training/trainer.py` (`train_seed`, `load_training_anchor`, `to_crop`) |
| masking | `tools/semantic_field/data/semantic_evidence.py` → `tools/semantic_completion/dataset.py::hidden_block_mask` |
| losses | `tools/semantic_field/losses/semantic_loss.py` |
| config | `tools/semantic_field/config/schema.py`; `artifacts/semantic_field/semantic_field_config.yaml` |
| run records | `artifacts/semantic_field/runs/train_B4_seed{0,1,2}.json`: config, per-step log, `draw_digest`, checkpoint sha, firewall |
| logs | `artifacts/semantic_field/logs/train_B4_seed*.log` |
| declaration / verification | `artifacts/semantic_field/source_predeclaration.md`; `verification.md` (10/10, unchanged) |
| anchor list | `tools/semantic_completion/evidence/build.py::anchor_ids`: files in `/media/SSD1/MINH_DATASETS/lingbot_gate12_suf/inputs/<drive>` |
| evidence cache | `/media/SSD1/MINH_DATASETS/lingbot_gate15_semantic/evidence/<drive>/<t>.npz` (flat, fp16 1024-d feature, count, frame keys) |
| delayed targets | `/media/SSD1/MINH_DATASETS/lingbot_delayed_semantic/delayed_targets/<drive>/<t>.npz`; rules `tools/delayed_semantic/targets.py`, `config.py::TargetConfig` |
| reliability | `artifacts/delayed_semantic_pilot/reliability/reliability.json` (gate: tiers [1, 2]) |
| stream definition | `core/datasets/kitti360_folds.py::_k360_train_segments` (every 5th SSCBench index with an image) |
| 7-drive manifests | `artifacts/kitti360_data_scaling_7drives/manifests/{full,coverage}.json` |
| poses | `/media/SSD1/MINH_DATASETS/sscbench_kitti360/data_poses/<drive>/poses.txt` (shared world frame: drives sit at distinct world translations) |
| audit script | `tools/semantic_field_data_audit/audit.py` |
| audit outputs | `artifacts/semantic_field_data_audit/*.csv`, `audit_summary.json`, `logs/` |

### B. Training data actually available and used (`training_pool.csv`, `sampling_by_seed.csv`)

| drive | raw images (image_00) | stream frames | eligible = cached anchors (clips, inputs, evidence, targets) | evidence voxels / anchor | stored delayed targets / anchor (p10–p90) | anchors with ≥ 1 delayed target | temporal blocks | 51.2 m cells | path at anchors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0003 | 1,031 | 202 | 193 | 19,056 | 2,952 (1,263–4,028) | 193 | 10 | 35 | 1.3 km |
| 0007 | 3,396 | 578 | 569 | 16,424 | 3,431 (2,115–5,041) | 568 | 29 | 117 | 4.9 km |
| 0010 | 3,836 | 605 | 596 | 23,688 | 3,381 (2,012–4,551) | 595 | 30 | 75 | 3.3 km |
| **total** | 8,263 | 1,385 | **1,358** | 27.14 M voxels | **4,536,663** | 1,356 | **69** | **227** | 9.5 km |

**Frame spacing.**
- The stream is every 5th SSCBench index, ≈ 0.5 s apart.
- History t−4..t.
- An anchor needs ≥ 4 preceding and ≥ 5 future stream frames, plus poses.

**Future horizon.**
- t+1..t+20 stream frames, capped at 10 s.
- Tier A: seen in history, ≥ 1 future sweep, ≥ 1 s static span, no free-space conflict.
- Tier B: not seen in history, ≥ 2 future sweeps.
- A target also needs ≥ 1 future camera view.
- B4 keeps non-evidence tier A+B targets.

**Delayed-target funnel.** Anchor-voxel instances, heavily overlapping across anchors:

| stage | count | retention |
|---|---:|---|
| future-hit voxels | 104.4 M | — |
| static, non-conflict tier A / B | 4.76 M / 7.40 M | — |
| with a future camera view | 4.61 M | 38 % of tier A+B |
| not evidence | 4.54 M | 98.4 % |

The stored delayed target is a view-averaged class distribution. No 1024-d target feature exists.

**Sampling per step.**
- 4 anchors, drawn uniformly with replacement (`torch.randint`, generator seed s).
- One optimizer update per step; no gradient accumulation; no dataloader. 6,000 updates = 24,000 anchor draws per seed.
- Crop: uniform 128×128×32 over the 256×256×32 grid.
- Masking: grid-aligned 16×16×8 cuboids hidden with p = 0.5.
- Hidden evidence inside the crop: capped at 8,192 per anchor.
- Delayed targets inside the crop: capped at 4,096 per anchor, from a separate generator s+1.

**What reaches the loss.**
- The crop keeps about 25 % of an anchor's hidden evidence (2,457–2,497 of ≈ 9,950) and 35 % of its delayed targets (1,181–1,191 of ≈ 3,340).
- About 9.7 k masked and 4.7 k delayed targets per step.

**Exposure per seed.**
- Every anchor is drawn, a mean of 17.7 times.
- By drive: 0003 3,360–3,492 draws, 0007 9,981–10,075, 0010 10,433–10,659.

**Distinct coverage.**

| target type | per seed | union over seeds |
|---|---:|---:|
| masked | 73.4–73.8 % of evidence voxels (19.9–20.0 M) | 24.56 M |
| delayed | 90.5–90.9 % of stored targets (4.11–4.12 M) | 4.38 M |

**Overlap between examples.**
- Consecutive anchors are 0.5 s apart and share most of their 51.2 m grid.
- Inputs include t−4..t.
- The delayed-target windows of anchors fewer than 20 stream frames apart share future frames.
- Independent units are therefore reported as 20-anchor temporal blocks (the 7-drive manifest convention, equal to the target horizon) and 51.2 m world cells.

**Issues found.**
- No silent sample cap, drive exclusion, missing cache or repeated small subset.
- The two in-crop caps rarely bind: 2.8–3.2 % of draws for masked, 0.1–0.2 % for delayed.
- The main discard is structural: the random crop drops ≈ 75 % of hidden evidence and 65 % of delayed targets per draw. This is by design, and coverage over repeated draws still reaches the fractions above.
- 1,270–1,314 of 24,000 draws (5.4 %) had no hidden target inside the crop.
- 79–109 draws (0.4 %) had no delayed target inside the crop.
- No step lacked either loss term.

### C. Semantic supervision that reached B4 (`class_supervision.csv`, seed 0 for distinct and block counts)

**Table columns.**
- **evidence**: teacher class of all 27.14 M training evidence voxels; the pool from which masked targets are drawn.
- **masked**: mean voxel-draws over the 3 seeds.
- **blocks**: temporal blocks (of 69) with ≥ 100 voxel-draws of the class.
- **delayed**: argmax of the stored distribution.
- **conf**: mean max-probability.
- **2-view**: top-1 agreement between the first and second future view, on voxels with ≥ 2 views. The pooled shuffled baselines from reliability.json are 0.38 (tier A) and 0.36 (tier B).
- **dev hidden**: share among the 5,264,367 hidden voxels of the frozen 0006 masked diagnostic. The replayed masks reproduce that count exactly.

| class | evidence share | masked share | masked blocks | masked distinct voxels | delayed stored | delayed share | delayed blocks | conf | 2-view | dev hidden |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| car | 2.64 % | 2.16 % | 62 | 510 k | 225 k | 4.96 % | 62 | 0.51 | 0.54 | 2.30 % |
| bicycle | 0.13 % | 0.09 % | 23 | 23 k | 9.0 k | 0.20 % | 24 | 0.57 | 0.37 | 0.01 % |
| motorcycle | 0.01 % | 0.01 % | **8** | 1.9 k | 0.8 k | 0.02 % | **7** | 0.41 | 0.25 | 0.00 % |
| truck | 0.48 % | 0.43 % | 39 | 94 k | 25 k | 0.55 % | 34 | 0.51 | 0.65 | 0.34 % |
| other vehicle | 0.90 % | 0.84 % | 52 | 182 k | 53 k | 1.16 % | 53 | 0.33 | 0.59 | 0.32 % |
| person | 0.05 % | 0.04 % | **28** | 9.9 k | 2.3 k | 0.05 % | **19** | 0.28 | **0.21** | 0.05 % |
| road | 9.49 % | 4.69 % | 69 | 1.43 M | 716 k | 15.8 % | 69 | 0.56 | 0.80 | 6.19 % |
| parking | 2.34 % | 2.07 % | 41 | 460 k | 222 k | 4.89 % | 40 | 0.35 | 0.62 | 1.17 % |
| sidewalk | 1.03 % | **0.56 %** | 41 | 161 k | 104 k | 2.30 % | 42 | 0.34 | 0.61 | **2.41 %** |
| other ground | 3.91 % | 2.45 % | 69 | 660 k | 384 k | 8.46 % | 69 | 0.33 | 0.52 | 3.38 % |
| building | 20.1 % | 21.5 % | 39 | 4.18 M | 303 k | 6.68 % | 37 | 0.38 | 0.80 | 23.97 % |
| fence | 1.21 % | 0.95 % | 53 | 229 k | 116 k | 2.55 % | 49 | 0.47 | 0.62 | 2.23 % |
| vegetation | 51.4 % | **57.4 %** | 69 | 10.7 M | 2.18 M | 48.1 % | 69 | 0.64 | 0.84 | **43.92 %** |
| terrain | 0.61 % | 0.58 % | 58 | 120 k | 64 k | 1.40 % | 46 | 0.27 | 0.37 | 0.42 % |
| pole | 2.05 % | 2.21 % | 69 | 430 k | 38 k | 0.83 % | 68 | 0.38 | 0.41 | 0.90 % |
| traffic sign | 0.56 % | 0.66 % | 64 | 119 k | 9.8 k | 0.22 % | 58 | 0.41 | 0.41 | 0.12 % |
| other structure | 2.72 % | **3.02 %** | 52 | 565 k | 71 k | 1.57 % | 47 | 0.22 | 0.49 | **11.67 %** |
| other object | 0.39 % | 0.36 % | 68 | 77 k | 12 k | 0.26 % | 45 | 0.19 | 0.38 | 0.59 % |

**Reading the table.**
- Several minority classes appear in a large share of blocks, but with few voxels per block, often below 100 voxel-draws.
- The 51.2 m cell counts (in the CSV) tell the same story: person reaches ≥ 100 delayed voxel-draws in 25 of 227 cells and motorcycle in 9.
- One repeatedly seen object is not many independent examples. Blocks and cells are the more honest unit, and even they overlap across adjacent blocks.

**Delayed-target reliability** (stored, `reliability.json`; label-free gate passed: G1, G2, G3 tiers A and B, G4).

*Registration residual (training drives pooled, median / p90):*

| future offset | median | p90 |
|---|---:|---|
| +1 frame | 0.027 m | 0.073 m |
| +5 frames | 0.055 m | 0.137 m |
| +10 frames | 0.069 m | 0.184 m |

The stricter per-drive reading fails at +10 frames on 0010 (and on 0006); this was disclosed at the time.

*Repeatability (pooled, agreement vs shuffled):*

| tier | agreement | shuffled | per drive |
|---|---:|---:|---|
| A | 0.752 | 0.381 | 0.84 (0003), 0.77 (0007), 0.72 (0010) |
| B | 0.733 | 0.365 | — |

*Re-observation agreement:* 0.69.

*Retention:* every stored target passed. B4's own filter (non-evidence tiers 1–2) removed only the 1.6 % that are evidence voxels.

*Previously measured 0006 GT diagnostic* (evaluation only, cited, not used here): delayed-target accuracy 0.63, mIoU 0.20 (tier A) / 0.18 (tier B).

**Loss reduction and balance.**
- `L = L_masked + 1.0 · L_delayed`.
- `L_masked = mean_voxels KL(P_teacher ‖ P_student) + mean_voxels (1 − cos(centred))`, both with weight 1.
- `L_delayed = mean_voxels KL(q ‖ P_student)`.
- Each term is a mean over the voxels pooled across the step's 4 anchors, so an anchor with more in-crop targets carries more weight within its term.
- Target counts do not change the balance between terms. The per-step count ratio (delayed/masked, median 0.50) has no effect except when a term is empty, which never happened.

**Logged magnitudes** (mean of seeds; last 500 steps):

| term | magnitude |
|---|---:|
| masked | 0.498 (text 0.160 + feature 0.337) |
| delayed | 0.419 |

The two terms are of similar size. Per-term gradient norms were not logged, so their gradient share is **unknown**.

### D. Training history (`training_curves.csv`, 500-step windows, mean of 3 seeds)

| steps | lr (mid) | loss | masked | masked feature | masked agree | delayed | delayed agree |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1–500 | 9.96e-4 | 1.889 | 1.267 | 0.837 | 0.745 | 0.622 | 0.556 |
| 1001–1500 | 9.14e-4 | 1.219 | 0.715 | 0.471 | 0.827 | 0.504 | 0.599 |
| 2501–3000 | 5.65e-4 | 1.024 | 0.565 | 0.372 | 0.849 | 0.460 | 0.620 |
| 4001–4500 | 1.70e-4 | 0.945 | 0.517 | 0.348 | 0.860 | 0.428 | 0.631 |
| 5501–6000 | 4.3e-6 | 0.917 | 0.498 | 0.337 | 0.861 | 0.419 | 0.633 |

**Reading.**
- Agreement is flat from ~3,000 steps.
- Losses still decline slightly, by ≤ 2 % per 1,000 steps, while the learning rate goes to zero.
- Verdict: plateaued under this schedule. The curves cannot tell whether a longer or re-warmed schedule, or more data, would move them.

**Agreement is class-mix sensitive.**
- The per-step majority-class share of sampled masked targets is 0.63–0.66 throughout.
- In seed 0 the agreement at steps 100–200 (0.89–0.95) already exceeded its end value.

**The 86 % vs 70.7 % comparison is unmatched.** They differ in every respect below, so no overfitting diagnosis is made:

| aspect | training 86 % | development 70.7 % |
|---|---|---|
| drives | training drives | 0006 |
| input | 128×128×32 crop | full grid |
| hidden voxels | ≤ 8,192 per anchor | all hidden voxels |
| masks | random | seeded by anchor id |
| metric | per-step mean over 4 anchors | voxel-pooled over 590 anchors |
| weights | running, pre-update | final checkpoint |
| majority share | vegetation 57 % | vegetation 44 % |

The architecture itself has no dropout or BatchNorm, so train and eval mode are identical.

**What is and is not supported.**
- **Supported:** a small number of independent source blocks (69); thin, low-agreement rare pseudo-classes; a teacher-class mix different from 0006; optimisation flat by the end of the schedule.
- **Not supported:** that 6,000 steps or three drives were insufficient.

### E. Additional accessible KITTI-360 data (`geography.csv`, `expansion_cost.csv`)

**Drive split** (`core/datasets/kitti360.py::OFFICIAL_SPLIT`):
- train: 0000, 0002, 0003, 0004, 0005, 0007, 0010;
- val: 0006;
- test: 0009, which is excluded (reserved, no local poses).

All drives were recorded 2013-05-28 between 08:46 and 12:08, sequentially and without temporal overlap. 0006 (10:37–10:54) immediately follows 0005 (ends 10:35).

| drive | stream frames | eligible anchors | temporal blocks | 51.2 m cells | anchors ≤ 57 m of current train | ≤ 25 / 57 / 114 m of 0006 | anchors kept (> 114 m from 0006) | blocks kept |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| 0000 | 2,103 | 2,094 | 105 | 128 | 0 % | 0 / 0 / 0 % | 2,094 | 105 |
| 0002 | 2,778 | 2,769 | 139 | 197 | 0.6 % | 0 / 0 / 0 % | 2,769 | 139 |
| 0004 | 2,211 | 2,202 | 111 | 182 | 0 % | 2.5 / 3.0 / 7.6 % | 2,034 | 104 |
| 0005 | 1,258 | 1,249 | 63 | 67 | 0 % | **18.2 / 24.1 / 35.1 %** | 811 | 44 |
| **candidate total** | 8,350 | 8,314 | 418 | 574 | | | **7,708** | **392** |

**How overlap was checked.**
- Anchor positions come from `poses.txt` at the anchor's native frame.
- The stream rebuild matches the manifest frame counts on all 7 drives.
- On 0003/0007/0010 the manifest's anchor ids, mapped through the rebuilt stream, reproduce B4's cached anchors and their native frames exactly.
- 114.4 m is twice the farthest grid point from the sensor, so beyond it two grids cannot intersect whatever the heading.
- Heading, elevation and changes between recordings are not modelled.
- Overlap among the candidate drives themselves is not quantified beyond cell counts.
- A new drive ID alone was not treated as a new environment. Novelty rests on the pose distances above.

**Already present for the candidate drives.**
- `image_00` images and Velodyne sweeps (`/media/welf/MINH/datasets/kitti360/KITTI-360`).
- Poses and calibration (SSD1).
- The Gate 8C-1 7-drive `samples/`, `targets/` and 18-class Gate 8B Trident probabilities. These are a different mapper and format with embedded GT fields, so B4 cannot use them.

**Missing for B4 / B3.**
- 1024-d Trident-H token maps (`tools/feature_first_3d_semantics/extract_features.py`).
- Local-reset LingBot + MoGe-2 clips (`tools/gate12_suf/gen_local_clips.py`, which asserts the three training drives and needs the drive registration used by `tools/kitti360_data_scaling_7drives/drives7.py`).
- Hard-map inputs (`tools/gate12_suf/build_inputs.py`).
- Gate 15 evidence (`tools/semantic_completion/evidence/build.py::build_drive`).
- Delayed targets (`tools/delayed_semantic/targets.py::build_drive`).
- The B3 train cache (`tools/delayed_semantic/train_cache.py`).
- Per-drive reliability statistics.
- A trainer loader that does not hold all anchors on the GPU.

**Cost estimate.** Measured per-unit rates on 0003/0007/0010 from the build logs, applied to the 7,708 kept anchors and the 8,350 frames. Extrapolation; the ranges are the per-drive min–max.

| stage | measured rate | single-GPU hours | storage |
|---|---|---:|---:|
| Trident-H tokens | 1.65–1.84 s / frame, 8.9 MB | 3.8–4.3 | 74 GB |
| local-reset clips | 0.151–0.158 s, 0.88–0.92 MB | 0.32–0.34 | 6.8–7.1 GB |
| hard-map inputs | 0.15–0.17 s, 0.06–0.10 MB | 0.32–0.36 | 0.5–0.7 GB |
| Gate 15 evidence | 0.19–0.25 s, 33.8–48.7 MB | 0.41–0.54 | **260–375 GB** |
| delayed targets | 0.146–0.153 s, 0.17–0.20 MB | 0.31–0.33 | 1.3–1.5 GB |
| B3 train cache | 0.051 s, 1.7–2.1 MB | 0.11 | 13–16 GB |
| **total** | | **≈ 5.3–6.0** (token extraction shards; ≈ 1.5–2 h on 4 GPUs if I/O permits) | **≈ 355–475 GB** |

**Storage and memory.**
- Free space: SSD1 54 GB, `/` 6 GB, `/media/welf` 4.0 TB (a SanDisk Extreme external drive; throughput unmeasured).
- Current trainer footprint: 27.14 M × 1024 × 2 B ≈ 56 GB of evidence features on the GPU.
- The expanded pool needs an estimated 154–210 M evidence voxels, i.e. 315–430 GB, above both the 98 GB GPU and the 125 GB host RAM. Its evidence voxels per anchor are unknown; the range is taken from the current drives.

**Training time.**
- Measured: B4 25.5–27.9 min per 6,000-step seed (in memory); B3 157 s per 3,000-step seed; 0006 source evaluation ≈ 1.9 s per anchor with 7 heads, ≈ 19 GPU-min for 590 anchors.
- On-demand loading overhead is **unknown**. At 4 files of 34–49 MB per step it may dominate on the external disk.

**Coverage gain, supported by metadata.**
- 69 → 461 temporal blocks (6.7×).
- 227 → ≤ 801 cells (574 candidate cells, before exclusion).
- 9.5 → ≈ 43 km of trajectory.

**Not supported by metadata.**
- Class balance of the new drives.
- Rare-class counts.
- Delayed-target reliability on them.

All three are unknown until the tokens and targets exist. More images ≠ more usable examples: the per-drive retention and reliability must be measured after preparation.

### F. Commands

```bash
cd /home/minh/workspace/lingbot-map_fork; PY=/home/minh/anaconda3/envs/cu128/bin/python
# per-drive teacher-class / target index (CPU, ≈ 12 min, writes /media/SSD1/MINH_DATASETS/lingbot_semantic_field/data_audit_cache, 0.1 GB)
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES= $PY -m tools.semantic_field_data_audit.audit index
# exact sampler replay (digest + per-step count check), classes, blocks, geography, curves, cost -> artifacts/semantic_field_data_audit/
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES= $PY -m tools.semantic_field_data_audit.audit analyse
# inventory (read-only)
ls /media/SSD1/MINH_DATASETS/{lingbot_gate12_suf/inputs,lingbot_gate15_semantic/evidence,lingbot_delayed_semantic/delayed_targets,lingbot_k360_7drives/samples}/*
ls /media/welf/MINH/datasets/kitti360/KITTI-360/data_2d_raw/*/image_00/data_rect | wc -l; df -h /media/SSD1 /media/welf/MINH; free -g
```

**Operational note.** The first `index` run stopped on the 0006 key format ("0000000704" vs "0007_…"). The parser was fixed and 0006 was re-indexed; the training-drive indices were unaffected.

### G. Missing information

- Per-term gradient norms (not logged).
- Candidate-drive evidence voxels per anchor, class mix, delayed-target counts and reliability (no caches).
- External-disk throughput and on-demand loader speed (not measured).
- Heading-aware overlap.
- GPU memory headroom of the current trainer beyond the ≈ 56 GB feature estimate (peak memory not logged).
