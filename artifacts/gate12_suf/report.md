# Gate 12 — does uncertainty-aware causal geometry improve learned completion?

**Decision B: S/U/F does not meaningfully improve completion. Keep H. The frozen monocular
geometric source is the limiting factor.**

KITTI-360 only: training on 0003 / 0007 / 0010, selection-free validation on 0006. SemanticKITTI
and Occ3D-nuScenes were not accessed. Verification passes 22/22 (`verification.md`).

**The result:**
- **Gain:** S/U/F beats H on every seed, but only by +0.24 IoU points on a 13.4 % baseline, below
  the predeclared +0.5-point bar.
- **Where:** the gain comes from recall at 5–30 m, including space unobserved at input.
- **Cost:** S/U/F loses 0.5 points beyond 30 m and on voxels the causal input already observed.

| 0006 completion occupancy IoU (590 anchors, final step) | seed 0 | seed 1 | seed 2 | mean | S/U/F − H [95 % CI] |
|---|---|---|---|---|---|
| H (hard) | 0.1343 | 0.1331 | 0.1341 | 0.1338 | |
| S/U/F | 0.1354 | 0.1358 | 0.1373 | 0.1362 | +0.0024 [+0.0013, +0.0035]; per seed +0.0012 / +0.0028 / +0.0032 (sd 0.0011) |
| precision H / S/U/F | | | | 0.181 / 0.181 | |
| recall H / S/U/F | | | | 0.339 / 0.354 | |

The CI is a paired clip-block bootstrap: blocks of 20 anchors, 2000 resamples, mean-over-seeds
difference. Tables: `results_table.md`, `tables/per_seed.csv`, `tables/per_range.csv`,
`tables/paired.csv`.

## What was trained (fixed before training; `code_freeze.json`)

**Inputs (deployable, dense, no LiDAR, no GT pose).** Everything below is identical for both
conditions:
- **Clips:** frozen local-reset five-frame LingBot inference (L) per anchor t (frames t−4..t).
- **Depth and confidence:** dense LingBot depth and confidence; the confidence gate ≥ 1.5 is unchanged.
- **Pose and camera:** LingBot local poses `inv(P_4)·P_q` and LingBot `pred_K`.
- **Scale:** the causal MoGe-2 clip scale, applied once by the mapper to depth and translation.
- **Clip provenance:** stride-1 clips for the training drives come from the unchanged local-reset
  script; the 454 anchors that overlap the audited stride-3 clips are bit-identical to them.
- **Firewall:** input generation ran under a file audit forbidding LiDAR, GT poses, targets,
  evaluation masks and the target datasets (0 violations).
- **Density:** the mapper accepted 137k–352k rays per clip for H. On 0006 that is about 3× the Gate 9
  LiDAR-support pixel count.

**The two conditions:**
- **H:** channels [S_H, 0, F_H] from the unchanged mapper (±0.2 m band, hard carving). S_H / F_H
  equal the core `IncrementalMapper` bit for bit.
- **S/U/F:** channels [S, U, F].
  - **Rays:** only those with predicted depth < 20 m.
  - **S:** the same ±0.2 m surface band.
  - **F:** conservative carving to `z − u_front`.
  - **U:** the Gate 11 U4 interval `[z − u_front, z + u_back]` wherever the voxel is neither S nor F.
  - **Widths:** `u3 = max(0.2, (0.0658 + 0.1270 z)·g(c))` with g = 1.502 / 1.164 / 0.978 / 0.879,
    `u_front = max(0.2, 0.3084·u3)`, `u_back = max(0.2, 1.3945·u3)`. These come unchanged from
    Gate 11's frozen file and round to the brief's values.
  - Its mean interval on kept rays is 0.38 m in front and 1.67 m behind (mean over all 1,948 anchors).
  - The 20 m cut removes a median 7.7 % of H's rays per anchor (5th–95th percentile 1.1–22 %, maximum 46 %).

**Model and recipe.** This is the adopted Gate 8C-1 recipe (`configs/gate8c1_sky/seed0.yaml`),
executed through the unchanged `step_loss`. It was the most recent validated matched completion
recipe, chosen without reference to any target dataset.
- **Network:** `CompletionUNet`, width 24, replicate padding, 3 input channels, 967,322 parameters.
- **Loss:** focal + Dice occupancy (γ 2, pos-weight cap 8, unknown weight 2).
- **Optimisation:** AdamW 1e-3, weight decay 0.01, cosine schedule, grad clip 1, batch 4, bf16.
- **Data:** uniform 128×128×32 crops; anchors drawn uniformly over 1,358 training anchors;
  6,000 steps (about 17.7 epochs).
- **Checkpoint:** the final step; nothing is selected.

**Geometry-only changes from the adopted recipe, identical in both conditions:**
- no semantic input and `w_sem = 0`;
- the input carries no log-odds, so the base log-odds is 0 and the residual lock is inactive;
- the loss's unknown-voxel weighting uses S_H ∪ F_H in both conditions.

**Target.** The unchanged raw-LiDAR KITTI-360 geometry target over the full 256×256×32 grid, with
no range crop.

**Pairing.** Per seed, H and S/U/F share byte-identical initial weights, anchor list, anchor draws,
crop origins, target crops, loss weights and learning-rate sequence. Digests match at every
1000-step checkpoint.

## Answers

1. **Does dense deployable S/U/F improve KITTI-360 completion over dense deployable H?** Not
   meaningfully. The gain is positive and reproducible, but +0.0024 IoU, about 1.8 % relative,
   against the predeclared bar of +0.005.
2. **Mean IoU difference across matched seeds:** +0.0024 (sd 0.0011) [+0.0013, +0.0035].
3. **Is the gain reproducible?** Yes, in sign: all 3 seeds improve, and S/U/F sits above H at
   steps 5000 and 6000 on every seed's learning curve. It is small, and both conditions plateau
   at the same level.
4. **Precision and recall:** precision is unchanged (0.181 / 0.181). Recall rises from 0.339 to
   0.354, most strongly at 10–20 m (0.464 → 0.518) and 20–30 m (0.268 → 0.302). Beyond 30 m recall
   falls (0.149 → 0.130).
5. **Where in range does the gain occur?** Full-region IoU difference by voxel range:

   | 0–5 m | 5–10 m | 10–20 m | 20–30 m | > 30 m |
   |---|---|---|---|---|
   | −0.0008 [−0.0034, +0.0016] | +0.0023 [+0.0005, +0.0042] | +0.0035 [+0.0019, +0.0052] | +0.0040 [+0.0015, +0.0068] | −0.0048 [−0.0063, −0.0036] |

   Beyond 30 m S/U/F is worse on all 3 seeds. That band holds 35 % of the target-occupied voxels.
6. **Does it improve genuinely unobserved / future-measured space?** Slightly.
   - **Unobserved later** (not measured by the five causal LiDAR sweeps, resolved by the future
     sweeps): +0.0027 [+0.0016, +0.0038], all seeds, mostly at 10–30 m (+0.0047 and +0.0058).
   - **Voxels with no H input evidence:** +0.0032 [+0.0022, +0.0044].
   - **Voxels the input already observes:** S/U/F is worse. The past-5-measured region shows
     −0.0051 [−0.0068, −0.0035], and voxels with H input evidence −0.0134. There S/U/F has fewer
     definite claims (conservative carving) or none (beyond 20 m).
7. **Does the 20 m observation cutoff prevent useful far-range completion?** Yes, beyond 30 m.
   - The model carries the benefit about 10 m past the cutoff (20–30 m: +0.0040).
   - Beyond 30 m, H's hard far evidence, though poor, is worth −0.0048 IoU when it is withheld.
8. **Does U provide useful learned information despite its low raw containment efficiency?** A
   little, and it cannot be isolated here.
   - The recall gain at 10–30 m is consistent with the network using U as a "surface somewhere
     along this ray" cue.
   - But the two conditions differ in three bundled ways: U, conservative carving, and the 20 m cut.
     This two-arm design cannot attribute the +0.0024 to U alone.
   - Gate 11's 3× more true-surface evidence became only +0.24 IoU points.
9. **Should S/U/F replace H as the frozen geometry representation?** No. The net gain is below the
   predeclared meaningfulness bar, and it trades far-range and observed-space accuracy for a small
   recall gain at mid-range.
10. **Is the method ready for zero-shot evaluation?** No. Decision A was not reached, so
    SemanticKITTI and Occ3D-nuScenes remain untouched.

## Decision

**B — S/U/F does not meaningfully improve completion.** Strictly, the improvement is reproducible
but small. It fails the "meaningful gain" criterion predeclared in `analyze12.py` (+0.005 mean
full-region IoU), while meeting the other guide items: CI > 0 in unobserved space, no precision or
recall collapse, not harmful overall. It also carries a consistent loss beyond 30 m. C does not
apply: S/U/F does not hurt overall. D does not apply either: the small benefit is not confined to
≤ 20 m or to observed voxels.

**Failure mode.** The completion model can read the extra evidence: recall rises exactly where U
exists and slightly beyond. The limit is what the evidence contains.
- Gate 11 showed that 84–91 % of U4's claimed voxels are LiDAR-free.
- 87–96 % of hard surface voxels contradict LiDAR.
- Both conditions converge to the same about 13.5 % completion IoU plateau.

Representation changes on this depth move the result by tenths of a point. I therefore conclude
that **the frozen monocular geometric source is insufficient**, and H stays the frozen baseline
representation.

**Next action (source-side only; do not access target datasets):** train the identical paired
recipe on accurate geometry. Build the H channels from LiDAR depth through the same mapper, using
the same anchors, crops, targets, model and seeds, with GT relative poses. Compare against
monocular H on 0006.

This measures how much completion IoU a stronger geometric source would unlock with this model.
That decides whether a stronger source is worth pursuing, or whether the completion model is also
capped.

## Limitations and comparability

- **Single validation drive.** Evaluation is on one source drive (0006, 590 anchors), and the
  bootstrap covers anchor uncertainty only. Three seeds show the training spread.
- **Bundled conditions.** The two arms bundle U, conservative carving and the range cut, as the
  brief specified.
- **Not the historical 30.89 % Occ3D result.** That was a zero-shot target-domain score from the
  32-channel stream-mapper model with semantics and the residual lock, under the `past5` protocol.
  This is a KITTI-360 source-domain score of a 3-channel geometry-only model on dense local-reset
  inputs. The two are not comparable and are not placed side by side.
- **Not the data-scaling occupancy numbers.** Those used the 32-channel model and different
  inputs, so they are not comparable either.

## Figures

- `figures/0_inputs_H_vs_SUF.png`: dense deployable inputs for three 0006 anchors; no target shown.
- `figures/1_per_seed_region_range.png`: per-seed IoU, region differences, and range differences
  with CIs.
- `figures/2_learning_curves.png`: full IoU every 1000 steps on every 5th anchor, plus
  training-time validation loss.

## Verification and process notes

Verification is 22/22.
- **Pre-training sanity check (7/7).** The carving-stop test failed twice on a float-rounding tie
  in its interval clause, before any training. The chain refused to train both times. My first
  written diagnosis of that failure (voxel centres) was wrong. The test now uses explicit tolerances
  on voxel extents, and the failed logs are kept under `logs/`.
- **Verification check 12.** The first verification run failed on two clauses of my own:
  - an arbitrary "≥ 50 % of pixels are rays" threshold failed on a low-confidence frame;
  - a source scan matched the word "velodyne" in the builder's own firewall list.

  Both were replaced by substantive tests: the recount against confident dense pixels, a
  comparison with LiDAR-support counts, and a scan for LiDAR-access code. The failed record is
  `logs/verification_attempt1_21of22.md`. The training data, inputs and results were unaffected.

## Reproduce

```bash
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH="$PWD" CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH FLASHINFER_CUDA_ARCH_LIST="12.0a" HF_HUB_OFFLINE=1
PY=/home/minh/anaconda3/envs/cu128/bin/python; T=tools/gate12_suf; L=artifacts/gate12_suf/logs; mkdir -p $L
for d in 0003 0007 0010; do $PY $T/gen_local_clips.py --drive 2013_05_28_drive_${d}_sync --device cuda:1; done
for d in 0006 0003 0007 0010; do $PY $T/build_inputs.py --drive 2013_05_28_drive_${d}_sync --device cuda:0; done
$PY $T/build_regions.py --device cuda:0
bash $T/run_all.sh          # sanity gate -> code freeze -> 6 runs -> evaluation -> analyze12.py
$PY $T/verify12.py --device cuda:0
```
