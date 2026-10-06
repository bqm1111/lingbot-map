# Data-scaling extension for B4: study specification (proposal, not launched)

**Protocol version:** `semantic_field_b4_datascale/v1` (new). It follows the closed B4 experiment and does not reopen or edit it.

**Unchanged.**
- The B4 SOURCE FAIL, checkpoints, `source_predeclaration.md` and every existing artifact stay as they are.
- SemanticKITTI and Occ3D stay locked unless the frozen promotion rule passes.
- Evidence and motivation are in `data_audit.md`.
- Expected benefit is **unknown**: the audit found no demonstration that data quantity caused the failure.

## 1. Question

With the B4 method held fixed, does training on a 6.7× larger KITTI-360 pool in independent temporal blocks move completed_unobserved semantics on 0006 enough to pass the frozen S1–S6? Is any change due to data rather than optimisation or architecture?

## 2. Fixed (by reference, no replacement)

**B4.** Everything in `artifacts/semantic_field/source_predeclaration.md` and `tools/semantic_field/config/schema.py`:
- architecture: 1,357,008 parameters, 61 input channels;
- frozen geometry seed 0 / CompletionUNet / hard mapper / LingBot / MoGe-2;
- Trident-H feature space, text bank and logit scale 40;
- centering μ: the frozen `feature_mean_train_drives.npz`, **not recomputed**;
- masking 16×16×8, p = 0.5;
- losses and weights;
- crop 128×128×32, caps 8,192 / 4,096;
- 4 anchors per step, AdamW 1e-3, weight decay 0.01, cosine schedule, clip 1.0;
- final-step checkpoint;
- full-grid inference.

**B3.** The Gate 16 frozen configuration (`artifacts/gate16_candidate_selection/candidate_config.yaml`, pilot `TrainConfig`, τ = 1/40, K = 16).

**Delayed targets.** Pilot `TargetConfig` (10 s / 20 frames, tiers A+B) and the reliability tiers [1, 2].

**Evaluation.** Frozen `tools/semantic_field/evaluation/evaluate_source.py` and `analysis.py`: regions, candidate-supported / unsupported split, distance bins, masked diagnostic, paired block bootstrap (20-anchor blocks, 2,000 resamples, seed 0, seed-pooled confusions).

## 3. Source pools

| pool | definition | anchors | temporal blocks |
|---|---|---:|---:|
| **CUR** | the existing 1,358 anchors of 0003 / 0007 / 0010 | 1,358 | 69 |
| **EXP** | CUR ∪ all eligible anchors of 0000 / 0002 / 0004 / 0005 (frozen stream and anchor rule, `artifacts/kitti360_data_scaling_7drives/manifests/full.json`) whose `poses.txt` position is > 114.4 m from every 0006 pose | 1,358 + 7,708 = **9,066** | 69 + 392 = 461 |

**Exclusions.**
- 0006 and 0009.
- The 606 candidate anchors within 114.4 m of 0006: 168 in 0004 and 438 in 0005.

**Frame-level exclusion.** A kept anchor's delayed targets use only future frames of its own drive, so no 0006 frame can enter.

**Declared before preparation.** Anchor lists and their sha256 go in `manifests/exp.json`.

## 4. Preparation (unchanged tools; the only new code is drive registration and a loader)

1. **Drive registration.** Register the four drives for the unchanged tools, following the `tools/kitti360_data_scaling_7drives/drives7.py` pattern:
   - the k360_train stream source;
   - Gate 12 `TRAIN_DRIVES`;
   - the Gate 15 and pilot drive tuples.
   - Test: the rebuilt stream and anchor ids equal `full.json`; the existing drives' caches are untouched.
2. **Trident-H tokens.** All 8,350 stream frames, via `tools/feature_first_3d_semantics/extract_features.py`, with its determinism check (max |dp|, argmax agreement) logged.
3. **Local-reset clips** (`tools/gate12_suf/gen_local_clips.py`) → **inputs** (`build_inputs.py`) → **Gate 15 evidence** (`evidence/build.py`) → **delayed targets** (`tools/delayed_semantic/targets.py`) → **B3 train cache** (`train_cache.py`).
   - Firewalls: the existing `FileAudit` token lists.
4. **Reliability check (inclusion rule, frozen now).**
   - Run `reliability.target_statistics` and `registration` per new drive.
   - A new drive is included only if it meets the frozen pilot gate thresholds (`tools/delayed_semantic/rules.py::RELIABILITY_GATE`, the G1 train-pooled reading, G2, G3 tier A and B, G4) computed on that drive alone.
   - Failing drives are dropped before any training. The drop is recorded; thresholds are not changed.
   - No labels are used.
5. **On-demand loader.** B4 and B3 anchors are loaded per draw instead of all onto the GPU.
   - **Equivalence test (must pass before any new run):** on CUR, seeds 0–2, the new loader reproduces the original `draw_digest` and every logged `n_masked` / `n_delayed` of `artifacts/semantic_field/runs/train_B4_seed*.json` (the replay in `tools/semantic_field_data_audit/audit.py` already does this on CPU), and Gate 16's B3 draw digests.
   - Also record step time with a 200-step timing smoke on EXP. This measures cost only; no metric is read.
6. **Hash freeze** before any new checkpoint: this document, the manifests, the registration code, the loader and its equivalence record, and the reliability inclusion record.

## 5. Arms (seeds 0, 1, 2; initialisation `torch.manual_seed(s)`, data generator s, delayed generator s+1, as in the trainer)

**Update budgets.** B4's frozen budget is 6,000 updates = 17.67 draws per CUR anchor. The exposure-matched budget on EXP is U = round(9,066 × 24,000 / 1,358 / 4, −2) = **40,100 updates** (after any reliability drop, recomputed with the same formula before the freeze).

- For B3, the frozen budget is 3,000 updates at 8 anchors per step. Its exposure-matched budget is U_B3 = round(9,066 × 24,000 / 1,358 / 8, −2) = **20,000 updates**.

| arm | pool | updates | purpose |
|---|---|---:|---|
| B4-CUR-6k | CUR | 6,000 | existing checkpoints `B4_seed{s}_step06000.pt` (valid once loader equivalence passes) |
| B4-EXP-6k | EXP | 6,000 | data effect at the frozen optimisation budget |
| **B4-EXP-U (primary)** | EXP | 40,100 | data at the frozen per-anchor exposure |
| B4-CUR-U | CUR | 40,100 | optimisation-only control at the primary's update budget |
| B3-EXP-3k | EXP | 3,000 | B3 at its frozen budget on the same pool |
| **B3-EXP-U** | EXP | 20,000 | architecture control for the primary |
| B3-CUR-U | CUR | 20,000 | B3 optimisation-only control |

The cosine schedule spans each arm's own update count, as in the frozen trainer. Final-step checkpoint only; no checkpoint selection and no early stopping.

## 6. Decision rule (frozen S1–S6, unchanged thresholds)

**Promotion applies only to B4-EXP-U** (mean of seeds), on 0006 completed_unobserved:

| id | criterion |
|---|---|
| S1 | B4-EXP-U − B0 ≥ +2.0 mIoU |
| S2 | every seed's gain over B0 > 0 and bootstrap CI lower bound > 0 |
| S3 | B4-EXP-U − B3 ≥ +1.0 with CI lower bound > 0, **against both** the frozen Gate 16 B3 (as originally declared) **and** B3-EXP-U |
| S4 | ≥ 8 classes with positive IoU gain over B0 |
| S5 | the declared no-collapse checks |
| S6 | occupancy TP/FP/FN identical to Gate 15 seed 0 on every anchor |

**Pass.** SOURCE PASS requires S1–S6 all true. Only then are the frozen `final` stage and the conditional target protocol of `source_predeclaration.md` applied, once, unchanged.

**Attribution** (reported for every outcome; it decides nothing):

| effect | contrast |
|---|---|
| data effect at fixed optimisation | B4-EXP-6k − B4-CUR-6k |
| data effect at matched exposure | B4-EXP-U − B4-CUR-U |
| optimisation effect | B4-CUR-U − B4-CUR-6k |
| architecture effect on the same data | B4-EXP-U − B3-EXP-U |

- Each contrast is paired, with the frozen bootstrap.
- Candidate-supported / unsupported and distance-bin results are reported for every arm.
- A gain appearing only in B4-CUR-U is attributed to optimisation, not data.
- A gain matched by B3-EXP-U is attributed to data, not the generative architecture.

**Descriptive only.** Per-class teacher pseudo-label counts, temporal blocks and two-view agreement of EXP, by the audit script. They are never used for weighting or selection.

**Dev-drive reuse caveat.** 0006 has informed many earlier decisions. A pass is a source result, not independent confirmation.

## 7. Cost estimate (extrapolated from logs; I/O unknown)

**Preparation.**
- 5.3–6.0 single-GPU hours.
- 355–475 GB on `/media/welf` (4.0 TB free); SSD1 has 54 GB free.

**Training** (compute only, from 0.25 s per B4 update and 0.052 s per B3 update measured in memory):

| arm | seeds × time | GPU-h |
|---|---|---:|
| B4-EXP-6k | 3 × 0.45 h | 1.4 |
| B4-EXP-U | 3 × 2.8–3.1 h | 8.4–9.3 |
| B4-CUR-U | 3 × 2.8–3.1 h | 8.4–9.3 |
| B3 arms (all) | — | ≈ 1.8 |
| **total** | | **≈ 20** |

- On-demand loading adds unmeasured I/O. It could plausibly double the EXP arms on the external disk; the step 5 smoke measures it before the freeze.

**Evaluation.**
- ≈ 1.9 s per anchor with 7 heads.
- About 1–1.5 GPU-h for all arms on 590 anchors.

**Wall time.** About 1–2 days on the 4 GPUs.

## 8. Material unknowns (resolved only by preparation, not by this audit)

- **Delayed-target reliability and counts on the new drives.** A drive can fail the inclusion rule.
- **Class mix and rare-class coverage of the new drives.** Rare-class delayed targets on the current drives have two-view agreement of 0.21–0.41, so scaling may add mostly frequent-class supervision and rare-class noise.
- **Evidence voxels per anchor.** They set storage and loader I/O; the range is taken from the current drives.
- **External-disk throughput and loader step time.**
- **Magnitude of any effect.** B4 needs +2.58 mIoU over its current 8.22 on the primary region. Candidate transport (B3) also benefits from the same data. The earlier 7-drive scaling study of a different model (Gate 8C-1) gained +0.1 pp.
