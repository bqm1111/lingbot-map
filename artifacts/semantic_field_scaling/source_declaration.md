# B4 source-data scaling study: source declaration

**Protocol version:** `b4_source_scaling/2026-09-15`

**Status.** Written 2026-09-15, before any scaling training step and before any new semantic score. It is hash-frozen with the manifest, code, configuration and cache identities by `tools/semantic_field/scaling/study_freeze.py` (stage `source`).

**Authorization.** The user authorized one finite study with the unchanged B4 method. Everything not listed here is inherited unchanged from:
- `artifacts/semantic_field/source_predeclaration.md` (B4, S1–S6);
- `artifacts/gate16_candidate_selection/predeclaration.md` (B3);
- `artifacts/delayed_semantic_pilot` (delayed targets, reliability).

## Fixed

**Frozen system.**
- Geometry: seed 0 (`artifacts/baseline_reconstruction/checkpoints/geo_seed0_step06000.pt`), with the CompletionUNet trunk, hard mapper, five-frame local-reset LingBot and causal MoGe-2 clip scale.
- Teacher: Trident-H with its text bank and logit scale 40; teacher features are fp16, 1024-d, unit L2.
- Feature centre: the original μ (`feature_mean_train_drives.npz`), **not recomputed**.
- Pipeline rules: Gate 15 evidence lifting and readout; pilot delayed targets (10 s / 20 frames, tiers A+B); B3 candidate rule (K = 16, τ = 1/40).

**B4, unchanged.** Architecture, losses and weights, cuboid masking, crop distribution, 4 anchors per update, target caps, and AdamW with lr 1e-3, weight decay 0.01, clip 1.0 and cosine schedule.

**B3, unchanged.** Architecture, observed-only and selection terms, 8 anchors per update, hidden/delayed query counts, and the same optimiser settings.

**Not allowed.** No contrastive loss, B5, new architecture or teacher, class reweighting, recentering, compression, hybrid rule or semantic tuning.

## Source pools

| pool | anchors | sha256 |
|---|---:|---|
| current | 1,358 (0003 / 0007 / 0010, `G15B.anchor_ids`) | `56c033d5eced26020a1209a3c76e795289af98ba222d310dd5ec1f152ea16424` |
| expanded | 9,066 = current + 7,708 (`manifests/source_pool_expanded.json`, frozen with this document) | recorded in the manifest |

**Added anchors.** 0000: 2,094; 0002: 2,769; 0004: 2,034; 0005: 811.

**Membership rule** (declared in `artifacts/semantic_field_data_audit/data_scaling_proposal.md`, unchanged): an eligible anchor of the 7-drive manifest is kept iff its `poses.txt` position at t is more than 2 × grid reach (114.49 m) from every 0006 pose.

**Exact recheck** (`verification/geography_check.json`).
- **Method:** world-XY convex footprints of each anchor's grid box against all 590 development grids.
- **Result:** 0 kept anchors overlap (74 in 0004 and 261 in 0005 truly overlap and are screened out).
- **Windows:** every history/future window frame of a kept anchor has a pose.

**Excluded.**
- 0006, which remains the source development drive.
- 0009, reserved.
- SemanticKITTI and Occ3D, which remain inaccessible during the source study; the prior target-access disclosure is preserved.
- The Gate 8C-1 7-drive samples, which contain GT fields.

**Residual dependence** (disclosed, not resolved):
- same city and same morning;
- 0005 ends 1.5 min before 0006;
- kept 0004 / 0005 window frames pass within 23.6 m / 45.1 m of the 0006 route;
- the 20-anchor temporal blocks are not proven independent scenes.

**Label disclosure.** The frozen k360_train stream builder lists file names in the SSCBench label directory to mark labelled frames. Label contents are not read by preparation, training or sample selection. Human labels enter only the existing 0006 evaluation path.

## Preparation (unchanged producers; `tools/semantic_field/scaling/prepare.py`, `prepare_tokens.py`)

**Producers, in order:**
1. Trident-H tokens (`extract_features.py`);
2. local-reset clips (`run_local_lingbot.py`, stride 1, kept anchors only);
3. hard maps (`build_inputs.py`);
4. evidence (`evidence/build.py`);
5. provenance;
6. delayed targets (`targets.py`, 10 s);
7. B3 train cache (variant "" only).

**Registration.** The added drives are registered in-process only (`registration.py`), and split assertions stay in force.

**Storage** (`prep/storage_layout.json`):
- **Hot, NVMe:** what training reads (inputs, evidence, provenance, delayed targets, train cache).
- **Cold, USB MINH1:** tokens and clips.
- Per-drive symlinks sit at the fixed tool paths. Existing drives' caches are untouched.

**Failures.** Any failed or missing anchor is recorded per drive before the freeze. Anchors are never silently dropped; an unresolved failure is BLOCKED.

**Operational record of preparation.** None of these items changes a sample, value or rule.
- **Raw data path.** The USB disk was remounted by its owner at `/media/minh/MINH1`. `/media/welf/MINH` is now a symlink to it, so the hard-coded `KITTI360_ROOT` resolves unchanged.
- **Hot placement.**
  - drives 0005, 0000 and 0002: every hot kind on SSD2;
  - drive 0004: every hot kind on SSD1;
  - per-kind free-space reservation: 1.10 × the measured evidence size of the drives already built.
- **Space freed.** Rarely used existing directories, none read by this study and none containing SemanticKITTI or Occ3D data, were moved to MINH1 (`relocated/`):
  - `ROI`, `lingbot_gate8d`, `lingbot_feature_study`, `lingbot_gate8c1{,_sky,_sky2}`, `lingbot_gate14_must3r`, `lingbot_gate5_2`, `occany_checkpoints`, `lingbot_gate8b`;
  - `Replica`, `NYUDepthv2`.

  Each move was checksum-verified (`rsync -c`, 0 differing files), and a symlink at the old path keeps every tool path valid (`prep/cold_moves.jsonl`).
- **Memory guard.** At 19:50 the session's memory guard stopped the background preparation jobs while free memory (not available memory) was low because of page cache. Every writer is atomic, the newest outputs were re-validated, and the jobs resumed by skipping finished files. Since then `cache_trim.py` releases the page cache of the study's own files every 60 s.

**Reliability** (`prep/reliability_expanded.json`).
- **Rule:** the frozen pilot gate, with frozen thresholds, over the pooled 7 training drives plus 0006 dev statistics.
- **Required:** pass, with delayed-loss tiers [1, 2] unchanged; otherwise BLOCKED.
- **Descriptive only:** per-drive retention, teacher-class mix, delayed coverage and repeatability are reported but never used to tune anything.

## Training (12 runs; `tools/semantic_field/scaling/cli.py`)

| method | current pool | expanded pool | updates per run |
|---|---|---|---:|
| B4 | seeds 0, 1, 2 | seeds 0, 1, 2 | 40,056 = ceil(6,000 · 9,066 / 1,358) |
| B3 | seeds 0, 1, 2 | seeds 0, 1, 2 | 20,028 = ceil(3,000 · 9,066 / 1,358) |

- **Budgets:** recalculated once from the frozen pool size if it changes; identical duration and schedule for both pools within a method.
- **Initialisation:** the method's declared one (`torch.manual_seed(seed)` then construction; data generator seed, delayed generator seed + 1). No warm start.
- **Schedule:** cosine over the run's own update count. Final-step checkpoint only; no early stopping, no checkpoint or seed selection, no look at 0006 before all 12 runs finish.
- **Loader:** disk-backed and bounded (`anchor_store.py`), equivalent to the original trainers (`verification/loader_equivalence.json`).
  - **B3:** bit-identical.
  - **B4:** identical draws, inputs and targets. Later-step values differ only by the original trainer's own run-to-run GPU nondeterminism (`upsample_trilinear3d` backward).
- **Execution:** several runs may share a GPU. Resume is exact in draws (`resumable.py`).
- **Historical references:** the original short-budget runs (B4 6,000; B3 3,000) are kept.

## Source evaluation and decision

**Evaluator.** The frozen `evaluate_source.evaluate` per pool (`scaling/evaluate.py`; only checkpoint paths and output directory redirected). Frozen masks, classes, denominators, occupancy and paired 20-anchor block bootstrap (2,000 resamples, seed 0, seed-pooled confusions).

**Reported:**
- primary completed_unobserved mIoU;
- full SSC;
- candidate-supported / unsupported;
- per-class IoU and confusions;
- distance bins and masked diagnostic;
- collapse checks.

**Contrasts** (each with its own paired interval):

| # | contrast | what it measures |
|---|---|---|
| 1 | expanded B4 − current B4 (both 40,056) | source-pool effect under this recipe |
| 2 | current B4 (40,056) − original B4 (6,000) | optimisation budget / schedule effect |
| 3 | expanded B4 − expanded B3 | same-pool method comparison under the declared method budgets |
| 4 | expanded B4 − frozen B0 | absolute improvement |

No data-effect claim is made from expanded B4 − original B4.

**Promotion.** The candidate is expanded B4 only (mean of seeds); the original S1–S6 apply exactly, with expanded B3 as comparator:

| rule | requirement |
|---|---|
| S1 | ≥ +2.0 points over B0 |
| S2 | every seed positive, and CI lower bound > 0 |
| S3 | ≥ +1.0 point over expanded B3, with CI lower bound > 0 |
| S4 | ≥ 8 improved classes |
| S5 | no collapse (candidate-supported regression reported explicitly) |
| S6 | occupancy identical |

A positive source-pool effect does not replace S1–S6. Control successes are reported separately.

**Outcomes.**
- **PASS:** freeze the complete method, then run the previously authorised one-time target evaluation with the existing practical and official protocols, all three seeds, no target-based selection.
- **SCIENTIFIC FAIL:** end semantic training under this extension; hand off to the geometry / generalisation paper.
- **BLOCKED:** preserve resumable state, report what completed and what remains; no scientific claim.

**Resource caps:** ≤ 48 aggregate GPU-hours (readiness + preparation + training + source evaluation, `resources/ledger.jsonl`) and ≤ 750 GB additional storage.
