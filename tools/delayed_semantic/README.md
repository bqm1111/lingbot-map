# Causal semantic completion with delayed observation supervision — source pilot

**Question.** Can trustworthy observations that arrive *after* t teach a causal model to predict semantics
in space that has no semantic evidence anywhere in its available history H_t?

This package reuses Gate 15 (`tools/semantic_completion`, never modified) and the Current Geometry-Only
Baseline. It adds provenance-tracked evidence memory, delayed targets built from physically synchronized
future observations, a learned evidence-association head, and region-resolved evaluation. KITTI-360 only:
training on 0003/0007/0010, development evaluation on 0006. SemanticKITTI and Occ3D are never opened.

Results: `artifacts/delayed_semantic_pilot/report.md`. Predeclared rules: `predeclaration.md` and
`predeclaration_addendum_A.md` in the same directory, hash-frozen in `frozen_*.json`.

## What is available when, per component

| component | at prediction time t (student) | after t (targets only) |
|---|---|---|
| LingBot | images t-4..t, KV cache cleared per clip (local reset) | — |
| scale | MoGe-2 median over t-4..t | — |
| hard mapper (S, F) | the five depth maps and poses above | — |
| CompletionUNet (occupancy O, trunk d1) | [S, 0, F], frozen seed 0 | — |
| semantic evidence memory | Trident-H tokens of t-4..t lifted at accepted ray endpoints ∩ S (Gate 15 rule), with frame bits and times | — |
| completion head | trunk, O, S, F, evidence memory | — |
| delayed target | — | LiDAR of t+1..t+20 (≤ 10 s), GT poses, calibration, Trident-H tokens of those images |

Future frames can't reach the student in two ways:
- **Accessor guard.** `stream.CausalHistory` raises on an index > t.
- **File audit.** The prediction path runs inside a file audit that forbids LiDAR, poses, targets and labels.

Tests: `test_prediction_at_t_ignores_changed_and_appended_future_frames`. `verify.py` #13 re-runs LingBot
after it has processed future images.

## Mechanism

- **`memory.py`: observed vs predicted.**
  - `EvidenceMemory.observe` inserts one history frame at a time.
  - `snapshot` gives `ObservedEvidence`: flat index, feature [N, 1024], count, `frame_bits` (bit q = frame t-4+q), first/last time.
  - `PredictedSemantics` is a separate type; the memory refuses it.
- **`model.py`: EvidenceAssociation, ~25 k parameters.**
  - For each non-evidence query voxel, take its 16 nearest observed evidence voxels (Euclidean; Q×16, never all-voxel × all-evidence).
  - An MLP scores each candidate from:
    - the frozen trunk at query and candidate
    - relative position
    - the fraction of the segment through F, S and O
    - provenance (count, seen in frame t, number of frames)
    - a vocabulary-free 16-d projection of the evidence feature
  - Output = normalize(Σ softmax(score) · f_e), a 1024-d vector in the teacher space. Readout = argmax cosine to the 18 prompts.
  - **Limitation:** a convex combination of observed features can't express a class no candidate carries.
- **Shared evidence path:** every method keeps each evidence voxel's own observed class (checked per anchor).
- **Methods:**
  - **B0:** Gate 15 nearest evidence.
  - **B0g:** geodesic nearest evidence through predicted-occupied voxels.
  - **B1:** head trained with the observed-only term. Gate 15 hiding blocks; hidden evidence voxels predict their own teacher distribution.
  - **B2:** same draws and steps, plus a delayed term: KL to the delayed target on non-evidence target voxels.
  - **B2s:** B2 with 4 s targets (addendum A sensitivity).

## Delayed targets (`targets.py`)

1. **Camera view of a future sweep.** Future LiDAR sweep j is projected into image j; the timestamps are < 5 ms apart. Returns pass a 4 px z-buffer, then the Trident-H token at that pixel gives softmax(40 · cos) over the 18 prompts. That is one view per (voxel, frame).
2. **Registration.** GT poses move the returns into the grid of t.
3. **Static support.** Returns must span ≥ 1 s with no occupied/free conflict across sweeps (the `core/supervision/rawtarget.py` measurement).
   - **Tier A:** also hit by an H_t sweep (occupied at t).
   - **Tier B:** ≥ 2 future sweeps only (earlier occupancy uncertain).
4. **Target.** The mean of the future view distributions. The loss asserts that the student's class distribution at this voxel should match the average later camera observation; it is not an exact label.
5. **Moving objects.** Single-sweep or conflicting voxels are excluded from targets, never from evaluation.

**Reliability gate (`reliability.py`, label-free, before training):**
- G1: registration residual
- G2: coverage
- G3: cross-view repeatability vs a within-anchor shuffle
- G4: agreement with H_t camera views

Semantic GT only appears in a labelled diagnostic.

## Masks and denominators (`regions.py`)

| symbol | meaning |
|---|---|
| G | GT occupied, class 1..18 (SSCBench, evaluation only) |
| E, Ec | evidence anywhere in H_t / in frame t itself |
| Gobs | hard map S or F (the student geometry observed the voxel) |
| O | frozen predicted occupancy |
| R, Rh | 0006 delayed-target voxels not in E / of those, voxels an H_t camera saw a return in (privileged) |

- **Semantic-only regions** are scored on GT-occupied voxels:
  - `observed_current` = G∩Ec
  - `observed_past_only` = G∩E∖Ec
  - `never_observed` = G∖E
  - **`completed_never_observed` = G∩O∖E** (primary; identical to Gate 15 `completed_unobserved`)
  - `completed_geo_observed` / `completed_geo_unobserved` split it by Gobs
  - `completed_later_revealed` = …∩R
  - `completed_revealed_unseen` = …∩R∖Rh
  - `completed_not_revealed` = …∖R
- **SSC regions** score all labelled voxels with prediction = class where O, else empty.
- **Uncertainty:** block bootstrap over contiguous 20-anchor blocks, one head seed.

## Implementation map (verified in code; file references)

| stage | code | verified behavior |
|---|---|---|
| stream, images, timestamps | `core/datasets/sources.py` (0006), `core/datasets/kitti360_folds.py::_k360_train_segments`, `kitti360_transforms.DriveGeometry.timestamps`; `stream.py` | stream = SSCBench index stride 5 (native gap 5 for ≥ 96 % of frames, larger at pose gaps); image vs LiDAR timestamps ≤ 4.2 ms apart at stream frames |
| LingBot reset | `gates/gate7b/replay.py::replay_segment` (one `clean_kv_cache` per call), `tools/baseline_reconstruction/targets/clips.py::infer_clip`; caches `lingbot_local_registration_audit/local_lingbot` (0006), `lingbot_gate12_suf/local_lingbot` (train) | clip = frames t-4..t only; future images processed first leave it bit-identical (`verify.py` #13) |
| metric scale | `tools/baseline_reconstruction/current/geometry.py` (clip `scale`), applied once by `IncrementalMapper` `scale_state.force` | median of the clip's own MoGe-B log-scales |
| coordinates | `geometry.frame_inputs` (anchor camera = identity), `kitti360.Kitti360Calibration.rect_cam_to_velo`, grid docs in `kitti360_transforms.py` | velodyne grid of t, 256×256×32, 0.2 m, floor binning |
| surfaces and free space | `core/mapping/mapper.py` (conf ≥ 1.5, 1–60 m, ±0.2 m band, free carve stride 4) → `lingbot_gate12_suf/inputs` `S_H`, `F_H` | the frozen `build_local_geometry` reproduces the cached hard map (`verify.py` #13) |
| CompletionUNet | `core/model/net.py`, `tools/semantic_completion/frozen_geometry.py` | input [S, 0, F]; occupied = residual > 0; trunk d1 has 24 channels; sha256-checked checkpoint |
| Trident + language readout | tokens `lingbot_feature_first_3d/trident_tokens` (34×128×1024, unit norm), `evidence/trident.py` | readout argmax cosine, 18 prompts; scale 40 from `tools/feature_first_3d_semantics/common.py` |
| semantic evidence | `tools/semantic_completion/evidence/lift.py` | the frame-by-frame replay (`provenance.py`) reproduces Gate 15 voxels and counts on every anchor |
| splits and anchors | `core/datasets/kitti360_splits.py`, Gate 15 evidence listing | 1,358 training anchors, 590 dev anchors |
| LiDAR measurement | `core/supervision/rawtarget.py::build` | endpoint counts, free traversals, conflicts |

- **Assumptions:**
  - static world for tier B
  - Velodyne in-sweep motion distortion is not modelled
  - the ~11 px Trident token resolution
  - rawtarget conflicts = dynamics and thin structure
  - tier A "occupied at t" relies on hits before and after t
- **Missing capabilities:**
  - a validated persistent-streaming geometry control
  - motion correspondence for dynamic objects
  - any output beyond convex mixtures of observed features
  - evidence confidence
  - history longer than five frames

## Reproduce

```bash
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 CUDA_HOME=/usr/local/cuda-12.8 FLASHINFER_CUDA_ARCH_LIST="12.0a"
PY=/home/minh/anaconda3/envs/cu128/bin/python; T=tools/delayed_semantic/cli
$PY -c "from tools.delayed_semantic import rules; rules.freeze('predeclaration')"      # before any target statistic
for d in 0003 0006 0007 0010; do $PY $T/build_provenance.py --drive 2013_05_28_drive_${d}_sync --device cuda:0; done
$PY -c "from tools.delayed_semantic import rules; rules.freeze('reliability')"
for d in 0003 0006 0007 0010; do $PY $T/build_targets.py --drive 2013_05_28_drive_${d}_sync --device cuda:0; done
$PY $T/run.py reliability --device cuda:0                                                 # gate; Stop here if it fails
$PY -c "from tools.delayed_semantic import rules; rules.freeze('addendum_A')"
for d in 0003 0007 0010; do $PY $T/build_targets.py --drive 2013_05_28_drive_${d}_sync --variant 4s --device cuda:0
                            $PY $T/run.py train_cache --drive 2013_05_28_drive_${d}_sync --device cuda:0; done
$PY -c "from tools.delayed_semantic import rules; rules.freeze('decision')"               # before any trained head is evaluated
for m in B1 B2 B2s; do $PY $T/run.py train --method $m --device cuda:0; done              # ~minutes each after loading
for s in 0 1 2 3; do $PY $T/run.py evaluate --shard $s --shards 4 --device cuda:$s & done; wait
$PY $T/run.py analyse && $PY $T/run.py verify --device cuda:0 && $PY $T/run.py report
$PY -m pytest tests/delayed_semantic -q
```

Caches live under `/media/SSD1/MINH_DATASETS/lingbot_delayed_semantic`: provenance, `delayed_targets[_4s]`, `train_cache`, `predictions_at_t`.

## Gate 16 — candidate-level delayed supervision (`gate16/`)

Question: when the correct identity is among the 16 candidates, can delayed observations teach the scorer to pick it?

- **What stays fixed:** geometry, evidence, delayed targets, candidate rule, scorer and TrainConfig are all as in the pilot.
- **B3:** the observed-only term is unchanged; only B2's delayed term is replaced.
  - Compatibility: `r_i = <p_i, q_v>`, where `p_i` = softmax(40 cos(f_i, text)) and `q_v` = the delayed target.
  - Candidate target: `softmax(r/τ)` with τ = 1/40.
  - Loss: `KL(target || softmax(scorer logits))`.
  - Head seeds 0, 1, 2; seed 0 shares the pilot B2 draws.
- **B3geom:** the same, with the semantic projection removed from the scorer input (the one allowed ablation).
- **Controls:** B1/B2 are the pilot's single-seed checkpoints.

| module | role |
|---|---|
| `config.py` | seeds, τ, K = 16, paths |
| `candidates.py` | `CandidateSet` (rows, voxel ids, distances, readout distributions, classes) |
| `candidate_targets.py` | `CandidateTarget` (compatibility, soft target, best index) — label-free |
| `selection_loss.py` | selection KL and label-free selection statistics |
| `candidate_regions.py` | EVALUATION ONLY: candidate-hit / miss, oracle selection (take GT) |
| `candidate_metrics.py` | selection tallies, accuracy, recovered fraction |
| `context.py` | per-anchor student context (prediction audit) vs evaluation truth |
| `ceiling.py` | reconstructed C0 / C16 / oracle ceilings and the threshold T = N + ½(O − N) |
| `train.py` | B3 / B3geom, all seeds in one process, through `delayed_semantic/train.fit` |
| `evaluate.py`, `analysis.py`, `report.py`, `verify.py`, `rules.py`, `cli.py` | scoring, criteria / decision, tables and figures, checks, hash freezes, entry points |

```bash
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1; PY=/home/minh/anaconda3/envs/cu128/bin/python; X=tools/delayed_semantic/gate16/cli.py
$PY $X freeze --stage predeclaration
$PY $X ceiling --device cuda:0 && $PY $X freeze --stage ceiling          # before any B3 score
$PY $X train --variant B3 --device cuda:1 & $PY $X train --variant B3geom --device cuda:2 & wait
$PY $X freeze --stage code
for s in 0 1; do $PY $X evaluate --shard $s --shards 4 --device cuda:$s & done; wait   # two at a time (host memory guard)
for s in 2 3; do $PY $X evaluate --shard $s --shards 4 --device cuda:$s & done; wait
$PY $X analyse && $PY $X report --device cuda:0 && $PY $X verify --device cuda:0
$PY -m pytest tests/delayed_semantic -q
```

Results: `artifacts/gate16_candidate_selection/report.md`.
