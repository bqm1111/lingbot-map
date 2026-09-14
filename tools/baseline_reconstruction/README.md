# Baseline reconstruction — the historical post-ceiling-fix baseline, and the current geometry-only baseline

Two questions, answered in order, and the order matters:

1. **What was the real post-ceiling-fix zero-shot baseline?** Memory said SemanticKITTI 22.71 % / Occ3D 30.74 %.
   Memory is not evidence. Phase 1 reconstructs the numbers from the evaluation records, manifests, checkpoints,
   logs and cached targets on disk, identifies the exact model/protocol that produced them, and proves the
   ceiling fix was active (`historical/`). Result: **22.71 % / 33.88 %** — the Occ3D value was misremembered.
2. **What is the clean geometry-only baseline the semantic-completion work should build on?** Phase 2 trains the
   frozen local-reset pipeline (L + LB-M + H → `CompletionUNet` 3-channel, Gate 12/13 recipe, seeds 0/1/2),
   confirms it reproduces Gate 13 M on KITTI-360 0006, freezes everything with hashes, and only then runs one
   zero-shot evaluation per target (`current/`, `targets/`).

Historical reconstruction comes first because a comparison against a misremembered or unverifiable number is
worse than no comparison: the historical number defines what "regression" would even mean.

**What counts as the post-ceiling-bug baseline.** The adopted sky-fix model (`artifacts/gate8c1/frozen_manifest_sky.json`:
replicate padding + open-sky-is-FREE targets, three seeds retrained, thresholds reselected on 0006), evaluated by the
gate's own `core/run/eval_target.py` under Protocol A `past5` on the full official anchor sets, median of three seeds.
Not the OccAny-scorer subsets, not the rejected `sky2` variant, not the pre-fix headline. See `historical/provenance.py`.

## Layout

```
tools/baseline_reconstruction/
  config.py                  one source of truth: paths, drives, seeds, grid, threshold, target protocols, tolerances
  historical/records.py      parse eval JSONs / logs / manifests -> HistoricalResult (original metric names)
  historical/provenance.py   commits, hashes, the canonical decision, the historical pipeline description
  historical/verify_ceiling_fix.py   evidence the recovered result is the fixed pipeline (5 checks + firewall)
  current/geometry.py        LocalClip -> core IncrementalMapper -> (S, F); frames/units/scale stated once
  current/dataset.py         packed H inputs + raw-LiDAR targets -> crops for step_loss
  current/recipe.py          training_config.yaml = verbatim Gate 12 H recipe
  current/train.py           one seed, frozen recipe, file audit, run record
  current/evaluate_source.py 0006 regions x range bins; map-only reference
  targets/adapters.py        SemanticKITTI / Occ3D streams, anchors, masks, grids via the frozen core code
  targets/clips.py           local-reset LingBot clips on target streams (opens target images)
  targets/evaluate.py        the frozen zero-shot evaluation, all seeds in one pass
  metrics/occupancy.py       OccupancyCounts, IoU/P/R, range bins, accumulators, seed statistics
  metrics/comparability.py   property table + mechanical comparability classification
  freeze.py                  code_freeze.json writer and require_source_freeze() guard
  reports/                   render-only: historical / source / zero_shot / comparison / verification / final
  cli/                       one command per phase (all support --help)
tests/baseline_reconstruction/   metrics, historical parsing, ceiling-fix constants, scale-once, freeze guard, comparability
artifacts/baseline_reconstruction/   every deliverable; checkpoints/, runs/, source_eval/, zero_shot_eval/, tables/, figures/, logs/
```

Conventions that bit this project before and are therefore explicit in code: LingBot depth is **canonical** until the
mapper multiplies it by the clip scale (`current/geometry.py`); the scale is forced once via `scale_state.force`, which
scales depth and translation together; the anchor camera is the map's world frame; `T_cam_to_grid` is metres.

## Running it

```bash
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH FLASHINFER_CUDA_ARCH_LIST="12.0a"
PY=/home/minh/anaconda3/envs/cu128/bin/python; T=tools/baseline_reconstruction/cli

$PY $T/audit_history.py --device cuda:0                    # Phase 1: historical audit (no target dataset opened)
# predeclaration.md is written by hand before the next step
for s in 0 1 2; do $PY $T/train_current.py --seed $s --device cuda:$s & done; wait   # Phase 2a: ~9 min per seed
$PY $T/eval_source.py --device cuda:0 --curve --no-report  # Phase 2b: learning curves (stride-5 anchors)
$PY $T/eval_source.py --device cuda:0                      #           final checkpoints, 590 anchors -> source_results.md
$PY $T/freeze_source.py                                    # Phase 2c: code_freeze.json + source_freeze.md (refuses on a FAIL verdict)
# ---- target evaluation is unlocked only by code_freeze.json; every entry point re-hashes the frozen files ----
$PY $T/build_target_clips.py --dataset semantickitti --device cuda:0
for s in 0 1 2 3; do $PY $T/build_target_clips.py --dataset occ3d --shard $s --shards 4 --device cuda:$s & done; wait
$PY $T/eval_targets.py --dataset semantickitti --device cuda:0
$PY $T/eval_targets.py --dataset occ3d --device cuda:1
$PY $T/report.py --device cuda:0                           # Phase 4: zero_shot_results.md, historical_comparability.md, verification.md, report.md
$PY -m pytest tests/baseline_reconstruction -q
```

When target evaluation is unlocked: after `freeze_source.py` has written `code_freeze.json`, and only while every hashed
file (this package's evaluation code, the core modules it imports, the LingBot checkpoint, the recipe, the three final
checkpoints) is byte-identical to the record. Render-only code (`reports/`, `historical/`, `cli/report.py`) is outside the
hash set. After the first target evaluation nothing is rerun with alternatives; a genuine bug would be documented next
to the original result, never in place of it.

Where results are written: `artifacts/baseline_reconstruction/` (see `report.md` for the index). Target clips go to
`/media/SSD1/MINH_DATASETS/lingbot_baseline_reconstruction/target_clips/`.
