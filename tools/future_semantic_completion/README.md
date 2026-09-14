# Future-semantic-completion study

**Question.** Can a small causal head predict the semantic representation of voxels that
are currently unobserved but later confirmed occupied, using future observations as
privileged supervision? Geometry is frozen and never touched; occupancy is oracle at
evaluation, so semantic errors are not mixed with geometry errors.

## What is reused

| piece | where |
|---|---|
| causal 32-channel input, future-window teacher target `fut_p`, `observed`, `fut_observed`, raw-LiDAR `gt_occ` | `core/supervision/targets.py`; cached samples under `$G8C1_DATA_ROOT` (ceiling-fixed root) |
| frozen geometry (trunk + occupancy head + original semantic head) | adopted Gate 8C-1 checkpoints, `artifacts/gate8c1/frozen_manifest_sky.json` |
| teacher | Trident-H vocabulary-conditioned probabilities, lifted by the frozen future mapper (25-way union simplex; no language-aligned dense embedding is cached, and this is the representation's stated limit) |
| drive-0006 measured partition (`later_measured`, `measured_occ`, `map_observed`) | `tools/feature_study/build_eval_masks.py` output |
| transfer protocol (`past5`, official grids and masks, union -> benchmark map) | `core/run/eval_target.py` pieces, called directly |

## Supervision set

`~observed & fut_observed & gt_occ & gt_valid & fut_valid` -- unobserved by the causal
map at `t`, observed by the future mapper in `t+1..t+20`, occupied in the raw-LiDAR target,
and carrying a future teacher vector. Counts, class mix and the projection-consistency
check: `supervision_stats.py` -> `artifacts/future_semantic_completion/supervision.json`.

## Conditions

* **B0** -- the mean teacher vector over the training key set (`b0_mean.npz`).
* **B1** -- the probability vector of the nearest causal-map voxel carrying teacher
  evidence (Euclidean distance transform), B0 beyond 15 voxels (3 m).
* **B2** -- `head.SemanticCompletionHead`: two 3x3x3 convolutions + 1x1 (77 k parameters)
  on `concat(frozen trunk d1, causal input)`, trained with `KL(teacher || student)` on the
  key set only. Full grids, batch 2, 6 000 steps, AdamW + cosine, last step kept.
* **B2ref** -- the frozen checkpoint's *existing* semantic head (trained jointly with
  occupancy on every future-target voxel), reported as a free reference row.

Gradient flow: the frozen network runs under `torch.no_grad` with `requires_grad=False`
on every parameter; the head reads detached tensors. `verify.py` checks the state digest
before/after training, the absence of gradients after a semantic backward, and that the
occupancy residual is bit-identical.

## Run

```bash
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH="$PWD" G8C1_DATA_ROOT=/media/SSD1/MINH_DATASETS/lingbot_gate8c1_sky
PY=/home/minh/anaconda3/envs/cu128/bin/python
$PY tools/future_semantic_completion/supervision_stats.py --device cuda:0   # counts + B0
$PY tools/future_semantic_completion/train.py --seed 0 --device cuda:0
$PY tools/future_semantic_completion/evaluate_source.py --seed 0 --device cuda:0
$PY tools/future_semantic_completion/figures.py --seed 0
$PY tools/future_semantic_completion/verify.py --seed 0 --device cuda:0
# only after the decision rule says so:
$PY tools/future_semantic_completion/freeze_manifest.py --seeds 0 1 2
$PY tools/future_semantic_completion/transfer.py --seed 0 --dataset semantickitti
$PY tools/future_semantic_completion/transfer.py --seed 0 --dataset occ3d
```

Report: `reports/future_semantic_completion.md`.

## Result (2026-09-12, seed 0) -- Case C

B2 does not beat the trivial baselines on the primary set: cosine 0.491 vs B0 0.476 (CI
crosses 0), top-1 0.388 vs the 0.394 majority rate, 6 classes predicted, cosine to the
global mean 0.83. Diagnosis: the future teacher target is not repeatable at voxel level --
the causal map's own vector for the same LiDAR-occupied voxel agrees with it only 0.456 of
the time (shuffled 0.35), flat in evidence weight. Transfer measured once for the record
(SemanticKITTI unobserved top-1 B2 0.398 < B0 0.451; Occ3D 0.314 > 0.238). Verification
8/10, the two failures being the finding itself. Report: `reports/future_semantic_completion.md`;
diagnosis: `target_diagnosis.json`; every number: `results_table.md`, `results.json`.

## Target-consistency follow-up (2026-09-13) -- Decision B

`target_consistency/`: the frozen future mapper replayed per future frame; candidates from
frame-disjoint groups A+B, validated on independent group C. 92-95 % of teacher-covered
voxels are seen by exactly one future frame, so two-group support is 0.45-0.89 % of the
eligible population. Requiring A/B agreement lifts independent repeatability from 0.53 to
0.70 on drive 0006 (0.91 on the training drives) but the within-anchor shuffled reference
rises alongside (0.38 -> 0.51; 0.60 -> 0.82), coverage drops to 0.5 % and supported classes
from 14 to 4. No target frozen, nothing trained. Report:
`artifacts/future_semantic_completion/target_consistency/report.md`
(copy at `reports/target_consistency_study.md`); 10/10 verification checks.

```bash
for d in 0003 0007 0010 0006; do $PY tools/future_semantic_completion/target_consistency/build_evidence.py --drive 2013_05_28_drive_${d}_sync --device cuda:0; done
$PY tools/future_semantic_completion/target_consistency/analyze.py
$PY tools/future_semantic_completion/target_consistency/neighbourhood.py
$PY tools/future_semantic_completion/target_consistency/causal_diagnostic.py --device cuda:0
$PY tools/future_semantic_completion/target_consistency/verify.py
$PY tools/future_semantic_completion/target_consistency/figures.py
$PY tools/future_semantic_completion/target_consistency/summarize.py
```
