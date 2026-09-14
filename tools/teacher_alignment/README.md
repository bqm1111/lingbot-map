# Teacher-alignment control

**Question.** Does the *correct local assignment* of the future-teacher class target matter
for occupancy transfer, once the auxiliary gradient budget at the shared trunk is held fixed?

Two conditions, one difference:

| | auxiliary target | trunk gradient rule |
|---|---|---|
| **R** | the future-teacher class vector at its own voxel | `g_occ + rho·‖g_occ‖/‖g_aux‖ · g_aux` |
| **S** | same vectors, permuted within (z, map-observed) strata of each crop | same |

Everything else is the feature study's condition B: data, crops, initialisation, architecture,
optimizer, schedule, 6 000 steps, final checkpoint, `past5` inference. Heads receive their
own unscaled task gradient (occupancy head ← `L_occ`, semantic head ← `0.5·L_KL`).

## Modified code path

Nothing under `core/` changed for this study. `tools/feature_study/evaluate_source.py` got a
one-line generalisation (the semantic-threshold rule now applies to any condition listed in
`ST.SEMANTIC_CONDITIONS`, default `("B",)`, so the feature study's behaviour is unchanged).

* `study.py` — re-points the feature-study module at `artifacts/teacher_alignment_study/`
  and conditions `R/S`; every reused tool reads those attributes at call time.
* `shuffle.py` — the stratified permutation, its own RNG keyed by (seed, step, crop), the
  tiny-strata rule, the manipulation statistics, and the multiset-preservation check.
* `gradmatch.py` — task gradients over parameter groups and the matched combination rule.
* `calibrate_rho.py` — fixes `rho` from historical B seed 0 on 32 deterministic minibatches.
* `train.py` — the trainer; `evaluate_source.py` / `transfer.py` — thin wrappers that add the
  aligned-teacher agreement and the sparse per-voxel semantic record for the common-set
  naming comparison; `freeze_manifest.py`, `semantics.py`, `compare.py`, `figures.py`,
  `verify.py`, `dump_predictions.py`.

## Reproduce

```bash
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH="$PWD"  G8C1_DATA_ROOT=/media/SSD1/MINH_DATASETS/lingbot_gate8c1_sky
PY=/home/minh/anaconda3/envs/cu128/bin/python
$PY tools/teacher_alignment/calibrate_rho.py --device cuda:0        # rho, frozen
$PY tools/teacher_alignment/verify.py --device cuda:0 --pre         # rule == joint backprop at c=1
bash tools/teacher_alignment/run_study.sh                            # 6 runs
bash tools/teacher_alignment/run_source.sh                           # drive 0006
$PY tools/teacher_alignment/freeze_manifest.py
bash tools/teacher_alignment/run_transfer.sh                         # 12 frozen evaluations
$PY tools/teacher_alignment/semantics.py
$PY tools/teacher_alignment/compare.py
$PY tools/teacher_alignment/figures.py
$PY tools/teacher_alignment/dump_predictions.py --seed 0 --device cuda:0
$PY tools/teacher_alignment/verify.py --device cuda:0
```

Result: `reports/teacher_alignment_study.md`; numbers in
`artifacts/teacher_alignment_study/{comparison.json,comparison_table.md,semantics.json}`.

## Result (2026-09-11)

Budgets matched (achieved ratio = ρ = 3.564 on 5 999/6 000 steps per run); shuffle changed
46 % of supervision vectors and 31 % of argmax classes. R does not beat S anywhere:
drive 0006 later IoU R−S = −0.006 / −0.002 / +0.005; SemanticKITTI −0.022 / −0.019 / +0.013;
Occ3D −0.036 / −0.001 / +0.038 (sign follows the frozen threshold gap). Threshold-free, S is
equal or slightly ahead on 5/6 dataset×seed cells. R is 1.5 pp better at agreeing with the
aligned teacher and ~2 pp better at naming on SemanticKITTI, with no occupancy consequence.
Verification 10/10. Full numbers in `artifacts/teacher_alignment_study/comparison_table.md`,
interpretation in `reports/teacher_alignment_study.md`.
