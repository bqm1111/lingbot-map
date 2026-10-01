#!/usr/bin/env bash
# Gate 21, end to end. Assumes the Gate 20 population cache exists.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD
PY=/home/minh/anaconda3/envs/cu128/bin/python
LOG=artifacts/gate21_future_semantic_prior/logs

# 1. the training population: Gate 20's P0, restricted to its revealed rows, with the soft target
for i in 0 1 2; do
  d=$(echo "0003 0007 0010" | cut -d' ' -f$((i+1)))
  $PY -m tools.future_semantic_prior.targets --drive 2013_05_28_drive_${d}_sync --device cuda:$i &
done; wait
$PY -m tools.future_semantic_prior.targets --stats

# 2. train: 4 controls x 3 seeds, geometry seed 0 throughout
launch() { nohup $PY -m tools.future_semantic_prior.train --control $1 --seed $2 --device cuda:$3 > $LOG/train_$1_s$2.log 2>&1 & }
launch C3_full 0 0;      launch C3_full 1 0;      launch C3_full 2 0
launch C2_geometry 0 2;  launch C2_geometry 1 2;  launch C2_geometry 2 2
launch C4_semantics 0 1; launch C4_semantics 1 1
launch C4_semantics 2 3; launch C1_prior 0 3; launch C1_prior 1 3; launch C1_prior 2 3
wait

# 3. training curves (after training, so the training process never opens a 0006 file)
i=0
for c in C1_prior C2_geometry C3_full C4_semantics; do for s in 0 1 2; do
  nohup $PY -m tools.future_semantic_prior.curves --control $c --seed $s --device cuda:$((i % 4)) > $LOG/curves_${c}_s${s}.log 2>&1 &
  i=$((i+1)); done; done
wait

# 4. full development evaluation, all 590 anchors of 0006
i=0
for c in C1_prior C2_geometry C3_full C4_semantics; do for s in 0 1 2; do
  nohup $PY -m tools.future_semantic_prior.evaluate --control $c --seed $s --device cuda:$((i % 4)) > $LOG/eval_${c}_s${s}.log 2>&1 &
  i=$((i+1)); done; done
wait

# 5. analysis, verification, report
$PY -m tools.future_semantic_prior.selection_bias
$PY -m tools.future_semantic_prior.validate --device cuda:0
$PY -m tools.future_semantic_prior.analyze
$PY -m tools.future_semantic_prior.report
$PY -m pytest tests/future_semantic_prior -q
