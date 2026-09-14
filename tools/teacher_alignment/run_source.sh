#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
export G8C1_DATA_ROOT=/media/SSD1/MINH_DATASETS/lingbot_gate8c1_sky
PY=/home/minh/anaconda3/envs/cu128/bin/python
LOG=artifacts/teacher_alignment_study/logs
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2)
GPUS=(${TAS_GPUS:-2 3})
for seed in "${SEEDS[@]}"; do
  i=0
  for cond in R S; do
    $PY tools/teacher_alignment/evaluate_source.py --condition "$cond" --seed "$seed" \
        --device "cuda:${GPUS[$i]}" > "$LOG/source_${cond}_seed${seed}.log" 2>&1 &
    i=$((i+1))
  done
  wait
  echo "=== source seed $seed done ==="
  grep -h "^\[tas" "$LOG"/source_*_seed${seed}.log
done
