#!/usr/bin/env bash
# `run_study.sh 0` trains R and S for seed 0 in parallel; no argument runs seeds 0 1 2.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
export G8C1_DATA_ROOT=/media/SSD1/MINH_DATASETS/lingbot_gate8c1_sky
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
PY=/home/minh/anaconda3/envs/cu128/bin/python
LOG=artifacts/teacher_alignment_study/logs
mkdir -p "$LOG"
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2)
GPUS=(${TAS_GPUS:-0 1})
for seed in "${SEEDS[@]}"; do
  i=0
  for cond in R S; do
    $PY tools/teacher_alignment/train.py --condition "$cond" --seed "$seed" \
        --device "cuda:${GPUS[$i]}" > "$LOG/train_${cond}_seed${seed}.log" 2>&1 &
    i=$((i+1))
  done
  wait
  echo "=== seed $seed done ==="
done
