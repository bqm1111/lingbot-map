#!/usr/bin/env bash
# Train the study. `run_study.sh 0` runs seed 0's three conditions in parallel;
# with no argument it runs all three seeds, one seed at a time.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
export G8C1_DATA_ROOT=/media/SSD1/MINH_DATASETS/lingbot_gate8c1_sky
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
PY=/home/minh/anaconda3/envs/cu128/bin/python
LOG=artifacts/feature_completion_study/logs
mkdir -p "$LOG"
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2)
for seed in "${SEEDS[@]}"; do
  g=0
  for cond in A B C; do
    $PY tools/feature_study/train.py --condition "$cond" --seed "$seed" --device "cuda:$g" \
        > "$LOG/train_${cond}_seed${seed}.log" 2>&1 &
    g=$((g+1))
  done
  wait
  echo "=== seed $seed done ==="
done
