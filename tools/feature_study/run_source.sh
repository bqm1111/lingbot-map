#!/usr/bin/env bash
# Source measurement on KITTI-360 drive 0006. `run_source.sh 1 2` does those seeds only.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
export G8C1_DATA_ROOT=/media/SSD1/MINH_DATASETS/lingbot_gate8c1_sky
PY=/home/minh/anaconda3/envs/cu128/bin/python
LOG=artifacts/feature_completion_study/logs
mkdir -p "$LOG"
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2)
for seed in "${SEEDS[@]}"; do
  g=0
  for cond in A B C; do
    $PY tools/feature_study/evaluate_source.py --condition "$cond" --seed "$seed" \
        --device "cuda:$g" > "$LOG/source_${cond}_seed${seed}.log" 2>&1 &
    g=$((g+1))
  done
  wait
  echo "=== source seed $seed done ==="
  grep -h "^\[fcs" "$LOG"/source_*_seed${seed}.log
done
