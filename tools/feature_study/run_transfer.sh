#!/usr/bin/env bash
# One frozen transfer evaluation per checkpoint and dataset. Requires the manifests.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
export G8C1_DATA_ROOT=/media/SSD1/MINH_DATASETS/lingbot_gate8c1_sky
PY=/home/minh/anaconda3/envs/cu128/bin/python
LOG=artifacts/feature_completion_study/logs
mkdir -p "$LOG"
for ds in semantickitti occ3d; do
  for seed in 0 1 2; do
    g=0
    for cond in A B C; do
      $PY tools/feature_study/transfer.py --condition "$cond" --seed "$seed" \
          --dataset "$ds" --device "cuda:$g" \
          > "$LOG/transfer_${ds}_${cond}_seed${seed}.log" 2>&1 &
      g=$((g+1))
    done
    wait
    echo "=== $ds seed $seed done ==="
  done
done
