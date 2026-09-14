#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
export G8C1_DATA_ROOT=/media/SSD1/MINH_DATASETS/lingbot_gate8c1_sky
PY=/home/minh/anaconda3/envs/cu128/bin/python
LOG=artifacts/teacher_alignment_study/logs
for ds in semantickitti occ3d; do
  for seed in 0 1 2; do
    i=0
    for cond in R S; do
      $PY tools/teacher_alignment/transfer.py --condition "$cond" --seed "$seed" --dataset "$ds" \
          --device "cuda:$i" > "$LOG/transfer_${ds}_${cond}_seed${seed}.log" 2>&1 &
      i=$((i+1))
    done
    wait
    echo "=== $ds seed $seed done ==="
  done
done
echo "TRANSFER COMPLETE"
