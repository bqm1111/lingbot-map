#!/usr/bin/env bash
# The drive-0006 map-observability partition, four shards in parallel.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
export G8C1_DATA_ROOT=/media/SSD1/MINH_DATASETS/lingbot_gate8c1_sky
PY=/home/minh/anaconda3/envs/cu128/bin/python
LOG=artifacts/feature_completion_study/logs
mkdir -p "$LOG"
for s in 0 1 2 3; do
  $PY tools/feature_study/build_eval_masks.py --device "cuda:$s" --shard "$s" --shards 4 \
      > "$LOG/masks_s${s}.log" 2>&1 &
done
wait
echo "=== eval masks done ==="
