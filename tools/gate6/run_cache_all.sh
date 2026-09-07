#!/usr/bin/env bash
# Cache Trident-H semantics for every unique frame of all three benchmarks.
# One process per GPU; datasets run in sequence so each process holds one teacher.
set -u
REPO=/home/minh/workspace/lingbot-map_fork
TRI=/home/minh/workspace/third_party/Trident
PY=/home/minh/workspace/third_party/trident_env/bin/python
export PYTHONPATH=$REPO:$TRI
cd "$TRI" || exit 1
for DS in kitti360 semantickitti occ3d; do
  echo "=== $DS $(date -Is) ==="
  for S in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$S $PY "$REPO/tools/gate6/cache_semantics.py" \
        --dataset "$DS" --shard "$S" --num-shards 4 --device cuda:0 \
        > "$REPO/artifacts/gate6/logs/cache_${DS}_s${S}.log" 2>&1 &
  done
  wait
  echo "=== $DS done $(date -Is) ==="
done
echo "ALL DONE $(date -Is)"
