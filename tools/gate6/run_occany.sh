#!/usr/bin/env bash
set -u
REPO=/home/minh/workspace/lingbot-map_fork
PY=/home/minh/workspace/third_party/occany_env/bin/python
export PYTHONPATH=$REPO
cd /home/minh/workspace/third_party/OccAny || exit 1
for S in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$S $PY "$REPO/tools/gate6/occany_comparator.py" --stage cache \
     --shard $S --num-shards 4 --device cuda:0 \
     > "$REPO/artifacts/gate6/logs/occany_s${S}.log" 2>&1 &
done
wait
echo "OCCANY CACHE DONE $(date -Is)"
