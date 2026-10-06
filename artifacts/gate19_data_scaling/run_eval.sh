#!/usr/bin/env bash
set -u
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 PATH=/home/minh/anaconda3/envs/cu128/bin:$PATH
PY=/home/minh/anaconda3/envs/cu128/bin/python
LOGS=artifacts/gate19_data_scaling/logs; mkdir -p "$LOGS"
TAGS=($(ls artifacts/gate19_data_scaling/runs/ | sed 's/train_//;s/\.json//' | sort))
pids=()
for g in 0 1 2 3; do
  ( for ((i=g; i<${#TAGS[@]}; i+=4)); do
      $PY -m tools.data_scaling.evaluate --tag "${TAGS[$i]}" --device "cuda:$g" >> "$LOGS/eval_gpu$g.log" 2>&1 \
        || { echo "EVAL FAILED: ${TAGS[$i]}" >&2; exit 1; }
    done ) & pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ "$rc" -eq 0 ] || { echo "GATE19 EVAL FAILED -- see $LOGS/eval_gpu*.log" >&2; exit 1; }
echo "gate19 eval ok (${#TAGS[@]} tags)"
