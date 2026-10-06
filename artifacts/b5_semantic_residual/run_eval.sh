#!/usr/bin/env bash
set -u
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 PATH=/home/minh/anaconda3/envs/cu128/bin:$PATH
PY=/home/minh/anaconda3/envs/cu128/bin/python
LOGS=artifacts/b5_semantic_residual/logs; mkdir -p "$LOGS"
pids=()
for s in 0 1 2 3; do
  ( $PY -m tools.semantic_residual.evaluate --device "cuda:$s" --shard $s --shards 4 > "$LOGS/eval_s$s.log" 2>&1; \
    rc=$?; echo "exit $rc" >> "$LOGS/eval_s$s.log"; exit $rc ) &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
if [ "$rc" -ne 0 ]; then echo "EVAL FAILED -- see $LOGS/eval_s*.log" >&2; exit 1; fi
echo "eval ok"
