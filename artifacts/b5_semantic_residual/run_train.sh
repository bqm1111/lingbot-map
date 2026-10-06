#!/usr/bin/env bash
# B5: three seeds, one per GPU, same budget. Resumable; rerun to continue.
set -u
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 PATH=/home/minh/anaconda3/envs/cu128/bin:$PATH
PY=/home/minh/anaconda3/envs/cu128/bin/python
LOGS=artifacts/b5_semantic_residual/logs; mkdir -p "$LOGS"
pids=()
for s in 0 1 2; do
  ( $PY -m tools.semantic_residual.train --seed $s --device "cuda:$s" > "$LOGS/train_seed$s.log" 2>&1; \
    rc=$?; echo "exit $rc" >> "$LOGS/train_seed$s.log"; exit $rc ) &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
if [ "$rc" -ne 0 ]; then echo "TRAINING FAILED -- see $LOGS/train_seed*.log" >&2; exit 1; fi
echo "training ok"
