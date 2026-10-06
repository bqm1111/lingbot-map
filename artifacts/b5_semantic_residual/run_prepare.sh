#!/usr/bin/env bash
# B5 query cache: one drive per GPU. Resumable and fails loudly.
set -u
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 PATH=/home/minh/anaconda3/envs/cu128/bin:$PATH
PY=/home/minh/anaconda3/envs/cu128/bin/python
LOGS=artifacts/b5_semantic_residual/logs; mkdir -p "$LOGS"
DRIVES=(2013_05_28_drive_0003_sync 2013_05_28_drive_0007_sync 2013_05_28_drive_0010_sync 2013_05_28_drive_0006_sync)
pids=()
for i in 0 1 2 3; do
  d=${DRIVES[$i]}
  ( $PY -m tools.semantic_residual.prepare --drive "$d" --device "cuda:$i" --overwrite > "$LOGS/prepare_${d: -9:4}.log" 2>&1; \
    rc=$?; echo "exit $rc" >> "$LOGS/prepare_${d: -9:4}.log"; exit $rc ) &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
if [ "$rc" -ne 0 ]; then echo "PREPARE FAILED -- see $LOGS/prepare_*.log" >&2; exit 1; fi
echo "query cache ok"
