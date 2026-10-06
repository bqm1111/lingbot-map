#!/usr/bin/env bash
# Gate 19 learning curve. Jobs are queued round-robin across the 4 GPUs; each run is resumable (a finished run
# writes its record and is skipped on re-invocation).
set -u
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 PATH=/home/minh/anaconda3/envs/cu128/bin:$PATH
PY=/home/minh/anaconda3/envs/cu128/bin/python
LOGS=artifacts/gate19_data_scaling/logs; mkdir -p "$LOGS"

JOBS=()
for s in 0 1 2; do for sc in D_small D_medium D_large D_all; do JOBS+=("$sc fixed_steps $s"); done; done
# D_small fixed_epochs == fixed_steps (6000 steps), so it is not repeated
for s in 0 1 2; do for sc in D_medium D_large D_all; do JOBS+=("$sc fixed_epochs $s"); done; done

run_queue() {  # $1 = gpu, rest = job indices
  local gpu=$1; shift
  for i in "$@"; do
    set -- ${JOBS[$i]}
    $PY -m tools.data_scaling.train --scale "$1" --regime "$2" --seed "$3" --device "cuda:$gpu" \
        >> "$LOGS/gpu$gpu.log" 2>&1 || { echo "FAILED: ${JOBS[$i]}" >&2; return 1; }
  done
}

pids=()
for g in 0 1 2 3; do
  idx=(); for ((i=g; i<${#JOBS[@]}; i+=4)); do idx+=($i); done
  run_queue $g "${idx[@]}" & pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
if [ "$rc" -ne 0 ]; then echo "GATE19 TRAINING FAILED -- see $LOGS/gpu*.log" >&2; exit 1; fi
echo "gate19 training ok (${#JOBS[@]} jobs)"
