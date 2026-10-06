#!/usr/bin/env bash
# The fixed-compute control: D_small and D_large at D_all's step count, so data and optimisation can be separated.
set -u
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 PATH=/home/minh/anaconda3/envs/cu128/bin:$PATH
PY=/home/minh/anaconda3/envs/cu128/bin/python
LOGS=artifacts/gate19_data_scaling/logs; mkdir -p "$LOGS"
JOBS=(); for s in 0 1 2; do for sc in D_small D_large; do JOBS+=("$sc $s"); done; done
pids=()
for g in 0 1 2 3; do
  ( for ((i=g; i<${#JOBS[@]}; i+=4)); do set -- ${JOBS[$i]}
      $PY -m tools.data_scaling.train --scale "$1" --regime fixed_compute --seed "$2" --device "cuda:$g" \
        >> "$LOGS/control_gpu$g.log" 2>&1 || { echo "FAILED ${JOBS[$i]}" >&2; exit 1; }
    done ) & pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ "$rc" -eq 0 ] || { echo "CONTROL FAILED" >&2; exit 1; }
echo "fixed-compute control ok"
