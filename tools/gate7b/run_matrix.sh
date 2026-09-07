#!/usr/bin/env bash
# Gate 7B evaluation matrix. Twelve configurations per benchmark, distributed over the
# four GPUs. Every configuration was declared in the precommit before any was run.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/home/minh/anaconda3/envs/cu128/bin/python
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
mkdir -p artifacts/gate7b/logs
run () {  # dataset variant scale horizon conf gpu
  local ds=$1 var=$2 sc=$3 hz=$4 cf=$5 gpu=$6
  local tag="${var}_${sc}_h${hz}"
  [ "$var" = "S2" ] && tag="${tag}_c${cf}"
  $PY tools/gate7b/run_stream_eval.py --dataset "$ds" --variant "$var" --scale "$sc" \
      --horizon "$hz" --conf "$cf" --device "cuda:$gpu" \
      >> "artifacts/gate7b/logs/eval_${ds}_gpu${gpu}.log" 2>&1
}
for DS in semantickitti occ3d kitti360; do
  : > artifacts/gate7b/logs/eval_${DS}_gpu0.log
  : > artifacts/gate7b/logs/eval_${DS}_gpu1.log
  : > artifacts/gate7b/logs/eval_${DS}_gpu2.log
  : > artifacts/gate7b/logs/eval_${DS}_gpu3.log
  ( run $DS S1 G-A all  0.5 0; run $DS S1 G-A 50 0.5 0; run $DS S1 G-A 20 0.5 0 ) &
  ( run $DS S3 G-A all  0.5 1; run $DS S1 G-A 5  0.5 1; run $DS S1 G-A 1  0.5 1 ) &
  ( run $DS S4 G-A all  0.5 2; run $DS S2 G-A all 0.5 2; run $DS S2 G-A all 1.0 2 ) &
  ( run $DS S1 G-B all  0.5 3; run $DS S1 G-C all 0.5 3; run $DS S2 G-A all 0.0 3 ) &
  wait
  echo "=== $DS matrix done ==="
done
