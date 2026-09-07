#!/usr/bin/env bash
# Gate 8C-1: the locked target evaluation. 2 targets x 3 protocols x 3 seeds = 18 runs.
# Refuses to start unless the frozen manifest exists (eval_target.py enforces this too).
set -u
REPO=/home/minh/workspace/lingbot-map_fork
cd "$REPO"; source tools/gate8/gpus.sh
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST="12.0a" HF_HUB_OFFLINE=1
PY=/home/minh/anaconda3/envs/cu128/bin/python
mkdir -p artifacts/gate8c1/logs
[ -f artifacts/gate8c1/frozen_manifest.json ] || { echo "no frozen manifest; refusing"; exit 1; }
# one seed per GPU, all datasets/modes sequential within a seed
for S in 0 1 2; do
 (
  for D in semantickitti occ3d; do
    for M in past5 stream occany_fwd; do
      $PY tools/gate8c1/eval_target.py --dataset "$D" --mode "$M" --seed "$S" \
          --device "cuda:$(g8 $S)" >> "artifacts/gate8c1/logs/eval_seed${S}.log" 2>&1
    done
  done
 ) &
done
wait
echo "=== gate8c1 target evaluation done $(date -Is) ==="
