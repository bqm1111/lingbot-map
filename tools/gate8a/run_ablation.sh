#!/usr/bin/env bash
# Gate 8A Stage 2: train the three new cells of the 2x2, one per GPU of the pool.
# Cell A (occ_centred x focal_dice) is the existing Gate 8 run and is NOT retrained.
set -u
REPO=/home/minh/workspace/lingbot-map_fork
cd "$REPO"; source tools/gate8/gpus.sh
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST="12.0a" HF_HUB_OFFLINE=1
PY=/home/minh/anaconda3/envs/cu128/bin/python
mkdir -p artifacts/gate8a/logs
CELLS=(cellB_uniform_focal cellC_occcentred_bce cellD_uniform_bce)
for i in "${!CELLS[@]}"; do
  C=${CELLS[$i]}
  $PY tools/gate8a/train.py --config "configs/gate8a/${C}.yaml" --tag "$C" \
      --device "cuda:$(g8 $i)" > "artifacts/gate8a/logs/train_${C}.log" 2>&1 &
done
wait
echo "=== gate8a ablation training done $(date -Is) ==="
