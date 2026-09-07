#!/usr/bin/env bash
# Gate 8A Stage 1+3: score every ablation checkpoint on the two SOURCE validation sets.
# One map build per benchmark serves all checkpoints, so every candidate sees byte-identical
# maps, masks and anchors. KITTI-360 is not touched here.
set -eu
REPO=/home/minh/workspace/lingbot-map_fork
cd "$REPO"; source tools/gate8/gpus.sh
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST="12.0a" HF_HUB_OFFLINE=1
PY=/home/minh/anaconda3/envs/cu128/bin/python
mkdir -p artifacts/gate8a/logs
CK="cellA_best=artifacts/gate8/checkpoints/completion_best.pt"
CK="$CK,cellA_last=artifacts/gate8/checkpoints/completion_last.pt"
for C in cellB_uniform_focal cellC_occcentred_bce cellD_uniform_bce; do
  S=$(echo "$C" | cut -d_ -f1)
  CK="$CK,${S}_best=artifacts/gate8a/checkpoints/${C}_best.pt"
  CK="$CK,${S}_last=artifacts/gate8a/checkpoints/${C}_last.pt"
done
echo "candidates: $CK"
$PY tools/gate8a/evaluate.py --source semantickitti --checkpoints "$CK" --tag source \
    --device "cuda:$(g8 0)" > artifacts/gate8a/logs/eval_semantickitti_source.log 2>&1 &
$PY tools/gate8a/evaluate.py --source occ3d --checkpoints "$CK" --tag source \
    --device "cuda:$(g8 1)" > artifacts/gate8a/logs/eval_occ3d_source.log 2>&1 &
wait
echo "=== gate8a source scoring done $(date -Is) ==="
$PY tools/gate8a/selection.py --tag source --checkpoints "$CK" \
    2>&1 | tee artifacts/gate8a/logs/selection.log
