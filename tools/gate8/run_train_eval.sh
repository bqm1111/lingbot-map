#!/usr/bin/env bash
# Gate 8 primary experiment: train on sk_train + occ3d_train (genuine train splits), select
# on source validation (geometry loss + teacher KL only), evaluate held-out KITTI-360 once.
# Pool-aware: GPUs come from GATE8_GPUS (default "1 2 3").
set -u
REPO=/home/minh/workspace/lingbot-map_fork; cd "$REPO"; source tools/gate8/gpus.sh
PY=/home/minh/anaconda3/envs/cu128/bin/python
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
mkdir -p artifacts/gate8/logs
TRAIN_GPU="cuda:$(g8 0)"; EVAL_A="cuda:$(g8 0)"; EVAL_B="cuda:$(g8 1)"
echo "GPU pool ${G8[*]}  train=$TRAIN_GPU  eval=$EVAL_A,$EVAL_B"
# 1. wait for training samples
until ls artifacts/gate8/samples_sk_train_s*.json >/dev/null 2>&1 \
   && ls artifacts/gate8/samples_occ3d_train_s*.json >/dev/null 2>&1; do sleep 60; done
echo "=== samples ready $(date -Is) ==="
# 2. small-subset run (sanity), then the primary run
$PY tools/gate8/train.py --config configs/gate8/completion.yaml --device "$TRAIN_GPU" --tag subset --steps 300 --limit 200 \
    > artifacts/gate8/logs/train_subset.log 2>&1
$PY tools/gate8/train.py --config configs/gate8/completion.yaml --device "$TRAIN_GPU" --tag completion \
    > artifacts/gate8/logs/train_completion.log 2>&1
echo "=== training done $(date -Is) ==="
# 3. evaluation: mapper-only in the union vocabulary (apples-to-apples), then with completion
for DS in semantickitti occ3d kitti360; do
  $PY tools/gate8/evaluate.py --source $DS --device "$EVAL_A" --vocab union --tag mapper_union \
      > artifacts/gate8/logs/eval_${DS}_mapper_union.log 2>&1 &
  $PY tools/gate8/evaluate.py --source $DS --device "$EVAL_B" --vocab union --tag complete_union \
      --checkpoint artifacts/gate8/checkpoints/completion_best.pt \
      > artifacts/gate8/logs/eval_${DS}_complete_union.log 2>&1 &
  wait
done
echo "=== evaluation done $(date -Is) ==="
