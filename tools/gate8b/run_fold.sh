#!/usr/bin/env bash
# One Gate 8B fold end to end: train -> score {best,last} on the two SOURCE-validation
# domains -> freeze -> the locked target evaluation in both settings.
#     tools/gate8b/run_fold.sh semantickitti
set -eu
FOLD=$1
REPO=/home/minh/workspace/lingbot-map_fork
cd "$REPO"; source tools/gate8/gpus.sh
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST="12.0a" HF_HUB_OFFLINE=1
PY=/home/minh/anaconda3/envs/cu128/bin/python
mkdir -p artifacts/gate8b/logs
case "$FOLD" in
  semantickitti) VAL="occ3d kitti360" ;;
  occ3d)         VAL="semantickitti kitti360" ;;
  *) echo "unknown fold $FOLD"; exit 1 ;;
esac
STAGE=${STAGE:-all}
if [ "$STAGE" = all ] || [ "$STAGE" = train ]; then
  $PY tools/gate8b/train.py --config configs/gate8b/fold_$FOLD.yaml --device "cuda:$(g8 0)" \
      > artifacts/gate8b/logs/train_fold_$FOLD.log 2>&1
fi
if [ "$STAGE" = all ] || [ "$STAGE" = score ]; then
  CK="fold_${FOLD}_best=artifacts/gate8b/checkpoints/fold_${FOLD}_best.pt,fold_${FOLD}_last=artifacts/gate8b/checkpoints/fold_${FOLD}_last.pt"
  i=0
  for V in $VAL; do
    $PY -c "
import sys; sys.path.insert(0,'.'); import gate8b.sources
from tools.gate8a.evaluate import run
run('$V', 'cuda:$(g8 $i)', dict(kv.split('=') for kv in '$CK'.split(',')), tag='src_$FOLD', art='artifacts/gate8b')
" > artifacts/gate8b/logs/score_${FOLD}_${V}.log 2>&1 &
    i=$((i+1))
  done
  wait
  $PY tools/gate8b/select_fold.py --fold $FOLD 2>&1 | tee artifacts/gate8b/logs/select_$FOLD.log
fi
if [ "$STAGE" = all ] || [ "$STAGE" = target ]; then
  $PY tools/gate8b/eval_target.py --fold $FOLD --setting stream --device "cuda:$(g8 0)" \
      > artifacts/gate8b/logs/target_${FOLD}_stream.log 2>&1 &
  $PY tools/gate8b/eval_target.py --fold $FOLD --setting clips --device "cuda:$(g8 1)" \
      > artifacts/gate8b/logs/target_${FOLD}_clips.log 2>&1 &
  wait
fi
echo "=== fold $FOLD done $(date -Is) ==="
