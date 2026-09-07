#!/usr/bin/env bash
# Gate 8A Stage 4: the ONE locked KITTI-360 evaluation.
# Reads the checkpoint and both thresholds out of the frozen selection written before any
# KITTI-360 data was opened. Nothing here may be re-tuned afterwards.
set -eu
REPO=/home/minh/workspace/lingbot-map_fork
cd "$REPO"; source tools/gate8/gpus.sh
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST="12.0a" HF_HUB_OFFLINE=1
PY=/home/minh/anaconda3/envs/cu128/bin/python
F=configs/gate8a/frozen_selection.yaml
CKPT=$($PY -c "import yaml;print(yaml.safe_load(open('$F'))['checkpoint'])")
TAU=$($PY -c "import yaml;print(yaml.safe_load(open('$F'))['global_threshold_final_logodds'])")
MTAU=$($PY -c "import yaml;print(yaml.safe_load(open('$F'))['mapper_calibrated_threshold_logodds'])")
echo "locked: ckpt=$CKPT tau=$TAU mapper_tau=$MTAU"
$PY tools/gate8a/evaluate.py --source kitti360 --checkpoints "selected=$CKPT" --tag locked \
    --locked selected --threshold "$TAU" --mapper-threshold "$MTAU" \
    --device "cuda:$(g8 0)" 2>&1 | tee artifacts/gate8a/logs/eval_kitti360_locked.log
