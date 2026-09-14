#!/usr/bin/env bash
# Seven-drive KITTI-360 data-scaling experiment, end to end after the frozen-model caches.
# Stops at the first failing stage (verification failures stop the run before training).
set -u
REPO=/home/minh/workspace/lingbot-map_fork; cd "$REPO"
export PYTHONPATH=$REPO CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH FLASHINFER_CUDA_ARCH_LIST="12.0a" HF_HUB_OFFLINE=1
PY=/home/minh/anaconda3/envs/cu128/bin/python
T=tools/kitti360_data_scaling_7drives; W=$T/drives7.py
A=artifacts/kitti360_data_scaling_7drives; L=$A/logs
C=/media/SSD1/MINH_DATASETS/lingbot_gate8b
D=/media/SSD1/MINH_DATASETS/lingbot_k360_7drives
say() { echo "[$(date -Is)] $*"; }
die() { say "STOPPED: $*"; exit 1; }
waitjobs() { local bad=0; for p in "$@"; do wait "$p" || bad=1; done; return $bad; }
say "resume from target verification (check amended: semantic-evidence run-to-run magnitude envelope)"
say "target verification"
$PY $T/verify_targets.py --device cuda:0 > $L/verify_targets.log 2>&1 || die "target verification failed (see $L/verify_targets.log)"
say "manifests + budget"
$PY $T/manifests.py > $L/manifests.log 2>&1 || die "manifest freeze failed"
say "smoke test"
$PY $T/train.py --run full --seed 0 --steps 20 --device cuda:0 > $L/smoke_train.log 2>&1 || die "smoke training failed"
rm -f $A/checkpoints/init_seed0.pt   # the smoke run must not define the shared initial weights
$PY $T/evaluate.py --checkpoint $A/smoke/kds7_full_seed0_step00020.pt --stride 59 --out $A/smoke/eval_smoke.json --device cuda:0 > $L/smoke_eval.log 2>&1 || die "smoke evaluation failed"

say "four runs"
# the seed's initial weights are created by the first run of that seed and loaded by the second
$PY $T/train.py --run small --seed 0 --device cuda:0 > $L/train_small_seed0.log 2>&1 &
$PY $T/train.py --run small --seed 1 --device cuda:1 > $L/train_small_seed1.log 2>&1 &
until [ -f $A/checkpoints/init_seed0.pt ] && [ -f $A/checkpoints/init_seed1.pt ]; do sleep 5; done
sleep 30
$PY $T/train.py --run full --seed 0 --device cuda:2 > $L/train_full_seed0.log 2>&1 &
$PY $T/train.py --run full --seed 1 --device cuda:3 > $L/train_full_seed1.log 2>&1 &
wait || true
for r in small full; do for s in 0 1; do [ -f $A/runs/train_kds7_${r}_seed${s}.json ] || die "run $r seed $s did not finish"; done; done
say "training done"

say "evaluation"
STEPS=$($PY -c "import json;print(json.load(open('$A/budget.json'))['steps'])")
g=0
for r in small full; do for s in 0 1; do
  ( for ck in $A/checkpoints/kds7_${r}_seed${s}_step*.pt; do
      st=$(basename $ck .pt | sed 's/.*_step0*//')
      if [ "$st" = "$STEPS" ]; then $PY $T/evaluate.py --checkpoint $ck --device cuda:$g >> $L/eval_${r}_seed${s}.log 2>&1
      else $PY $T/evaluate.py --checkpoint $ck --stride 5 --device cuda:$g >> $L/eval_${r}_seed${s}.log 2>&1; fi
    done ) &
  g=$((g+1))
done; done
wait
say "analysis"
$PY $T/analyze.py > $L/analyze.log 2>&1 || die "analysis failed"
say "KDS7 CHAIN COMPLETE"
