#!/usr/bin/env bash
# Stages 4-9. The Qwen stage is split across the two designated free devices (GPU 2 and GPU 3);
# every other stage is cheap and runs on the first of them. Each stage is resumable and idempotent.
set -e
EXP=/media/SSD1/region_naming_pilot
export PYTHONDONTWRITEBYTECODE=1 HF_HOME=$EXP/cache/hf HF_HUB_OFFLINE=1 PYTHONPATH=$EXP/code
export TORCH_HOME=$EXP/cache/torch TRITON_CACHE_DIR=$EXP/cache/triton
PY=$EXP/venv/bin/python
GPUS=(${PILOT_GPUS:-2 3})

echo "=================== run_proposals  $(date -Is) ==================="
CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY $EXP/code/run_proposals.py cuda:0

echo "=================== run_qwen on GPUs ${GPUS[*]}  $(date -Is) ==================="
pids=()
for i in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} $PY $EXP/code/run_qwen.py cuda:0 "$i" "${#GPUS[@]}" \
      > $EXP/logs/qwen_gpu${GPUS[$i]}.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
grep -hv '^Loading weights' $EXP/logs/qwen_gpu*.log | tail -40
[ $fail -eq 0 ] || { echo "a Qwen shard failed"; exit 1; }

for s in run_predict evaluate figures report verify_final; do
  echo "=================== $s  $(date -Is) ==================="
  if [ "$s" = "report" ]; then $PY $EXP/code/$s.py
  else CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY $EXP/code/$s.py cuda:0; fi
done
echo "=================== ALL STAGES DONE  $(date -Is) ==================="
