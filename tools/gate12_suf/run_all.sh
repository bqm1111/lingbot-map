#!/usr/bin/env bash
# Gate 12 chain after inputs exist: sanity -> code freeze -> 6 training runs -> evaluation -> analysis.
set -u
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1
PY=/home/minh/anaconda3/envs/cu128/bin/python; T=tools/gate12_suf; L=artifacts/gate12_suf/logs; A=artifacts/gate12_suf
$PY $T/sanity_inputs.py --device cuda:0 > $L/sanity_inputs.log 2>&1 || { echo "SANITY FAILED -- not training"; tail -20 $L/sanity_inputs.log; exit 1; }
$PY - <<'PYEOF'
import hashlib, json, datetime, subprocess
h = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
inp = ["tools/gate12_suf/g12.py", "tools/gate12_suf/build_inputs.py", "tools/gate12_suf/gen_local_clips.py", "tools/gate11_uncertainty/channel_mapper.py", "tools/gate11_uncertainty/g11.py",
       "tools/gate9_depth_occupancy/variant_mapper.py", "core/mapping/mapper.py", "artifacts/gate11_uncertainty/uncertainty_rules.json", "tools/local_registration_audit/run_local_lingbot.py"]
trn = ["tools/gate12_suf/train12.py", "tools/gate12_suf/eval12.py", "tools/gate12_suf/build_regions.py", "tools/gate12_suf/sanity_inputs.py", "core/run/_gate8a_train.py", "core/model/net.py",
       "core/model/losses.py", "core/model/sampler.py", "artifacts/gate12_suf/training_config_H.yaml", "artifacts/gate12_suf/training_config_SUF.yaml"]
json.dump({"frozen_at": datetime.datetime.now().isoformat(timespec="seconds"), "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "note": "input generation frozen after the pre-training sanity checks passed; training / evaluation code and configs frozen before any training run; analysis thresholds predeclared in analyze12.py",
           "input_generation": {f: h(f) for f in inp}, "training_evaluation": {f: h(f) for f in trn}, "analysis": {"tools/gate12_suf/analyze12.py": h("tools/gate12_suf/analyze12.py")}},
          open("artifacts/gate12_suf/code_freeze.json", "w"), indent=1)
PYEOF
echo "sanity passed, code frozen $(date -Is)"
declare -A GPU=([H0]=0 [SUF0]=1 [H1]=2 [SUF1]=3 [H2]=0 [SUF2]=1)
for s in 0 1 2; do for c in H SUF; do
  $PY $T/train12.py --cond $c --seed $s --device cuda:${GPU[$c$s]} > $L/train_${c}_seed${s}.log 2>&1 &
done; done; wait
echo "training done $(date -Is)"
i=0
for s in 0 1 2; do for c in H SUF; do
  ( $PY $T/eval12.py --checkpoint $A/checkpoints/g12_${c}_seed${s}_step06000.pt --device cuda:$((i % 4)) > $L/eval_${c}_seed${s}.log 2>&1
    for st in 01000 02000 03000 04000 05000 06000; do $PY $T/eval12.py --checkpoint $A/checkpoints/g12_${c}_seed${s}_step${st}.pt --stride 5 --device cuda:$((i % 4)) >> $L/eval_curve_${c}_seed${s}.log 2>&1; done ) &
  i=$((i+1))
done; done; wait
echo "evaluation done $(date -Is)"
$PY $T/analyze12.py > $L/analyze.log 2>&1; tail -40 $L/analyze.log
echo "GATE12 CHAIN COMPLETE $(date -Is)"
