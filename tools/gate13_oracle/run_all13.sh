#!/usr/bin/env bash
# Gate 13 chain after M (Gate 12 cache) and O inputs exist and input quality passed: freeze -> 6 runs -> evaluation -> analysis -> figures.
set -u
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1
PY=/home/minh/anaconda3/envs/cu128/bin/python; T=tools/gate13_oracle; L=artifacts/gate13_oracle/logs; A=artifacts/gate13_oracle
$PY - <<'PYEOF'
import hashlib, json, datetime, subprocess
h = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
inp = ["tools/gate13_oracle/g13.py", "tools/gate13_oracle/build_oracle.py", "tools/gate9_depth_occupancy/audit.py", "tools/gate9_depth_occupancy/common.py", "tools/gate9_depth_occupancy/variant_mapper.py",
       "core/mapping/mapper.py", "tools/gate12_suf/build_inputs.py", "tools/gate12_suf/g12.py"]
trn = ["tools/gate13_oracle/train13.py", "tools/gate13_oracle/eval13.py", "tools/gate13_oracle/quality13.py", "tools/gate12_suf/train12.py", "tools/gate12_suf/eval12.py", "core/run/_gate8a_train.py",
       "core/model/net.py", "core/model/losses.py", "core/model/sampler.py", "artifacts/gate13_oracle/training_config_M.yaml", "artifacts/gate13_oracle/training_config_O.yaml"]
ana = ["tools/gate13_oracle/analyze13.py", "tools/gate13_oracle/figures13.py"]
json.dump({"frozen_at": datetime.datetime.now().isoformat(timespec="seconds"), "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "note": "frozen after the oracle input build, the input-quality check (O == Gate 9 LD on 590/590) and the M/O smoke pairing test; before any Gate 13 training run. M reproduction tolerance and decision guide predeclared in analyze13.py",
           "input_generation": {f: h(f) for f in inp}, "training_evaluation": {f: h(f) for f in trn}, "analysis": {f: h(f) for f in ana}}, open("artifacts/gate13_oracle/code_freeze.json", "w"), indent=1)
PYEOF
echo "code frozen $(date -Is)"
declare -A GPU=([M0]=0 [O0]=1 [M1]=2 [O1]=3 [M2]=0 [O2]=1)
for s in 0 1 2; do for c in M O; do $PY $T/train13.py --cond $c --seed $s --device cuda:${GPU[$c$s]} > $L/train_${c}_seed${s}.log 2>&1 & done; done; wait
echo "training done $(date -Is)"; grep -h "done" $L/train_*.log
i=0
for s in 0 1 2; do for c in M O; do
  ( $PY $T/eval13.py --checkpoint $A/checkpoints/g13_${c}_seed${s}_step06000.pt --device cuda:$((i % 4)) > $L/eval_${c}_seed${s}.log 2>&1
    for st in 01000 02000 03000 04000 05000 06000; do $PY $T/eval13.py --checkpoint $A/checkpoints/g13_${c}_seed${s}_step${st}.pt --stride 5 --device cuda:$((i % 4)) >> $L/eval_curve_${c}_seed${s}.log 2>&1; done ) &
  i=$((i+1))
done; done; wait
echo "evaluation done $(date -Is)"; tail -qn1 $L/eval_M_seed*.log $L/eval_O_seed*.log
$PY $T/analyze13.py > $L/analyze.log 2>&1; tail -5 $L/analyze.log
$PY $T/figures13.py --device cuda:0 > $L/figures.log 2>&1; tail -2 $L/figures.log
echo "GATE13 CHAIN COMPLETE $(date -Is)"
