# Gate 8 — Causal semantic memory with privileged completion

**Date:** 2026-09-02 · **Commit:** `{{COMMIT}}` · **Status:** {{STATUS}}

## 0. Decision summary

{{DECISION}}

| question | answer |
|---|---|
| What was implemented | {{IMPLEMENTED}} |
| Is the mapper genuinely incremental and causal? | {{INCREMENTAL}} |
| Did the completion module learn? | {{LEARNED}} |
| Did it transfer to KITTI-360 (held out, untuned)? | {{TRANSFER}} |
| Did coverage improve without destroying precision or naming? | {{COVERAGE}} |

---

## 1. What was built

{{TABLE_IMPLEMENTED}}

**Frozen and unchanged:** LingBot-Map (direct streaming mode, native), MoGe-2 with the
calibrated FOV (scale only; depth rescue disabled), Trident-H (cached; never re-run for
the benchmarks), the union vocabulary, the official evaluation protocol and every prior
baseline result.

**Semantic representation — stated plainly.** Trident-H is cached as per-benchmark,
vocabulary-conditioned probability maps, not dense language-aligned features. The first
version therefore fuses and predicts a **declared 25-class union vocabulary**
(`gate8/vocab.py`), fixed before any result and mapped one-to-one into each benchmark's
classes at evaluation only. **This version is not query-time open-vocabulary.** The map
and the completion head must be moved to fixed-dimensional language-aligned features
before an open-vocabulary claim can be made.

---

## 2. The incremental mapper

{{TABLE_MAPPER}}

{{MAPPER_PROSE}}

---

## 3. Privileged training targets

{{TABLE_SAMPLES}}

{{SAMPLE_AUDIT}}

---

## 4. The completion module

{{TABLE_NET}}

---

## 5. Training

{{TABLE_TRAIN}}

![training](../../artifacts/gate8/fig_training.png)

{{TRAIN_PROSE}}

---

## 6. Results

{{TABLE_MAIN}}

![raw results](../../artifacts/gate8/fig_results_raw.png)

![dilated results](../../artifacts/gate8/fig_results_dil.png)

{{TABLE_COUNTS}}

{{TABLE_BOOTSTRAP}}

{{RESULTS_PROSE}}

### Classwise IoU (KITTI-360, held out)

{{TABLE_CLASSWISE}}

---

## 7. Runtime and memory

{{TABLE_RUNTIME}}

{{RUNTIME_PROSE}}

---

## 8. Training-integrity audit

{{TABLE_INTEGRITY}}

---

## 9. Recommendation (one action; awaiting approval)

{{RECOMMENDATION}}

---

## 10. Files, commands and manifest

{{TABLE_FILES}}

```bash
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH FLASHINFER_CUDA_ARCH_LIST="12.0a" HF_HUB_OFFLINE=1
PY=/home/minh/anaconda3/envs/cu128/bin/python
$PY tools/gate8/stage0.py                                   # provenance hashes
tools/gate8/run_caches.sh                                   # LingBot + MoGe + Trident on the train sources
for S in semantickitti occ3d kitti360; do $PY tools/gate8/evaluate.py --source $S --tag mapper_native; done
$PY tools/gate8/build_samples.py --source sk_train  --shard 0 --shards 2 &  # (x2)
$PY tools/gate8/build_samples.py --source occ3d_train --anchor-stride 2 --shard 0 --shards 2 &   # (x2)
$PY tools/gate8/build_samples.py --source semantickitti; $PY tools/gate8/build_samples.py --source occ3d --anchor-stride 4
tools/gate8/run_train_eval.sh                               # subset run, primary run, evaluation
$PY tools/gate8/runtime_audit.py --source kitti360 --checkpoint artifacts/gate8/checkpoints/completion_best.pt
(cd /home/minh/workspace/third_party/Trident && PYTHONPATH=$PWD:/home/minh/workspace/lingbot-map_fork \
   /home/minh/workspace/third_party/trident_env/bin/python /home/minh/workspace/lingbot-map_fork/tools/gate8/time_trident.py)
$PY tools/gate8/aggregate.py && $PY tools/gate8/figures.py && $PY tools/gate8/report.py --write
$PY -m pytest tests/gate6 tests/gate7a tests/gate7b tests/gate8 -q
```

{{MANIFEST}}

---

## 11. Deviations and limitations

{{DEVIATIONS}}
