# SemBypass — a lightweight MaskCLIP semantic bypass for frozen LingBot-Map

**Status: `FAIL_REPRODUCIBILITY` at Gate 1.** Gates 2–3 and Phase 4 were not run, per the
stop-at-first-failure protocol. See [`reports/final_decision.md`](reports/final_decision.md).

## Idea

```python
geometry, encoder_tokens = frozen_lingbot(images)   # pre-GCT x_norm_patchtokens, 1024-d
semantic_features        = sidecar(encoder_tokens)  # 64-d orthonormal PCA of MaskCLIP-512
```

Decode already-computed pre-GCT encoder tokens into MaskCLIP's language-aligned space, so
no semantic encoder runs at inference and LingBot geometry is untouched.

## Result

| model | matched mIoU (seeds 0/1/2) | mean | σ | retention |
|---|---:|---:|---:|---:|
| `maskclip_direct` (ceiling) | — | **10.5027** | — | 1.000 |
| **`sem_bypass`** | 8.7735 / 7.5195 / 7.7050 | **7.9994** | 0.553 | **0.762** |
| `sem_bypass_consensus` | 9.9883 / 8.3084 / 8.6702 | 8.9890 | 0.722 | 0.856 |
| control `[11, 23]` GCT | 8.7636 / 6.8439 / 7.7464 | 7.7847 | 0.784 | 0.741 |

Gate 1 required ≥ 0.90 retention, ≤ 1.5 absolute gap and σ ≤ 0.5. Three of four failed.

**What the diagnostics showed.** The pre-GCT representation is *not* the problem — with
matched seeds the audited `[11, 23]` tokens do slightly worse. The prior published 9.02
reproduces exactly at its own teacher-draw seed (replay: 9.0683) but lies *above the
maximum* of three independent draws. The 25 % teacher-frame draw contributes ~±1 mIoU,
larger than any architectural effect in the earlier study.

**What is supported:** sub-1 % parameters (6 156 864 = 0.529 %), bit-exact geometry
preservation, no labels in training, no semantic encoder at inference.
**What is not:** recovery of MaskCLIP-quality open-vocabulary features, and transfer
(never tested — nuScenes semantic GT is absent, see audit §14).

## Layout

| path | role |
|---|---|
| `reports/phase0_audit.md` | audit of the prior implementation; where 10.04 really came from |
| `reports/gate1_reproducibility.md` | Gate 1 protocol, results, diagnostics, verdict |
| `reports/final_decision.md` | final decision and paper-claim assessment |
| `configs/gate1.yaml` | primary: pre-GCT tokens, 25 % teacher, pixel-only loss |
| `configs/control_gct.yaml` | diagnostic: same seeds, representation reverted to `[11, 23]` |
| `encoder_features.py` | frozen LingBot restricted to pre-GCT `patch_embed` tokens |
| `cache_encoder.py` | writes the pre-GCT feature cache in the existing shard format |
| `model.py` | `LingBotSemBypass` wrapper + `build_sidecar` (6 156 864 params) |
| `_tool_shim.py` | runs the existing `tools/` scripts against a 1024-ch cache, unmodified |
| `run_gate1.py` | tracks → train → infer → eval, per seed and variant |
| `report_gate1.py` | `outputs/gate1/metrics.json` + `per_class.csv` + gate decision |
| `benchmark.py` | Gate 2 latency harness (implemented, **never run**) |
| `tests/` | 16 tests: freezing, shapes, normalisation, geometry invariance, determinism |

Nothing under `semantic_sidecar/`, `tools/`, `semantic/` or `output/` was modified; the
previous study's artifacts are intact and its scripts still reproduce its numbers.

## Commands

```bash
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH=$PWD CUDA_HOME=/usr/local/cuda-12.8 FLASHINFER_CUDA_ARCH_LIST="12.0a"
PY=/home/minh/anaconda3/envs/cu128/bin/python

$PY -m pytest research/sem_bypass/tests -q

# 1. pre-GCT encoder cache (3 982 frames, 49.5 ms/frame, 4.3 GB)
$PY -m research.sem_bypass.cache_encoder \
    --config research/sem_bypass/configs/gate1.yaml --roles train eval --device cuda:0

# 2. smoke test (one seed, 300 steps)
$PY -m research.sem_bypass.run_gate1 --seeds 0 --variants sem_bypass \
    --stages tracks train --steps 300 --device cuda:0 --overwrite

# 3. Gate 1 (3 seeds x 2 variants, 909 s)
$PY -m research.sem_bypass.run_gate1 --seeds 0 1 2 \
    --variants sem_bypass sem_bypass_consensus \
    --stages tracks train teacher infer eval --device cuda:0 --overwrite

# 4. representation control (476 s)
$PY -m research.sem_bypass.run_gate1 --config research/sem_bypass/configs/control_gct.yaml \
    --seeds 0 1 2 --variants sem_bypass \
    --stages tracks train teacher infer eval --device cuda:0 --overwrite

# 5. collect
$PY -m research.sem_bypass.report_gate1
```

The teacher feature cache and PCA basis are **symlinked** into
`outputs/cache_encoder/` from the previous study, so they are reused byte-for-byte
rather than recomputed.

## Next action

Re-run the previous study's key contrasts at **5+ seeds** before drawing any
architectural conclusion. Every single-seed contrast of 1–2 mIoU in that study sits inside
the ±1 mIoU teacher-draw noise measured here. Do not enlarge the sidecar, add losses, or
acquire Gate 3 data until a configuration clears 90 % retention with σ well under 0.5.
