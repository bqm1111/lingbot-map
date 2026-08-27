# Gate 1 — SemanticKITTI reproducibility

## Result: **FAIL_REPRODUCIBILITY**

`sem_bypass` retains **76.2 %** of the direct MaskCLIP teacher (gate: ≥ 90 %), sits
**2.50 mIoU** below it (gate: ≤ 1.5), and varies by **σ = 0.553** across three seeds
(gate: ≤ 0.5). Three of the four criteria fail. Only the coverage criterion passes.

Per the protocol this stops the study. Gates 2 and 3 were not run.

## 1. What was run

| | |
|---|---|
| representation | pre-GCT `aggregator.patch_embed → x_norm_patchtokens`, 1024-ch |
| sidecar | unmodified `SemanticSidecar`, **6 156 864** trainable params (**0.529 %** of LingBot's 1 157 943 540) |
| LingBot | frozen, 0 trainable; geometry **bit-exact** vs the legacy cache (§6) |
| teacher | frozen MaskCLIP `ViT-B-16-quickgelu/openai`, 512-d → 64-d orthonormal PCA |
| teacher frames | 25 %, frame-level uniform draw, seed varied per run |
| training data | KITTI odometry 13/14/15/16, **2 982 frames, no labels exist for these sequences** |
| evaluation | SemanticKITTI seq 08, 1 000 frames, 200 scored, 19 classes, labels used **only** here |
| loss | pixel-only feature distillation (`lambda_pixel = 1.0`); no consensus in the primary |
| steps | 20 000 per run; 105–191 s per run; peak VRAM 0.26 GB |
| seeds | 0, 1, 2 — each varies `seed`, `train.seed` **and** `teacher_density_seed` together |

## 2. Headline table

All rows share coverage **87.004604 %** to 6 decimals — identical support, so no model
gains from covering more points.

| model | teacher at inference | matched mIoU (3 seeds) | mean | σ | retention | gap |
|---|---|---|---:|---:|---:|---:|
| `teacher_direct` (ceiling) | **yes** | — | **10.5027** | — | 1.000 | — |
| **`sem_bypass`** (primary) | no | 8.7735 / 7.5195 / 7.7050 | **7.9994** | 0.553 | **0.762** | 2.503 |
| `sem_bypass_consensus` (secondary) | no | 9.9883 / 8.3084 / 8.6702 | **8.9890** | 0.722 | 0.856 | 1.514 |

### Gate arithmetic

| criterion | required | observed | verdict |
|---|---|---:|:--:|
| retains ≥ 90 % of direct MaskCLIP | ≥ 0.90 | **0.762** | ✗ |
| within 1.5 absolute mIoU | ≤ 1.5 | **2.503** | ✗ |
| seed σ ≤ 0.5 mIoU | ≤ 0.5 | **0.553** | ✗ |
| no coverage advantage | identical | identical (87.004604) | ✓ |
| three seeds present | 3 | 3 | ✓ |

`sem_bypass_consensus` would also fail (retention 0.856), and it is a secondary
diagnostic that the protocol forbids promoting into the method on a failed primary.

## 3. Why it failed — the representation is **not** the cause

Two diagnostics were run to separate "pre-GCT tokens are bad" from "the 3-seed protocol
is stricter than the prior single-seed evidence".

### 3.1 Matched-seed control on the audited representation

Identical protocol, identical seeds, identical teacher / PCA / tracks / evaluation. The
**only** change is the representation, reverted to the audited mid+final GCT blocks
`[11, 23]`, read from the existing legacy cache (no re-inference, prior artifacts
untouched):

| representation | seed 0 | seed 1 | seed 2 | mean | σ | retention |
|---|---:|---:|---:|---:|---:|---:|
| pre-GCT encoder (SemBypass) | 8.7735 | 7.5195 | 7.7050 | **7.9994** | 0.553 | 0.762 |
| `[11, 23]` mid+final GCT | 8.7636 | 6.8439 | 7.7464 | **7.7847** | 0.784 | 0.741 |
| delta (GCT − pre-GCT) | −0.010 | −0.676 | +0.041 | **−0.215** | | |

**Both representations fail, and pre-GCT is if anything marginally better** (+0.215 mean,
well inside seed noise). The failure is not a property of the encoder tokens. The
project's premise — that pre-GCT tokens carry the semantics — is *not* refuted here; it
is simply not enough to clear the gate, and neither is the alternative.

### 3.2 The prior 9.02 was a favourable teacher draw

Exact replay of the published run — `[11, 23]`, `seed=0`, `train.seed=0`,
`teacher_density_seed=1234`:

| | matched mIoU | coverage |
|---|---:|---:|
| published `mlp_pixel_d025` | 9.0239 | 87.004604 |
| **replay here** | **9.0683** | 87.004604 |
| delta | **+0.044** | 0 |

**The prior implementation reproduces essentially exactly at its own seed.** Nothing was
fabricated or mis-recorded. But `teacher_density_seed = 1234` is a *lucky draw*: three
independent draws of the same 25 % budget give 6.8439 / 7.7464 / 8.7636, and 9.07 lies
**above the maximum** of that range.

So the single number the project was asked to reproduce is real but unrepresentative.
Under the three-seed protocol this project itself mandates — and which the audit flagged
as necessary — the method lands ~1.3 mIoU lower than the published figure suggests.

**This is the substantive finding of Gate 1:** the headline was not wrong, it was
under-sampled. The 25 % teacher-frame draw contributes roughly ±1 mIoU of variance,
which is larger than every architectural effect measured in the previous study.

## 4. Consensus stability (secondary, reported separately)

The audit noted the prior report called the consensus advantage "unstable". With three
seeds it is not:

| seed | `sem_bypass` | `sem_bypass_consensus` | delta |
|---|---:|---:|---:|
| 0 | 8.7735 | 9.9883 | **+1.215** |
| 1 | 7.5195 | 8.3084 | **+0.789** |
| 2 | 7.7050 | 8.6702 | **+0.965** |
| mean | 7.9994 | 8.9890 | **+0.990** (σ 0.175) |

Consensus is **positive on all three seeds with low variance** — it would satisfy the
adoption rule the protocol sets. It is nonetheless *not* adopted, because the primary
gate failed and the protocol forbids using a secondary diagnostic to rescue it. Recorded
as a genuine, reproducible +0.99 mIoU effect for whoever picks this up.

Note the direction contradicts the earlier study's conclusion that consensus loses at
high teacher density and wins only at low density; at 25 % on pre-GCT tokens it wins
consistently.

## 5. Per-class and rare-class analysis

Matched IoU, `teacher_direct` vs `sem_bypass` (seed 0), classes where the teacher scores > 1:

| class | teacher | sem_bypass |
|---|---:|---:|
| **car** | 54.67 | **61.34** |
| **road** | 37.55 | **44.14** |
| building | 27.88 | 11.45 |
| vegetation | 32.69 | 18.69 |
| other-vehicle | 2.54 | 4.46 |
| parking | 4.53 | **0.00** |
| trunk | 3.35 | **0.00** |
| pole | 1.99 | **0.00** |
| fence | 1.18 | 0.07 |
| other-ground | 1.12 | **0.00** |

| | dominant-class mean IoU | rare-class mean IoU | rare classes with IoU > 0.1 |
|---|---:|---:|---:|
| `teacher_direct` | 30.62 | 1.07 | — |
| `sem_bypass` (3-seed mean) | 24.23 | **0.49** | **3.3 / 14** |
| `sem_bypass_consensus` | 28.00 | **0.27** | **2.7 / 14** |

The sidecar *beats* the teacher on the two dominant classes (car +6.7, road +6.6) and
collapses to exactly 0.00 on parking, trunk, pole and other-ground. Its mean is carried
almost entirely by car / road / building / vegetation. Consensus raises the dominant-class
mean (24.2 → 28.0) while **lowering** the rare-class mean (0.49 → 0.27) — it buys mIoU by
sharpening the head classes, which is worth knowing before anyone adopts it.

Absolute levels are low for every model including the teacher (10.50), as the previous
report also observed; differences of ±0.5 mIoU between near-floor models should not be
over-read. That is a further reason the three-seed spread matters.

## 6. Geometry invariance

The decisive check: the SemBypass token cache was produced by an **independent inference
run** with a different forward hook, and its geometry is compared tensor-by-tensor
against the legacy `[11, 23]` cache for the same scene.

`depth`, `depth_conf`, `extrinsic`, `intrinsic`, `pose_enc`, `frame_index` and
`image_patch_rgb` are **bit-identical** (`torch.equal`), asserted in
`tests/test_sem_bypass.py::test_geometry_is_bit_exact_against_the_existing_gct_cache`.
Registering the sidecar hook cannot move geometry: the hook returns `None`.

Independently, `teacher_direct` re-evaluated in this package returns **10.5027** with
coverage **87.004604** — identical to the previous study's stored value — confirming the
evaluation support is the same one the audited numbers used.

## 7. Anomalies

1. **`sem_bypass` beats the teacher on `car` and `road`** (61.3 vs 54.7, 44.1 vs 37.6). A
   distilled student exceeding its teacher on the two easiest, most frequent classes is
   consistent with the student regressing toward dominant modes; it is not evidence of
   superior semantics, and the rare-class collapse in §5 is the other side of it.
2. **Seed 0 is the best seed for every configuration** (8.77 / 9.99 / 8.76). With three
   seeds this cannot be separated from chance, but any future work should not report
   seed 0 alone.
3. **Consensus reverses sign versus the previous study** (§4). Different representation,
   different density, three seeds instead of one — not diagnosed further.

## 8. Exact commands

```bash
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH=$PWD CUDA_HOME=/usr/local/cuda-12.8 FLASHINFER_CUDA_ARCH_LIST="12.0a"
PY=/home/minh/anaconda3/envs/cu128/bin/python

$PY -m pytest research/sem_bypass/tests -q                      # 16 passed

# pre-GCT encoder cache (3 982 frames, 49.5 ms/frame, 4.3 GB)
$PY -m research.sem_bypass.cache_encoder \
    --config research/sem_bypass/configs/gate1.yaml --roles train eval --device cuda:0

# Gate 1: 3 seeds x 2 variants  (909 s total)
$PY -m research.sem_bypass.run_gate1 --seeds 0 1 2 \
    --variants sem_bypass sem_bypass_consensus \
    --stages tracks train teacher infer eval --device cuda:0 --overwrite

# diagnostic control: same seeds, representation reverted to [11, 23]  (476 s)
$PY -m research.sem_bypass.run_gate1 --config research/sem_bypass/configs/control_gct.yaml \
    --seeds 0 1 2 --variants sem_bypass \
    --stages tracks train teacher infer eval --device cuda:0 --overwrite

# exact replay of the published single-seed run (teacher_density_seed=1234)
$PY scratchpad/replay.py

$PY -m research.sem_bypass.report_gate1
```

Artifacts: `outputs/gate1/metrics.json`, `outputs/gate1/per_class.csv`,
`outputs/maps/*/eval.json`, `outputs/control_gct/maps/*/eval.json`,
logs `outputs/gate1_full.log`, `outputs/control_gct.log`, `outputs/replay_prior.log`.

## 9. Verdict

**`FAIL_REPRODUCIBILITY`.** Stop. Gates 2 and 3 were not run, and no later phase was
built to rescue the result.
