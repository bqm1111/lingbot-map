# B5 semantic residual: SOURCE FAIL

**The correction makes semantics worse, not better.** On the pre-registered decision population it loses
**2.04 mIoU** against B0 (seeds -1.96, -2.11, -2.04), with a paired 95% CI of [-3.31, -1.52] that excludes zero on the wrong side.
The rule required at least +1.0. Nothing about this is marginal.

Everything ran: supervision was valid, all reliability gates passed, three seeds trained to budget and agree with
each other. This is a model result, not an infrastructure failure.

## 1. What B5 changed relative to the actual B0 / B4 code

    corrected = b0_feature + valid * residual_head(context)
    prediction = normalize(corrected) @ text.T        # frozen readout, unchanged

B4 (`tools/semantic_field/model/semantic_field_head.py`) has **no** such skip -- its docstring calls its output
"a free vector, not a combination of input features", its `semantic_head` is a default-initialised `nn.Linear`,
and its `ResidualBlock`s are internal convolutional residuals. B0's feature never reaches a query voxel there.
B5 adds the missing path and zero-initialises only the output projection, so it starts exactly at B0
(max deviation at initialisation **1.5e-8**). 226,752 parameters. Geometry, occupancy, teacher, vocabulary,
centering and readout are untouched.

## 2. Was the supervision valid?

Yes, and this is the part that worked.

| gate | requirement | measured | verdict |
|---|---|---|---|
| G1 registration, every offset +1..+5, every drive incl. dev | median <= 0.10 m, p90 <= 0.20 m | max median 0.066, max p90 0.156 | **pass** |
| G2 coverage | mean \|D\| >= 2000/anchor; >= 8 classes >= 10k | 3391; 14 classes | **pass** |
| G3 repeatability | train >= 0.65 (+0.10 margin), dev >= 0.60 | train 0.749 (+0.365), dev 0.738 (+0.408) | **pass** |

**Horizon, and what it excludes.** One stream frame is **0.520 s**, not 0.1 s -- the stream subsamples KITTI-360's
10 Hz by 5x. Measured twice, independently: modal gap in the frozen caches' `first_dt_s`, and directly from
KITTI-360 timestamps once the data was reachable (+1 = 0.522 s, +5 = 2.61 s, native step exactly 5).

The brief named "+1 and +5". Taken as exactly those two frames the pilot **fails its own coverage requirement**
(mean \|D\| 1,370 vs a threshold of 2,000), which would have forced DATA BLOCKED with nothing trained. Taken as the
window they bound, \|D\| is 3,391 and G2 passes. Both readings were measured before choosing. The brief also
requires the thresholds to hold "at each included offset", so +2/+3/+4 were measured rather than assumed monotone;
all pass on all four drives. The +1 and +5 values reproduce the frozen pilot record to four decimals.

**+10 and longer remain excluded** and nothing here speaks to them: per-drive p90 at +10 is 0.2072 m (0010) and
0.2174 m (0006), over threshold, while the training-pooled 0.184 m passes and hides it. This pilot neither repairs
nor supersedes the BLOCKED B4 scaling study, and claims nothing about reveal delays beyond 2.61 s.

## 3. B0 versus B5

594 development anchors on drive 0006. Ground truth is the held-out source semantics, opened only after each
prediction was made. mIoU, unweighted over classes present in the region.

| region | B0 | B5 (seed mean) | delta |
|---|---:|---:|---:|
| observed_current | 15.22 | 15.22 | +0.00 |
| observed_past_only | 17.37 | 17.37 | +0.00 |
| never_observed | 8.38 | 6.32 | -2.06 |
| completed_never_observed | 8.79 | 6.43 | -2.37 |
| completed_geo_observed | 9.70 | 8.99 | -0.71 |
| completed_geo_unobserved | 7.74 | 5.76 | -1.98 |
| completed_later_revealed | 8.35 | 6.05 | -2.30 |
| completed_revealed_unseen **<- decision** | 7.25 | 5.21 | -2.04 |
| completed_not_revealed | 8.78 | 6.46 | -2.32 |
| ssc_full | 6.42 | 6.13 | -0.29 |
| ssc_never_observed | 6.29 | 5.74 | -0.54 |

Per seed on the decision region: seed0 -1.96 pp (CI [-3.24, -1.47]), seed1 -2.11 pp (CI [-3.40, -1.59]), seed2 -2.04 pp (CI [-3.31, -1.52]).

**The regression is uniform.** Every completion region loses ~2 mIoU. The two evidence regions are identical to
the last digit (+0.00), which is the shared evidence path doing its job; occupancy is identical by construction.
So B5 changes exactly what it was designed to change, and changes it for the worse.

### Teacher agreement versus being right

This is the substantive finding, and it is why the brief was right to forbid deciding on cosine.

| measured on | quantity | B0 | B5 | change |
|---|---|---:|---:|---|
| dev supervised voxels (116125) | cosine to future teacher | 0.8784 | 0.9299 | **+0.0515** |
| dev supervised voxels | top-1 agreement with teacher | 0.3793 | 0.3948 | +0.0155 |
| decision region, vs GT | **semantic mIoU** | 7.25 | 5.21 | **-2.04** |
| decision region, vs GT | accuracy | 49.87 | 36.68 | **-13.19** |

B5 did exactly what it was trained to do -- it moved features measurably closer to the future teacher, and that
generalised to the development drive. It still got worse against ground truth. Training cosine rose from
0.8882 to 0.9530 (all three seeds within 0.002 of each other).

**Mechanism: class collapse.** On the decision region B0 predicts 18 of 19 classes; B5 predicts 13. Its top-5
share rises from 0.916 to 0.975. Per-class IoU shows the trade: class 11 collapses 37.34 -> 7.92 (36,465 GT
voxels), classes 9 and 14 go to zero, class 17 falls 5.32 -> 1.85, while class 7 gains +7.99 and the dominant
class 13 is flat. Regressing to a mean embedding under a cosine loss maximises expected similarity by moving
toward the centre of the teacher distribution, which is near the dominant class -- so mean cosine improves while
the decision boundaries that separate rarer classes are destroyed. Accuracy falls too, so this is not an artefact
of mIoU's unweighted averaging.

## 4. Decision

**SOURCE FAIL.** Against the rule recorded in `config.DECISION_RULE` before training:

| requirement | required | measured | met |
|---|---|---|---|
| mean mIoU gain on the decision population | >= +1.0 pp | **-2.04 pp** | no |
| all three seeds positive | yes | no (-1.96, -2.11, -2.04) | no |
| paired 95% CI lower bound | > 0 | -3.31 | no |
| candidate-supported change | >= -0.5 pp | -0.71 pp | no |
| full-source SSC change | >= -0.5 pp | -0.29 pp | yes |
| direct evidence preserved | identical | identical | yes |
| occupancy identical | identical | identical | yes |

What improved: feature cosine to the teacher, and top-1 agreement with the teacher, both on held-out data.
What regressed: every semantic region against real labels, and accuracy by 13 points.

Per the brief, B5 stops here. No B6, no attention variant, no data scaling and no target evaluation were started.

## 5. Cost and state

| item | used | cap |
|---|---|---|
| GPU-hours | ~0.9 (targets 0.27, query cache 0.26, training 0.25, evaluation 0.09, checks 0.05) | 12 |
| additional storage | 6.8 GB | 50 GB |

**Nothing is running.** Three final checkpoints at step 6000 are on disk.

**Smallest next action:** none required. If the question is pursued, the finding says the cosine-to-teacher
objective is the thing to change, not the skip connection -- the skip works exactly as designed and the model
optimises the stated objective successfully. An objective that preserves class separability (for example a
readout-space loss rather than a feature-space one) would be a different experiment and would need its own
declaration.

## 6. Code

| file | what |
|---|---|
| `tools/semantic_residual/config.py` | every fixed choice, the horizon justification, the decision rule |
| `tools/semantic_residual/model.py` | the residual head; B0 skip, zero-init output, `valid` gate |
| `tools/semantic_residual/targets.py` | feature-valued supervision; verifies its teacher against the frozen cosines |
| `tools/semantic_residual/prepare.py` | query cache; causal query set defined before targets are read |
| `tools/semantic_residual/data.py` | bounded loader, one small file pair per anchor |
| `tools/semantic_residual/train.py` | cosine loss, resumable, atomic writes |
| `tools/semantic_residual/evaluate.py` | mirrors the pilot evaluator, B5 beside B0 |
| `tools/semantic_residual/analysis.py` | mIoU, paired block bootstrap, the rule |
| `tests/semantic_residual/test_model.py` | 9 checks (B0 equivalence, zero-init, gating, gradients, errors) |
| `artifacts/b5_semantic_residual/method.md` | input tensors, frames, masks, one worked sample |

Reproduce: `run_targets.sh` -> `run_prepare.sh` -> `run_train.sh` -> `run_eval.sh` -> `python -m tools.semantic_residual.analysis`.

## 7. Corrections made during this run

- My first evaluator scored teacher agreement over **all** revealed voxels. B5 only writes labels on the query
  set, so voxels outside it were compared as blank for B5 against a real label for B0, which made B5 look far
  worse on that metric (0.45 -> 0.27). Restricted to voxels both methods label, it is 0.4500 -> 0.3890. Region mIoU
  was never affected -- every scored region is a subset of the query set -- but the number was wrong and the
  evaluation was re-run after fixing it.
- Ten stale symlinks under `/media/SSD1/MINH_DATASETS` pointed at an unmounted disk; the KITTI-360 root and the
  development drive's LingBot cache were among them. Repointed to the current mount; no data was moved or deleted.
