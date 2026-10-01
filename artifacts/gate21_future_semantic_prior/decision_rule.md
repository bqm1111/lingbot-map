# Gate 21 decision rule — pre-registered

Written and hashed **before the first Gate 21 training run**. The hash is in `decision_rule.sha256`.
`tests/future_semantic_prior/test_gate21.py` recomputes it and fails if this file is edited afterwards, and
asserts that `tools/future_semantic_prior/config.Thresholds` carries exactly the numbers written here.

## 1. The question

Trained only on hidden voxels that a later frame happens to re-observe, does a small semantic-completion
head generalise to hidden voxels that are **never** re-observed?

Gate 20 established that 27.21 % of the correctly-completed-but-never-observed population becomes
camera-visible again inside the frozen 10 s privileged window, and that where it does, the future
observation is worth +10.56 mIoU over B0. Gate 21 asks whether that supervision teaches anything about the
other 72.79 %.

## 2. Populations (0006, from Gate 20's frozen per-anchor record, read not recomputed)

| name | definition | voxels |
|---|---|---|
| `E0` | GT-occupied ∧ frozen-predicted-occupied ∧ never causally observed (== the frozen region `completed_never_observed` == Gate 20's `P0`) | 2,743,212 |
| `E1` | `E0` ∧ Gate 20 marks the voxel future-visible | 876,254 |
| **`E2`** | **`E0` ∧ the voxel never becomes future-visible** | **1,866,958** |
| `E1_P1` | `E1` ∧ B0's class is wrong | 530,208 |
| `E2_P1` | `E2` ∧ B0's class is wrong | — |

`E1` and `E2` partition `E0`. Future visibility is Gate 20's `visible` level, unchanged and unrelaxed: a
future LiDAR return in the voxel that passes the frozen 4 px z-buffer test and carries a bilinear
Trident-H token. It is used **only** to split the evaluation population; the model never receives it, and
one forward over the whole grid produces both halves' predictions.

## 3. Primary metric

**Semantic mIoU on `E2`**, by the frozen metric (`semantic_metrics.summarize(conf, semantic_only=True)`:
classes 1..18 with GT support in the region, class 0 excluded), pooled over all 590 anchors of drive 0006,
averaged over the three head seeds.

Reference, read from Gate 20's `analysis.json` before this file was written:

```
B0 on E2 (never revealed)   10.180 mIoU   49.06 % top-1   1,866,958 voxels
B0 on E1 (revealed)          6.389 mIoU   39.49 % top-1     876,254 voxels
B0 on E0 (all of P0)         8.796 mIoU   46.01 % top-1   2,743,212 voxels
```

`E2` is where B0 is *strongest*, because the never-revealed voxels sit closer to causal evidence. A gain
there cannot be had by being handed an easier population.

## 4. Thresholds

Let `Δ_E2` = mean over seeds 0/1/2 of (model `E2` mIoU) − (B0 `E2` mIoU), in absolute percentage points.

### Outcome A — semantic completion prior demonstrated

All of:

- **A1** `Δ_E2 ≥ +1.00` pp
- **A2** every one of the three seeds has a positive `E2` delta
- **A3** the lower end of the 95 % block-bootstrap interval on `Δ_E2` is above `0.0`

### Outcome C — no transfer

Either of:

- **C1** `Δ_E2 ≤ +0.25` pp
- **C2** any seed has a negative `E2` delta while `Δ_E2 < +1.00` pp

### Outcome B — weak transfer

Anything else: `+0.25 < Δ_E2 < +1.00` with all seeds positive, or `Δ_E2 ≥ +1.00` with a bootstrap interval
that includes zero.

### Integrity (not an outcome)

If either of these fails the run is **void** and no outcome is declared:

- the frozen causal state on every 0006 anchor reproduces Gate 20's record exactly (occupancy count,
  evidence count, `O ∧ ¬E` on every population row, and B0's label voxel for voxel);
- every leakage check in `validate.py` passes.

## 5. Why +1.0 pp

- The brief proposes it, and it is the smallest gain worth developing into a paper mechanism.
- It is **~10 % relative** on B0's 10.18, which is the scale at which a semantic mechanism in this stack
  has never yet moved: Gate 15's B1 lost 2.68 pp on this population, Gate 16's B3 gained 0.58, B5 lost
  2.38. Nothing has gained a point.
- It is far outside head-seed noise. With geometry fixed at seed 0, B5's three head seeds spread 0.011 pp
  on this exact population (6.417 / 6.433 / 6.407); Gate 15's 0.633 pp spread came from varying the
  *geometry* seed, which Gate 21 does not do. +1.0 pp is therefore a very large multiple of the relevant
  σ, and A3's bootstrap interval — which covers anchor sampling, not seed noise — is the binding
  statistical test, not A1.

## 6. Bootstrap

Paired block bootstrap over the 590 dev anchors: 20-anchor contiguous temporal blocks, 2,000 resamples,
seed 0. Per resample the per-anchor confusions are **summed** and mIoU recomputed from the sum (mIoU is a
ratio of sums, so resampling cannot average per-anchor mIoUs). The statistic is the seed-mean model mIoU
minus B0's mIoU on the same resample.

## 7. Fixed training protocol

Fixed here so it cannot be tuned against the result. **No learning-rate sweep will be run** and no
architecture search will be performed.

- supervision: `U^R` = Gate 20's `P0` ∧ future-visible, on drives 0003/0007/0010 only; 1,274 anchors with
  a non-empty `U^R`; target = Gate 20's `mean_probability` vector, soft, never hardened
- loss: `-Σ_c q_c log softmax(logits)_c`, averaged over supervised voxels
- optimiser: AdamW, lr 1e-3, weight decay 0.01, cosine schedule, grad clip 1.0 (the Gate 15 / B5 / pilot
  convention, unchanged)
- 3,000 steps (the pilot's schedule), 2 full-grid anchors per step, no crop
- seeds 0, 1, 2; geometry seed fixed at 0 for every run so B0 cannot move
- the reported checkpoint is the **final step**. No checkpoint is selected on a dev metric.
- controls: `C1_prior` (18 params, input-independent), `C2_geometry` (29 channels, no semantic evidence),
  `C3_full` (63 channels, the primary model), `C4_semantics` (35 channels, evidence + occupancy only)

## 8. Secondary, reported but not decisive

`E1`, `E0`, `E1_P1`, `E2_P1`; per-class IoU on `E2`; thing/stuff on `E2`; the five distance-to-evidence
bins on `E2`; the selection-bias comparison between `U^R_train` and `U^N_dev`; the composite SSC read over
all labelled voxels (a change worse than −0.5 pp is flagged in the report, and decides nothing).

## 9. Deliberately not consulted before this file was hashed

- any `E1`, `E2` or `E0` metric from a Gate 21 head trained for its full schedule, for any control or seed;
- the selection-bias statistics of §8;
- the target class distribution of `U^R_train`.

**One disclosure.** A 60-step throughput probe of `C3_full` seed 0 — run to measure seconds per step
before committing to the 3,000-step budget — printed its dev subsample row, and it was seen:
`E1 4.00, E2 4.27` on the 50-anchor stride-12 subsample, against `B0 E2 11.94` on the same subsample. That
is a head 2 % of the way through its schedule, scoring less than half of B0; it carries no information
about where a trained head lands, and the +1.0 pp bar comes from the brief, from B0's pooled 10.18 and
from the prior gates' seed spreads, all of which are §3 and §5 above. It is recorded here rather than left
out, and this file was re-hashed after adding it. No other trained-head number has been seen.

B0's three mIoU values in §3 and the prior gates' seed spreads in §5 **were** consulted: the brief directs
the threshold to be set from "B0 E2 performance" and "existing seed variance", which cannot be done
without reading them. They are properties of the frozen baseline, not of any Gate 21 model.
