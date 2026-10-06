# Gate 17 — instance grounding: OUTCOME C (insufficient headroom)

**Perfect instance association does not fix semantic propagation.** With ground-truth grouping between visible evidence and completed unobserved voxels, and semantics still taken from the frozen teacher, mIoU on correctly completed voxels moves from **8.80 to 8.88** (+0.09, paired 95% CI [-0.27, +0.34] — includes zero). On the deployable SSC read it moves -0.04. **Do not build the instance transformer on this evidence.**

The oracle is wired correctly — destroying the grouping costs 2.80 mIoU (CI [-3.28, -2.42]), so the association carries real signal. It simply does not carry signal that B0 was missing.

## A. What was reused

The causal 5-frame state, hard map, frozen CompletionUNet occupancy (seed 0), Trident-H evidence, B0's nearest-evidence rule, `regions.build_regions`, `semantic_metrics.summarize`, the Gate 15 eval masks and the predict/eval file firewalls — all unchanged. Only the oracle grouping is new. See `plan.md`.

## B. What is oracle and what stays deployable

**Ground truth supplies only *which voxels belong together*.** For every method except the `C2` ceiling, the class still comes from the frozen teacher read through the frozen text embeddings.

- `C0_b0` — none -- the deployed baseline
- `A_oracle_instance` — grouping only; the label is the teacher's, aggregated over the instance's own visible evidence
- `C1_shuffled_instance` — grouping only, then aggregates permuted between instances (control)
- `C3_visible_majority` — grouping only; majority teacher class over the instance's visible evidence
- `C2_gt_class` — FULL semantic oracle -- the ceiling, never a method

**KITTI-360 instance annotations are not on this machine.** The tree holds only `data_2d_raw` and `data_3d_raw`; the SSCBench voxel labels are `float32` semantic-only with no instance channel; the sole instance annotations anywhere belong to nuScenes, which is firewalled out of source work. So instances are **6-connected components of same-class GT-occupied voxels** — the strongest legitimate substitute, and an *upper bound*: it knows GT class boundaries exactly, and adjacent same-class objects merge. A low ceiling under an optimistic oracle is decisive; a high one would have needed re-checking against real annotations before building anything.

## C. Baseline reproduction (gate passed)

B0 reproduces the frozen pilot record exactly on all eight regions that do not depend on the revealed set: observed_current 15.21, observed_past_only 17.38, never_observed 8.39, completed_never_observed 8.80, completed_geo_observed 9.72, completed_geo_unobserved 7.73, ssc_full 1.55, ssc_never_observed 1.39.

Getting there found two bugs in my own Gate B5 analysis, both fixed: a reimplemented mIoU that included the empty class (inflating SSC regions ~4x), and a shard glob that merged a leftover smoke-test file (4 duplicated anchors). B5's conclusion is unchanged.

## D. Oracle results

590 anchors, drive 0006. mIoU by the repository's frozen definition (classes 1..18, empty excluded).

| method | P1 deployable SSC | P2 correct completion | P3 oracle geometry |
|---|---:|---:|---:|
| C0_b0 | 1.39 | 8.80 | 8.39 |
| A_oracle_instance | 1.35 | 8.88 | 8.59 |
| C1_shuffled_instance | 0.96 | 5.99 | 6.42 |
| C3_visible_majority | 1.34 | 8.81 | 8.54 |
| C2_gt_class | 38.13 | 100.00 | 100.00 |

Populations: P1 553,275,690 voxels, P2 2,743,212, P3 7,834,276. Instance coverage of P2 is **36.7%**.

`C2_gt_class` reads 100.00 on P2 and P3 **by construction** — those populations are GT-occupied, so a GT label is perfect. It is not an informative ceiling there. The informative one is P1 = 38.13: even perfect semantics on the current predicted occupancy reaches only that, because occupancy errors cap the deployable number.

Accuracy tells the same story as mIoU on P2: B0 46.01, oracle instance 47.33, shuffled 36.15. The oracle buys about **1.3 points of accuracy and 0.09 of mIoU**.

### Per-class on P2 (B0 → oracle instance)

| class | B0 | oracle | delta |
|---|---:|---:|---:|
| car | 9.06 | 9.28 | +0.22 |
| truck | 7.96 | 7.45 | -0.51 |
| person | 0.82 | 1.10 | +0.27 |
| bicycle | 0.20 | 0.00 | -0.20 |
| building | 35.05 | 35.32 | +0.27 |
| road | 17.52 | 16.47 | -1.05 |
| sidewalk | 8.33 | 8.41 | +0.08 |
| vegetation | 52.77 | 54.89 | +2.12 |
| terrain | 0.26 | 0.26 | -0.00 |
| fence | 4.50 | 4.54 | +0.04 |
| pole | 0.03 | 0.03 | +0.00 |

## E. Why it does not help — geometry vs semantics vs association

Three measurements, in the order that matters:

**1. The association is mostly unavailable.** Only **5.5% of instances per anchor have any visible evidence** (121 of 2199), and only **36.7% of P2 voxels** belong to an instance the model saw at all. For the other two thirds the oracle has nothing to offer and falls back to B0. You cannot ground to an object you never saw.

**2. Where it IS available, it changes the answer but not the quality.** I expected B0's nearest evidence to already come from the same instance, which would have made the oracle a no-op. It does not: among P2 voxels whose instance has evidence, B0's nearest evidence is in the same instance only **24.1%** of the time (**8.6%** across all of P2). So the oracle really does relabel most covered voxels from a different, instance-correct source — and mIoU still barely moves. The label it hands over is no better than the one proximity was handing over.

**3. The teacher is the binding constraint.** Perfect grouping inherits the teacher's read of the instance's visible part, and that read is right about **47%** of the time. That ceiling, not the association, is what caps the result — the same teacher-capped conclusion the frozen zero-shot evaluation reached.

**Geometry's share, separately:** perfect semantics on the current occupancy gives P1 38.13 against B0's 1.39 (paired +36.74, CI [+34.20, +39.55]). So the deployable gap is bounded by geometry and dominated by semantic quality; instance association is not where either bottleneck lives.

## F. Distance from visible evidence (P2 mIoU)

The hypothesis was that instance grounding would pull ahead exactly where Euclidean propagation degrades. It does not.

| distance | C0 | A | C1 | C3 | C2 |
|---|---:|---:|---:|---:|---:|
| 0-1 m | 12.39 | 12.55 | 6.57 | 12.29 | 100.00 |
| 1-2 m | 9.82 | 9.86 | 7.39 | 9.81 | 100.00 |
| 2-4 m | 7.72 | 7.81 | 6.72 | 7.77 | 100.00 |
| 4-8 m | 4.23 | 4.28 | 4.00 | 4.27 | 100.00 |
| >8 m | 3.07 | 3.16 | 3.05 | 3.16 | 100.00 |

B0 and the oracle track each other within ~0.2 mIoU at every distance; both decay from ~12 to ~3 as distance grows. The advantage does not appear where proximity fails — it is not there.

## G. Instance visibility — the one real positive

| visible fraction | C0 | A | C1 | C3 | C2 |
|---|---:|---:|---:|---:|---:|
| 0-20% | 8.33 | 8.24 | 6.16 | 8.17 | 100.00 |
| 20-40% | 15.05 | 17.44 | 4.92 | 16.90 | 100.00 |
| 40-60% | 16.09 | 19.53 | 4.76 | 19.55 | 100.00 |
| 60-80% | 14.41 | 17.04 | 4.93 | 17.66 | 100.00 |
| 80-100% | 18.44 | 21.50 | 5.08 | 23.40 | 100.00 |

**When an object is actually well observed, instance grounding does work**: +2.4 at 20-40% visible, +3.4 at 40-60%, +3.1 at 80-100%. But the 0-20% bin dominates the population and shows -0.09. The mechanism is real; the data is not in the regime where it pays.

### thing vs stuff

| split | voxels | B0 | oracle | delta |
|---|---:|---:|---:|---:|
| thing | 41,560 | 17.45 | 16.11 | -1.34 |
| stuff | 2,701,652 | 11.40 | 11.77 | +0.36 |

On **things** — the classes where an instance is genuinely an object — the oracle is **worse** than B0 (16.11 vs 17.45). Things are 41,560 of 2,701,652 P2 voxels. The small positive on stuff comes from large connected regions (vegetation +2.12, building +0.27), which is region smoothing, not object grounding.

## H. Qualitative examples

`qualitative.png`. Selection rule fixed in advance: anchors at quantiles [0.05, 0.35, 0.65, 0.95] of per-anchor (oracle − B0) mIoU on P2, so the set spans anchors where the oracle hurts as well as helps: t=00661 -2.66, t=00043 -0.13, t=00604 +0.83, t=00838 +3.82. The coverage panel is the one to look at — grey is every completed voxel the oracle could not reach.

## I. Decision

**OUTCOME C — instance grounding has insufficient headroom.** The brief's own reading applies: this is a valid and useful negative result. Do not build the persistent-instance transformer. Per the brief no next architecture is proposed, because a design document is warranted only under Outcome A or B.

Scope: source domain only, KITTI-360 drive 0006, connected-component instances. No target dataset was opened.

## J. Next experiment (one, smallest)

**Measure the teacher's ceiling on visible evidence directly, and nothing else.** Every completion result so far — B1, B4, B5 and now the instance oracle — lands on the same wall: whatever you propagate, you propagate a teacher label that is right about half the time. Gate 17 shows association is not the missing piece; B5 showed the feature-space objective is not either. The cheapest decisive next measurement: on observed evidence voxels that have GT labels, what is frozen Trident-H's per-class accuracy, and how much of the completed-region error is inherited from it rather than created by propagation? That is a scoring exercise on caches that already exist, needs no training, and decides whether any propagation mechanism can pay before another one is built.

## K. Reproduce

```bash
bash artifacts/gate17_instance_oracle/run_eval.sh
python -m tools.instance_oracle.analysis
python -m tools.instance_oracle.visualize --device cuda:0
```

Code: `tools/instance_oracle/{config,oracle,evaluate,analysis,visualize,report}.py`. Commit in `git_commit.txt`. Deterministic (seed 0), resumable per shard, no frozen output modified. Cost: ~0.1 GPU-h.
