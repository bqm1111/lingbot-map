# Gate 17: instance-grounding oracle — repository map and plan

## A. What exists and what I will reuse

| need | existing component | reuse |
|---|---|---|
| causal 5-frame state at t | `tools/delayed_semantic/stream.py` (`CausalHistory`, raises on future access) | as is |
| hard map S/F | `frozen_geometry.load_hard_map` | as is |
| frozen occupancy O + trunk | `frozen_geometry.occupancy_and_trunk`, geometry seed 0 | as is |
| semantic evidence at t | `tools/delayed_semantic/memory.load_observed` (1024-d Trident-H, unit) | as is |
| **B0** | `semantic_completion/evaluation/baseline_nearest`: 3D Euclidean distance transform, label of the nearest evidence voxel; evidence voxels keep their own teacher class | as is |
| evaluation regions | `tools/delayed_semantic/regions.build_regions` | as is |
| metric | `semantic_metrics.summarize` (mIoU over classes 1..18 with GT support; **class 0 excluded**) | as is |
| GT semantics on 0006 | Gate 15 eval masks, `sem_gt` | as is |
| predict/eval firewall | `core/datasets/audit.FileAudit` with `PREDICT_FORBIDDEN` / `EVAL_FORBIDDEN` | as is |
| shard/analysis pattern | `tools/semantic_residual/{evaluate,analysis}.py` (Gate B5) | adapt |

Nothing about geometry, teacher, vocabulary, grid, class mapping or evaluation population is redefined.

## B. B0 reproduction (gate passed)

Reproduced on 0006, 590 anchors, against `artifacts/delayed_semantic_pilot/analysis.json`:

| region | frozen | reproduced |
|---|---:|---:|
| observed_current | 15.21 | 15.21 |
| observed_past_only | 17.38 | 17.38 |
| never_observed | 8.39 | 8.39 |
| completed_never_observed | 8.80 | 8.80 |
| completed_geo_observed | 9.72 | 9.72 |
| completed_geo_unobserved | 7.73 | 7.73 |
| ssc_full | 1.55 | 1.55 |
| ssc_never_observed | 1.39 | 1.39 |

Exact. Two bugs in my own Gate B5 analysis were found and fixed getting here: a reimplemented mIoU that included the
empty class (inflating SSC regions ~4x), and a shard glob that merged a leftover smoke-test file (4 duplicated
anchors). B5's conclusion is unchanged; its corrected table is in its report.

The `completed_later_revealed` / `completed_revealed_unseen` / `completed_not_revealed` regions still differ from the
pilot, by design: they depend on the revealed set R, and B5 built R from a +1..+5 window where the pilot used 20
frames. Gate 17's primary region, `completed_never_observed`, does not depend on R.

## C. The instance-annotation problem — read this before the numbers

**KITTI-360 instance annotations are not present on this machine.** Verified:

- `/media/welf/MINH/datasets/kitti360/KITTI-360/` contains only `data_2d_raw` and `data_3d_raw`. There is no
  `data_3d_bboxes`, no `data_3d_semantics`, no `data_2d_semantics`.
- The SSCBench-KITTI360 voxel labels (`preprocess/labels/<drive>/*.npy`) are `float32 [256,256,32]` with 14-15
  distinct values — semantic class and 255 ignore. **No instance channel.**
- The only instance annotations anywhere on the disks belong to nuScenes, which is a target dataset and is
  firewalled out of source work.

So `voxel -> instance ID` cannot come from annotation. The best defensible substitute available is
**6-connected components of same-class GT-occupied voxels**, computed per anchor in the evaluation grid.

What that is and is not:

- For **thing** classes (car, truck, bus, motorcycle, bicycle, person) a connected same-class blob is a good proxy
  for an object; adjacent same-class objects (parked cars in a row) merge, which makes the oracle *optimistic* in
  coverage and *pessimistic* in purity.
- For **stuff** classes (road, sidewalk, building, vegetation, terrain, fence, pole, traffic sign) a component is a
  large connected region, not an object. Reported separately throughout, never pooled into an "instance" claim.
- The grouping knows GT class boundaries exactly, which real instance annotation would also imply but a learned
  model never has. This makes the result an **upper bound**, which is the right direction for a go/no-go gate: a low
  ceiling is decisive, a high ceiling would need re-checking against real annotations before building anything.

This substitution is the one material deviation from the brief, and it is recorded here rather than buried.

## D. Experiment plan

Populations, all on 0006, all inside the existing regions:

- **P1** current SSC: `completed_never_observed` = GT-occupied & predicted-occupied & no evidence.
- **P2** correct completion: P1 restricted to voxels where predicted occupancy == GT occupancy.
- **P3** oracle geometry: GT-occupied & no evidence, regardless of prediction. Labelled oracle, never deployable.

Methods on each population (semantics always from the frozen teacher unless the row says GT):

| id | method | oracle used |
|---|---|---|
| C0 | B0 nearest evidence | none |
| A | oracle-instance transfer: each component takes the teacher label aggregated over its own visible evidence | grouping only |
| C1 | shuffled instance: the same aggregates, permuted between components within an anchor | grouping only (control) |
| C3 | visible-majority: per component, majority teacher class over its visible evidence | grouping only |
| C2 | GT class per voxel | full semantic oracle (ceiling, not a method) |

A and C3 differ only in aggregation (mean unit feature then readout, vs majority vote of per-voxel labels); both
take semantics from the teacher, never from GT.

Breakdowns: distance-to-evidence bins (existing `DISTANCE_EDGES_M`), instance visible fraction bins (0-20 .. 80-100),
per class, thing vs stuff. Controls, per-anchor records, aggregate JSON, deterministic seeds, git commit recorded.

Decision: Outcome A / B / C per the brief, from P1 and P2 on `completed_never_observed`.
