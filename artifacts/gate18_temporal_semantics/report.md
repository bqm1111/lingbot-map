# Gate 18 — temporal semantic evidence scaling: OUTCOME C (saturates), with a structural reason

Accumulating more causal history **cannot** improve semantics of the current volume, and the reason is geometric rather than a property of the teacher: **the past is behind you.** Evidence beyond about four anchors back lies entirely outside the forward-looking grid, so there is nothing left to fuse.

Within the existing five-frame window history does help — T3 completed-unobserved mIoU rises 7.09 → 8.80 from K=1 to K=5 — but essentially all of that is **new voxels being observed**, not better fusion of the same voxels. Beyond K=5 everything is flat: 8.80 → 8.84.

## 1. Raw teacher quality (deployable, no oracle)

Frozen Trident-H on drive 0006, at voxels where it has evidence **and** ground truth has a label (434,683 voxels; only 4.1% of the 10,553,472 evidence voxels carry a label, because the benchmark labels a smaller volume than the map covers).

| | top-1 accuracy | mIoU |
|---|---:|---:|
| all classes | **62.57** | **16.91** |
| thing | 63.62 | 32.58 |
| stuff | 62.54 | 17.83 |

**This is the ceiling every propagation method inherits.** The teacher is right 63% of the time on what it directly sees, and B0 reaches 8.80 mIoU on completed-unobserved against the teacher's own 16.91 — propagation loses roughly half, and the half that remains is bounded by a teacher that is already wrong a third of the time.

Per-class IoU is extremely uneven: vegetation 73.5, building 45.9, car 36.2, road 35.6, but terrain 0.36, pole 0.76, parking 4.77. Vegetation alone is 51% of the scored voxels.

## 2. The K ladder

Rule is the frozen one (normalised count-weighted mean of unit features). K≤5 is fully deployable; K>5 uses GT-pose registration and is an upper bound (see `plan.md`).

| K (frames) | seconds | evidence voxels | frames per voxel | T1 mIoU | T2 mIoU | **T3 completed-unobserved** |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.52 | 5,300 | 1.00 | 15.04 | 15.04 | **7.09** |
| 3 | 1.56 | 14,083 | 1.24 | 16.86 | 15.12 | **8.64** |
| 5 | 2.60 | 17,887 | 1.38 | 16.91 | 15.21 | **8.80** |
| 10 | 5.20 | 18,277 | 1.40 | 16.83 | 15.21 | **8.83** |
| 20 | 10.40 | 18,289 | 1.40 | 16.87 | 15.23 | **8.84** |

- **T1** population grows with K (all evidence voxels with a GT label).
- **T2** population is *fixed* at the K=1 evidence voxels, so only the fused feature changes.
- **T3** is `completed_never_observed`, fixed, propagated with the existing B0 rule.

**The T2 column is the one that matters, and it is flat.** Holding the voxel set fixed and only fusing more observations of those same voxels moves mIoU 15.04 → 15.23 — **+0.19 over the whole range**. Fusing more views of a voxel does almost nothing for its semantics.

All of the 1→5 gain is coverage: evidence voxels go 5,300 → 17,887 (3.4x). More frames see *more places*, not the same place better.

### A correlation that does not survive control

Stratifying the teacher by how many distinct frames saw a voxel suggests a large effect:

| distinct frames | accuracy | mIoU | voxels |
|---|---:|---:|---:|
| 1 | 61.81 | 15.27 | 284,015 |
| 2 | 63.44 | 19.39 | 102,155 |
| 3-5 | 65.19 | 21.62 | 48,513 |

That looks like +6 mIoU from being seen more often. It is **confounded**: voxels seen by more frames are nearer and easier, not better estimated. The controlled version of the same question is T2 above — same voxels, more observations — and it gives +0.19. The stratified number should not be quoted as a temporal effect.

Binning by *ray* count instead of distinct frames raises mIoU (12.64 → 19.82) while accuracy **falls** (61.41 → 60.96), which is the same confound seen from another angle.

## 3. Why history saturates: the past is behind you

For each earlier anchor, how much of its evidence can reach the current grid at all (40 anchors):

| anchors back | frames | seconds | ego moved | lands in the grid | and on the hard map |
|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 1.56 | 13.1 m | 30.8% | 25.6% |
| 2 | 6 | 3.12 | 26.1 m | 6.8% | 6.5% |
| 3 | 9 | 4.68 | 38.9 m | 2.6% | 1.4% |
| 4 | 12 | 6.24 | 51.4 m | 0.8% | 0.6% |
| 5 | 15 | 7.80 | 63.5 m | 0.0% | 0.0% |
| 6 | 18 | 9.36 | 75.5 m | 0.0% | 0.0% |
| 7 | 21 | 10.92 | 87.4 m | 0.0% | 0.0% |
| 8 | 24 | 12.48 | 98.2 m | 0.0% | 0.0% |

The ego vehicle moves **13.1 m per anchor** while an anchor's own evidence reaches only **28.5 m forward** (the depth range of the reconstruction) into a grid that extends 51.2 m forward. After four anchors — 12 frames, 6.2 s, ~51 m — **0.0%** of the older evidence is still inside the current volume.

This is why K=10 and K=20 add only ~2% more evidence voxels than K=5. It is not that old observations are uninformative; it is that they are observations of space the vehicle has already driven past. Forward-facing driving means the volume you must complete is the volume you have *least* history for.

## 4. Distance and class breakdowns

T3 mIoU by distance to the nearest evidence:

| distance | K=1 | K=3 | K=5 | K=10 | K=20 |
|---|---:|---:|---:|---:|---:|
| 0-1 m | 11.04 | 12.31 | 12.39 | 12.38 | 12.42 |
| 1-2 m | 9.33 | 9.68 | 9.82 | 9.94 | 9.95 |
| 2-4 m | 6.50 | 7.58 | 7.72 | 7.72 | 7.72 |
| 4-8 m | 4.24 | 4.31 | 4.23 | 4.24 | 4.22 |
| >8 m | 3.09 | 3.06 | 3.07 | 3.09 | 3.11 |

History helps only where evidence is already close (0–1 m: 11.04 → 12.42) and does nothing past 4 m (>8 m: 3.09 → 3.11). Better-fused nearby evidence does not reach distant completed voxels.

thing vs stuff on T3: thing 12.53 → 17.83 → 17.45 → 17.06 → 17.21; stuff 9.51 → 11.15 → 11.40 → 11.43 → 11.45. Both saturate by K=3–5.

Per-class T3 (K=1 → K=5 → K=20) is in `scaling.json`. The 1→5 gains concentrate in the frequent classes (building 28.00 → 35.05, road 12.20 → 17.52, vegetation 49.29 → 52.77); the classes the teacher cannot read at all stay at zero (pole 0.03 → 0.04, traffic-sign 0.03 → 0.04, terrain 0.38 → 0.26).

## 5. Aggregation rules (K=5)

| rule | T1 | T2 | T3 |
|---|---:|---:|---:|
| latest | 16.58 | 15.14 | 8.75 |
| mean_feature | 16.91 | 15.21 | 8.80 |
| mean_probability | 16.98 | 15.16 | 8.81 |

`count_weighted` is not listed because it is **identical** to `mean_feature` on this evidence: every accepted ray contributes one unit feature, so summing rays already is the count weighting. The rule choice is worth at most ~0.4 mIoU (`latest` is the worst, as expected), which is small beside everything else here.

## 6. Interpretation

Three things are now measured rather than assumed:

1. **The teacher is the ceiling.** 63% top-1, 16.91 mIoU on its own evidence. Nothing downstream can exceed it, and B0 already loses half of it in propagation.
2. **Fusing more views of the same voxel is worth almost nothing** (+0.19 mIoU from 1 to 20 frames on a fixed population). The apparent temporal effect in the stratified view is a confound.
3. **Longer history cannot reach the volume being completed**, because forward motion carries past evidence out of the forward grid within ~6 seconds.

Together these say the streaming nature of the map is not, by itself, a source of better semantics for forward completion. A persistent memory would retain space the vehicle has passed — useful for revisiting or for a surround volume, not for the benchmark's forward cone.

## 7. Outcome

**OUTCOME C — temporal evidence saturates.** Additional history barely improves teacher or map semantics, and the completed-unobserved region is flat beyond K=5. Per the brief: the teacher/representation itself is the thing that needs replacing or adapting, not the amount of temporal evidence.

No persistent-memory formulation is proposed, because that is warranted only under Outcome A.

## 8. Recommended next experiment (exactly one)

**Test whether the teacher can be improved at all on this data, before anything else is built.** Gate 17 ruled out association, Gate B5 ruled out the feature-space objective, and Gate 18 now rules out temporal accumulation — all three land on the same 63% teacher. The smallest decisive next step is a **teacher comparison on the frozen evidence voxels of 0006**: score one or two alternative open-vocabulary dense encoders, and a prompt/vocabulary variation of Trident-H itself, against the same GT on the same voxels. It needs no training and no new geometry — only a forward pass and the existing scorer. If a different teacher moves the 62.57% / 16.91 numbers materially, the whole completion stack inherits it; if none does, the bottleneck is the open-vocabulary readout itself and that is what must change.

Gate 19 (training-data scaling) is a separate hypothesis and is untouched by this result — it should be run on its own terms, as the brief specifies, now that Gate 18 is frozen.

## 9. Reproduce

```bash
python -m tools.temporal_semantics.teacher_quality --device cuda:0 --shard 0 --shards 4
python -m tools.temporal_semantics.scaling --device cuda:0 --shard 0 --shards 4
python -m tools.temporal_semantics.report
```

Code: `tools/temporal_semantics/{config,accumulate,teacher_quality,scaling,report}.py`. **Validation:** K=5 with the frozen rule reproduces the frozen evidence cache exactly — identical voxel index sets, cosine 1.000000, identical counts. No training. ~0.3 GPU-h. Commit in `git_commit.txt`.
