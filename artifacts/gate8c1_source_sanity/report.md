# Source-only sanity: can the Gate 8C-1 pipeline learn KITTI-360 occupancy?

KITTI-360 only. Both phases ran inside the Gate 8C-1 `FileAudit`: 22,131 paths opened, **0 SemanticKITTI / Occ3D / nuScenes / SSCBench-label paths** ([phase 1](phase1_file_access_check.json), [2](phase2_file_access_check.json)).

## Phase 1 — original seed-0 checkpoint, all 590 drive-0006 IDs

Checkpoint sha256 `5227c672…` matches the frozen manifest; τ = −0.125, zero padding, `pad_z=0`, raw-LiDAR targets, 256×256×32 @ 0.2 m.

| Condition | TP | FP | FN | TN | SC IoU % | P % | R % | FPR % | pred/GT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mapper | 781,192 | 5,921,044 | 17,412,932 | 243,384,715 | 3.24 | 11.66 | 4.29 | 2.38 | 0.37 |
| mapper + 0.4 m dilation | 2,664,802 | 24,577,700 | 15,529,322 | 224,728,059 | 6.23 | 9.78 | 14.65 | 9.86 | 1.50 |
| completion | 4,790,422 | 26,565,464 | 13,403,702 | 222,740,295 | **10.70** | 15.28 | 26.33 | 10.66 | 1.72 |

10.70 % reproduces the manifest's `source_iou_at_threshold` 0.10703704819534747, independently validating this scoring path. Per-case completion IoU: median 11.12 %, p10 7.61 %, p90 15.22 %, **max 20.75 %** — no subset of drive 0006 is handled well. Boundary FPR is 57.93 % with pred/GT 4.38 against interior 2.81 %; 62 % of all false positives sit in the four lowest z layers (−1.9 to −1.1 m) and 5.5 % in the four highest. So the boundary slab seen on SemanticKITTI is already present on the training domain. Occupied prevalence is 6.80 %.

## Phase 2 — 16-clip overfit sanity test

Sixteen clips were frozen before training ([IDs](phase2_training_clips.json)), six from drive 0003 and five each from 0007 and 0010. Fresh head, original seed-0 initialization, original objective (focal+dice+0.5 KL), AdamW 1e-3, cosine, batch 4, crop 128³×32, exactly 2,000 steps, no checkpoint selection.

* Masked occupancy loss **1.1599 → 1.0437 = 10.0 % decrease** (criterion ≥90 %): **failed**.
* Pooled training SC IoU over the 16 full clips **12.25 %** (criterion ≥80 %): **failed**.

The head cannot memorise sixteen clips it sees ~500 crops of each. Its drive-0006 score, 9.14 % (P 15.15, R 18.74), is the original checkpoint's regime. [Curve](phase2_training_curve.csv).

## Which branch applies

**"Fails to fit the 16 training clips: supervision, feature alignment or training implementation is broken."** The fourth branch's condition also holds — the original checkpoint is poor on all 590 drive-0006 cases, so Gate 8C-1 is not a valid Scene Completion baseline — but that is a consequence, not a separate finding. Cross-dataset transfer is *not* implicated: SemanticKITTI 12.74 % is **higher** than the source 10.70 %.

The sharpest clue is the input itself. The incremental mapper — built from the same poses, depths and scale that produce the network's 32 channels — has **11.66 % precision** against the raw-LiDAR target (median 12.80 % per case). Nearly nine in ten voxels the map calls occupied are not occupied in the supervision. Whatever the head does downstream, its input geometry and its target do not agree. (This mapper omits the MoGe-confirmation term, unavailable in the cached samples, which would only raise precision.)

## Recommended next experiment

Exactly one, source-only: a geometric alignment audit on a small deterministic set of drive-0006 clips. For each mapper-occupied voxel, measure the distance to the nearest target-occupied voxel, and recompute SC IoU under a small fixed grid of rigid translations and scale factors applied to the target. That decides whether map and supervision share a metric frame, before any architecture or loss work resumes.

[config](config.json) · [environment](environment.json) · [evaluated IDs](evaluated_ids.json) · [phase-1 pooled](phase1_pooled.json), [per case](phase1_per_case.csv), [FP by z](phase1_fp_by_z.csv) · [phase-2 result](phase2_result.json). Stopped for review.

