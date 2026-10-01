# Gate 20 — implementation plan and what was reused

No training. No new model. One question: **does ordinary future driving observation provide enough
high-quality semantic supervision for the geometry our causal system correctly completes but has never
observed?**

## Reused unchanged

| from | what |
|---|---|
| `tools/semantic_completion` | grid convention, frozen `CompletionUNet` loader (seed 0, step 6000, sha256-checked), Trident-H text embeddings and readout, cached evidence, `baseline_nearest` (B0 + the EDT distance), `semantic_metrics` (the frozen mIoU definition) |
| `tools/delayed_semantic` | `DriveStream` / `stream.future`, `TargetConfig` (the privileged window: `t+1..t+20`, ≤ 10 s), `targets.camera_view` (z-buffer occlusion + bilinear Trident-H token), `targets.ViewCache`, `targets._voxel_of` (floor binning), `regions.build_regions` (P0 **is** `completed_never_observed`), the frozen delayed-target cache (the `deployable` level) |
| `tools/instance_oracle` | `oracle.instance_map` — per-class 6-connected GT components, for the region analysis |
| `tools/kitti360_data_scaling_7drives/build_eval_masks.py` | the GT mask logic and its label-frame conversion, generalised to the training drives |
| `core.datasets.kitti360_transforms` | `DriveGeometry`, poses, calibration, velodyne reads |
| `core.datasets.audit.FileAudit` | the prediction-time firewall |

**No coordinate transform, occlusion test, voxel binning, evidence-lifting rule, class mapping or metric
is re-implemented in this gate.** The two genuinely new pieces are (a) a per-voxel *first-reveal time* at
three nested levels, and (b) the aggregation.

## New code

```
tools/future_revealability/
  config.py             horizons, levels, bins, pre-registered thresholds, paths
  eval_masks.py         GT (valid / occupied / sem_gt) for 0003/0007/0010; bit-exact rebuild of frozen 0006
  future_visibility.py  per-voxel first-reveal time, three nested levels; reproduces the frozen target cache
  build_targets.py      per-anchor P0/P1 record, in the order causal -> GT -> future -> no-feedback check
  analyze.py            micro/macro aggregation, block bootstrap, the pre-registered decision
  validate.py           the eight validation checks
  report.py             figures and report.md
```

## Drives

GT voxel labels on the anchor grid were available only for 0006 (590 anchors) when this gate started, so
they were built for the training drives with the frozen builder's own logic — verified by rebuilding 24
0006 anchors and requiring `valid`, `occupied` and `sem_gt` to match the frozen masks bit-exactly.

| drive | role | anchors | with a GT label |
|---|---|---:|---:|
| 0003 | training | 193 | 193 |
| 0007 | training | 569 | 529 |
| 0010 | training | 596 | 554 |
| 0006 | source validation, **never used to set a threshold** | 590 | 590 |

The two GT occupancy caches on this machine (`lingbot_gate8c1/targets`, pre-sky-rule, and
`lingbot_k360_7drives/targets`, post-sky-rule) disagree on `valid` but their `occupied` arrays are
identical and `occupied ⊆ valid` in both, so P0 — which needs only GT-occupied-with-a-class — is the same
whichever cache is used. The post-sky-rule cache is used, matching the frozen 0006 masks.

## Levels

See `decision_rule.md` §2. `visible` is the decision level and is exactly the evidence rule the semantic
lifting pipeline already uses, applied to a future frame; `geometric` and `deployable` bracket it.

## Leakage discipline

Per anchor, in this order: (1) the causal state is computed inside a `FileAudit` that forbids LiDAR, GT
poses, delayed targets and human labels, and is digested; (2) GT is opened, P0/P1 formed, and P0 is
asserted equal to the frozen `completed_never_observed` region and disjoint from the evidence mask; (3) the
future reveal record is built, restricted to the P0 rows; (4) the step-1 digest is recomputed and must be
unchanged. The future can therefore only answer "does this hidden voxel become observable later, and what
would it then say" — it cannot touch occupancy, evidence, the never-observed definition or B0.

## One change outside this gate's directory

`tools/delayed_semantic/targets.py` gained a single additive field on `CameraView`:

```python
#: rows of ``pts_velo`` inside the image at z >= min depth, BEFORE the z-buffer occlusion test.
in_image_index: torch.Tensor = None
```

It is set from a value `camera_view` already computed and is read only by the Gate 20 `in_frustum`
diagnostic. Nothing else reads it, no stored output changes: `tests/delayed_semantic` (20 tests, including
`test_delayed_targets_reproduce_the_pilot_records`) still passes, and the `visible` level still reproduces
the frozen delayed-target cache exactly. It is **not** included in the Gate 20 commit, because
`tools/delayed_semantic/` is untracked at `HEAD` — as every gate tool since the pilot is — and committing
one file of an untracked package would leave a broken partial package in the tree. Whether those packages
should be tracked is a repository decision, not Gate 20's to take.
