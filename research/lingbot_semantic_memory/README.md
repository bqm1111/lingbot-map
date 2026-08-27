# Geometry-Conditioned Semantic Memory — Phase 1 (gated feasibility)

Isolated research package. It **does not modify LingBot-MAP**: the model is loaded
frozen, read with forward hooks that cannot change a module's return value, and every
parameter is `requires_grad=False`. The external DINO teacher is frozen too. The only
trainable things in this package are small probes.

## Hypothesis under test

> Intermediate or post-GCT LingBot-MAP tokens are more consistent across viewpoints
> than frame-independent visual features, while retaining enough semantic information
> for a lightweight sidecar to decode DINO-space features.

This is a falsification experiment. The pre-registered decision rule below was written
before any number was produced.

## Pre-registration

**Primary cross-view metric.** Mean cosine between probe outputs at geometrically
corresponding patches, computed on held-out sequence 08, at the **100 % teacher
budget**, using **predicted** geometry, **without** depth-confidence filtering. Probe
outputs are chosen over raw tokens because all representations are mapped into the one
shared 768-d teacher space, so the numbers are comparable; raw-token cosine is not
comparable across representations of different width and anisotropy. Raw-token
consistency is reported as a secondary diagnostic.

**Primary fidelity metric.** Mean cosine to the frozen DINO teacher on the same
held-out frames, at the 100 % budget.

**Gate.** `PASS` requires one GCT representation to satisfy **both**:

1. cross-view `pos` cosine ≥ 1.15 × the `lingbot_encoder` value, on the *same*
   correspondence set; and
2. teacher cosine ≥ 0.95 × the `lingbot_encoder` value (text-query mIoU is
   unavailable — see below — so the 5 %-relative-fidelity arm of the criterion applies).

`SPARSE_PASS` is marked separately if the 10 %-budget probe of the best representation
reaches ≥ 90 % of its own 100 %-budget teacher cosine.

Verdicts: `PASS`, `FAIL_SEMANTICS`, `FAIL_CONSISTENCY`, `FAIL_GEOMETRY`, `INCONCLUSIVE`.

**Anti-degeneracy guard, declared up front.** Mean cross-view cosine is trivially 1.0
for a probe that emits a constant vector — the earlier study in this repository hit
exactly that (`docs/semantic_sidecar_feasibility_report.md` §2). Every cross-view
number is therefore reported with a negative control (cosine between non-corresponding
patches of the same pair), the `margin = pos − neg`, retrieval `recall@1`, and feature
`diversity`. **If a representation wins on `pos` while losing on `margin` and
`recall@1`, the win is reported as collapse, not as a pass.** This qualifier is part of
the pre-registration, not a post-hoc reinterpretation.

## Two things to know before reading any result

1. **The encoder baseline is the teacher's sibling.** LingBot's pre-GCT patch embedder
   *is* DINOv2 ViT-L/14 with registers; the external teacher is DINOv2 ViT-B/14 with
   registers. Different weights, different width, but the same family and training
   recipe. So `lingbot_encoder` is an unusually strong fidelity baseline — close to a
   linear re-encoding of the target. That makes criterion 2 a genuinely hard test and
   makes criterion 1's baseline honest (it is a real, strong frame-independent feature,
   not a straw man). It also means the fidelity column should not be read as "how
   semantic is this layer" in an absolute sense.
2. **No text-to-DINO bridge exists in this repository.** Searching for one turns up
   only LingBot's own DINOv2 backbone. Text-query and semantic mIoU evaluation is
   therefore **unavailable** and is marked as such rather than substituted with the
   unrelated MaskCLIP bridge in `semantic/`. Semantic labels are still used — for
   evaluation only — in the boundary-preservation metric.

## Representations compared

All read at the same 11×37 patch lattice, all with parameter-matched probes.

| name | where it comes from | width |
|---|---|---|
| `external_dino_teacher` | frozen DINOv2 ViT-B/14-reg, `x_norm_patchtokens` | 768 |
| `lingbot_encoder` | `aggregator.patch_embed` → `x_norm_patchtokens`, **pre-GCT** | 1024 |
| `lingbot_gct_b04` | aggregator block 4, `frame_blocks[4] ‖ global_blocks[4]` | 2048 |
| `lingbot_gct_mid` | aggregator block 11 | 2048 |
| `lingbot_gct_b17` | aggregator block 17 | 2048 |
| `lingbot_gct_final` | aggregator block 23 — last tokens before the depth head | 2048 |

Blocks 4/11/17/23 are exactly the `selected_idx` that `GCTStream._aggregate_features`
hands to the camera, depth and point heads, so the captured tokens are the ones the
model actually uses.

## Fairness

Identical train/val frames, identical teacher targets, identical optimiser and
schedule, no augmentation for any representation, identical feature resolution. Probe
**capacity is matched by parameter count**, not by hidden width: a fixed hidden width
would give the 2048-d GCT probes ~60 % more parameters than the 1024-d encoder probe.
`probes.solve_hidden` sizes each probe to a shared budget instead; the counts agree to
under 0.1 % (asserted in `tests/test_probes_metrics.py`).

## Commands

```bash
export PYTHONPATH=/home/minh/workspace/lingbot-map_fork
export CUDA_HOME=/usr/local/cuda-12.8 FLASHINFER_CUDA_ARCH_LIST="12.0a"
PY=/home/minh/anaconda3/envs/cu128/bin/python

# tests
$PY -m pytest research/lingbot_semantic_memory/tests -q

# one-command smoke test (2 short chunks, 200 probe steps)
$PY -m research.lingbot_semantic_memory.run_phase1 \
    --config research/lingbot_semantic_memory/configs/phase1.yaml --smoke

# one-command full phase-1 experiment
$PY -m research.lingbot_semantic_memory.run_phase1 \
    --config research/lingbot_semantic_memory/configs/phase1.yaml
```

Every script takes `--config`, `--device`, `--output-dir` and `--seed`.

## Layout

| file | role |
|---|---|
| `config.py` | dataclass config, seeding, provenance |
| `dataset_adapter.py` | SemanticKITTI chunks, oracle poses/depth/labels |
| `hooks.py` | frozen LingBot + frozen DINO teacher, representation capture |
| `feature_cache.py` | one `.npz` per chunk so probe training never re-runs inference |
| `reprojection.py` | patch-level cross-view correspondences + validity filtering |
| `probes.py` | parameter-matched probe, distillation loss |
| `metrics.py` | fidelity, cross-view (+ collapse guards), boundary |
| `run_phase1.py` | the experiment |
| `report_tables.py` | markdown tables incl. the cross-dataset comparison |
| `visualize.py` | correspondence and similarity figures |
| `reports/` | `phase0_inventory.md`, `phase1_verdict.md`, `phase1_replica_verdict.md` |
