# Target-dataset access for the IoU vs mIoU visualisation (2026-09-17)

**What was accessed.** SemanticKITTI 08 and Occ3D-nuScenes val, through the artifacts of the **already-completed, frozen**
final zero-shot evaluation:
- our canonical predictions (`lingbot_zero_shot_eval/predictions`);
- OccAny's stored per-sample labels and released geometry predictions;
- the frozen per-sample score shards (`artifacts/final_zero_shot/scores`).

**Why this is not new target information for a pending decision.**
- Every number shown was computed and frozen during the final zero-shot evaluation (PAPER C, 2026-09-15).
- Nothing was re-scored, retrained or tuned.
- The figures are explanatory only: they show why a class-agnostic occupancy IoU can favour us while an unweighted
  per-class mean favours OccAny.

**Relation to the open scaling study.** `artifacts/semantic_field_scaling/source_declaration.md` keeps SemanticKITTI and
Occ3D closed for the B4 source-data scaling study. That study is **BLOCKED before its freeze** on a reliability-gate
failure, and no B4 scaling checkpoint, source score or promotion decision exists. This access:
- did not touch any scaling-study artifact, manifest, checkpoint or pool;
- used only B0 / B3 predictions that already went through the frozen target protocol;
- is disclosed here and must be carried into the scaling study's disclosure if that study is ever unblocked, so the record
  stays complete.

**Requested by the user** on 2026-09-17 ("visualize the semantic scene completion output ... our method vs OccAny"), after I
flagged the constraint.

**Produced.**
- `iou_vs_miou.png` — per-class IoU and the two metrics side by side (frozen numbers only, no target access).
- `semantic_maps_<dataset>.png` (+ `_pair*`) — per-voxel semantic maps, ours vs GT, with OccAny occupancy.
- `semantic_maps_index.json` — the samples drawn and their per-sample statistics.

**Code.** `tools/metric_explainer/` (`iou_vs_miou.py`, `semantic_maps.py`). Nothing under `tools/zero_shot_eval` or
`tools/occany_comparison` was modified; their renderers and loaders are imported unchanged.

## Caveat on the row labels

Rows are ordered by **Track P** per-sample occupancy IoU difference (the frozen score shards, region-restricted: camera
field of view, within 40 m). The IoU printed inside each row is computed over the **whole valid label volume**, because the
figure draws every voxel as requested. The two can disagree for a single sample — e.g. SemanticKITTI `08_000970` sits at
Δ-rank 15 % (OccAny better in Track P) yet shows 33.7 vs 28.4 over the full volume — since the region, not the method,
differs. Aggregate conclusions in the report use Track P and Track O only.
