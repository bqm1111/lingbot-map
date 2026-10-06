# Final zero-shot semantic completion — evaluation predeclaration (written 2026-09-15, before any target semantic prediction or score existed)

Hash-frozen with `tools/zero_shot_eval/practical/protocol.py` by `tools/zero_shot_eval/freeze.py` (stage `protocol`) before B0, B3
or OccAny is scored under this protocol. Nothing below was chosen by looking at a method comparison. Target semantic labels have
not been opened by this study at the time of writing.

## Frozen system (no retraining, no target tuning)

Causal five-frame history t-4..t, local-reset LingBot, LB-M depth with causal MoGe-2 clip scale, local predicted poses, hard H
mapper (frozen thresholds), frozen CompletionUNet **geometry seed 0** (the seed the semantic heads were trained on; occupancy
threshold 0), frozen Trident-H (official Cityscapes configuration; token features, Gate 15 evidence lifting and aggregation),
evidence memory, candidate rule (16 nearest), and the Gate 16 **B3** heads, seeds 0/1/2. **B0** = nearest observed semantic
evidence. One camera: SemanticKITTI `image_2`; Occ3D-nuScenes `CAM_FRONT`.

Readout vocabulary: each benchmark's official non-empty classes with the frozen Gate 6 phrases (`core/datasets/vocab.py`),
encoded by Trident's own text tower; label = argmax cosine. No prompt, class map, threshold, checkpoint or candidate rule may
change after a target score is visible.

## Samples

Every sample the released OccAny model scored whose causal window t-4..t exists: SemanticKITTI 08 **801** of 805; Occ3D-nuScenes
val **4,219** of 4,819 (front camera). Labels: OccAny's stored `voxel_label` (verified per sample against an independent loader of
the official files in `tools/occany_comparison`). Occ3D labels carry the official camera-visibility mask and OccAny's
single-camera front-half rule (x index ≥ 100).

## Canonical prediction (one artifact per sample, read by every evaluator)

On the official evaluation grid (SemanticKITTI 0.2 m = our prediction grid; Occ3D 0.4 m, reduced from our canonical 0.2 m grid):
occupancy (Occ3D: a native voxel is occupied iff any of its eight subvoxels is — the frozen rule); per method a class label for
every predicted-occupied voxel and every GT-occupied voxel (the latter only as a query list; predictions are independent per query);
Occ3D semantic reduction: readout of the normalised mean 1024-d feature of the occupied subvoxels (all eight if none is occupied);
evidence mask E (any subvoxel carries semantic evidence); geometric observation mask Gobs (any subvoxel in hard-map S ∪ F);
distance to the nearest evidence voxel (min over subvoxels, metres); grid origin, voxel size, frame, `T_cam_to_grid`, intrinsics.

## Track P — practical causal evaluation (PRIMARY)

**Rationale.** The deployed system is a single forward camera with five past frames. The space it is responsible for is what lies
in front of that camera within a working range; whole-grid averages mix in regions no single camera can see and let observed
surfaces dominate. Track P therefore scores the camera's horizontal field of view and separates observed from completed space.

* **Region P** = valid label (≠ 255) ∩ horizontal field of view of the target camera (voxel centre in front of the camera and between
  the rays through image columns 0 and W, W = native width; no vertical limit beyond the grid) ∩ horizontal range ≤ **40 m**
  (in the camera x–z plane). 40 m is the largest forward range fully inside both benchmark grids' field of view (Occ3D extends
  ±40 m) and covers urban stopping distances; it was fixed from grid geometry, not results.
* **Range bins**: 0–10, 10–20, 20–30, 30–40 m.
* **Free / unknown**: label free = free (a predicted occupied voxel there is a false positive); 255 is ignored; voxels outside P are
  not scored in Track P.
* **Occupancy**: IoU, precision, recall over P, pooled over samples; per range bin; within Gobs (geometrically observed) and ~Gobs.
* **Semantic metrics** (pooled confusions; mIoU = mean IoU over classes with ≥ 1 GT voxel in that population; per-class IoU; overall
  accuracy; mean class accuracy; number of classes predicted; predicted-class frequencies):
  * `ssc_miou` — predicted occupancy: label where predicted occupied, else empty; all voxels of P;
  * `ssc_no_evidence_miou` — the same over P ∩ ~E;
  * `gt_occupied_miou` — diagnostic, GT-occupied voxels of P, label regardless of predicted occupancy;
  * `observed_evidence_miou` — GT occupied ∩ E;
  * `no_evidence_miou` — GT occupied ∩ ~E (oracle occupancy);
  * **`completed_no_evidence_miou` (CENTRAL)** — GT occupied ∩ predicted occupied ∩ ~E;
  * `geometrically_unobserved_miou` — GT occupied ∩ predicted occupied ∩ ~Gobs;
  * by distance to nearest evidence (0–1, 1–2, 2–4, 4–8, > 8 m) on the central population, and by range bin.
* **Classes**: all official non-empty classes (SemanticKITTI 19; Occ3D 17 including `others`). Dynamic classes are kept and flagged.
  Superclass mIoU with OccAny's official mapping as a secondary metric.
* **Secondary full-region variant**: the same metrics over all valid voxels.

## Track O — official benchmark protocol (SECONDARY)

Same predictions, all valid voxels of the official grid (labels as above), OccAny's `SSCMetrics` definitions: SC precision / recall /
IoU on valid voxels; SSC mIoU = mean over non-empty classes of TP / (TP + FP + FN + 1e-5) over the pooled dataset (SemanticKITTI
excludes no other class; Occ3D includes `others`), superclass mIoU. Raw predictions, no pooling.

## Statistics and B0 / B3

B3 = mean of seeds 0/1/2 (per-seed values reported). Δ = B3 − B0 for every metric above and per class / distance / range bin.
Paired block bootstrap (2,000 resamples, seed 0) of seed-pooled B3 confusions against 3 × B0; blocks = 25 consecutive samples
(SemanticKITTI), one nuScenes scene (Occ3D). Occupancy is identical for B0 and B3 by construction (checked).

## OccAny

* **Level A (geometry)**: OccAny's released predictions (its five forward frames, front camera, 512 px) through Track P occupancy
  and Track O, raw and with its published pooling.
* **Level A (semantics)**: attempted with OccAny's official semantic presets (SemanticKITTI five-frame distill, Occ3D surround-view
  distill — the only published nuScenes semantic setting) if they run on this machine; SemanticKITTI on all 801 samples; Occ3D on
  all 4,219 if the projected wall time on four GPUs is ≤ 12 h, otherwise on the samples that are also our official anchors. Outputs
  pass through the same evaluators unchanged. Otherwise **Level B**: published Table 4 numbers beside our Track O numbers, with
  every input/protocol difference listed; **Level C**: methodological comparison.
* No OccAny setting is tuned; no Track P choice changes after OccAny is scored.

## Paper decision (fixed now)

* **Generalises** (per dataset): central metric B3 − B0 > 0, bootstrap CI lower bound > 0, positive on every seed.
* **PAPER C** if B3 does not generalise on both datasets.
* **PAPER A** if it generalises on both, the central gain is ≥ +1.0 mIoU point on both, and our Track O SSC mIoU (B3 mean) is at
  least OccAny's published semantic mIoU on both (7.28 SemanticKITTI / 6.66 Occ3D) — or, where Level A semantics exist, not
  lower than OccAny's under the same evaluator — with every input difference disclosed.
* **PAPER B** otherwise.
* Practical usefulness (Q2) is reported descriptively from Track P occupancy and semantic numbers by range; it does not change the
  decision rule.
