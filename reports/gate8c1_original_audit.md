# Original Gate 8C-1 audit — 2026-09-06

1. **The historical 12.74% / 30.89% measurements reproduce to their reported precision.** They are **medians of three seeds with sample standard deviations**, despite the prompt describing means. The reproduced means are **13.1127% / 30.9536%**. The original checkpoints and thresholds were recovered; original spatial predictions were not. Saved score histograms were recounted first, followed by inference from the existing geometry caches. Completion is numerically reproducible, not bitwise identical.
2. **The original Occ3D superiority claim was not a compatible comparison.** It compared our causal inputs and raw outputs with published forward-looking, pooled OccAny results. A verified common forward subset now shows our fixed checkpoints ahead on Occ3D, but extra image history and different camera-to-grid calibration prevent calling this a fully matched five-image comparison. The conditional ranking is supported; strict superiority under compatible inputs remains unverified.
3. **SemanticKITTI precision is lower because completion introduces far more scored false occupancy, while Occ3D evaluates a denser and much more selectively visible population.** The mapper already has a precision gap at nearly equal recall and false-positive rate. Completion raises SemanticKITTI false-positive rate much more. High and boundary-adjacent errors dominate; many corresponding Occ3D predictions are outside its scoring mask. A fixed-checkpoint uniform-input probe demonstrates a spatial response without observed geometry, but does not isolate padding as the root cause. The demonstrated SemanticKITTI label defect is too small to explain the gap.

All evidence is in [the audit directory](../artifacts/gate8c1_original_audit/). No training, checkpoint selection, threshold search, target reconstruction, or model correction was performed. Later plain-column, sky, neighbourhood and Gate 8D variants were excluded.

## Reproduction and aggregation

The original [manifest](../artifacts/gate8c1/frozen_manifest.json), frozen on 2026-09-03, matches all three checkpoint hashes and source-selected thresholds. Each seed uses the same checkpoint and threshold on both datasets. Scores below are percentages, pooled over the original **163 SemanticKITTI sequence-08 anchors** and **1,182 Occ3D anchors** separately for each seed.

| Seed / aggregation | SK saved histogram | SK replay | SK corrected labels | Occ3D saved histogram | Occ3D replay / corrected |
|---|---:|---:|---:|---:|---:|
| 0 | 12.32374 | 12.32329 | 12.34453 | 30.88701 | 30.88716 |
| 1 | 12.73801 | 12.73847 | 12.76189 | 31.27335 | 31.27330 |
| 2 | 14.27605 | 14.27634 | 14.30488 | 30.70099 | 30.70039 |
| Median ± sample SD | 12.73801 ± 1.02865 | **12.73847 ± 1.02890** | 12.76189 ± 1.03263 | 30.88701 ± 0.29196 | **30.88716 ± 0.29218** |
| Mean ± sample SD | 13.11260 ± 1.02865 | **13.11270 ± 1.02890** | 13.13710 ± 1.03263 | 30.95378 ± 0.29196 | **30.95362 ± 0.29218** |

`aggregate.py` calculates mean, median and SD (`ddof=1`); `report.py` prints the median. Precision, recall and volume ratio were each independently reduced across seeds. Thus the historical SK precision/recall, 15.88%/39.15%, correspond to seed 1, whereas Occ3D's 67.14%/36.39% correspond to seed 0. SK's median volume ratio comes from seed 0. These summary fields must not be combined as one checkpoint's confusion matrix.

Each metric here is computed from summed TP/FP/FN/TN. Equal-weight averages of frame IoUs differ: SK seeds 0/1/2 give 12.41959/12.73975/14.33402%, and Occ3D gives 30.93362/31.64616/30.63505%. Pooling all seeds' voxels gives 13.07821/30.96161%, also different from averaging seed IoUs. Every convention is recorded in [results.json](../artifacts/gate8c1_original_audit/results.json).

Historical `scores_*.npz` contain 1,024-bin positive/negative score histograms, not spatial predictions. The thresholds fall exactly on their 1/32-logit bin edges. These recovered the original numbers before new scoring. Original binary-count files and histogram thresholding differ by only a few counts. No original spatial prediction volumes were recovered in the inspected Gate 8C-1 output/cache locations, so all required spatial predictions were regenerated. First/middle/last anchors were frozen for verification before replay. Full mapper and fixed-dilation TP/FP/FN reproduce **exactly on every anchor**, including both forward sets. Completion's largest causal IoU discrepancy is **0.00061 percentage points**; every per-sample count delta is saved. GPU reduction variability and histogram boundary rounding limit bitwise reproduction; no settings were changed to force the old numbers.

## Recovered implementation and inputs

| Seed | Original checkpoint | Full SHA-256 | Occupancy logit threshold |
|---|---|---|---:|
| 0 | `artifacts/gate8c1/checkpoints/seed0_last.pt` | `5227c672012024786561e615b755edd938d551e62d05ad52d2ad23813825e7d0` | −0.125 |
| 1 | `artifacts/gate8c1/checkpoints/seed1_last.pt` | `13640b7664997abb90fce5da733a8ea60e1ed4b0b48e6a5dcb2a1b7132587752` | −0.09375 |
| 2 | `artifacts/gate8c1/checkpoints/seed2_last.pt` | `68e67f28a1e4b764d14d1bc7d0f256ef5dd5f5be82d826884ada83ace0f65a67` | −0.15625 |

All are step-6,000 checkpoints: 986,114 parameters, 32 input channels (seven map features plus 25 semantic evidence channels), 24/48/96-channel three-level 3D U-Net, paired 3×3×3 convolutions, GroupNorm(8), GELU, stride-two down/up operations and occupancy/semantic heads. The occupancy residual is applied only where `abs(map_logodds) < 2`. Inference uses CUDA bfloat16 autocast, float32 final logits, zero convolution padding, `pad_z=0`, and `final >= threshold`. Primary completion has no spatial post-processing. The fixed mapper baseline requires occupied map evidence and positive semantic weight; dilation is the original radius-two cube on the 0.2 m map.

The recorded project commit is `9a648322e6dc4e34995f0530650c2fa469779676`, but Gate 8C-1 files were in a dirty/untracked tree and cannot be recovered by checking out that commit. In particular, frozen `gate8/net.py` hash `4cbfced792c8f255fc808c66819cc7f553777157e6699c01d4c26076af7a4ea4` differs from current `56852e4a11999c3ac8e817aaa7c7f8c0381fc71f734537226084750aedda578c`. Current additions expose optional padding variants; the old checkpoint has no `padding_mode`, so loading selects original zero padding. The audit explicitly disables added z padding. Mapper/feed/vocabulary hashes match the manifest. The exact original dirty network file remains unavailable; preserved current code plus numerical replay establishes the recovered behaviour, not byte-for-byte historical source identity. [Artifact references](../artifacts/gate8c1_original_audit/artifact_references.json) record matches and mismatches, and [code_used](../artifacts/gate8c1_original_audit/code_used/) preserves the execution sources.

The minimal file chain is:

| File | Responsibility |
|---|---|
| `tools/gate8c1/eval_target.py` | Original manifest loading, temporal windows, map query, prediction and scoring |
| `gate8/sources.py`, `gate7b/streams.py` | Dataset anchors, image sequence, calibration and cache paths |
| `gate7b/replay.py`, `lingbot_map/models/gct_stream.py` | Cached LingBot geometry and streaming initialization |
| `gate8/feed.py`, `gate8/mapper.py` | Cached RGB-derived geometry/semantics, fixed scale and map accumulation |
| `gate8/net.py`, `gate8a/regions.py` | Checkpoint inference, residual lock and Occ3D grid reduction |
| `gate6/targets.py`, `prompted_lingbot/occ_datasets.py`, `gate6/metrics.py` | Historical labels, evaluated population and counts |
| `gate8c1/occany_eval.py`, external `OccAny/occany/metrics/ssc.py` | Official counter comparison |
| `tools/gate8c1/original_audit.py` | Independent labels/counts, fixed replay, diagnostics and evidence export |

Geometry comes from `/media/SSD1/MINH_DATASETS/lingbot_gate7b/stream/{dataset}/{segment}.npz`; scales from the parallel `scale/` directory; Trident-H evidence from `/media/SSD1/MINH_DATASETS/lingbot_gate6/semantics/`. MoGe supplies the first-five-frame scale gauge, not target labels. Scale is `exp(median(log_s[:5]))`, fixed per segment and applied to both depths and pose translations. Recomputing all five scale candidates for the first segment of each dataset agrees exactly. The inputs, caches, labels, OccAny predictions and supporting files have **17,961 consumed-file hash records**, all present, in [consumed_artifact_hashes.jsonl](../artifacts/gate8c1_original_audit/consumed_artifact_hashes.jsonl); stream and scale hashes are also in the artifact references.

The primary `past5` evaluation creates a **fresh map for each anchor**, integrates five observations once, and resets between anchors. It does not retain an all-past MapState. LingBot's geometry cache nevertheless retains earlier image context: five jointly processed initialization images, then causal frame-by-frame streaming with a 64-frame cache and five pinned initialization frames; target segments use keyframe interval one and normalization off. All scored anchors are at index ≥4, so initialization does not introduce images after the anchor. Actual LingBot replay of nine versus eleven images on both datasets produced identical first-nine depth/confidence/pose/intrinsics/pose-encoding arrays, also bit-identical to saved caches. This supports causality for the inspected path and prefixes; “five frames” describes the map window, not an exclusive five-RGB input budget.

SK uses 815 stream frames at raw stride five, with 163 anchors at raw IDs 20…4070, step 25. Its causal raw offsets are −20/−15/−10/−5/0, about 2.08 seconds. Occ3D uses 150 scenes and five consecutive approximately 2 Hz CAM_FRONT frames ending at each of 1,182 anchors. Forward offsets are stream 0/2/4/6/8: SK raw 0/10/20/30/40, about 4.15 seconds; Occ3D about four seconds. The forward sets contain 161/882 anchors: the last two SK anchors and last two anchors in each of 150 Occ3D scenes lack enough future context. All IDs, exclusions, image paths, cache prefixes, scale frames and transforms are frozen in [samples.json](../artifacts/gate8c1_original_audit/samples.json); exact camera exposures, reference timestamps and offsets are in [input_timestamps.json](../artifacts/gate8c1_original_audit/input_timestamps.json). No 400-anchor evaluation was substituted.

## Scoring and input integrity

Arrays use XYZ order. SK is 256×256×32 at 0.2 m, origin (0,−25.6,−2), upper extent (51.2,25.6,4.4). Occ3D prediction is 400×400×32 at 0.2 m, origin (−40,−40,−1); scoring is 200×200×16 at 0.4 m, upper extent (40,40,5.4). Occ3D reduces each eight-child block by occupancy ANY / logit MAX. Query centres are transformed by `scaled_pose_c2w @ inverse(T_cam_to_grid)` into the world map. Coordinate indices are floored; out-of-grid points are rejected. The map's search-index clamp is followed by an equality check and does not clamp coordinates onto grid faces. Explicit lower/upper outside-point tests pass.

The independent counter reads SK raw uint16 labels and big-endian packed `.invalid` bits directly. Historical scoring uses the YAML learning map, raw zero as free and invalid voxels ignored. **Defect:** raw labels 1 and 99 map to zero and were incorrectly scored as free; official OccAny remapping ignores nonzero raw labels mapped to zero. Correcting only the audit mask excludes **326,492** causal voxels and gives the corrected scores above. Occupied counts do not change. For Occ3D, labels 0…16 are occupied, 17 free, 255 ignored; scoring requires `mask_camera` and x index ≥100. `mask_lidar` is unused. This is surround-camera visibility restricted to the front half-grid, **not a CAM_FRONT frustum**.

Independent historical label arrays and masks agree elementwise with the existing evaluator on every sample. Corrected arrays agree elementwise with saved official OccAny labels on all 161/882 common forward samples. Thirty predetermined cases (three anchors × five methods × two datasets) agree exactly with both the existing counter and official OccAny counter, including valid and GT occupied populations. [Evaluator equivalence](../artifacts/gate8c1_original_audit/evaluator_equivalence.json) and per-sample CSVs retain the evidence.

Metrics use TP/(TP+FP+FN), TP/(TP+FP), TP/(TP+FN), FP/(FP+TN), (TP+FN)/valid and (TP+FP)/(TP+FN); zero denominators return zero. The causal valid/GT counts are **239,805,138 / 18,762,930** for historical SK and **52,095,614 / 11,957,590** for Occ3D. Thus predict-everything-occupied IoU is **7.82424% / 22.95316%**, reproducing both claims. Corrected SK prevalence is 7.83491%.

An actual `eval_target.run` forward pass was repeated on the first anchor of each dataset using original scoring data, all-free/all-valid data, and all-occupied/half-valid data. Map queries, editable masks, raw logits, probabilities and unmasked occupancy hashes remain identical. A first attempt exposed nondeterministic GPU semantic-evidence reductions; enabling deterministic algorithms and the existing cuBLAS workspace setting made the controlled check bitwise stable. The stored [SK](../artifacts/gate8c1_original_audit/integrity_semantickitti.json) and [Occ3D](../artifacts/gate8c1_original_audit/integrity_occ3d.json) checks pass. This check addresses the raw completion path.

**A separate defect affects historical auxiliary pooling:** `comp = prediction & valid` is computed before 3×3×3 pooling. Consequently scoring masks influence post-processing. Official OccAny pools the full prediction and masks only when counting. The audit corrects that ordering solely for the auxiliary comparison below; raw primary predictions are unchanged. Historical forward pooled IoUs by seed were SK **12.65638/12.05398/13.59069%** and Occ3D **30.63196/29.89108/31.01981%**. Corrected values are reported separately below. The SK correction also uses official ignored-label handling; Occ3D's much larger change comes from pooling order. Historical measurements remain preserved.

Image preprocessing uses the full RGB image resized to width 518 and patch-aligned height (154 SK, 294 Occ3D), with predicted intrinsics on that lattice. Trident evidence is resized to the same full-image lattice. No inconsistent crop/intrinsic pairing was demonstrated. OccAny uses known-intrinsic, principal-point cropping to 512×160/288; reapplying its crop to all five images of three fixed forward anchors per dataset reproduces saved pixels and intrinsics exactly.

One calibration mismatch remains material: our Occ3D camera-to-grid transform compensates camera/LiDAR timestamp differences using annotated ego poses; OccAny uses static camera extrinsics. Three fixed samples have maximum transform-element differences 0.00017, 0.31034 and 0.14939. Those annotations place the benchmark grid; inter-image mapper motion remains predicted. Therefore the broad claim “GT poses only enter training targets” is not true of this benchmark-placement path. SK shares OccAny's inverse-`Tr` convention without a separate camera-0-to-camera-2 translation. No target-specific geometric correction was applied.

Checkpoint metadata and frozen records support KITTI-360 drives 0003/0007/0010 for training and 0006 for validation/selection. Training sample directory name/size fingerprints still match; separate raw-LiDAR target directory fingerprints do not (26 additional bytes per file). Their exact historical bytes and the complete dirty training source are unavailable. The old firewall records zero target accesses, but this audit did not rerun or certify training. It did not reconstruct targets or read rejected SSCBench `_1_1.npy` files. Reproducible inference does not by itself establish sound supervision or whole-system training provenance.

## OccAny: verified subset and remaining mismatches

The **published** sequence values are 25.91% SK and 23.55% Occ3D, from Table 1 of [OccAny v2](https://arxiv.org/html/2603.23502v2). The paper lists Waymo, DDAD, PandaSet, VKITTI2 and ONCE as training datasets; the old local report's claim that OccAny was trained on SemanticKITTI and nuScenes is incorrect. The relevant model is OccAny with MUSt3R and SAM2 features, including novel-view completion, rather than OccAny+ or reconstruction-only output.

The actual local reproduction uses `/home/minh/workspace/OccAny`, commit `04d4616e5a0fc3b0124b404be2f61d378373e8d8`, with checkpoint SHA-256 **`05155377a79cff430b6dfe45758cce89ff357c7e80b65438a66bea351c172b80`**. Extraction logs and full-completion prediction keys establish the variant. Local modifications are preserved in `occany_diff_final.txt`; they include a point-label bounds guard for corrupt SK frame 001865, SAM2 path handling and dataset-loader compatibility changes. The audit consumed existing outputs and did not reinstall or rerun OccAny.

Predictions are under `/media/SSD1/MINH_DATASETS/occany_out/ssc_voxel_pred/OccAny_5frames_{kitti,nuscenes}512_rot60_vpi10_fwd3_sTrans2/`. The full reconstruction-plus-generation keys use recorded confidence thresholds 2.5/1.1. The recipe includes ±60° rotations, ten views per interval, 3 m forward and 2 m lateral translations. Reference frame is the first input; image identities agree on the deterministic checks. Official voxelization distributes positive contributions to up to eight neighbouring native voxels; out-of-grid contributions are discarded. Its geometry-only `apply_majority_pooling` with `use_dilation=True` is **3×3×3 max pooling, stride one, padding one**, not voting. Source: [official OccAny repository](https://github.com/valeoai/OccAny), with exact local files hashed in the evidence.

The common IDs were frozen before fresh scoring: all 161 SK and 882 Occ3D original forward anchors have saved OccAny predictions. These are **subset comparisons**, drawn from 805/4,819 available local OccAny outputs; they do not reproduce the full published evaluations. Every row below uses identical samples, corrected label arrays and scoring populations within its dataset. “Pooled” means the same native-grid max operation before masking for both methods.

| Method | SK raw IoU | SK pooled IoU | Occ3D raw IoU | Occ3D pooled IoU |
|---|---:|---:|---:|---:|
| OccAny, full completion | 25.27627 | 25.92437 | 20.64007 | 23.48445 |
| Original Gate 8C-1 seed 0 | 13.96900 | 13.13528 | 27.55695 | 40.46975 |
| Original Gate 8C-1 seed 1 | 13.04135 | 12.47533 | 27.61267 | 38.88668 |
| Original Gate 8C-1 seed 2 | 15.28284 | 14.03800 | 27.50747 | 41.05076 |

This establishes the ranking for the saved pipelines on the common subsets. It does **not** establish a controlled five-RGB advantage: our geometry includes prior/unsampled streaming images and first-five-frame scale initialization; Occ3D calibration differs; native voxelization and image preprocessing differ. These conditions are explicit in [occany_provenance.json](../artifacts/gate8c1_original_audit/occany_provenance.json). A strict comparison would require geometry generated from exactly the common five images and a common benchmark-placement convention; no such verified paired artifact was recovered. The causal 163/1,182-anchor results remain the primary historical reproduction.

## Precision gap: measured contributions

All following diagnostics use the original causal samples and historical population, preserving the question being investigated. Completion rows average independently pooled seed metrics; they are not a combined confusion matrix.

| Dataset / method | IoU % | Precision % | Recall % | FPR % | Predicted / GT volume |
|---|---:|---:|---:|---:|---:|
| SK mapper | 8.904 | 33.320 | 10.835 | 1.841 | 0.325 |
| SK mapper + fixed 0.4 m dilation | 15.415 | 23.584 | 30.796 | 8.470 | 1.306 |
| SK completion, mean of seeds | 13.113 | 16.584 | 38.552 | 16.533 | 2.333 |
| Occ3D mapper | 9.987 | 65.082 | 10.552 | 1.687 | 0.162 |
| Occ3D mapper + fixed 0.4 m dilation | 21.641 | 57.543 | 25.753 | 5.661 | 0.448 |
| Occ3D completion, mean of seeds | 30.954 | 65.493 | 37.109 | 5.915 | 0.570 |

The mapper's similar recall/FPR already yields different precision because occupied prevalence differs. Completion then introduces the major additional SK false-positive burden. The exact mapper-to-completion transitions are:

| Dataset / seed | Added TP | Added FP | Retained TP | Removed FP | Incorrectly removed TP |
|---|---:|---:|---:|---:|---:|
| SK 0 | 4,959,670 | 33,817,851 | 1,988,857 | 263,825 | 44,075 |
| SK 1 | 5,335,748 | 35,024,736 | 2,009,506 | 194,037 | 23,426 |
| SK 2 | 5,392,061 | 29,221,163 | 2,014,334 | 173,641 | 18,598 |
| Occ3D 0 | 3,105,222 | 1,510,476 | 1,245,834 | 58,076 | 15,920 |
| Occ3D 1 | 3,452,996 | 2,436,995 | 1,241,536 | 60,224 | 20,218 |
| Occ3D 2 | 3,016,622 | 1,308,753 | 1,249,847 | 46,174 | 11,907 |

For seed 0, added occupancy is 12.79% correct on SK versus 67.28% on Occ3D. The precision identity

`precision = prevalence × recall / [prevalence × recall + (1 − prevalence) × FPR]`

accounts exactly for each paired seed. For seed 0, SK precision starts at 15.59%; substituting Occ3D prevalence alone gives 39.33%, then Occ3D recall gives 38.91%, then Occ3D FPR gives the actual 67.14%. This is an algebraic accounting, not a causal intervention: changes in population and FPR need not be independent. All three seeds' decompositions are saved in `results.json`.

Bins were fixed before full analysis: height edges 0/1/2/3 m; horizontal range 10/20/30/40 m; nearest grid-face distance 0.4/1/2 m; distance to nearest **valid GT occupied voxel centre** 0.4/1/2/5 m. End bins extend to infinity. Distances use physical metres and voxel centres; nearest-occupied distance is only a diagnostic proxy for surface distance. Height is benchmark-grid z, not ground-relative height. [SK bins](../artifacts/gate8c1_original_audit/bins_semantickitti.csv) and [Occ3D bins](../artifacts/gate8c1_original_audit/bins_occ3d.csv) contain all methods/seeds, confusion counts, populations and rates. Selected seed-0 evidence follows; categories overlap and must not be added together.

| Region | SK FP | SK FPR % | Occ3D FP | Occ3D FPR % |
|---|---:|---:|---:|---:|
| Height ≥3 m | 25,247,973 | 34.597 | 997,173 | 8.194 |
| Grid boundary distance <0.4 m | 21,602,334 | 72.106 | 554,383 | 64.717 |
| Grid boundary distance ≥2 m | 1,767,334 | 2.441 | 757,248 | 2.817 |
| GT-voxel distance 2–5 m | 19,578,631 | 21.404 | 626,625 | 3.770 |
| GT-voxel distance ≥5 m | 4,265,355 | 32.541 | 71,267 | 20.453 |

SK height ≥3 m contributes 67.11% of seed-0 FP, but contains 4,572 TP too: high voxels were not assumed empty. Of the added SK occupancy there, 24,519,893 are FP and 3,776 TP. Approximately 63.38% of SK FP are ≥2 m from a scored occupied voxel, versus 32.77% for Occ3D. These are not all near-surface thickening errors. SK range-bin FPR is roughly 14–21%, without a single abrupt range boundary; full count/rate rows are retained. The first GT-distance bin has no Occ3D FP because distinct voxel centres are at least 0.4 m apart, a quantization effect.

Visibility explains why similar-looking unmasked responses can score very differently:

| Seed 0, height ≥3 m | SK | Occ3D |
|---|---:|---:|
| Valid voxels / all grid voxels in band | 73,021,147 / 74,776,576 (97.65%) | 14,511,136 / 283,680,000 (5.12%) |
| Scored predicted occupied / unmasked predicted occupied | 25,252,545 / 25,567,547 (98.77%) | 1,589,916 / 58,164,889 (2.73%) |

Ignored Occ3D voxels have not been relabelled free. These measurements show that most of its high predictions are simply unscored. Its scored high band also has much greater occupied prevalence. The mask and grid distributions are therefore part of the measured cross-dataset difference, not evidence of superior geometry everywhere.

The lower band also differs: below grid z=0, occupied prevalence is 26.15% on SK versus 92.56% on Occ3D. Seed 0 has 9,758,825 FP and 6,451,523 TP there on SK, versus 193,341 FP and 2,322,481 TP on Occ3D. Dense lower-layer predictions therefore receive very different treatment by the actual labels and evaluated populations. Because the grids have different origins, these are documented physical-coordinate bins, not identical ground-relative regions.

A six-pass diagnostic used each original checkpoint on a spatially uniform, entirely unknown map with zero observed geometry and semantic evidence. It produces dense occupancy near top/bottom grid indices on both datasets. Seed 0 predicts about 99% occupancy in the top two SK layers, and about 99.5% in the top native Occ3D layer. Locations occupied in this synthetic probe overlap 69.96% of real SK FP and 34.16% of real Occ3D FP. This demonstrates an input-independent spatial response and its overlap with errors; it does not causally attribute that percentage of errors to padding. [Profiles and overlap](../artifacts/gate8c1_original_audit/mechanism.json) preserve the probe independently of all benchmark predictions.

The network and original mapper dilation operate at **0.2 m on both datasets**: a 3-voxel convolution spans 0.6 m in either, so the coarser Occ3D scoring resolution does not double the network neighbourhood. Occ3D's later eight-child reduction, different z origin/XY extent, and the auxiliary native-grid pool do differ physically. Zero padding, stride phase, learned biases and whole-volume GroupNorm are plausible contributors to the spatial response; their separate effects are unresolved. The smallest additional causal check would be one source-only fixed-crop diagnostic varying only the outside-grid encoding with weights and threshold fixed. It is not necessary to establish the measured error burden, and was not turned into a target-specific prediction rule here.

Four examples use seed 0 and the first/middle causal IDs fixed at preparation: [SK first](../artifacts/gate8c1_original_audit/example_semantickitti_08_000000_000020_s5.png), [SK middle](../artifacts/gate8c1_original_audit/example_semantickitti_08_002025_002045_s5.png), [Occ3D first](../artifacts/gate8c1_original_audit/example_occ3d_scene-0003_CAM_FRONT_fd842039_f56a5440.png), [Occ3D middle](../artifacts/gate8c1_original_audit/example_occ3d_scene-0633_CAM_FRONT_9fde5c65_22a2197b.png). Each includes RGB, mapper geometry, added TP/FP, and unprojected single-voxel slices at fixed x/y/z positions. Three-dimensional points are subsampled; slices are not. Grey denotes ignored voxels, so sparse projections cannot be mistaken for fully scored surfaces.

## Commands and evidence

Run from `/home/minh/workspace/lingbot-map_fork` with the existing Python 3.10 / PyTorch 2.7.1+cu128 environment. RTX PRO 6000 Blackwell GPUs were available; GPU work used `cuda:2`. No packages were upgraded. These commands describe the executed audit stages; preparation intentionally refuses to overwrite the existing sample freeze, and inference reuses its saved unmasked outputs.

```bash
/home/minh/anaconda3/envs/cu128/bin/python tools/gate8c1/original_audit.py prepare
for audit_ds in semantickitti occ3d; do
  /home/minh/anaconda3/envs/cu128/bin/python tools/gate8c1/original_audit.py infer --dataset "$audit_ds" --mode past5 --verify-only --device cuda:2
  /home/minh/anaconda3/envs/cu128/bin/python tools/gate8c1/original_audit.py analyze --dataset "$audit_ds" --mode past5 --verify-only
  for audit_mode in past5 occany_fwd; do
    /home/minh/anaconda3/envs/cu128/bin/python tools/gate8c1/original_audit.py infer --dataset "$audit_ds" --mode "$audit_mode" --device cuda:2
    /home/minh/anaconda3/envs/cu128/bin/python tools/gate8c1/original_audit.py analyze --dataset "$audit_ds" --mode "$audit_mode"
  done
  /home/minh/anaconda3/envs/cu128/bin/python tools/gate8c1/original_audit.py integrity --dataset "$audit_ds" --device cuda:2
  /home/minh/anaconda3/envs/cu128/bin/python tools/gate8c1/original_audit.py streamcheck --dataset "$audit_ds" --device cuda:2
done
/home/minh/anaconda3/envs/cu128/bin/python tools/gate8c1/original_audit.py provenance
/home/minh/anaconda3/envs/cu128/bin/python tools/gate8c1/original_audit.py mechanism --device cuda:2
/home/minh/anaconda3/envs/cu128/bin/python tools/gate8c1/original_audit.py finish
/home/minh/anaconda3/envs/cu128/bin/python tools/gate8c1/original_audit.py seal
```

`counts_*.csv` and `corrected_counts_*.csv` contain each sample's TP/FP/FN/TN, valid, GT/predicted occupied counts and metrics, separately for all five fixed methods. `occany_counts_*.csv` retain both raw and pooled common-subset counts. `summary_*.json` include transition totals; `verification_*.json` contain historical deltas. Packed unmasked volumes are under `predictions/`. [Reproduction provenance](../artifacts/gate8c1_original_audit/reproduction_provenance.json) includes checkpoint metadata, historical pooled counts and accounting checks. [Audit file hashes](../artifacts/gate8c1_original_audit/audit_file_hashes.json) cover every output, prediction, execution snapshot and this report; external-input hashes are in the two provenance inventories above. LingBot checkpoint SHA-256 is `ee665103348e07e6b826d529b8e61de8f413d5432a4f2e84970d6c8fd2e1cd72`.

The investigation establishes reproducible performance and a measured explanation of the precision gap, identifies two scoring defects, and limits the OccAny claim. It does not certify the unavailable historical training bytes or establish a unique learned mechanism for the boundary response. Existing work and historical outputs were preserved; the audit stops here.
