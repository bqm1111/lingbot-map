# CURRENT_SYSTEM — the current best system, traced from its entry command to its final SC/SSC metrics

Written 2026-10-07 on branch `feature-completion-study` (HEAD `148c46e`). No code was modified to produce it.

Everything below comes from reading the code on the path and from a **runtime trace**: one sample run through the real
entry points with `sys.setprofile`, outputs written only to the session scratchpad. The trace results are in § 0.

---

## 0. Which system, and what was verified

**The current best system** is the final zero-shot pipeline: `reproduce.sh` stage 11 (`tools/zero_shot_eval`),
fed by stage 10's causal clips.

- **Geometry:** the frozen geometry baseline, `CompletionUNet`, seed 0.
- **Semantics:** Trident-H evidence plus the Gate 16 **B3** association head (seeds 0/1/2). **B0** (nearest evidence) is
  the control.

It is "best" because nothing after it beat it on the frozen protocol:

| study | outcome |
|---|---|
| B4 (`semantic_field`) | SOURCE FAIL |
| B5 (`b5_semantic_residual`) | BLOCKED |
| Gate 19 | OUTCOME C |
| Gate 21 | OUTCOME C |
| Waymo pilot | BLOCKED |

The older `feature_completion_study` and `teacher_alignment_study` transfer IoUs (~22–23 SemanticKITTI past5 on 163 anchors)
sit below this system's geometry on a different protocol.

**Headline numbers** (`artifacts/final_zero_shot/results_numbers.md`, decision `PAPER C` in `analysis.json`):

| dataset (samples) | Track P SC IoU | Track P SSC mIoU B0 / B3 | Track O SC IoU | Track O SSC mIoU B0 / B3 |
|---|---:|---:|---:|---:|
| SemanticKITTI 08 (801) | 32.65 | 5.45 / 5.64 | 26.40 | 4.32 / 4.43 |
| Occ3D-nuScenes val (4,219) | 36.87 | 4.36 / 4.54 | 31.94 | 3.71 / 3.80 |

"B3" is the **mean of three per-seed metrics**, not an ensemble prediction. Every sample stores four label sets
(`B0`, `B3_s0`, `B3_s1`, `B3_s2`). A single deployable system would be one seed, e.g. `B3_s0`: 5.66 / 4.54 Track P.

**Verified in this session:**

| check | result |
|---|---|
| Freeze hashes: `frozen_protocol.json` / `frozen_evaluators.json` (`freeze.check`) | unchanged since 2026-09-15 |
| Freeze hashes: geometry checkpoints vs `baseline_reconstruction/code_freeze.json` | match |
| Freeze hashes: B3 checkpoints and Gate 16 code vs `gate16_candidate_selection/code_freeze.json` | match |
| Runtime trace of SemanticKITTI sample `08_000020`, raw PNGs → LingBot clip | `pred_depth`, `pred_depth_conf`, `pred_pose_c2w`, `pred_K`, `scale`, `log_s_frames`: **bit-identical** to the stored clip |
| Same sample, clip → canonical prediction | `occupied`, `observed`, `evidence`, `query_index`, all four label sets: **bit-identical** |
| Same sample, prediction → per-sample scores | every `ps__*` array **identical** to row 0 of `scores_semantickitti.npz` |
| KITTI-360 anchor 0006 / t = 4, raw PNGs → clip → hard map → occupancy | **bit-identical** to the frozen caches (§ 9) |
| `pytest tests/zero_shot_eval` | 9 passed |
| Working-tree state of the 133 files in § 7 | all tracked; the only modification is a whitespace-only diff in `tools/occany_comparison/scoring.py` (a removed blank line, an added trailing newline) |

---

## 1. Entry command

```bash
# environment (reproduce.sh header)
REPO=/home/minh/workspace/lingbot-map_fork; PY=/home/minh/anaconda3/envs/cu128/bin/python
TPY=/home/minh/workspace/third_party/trident_env/bin/python; TRI=/home/minh/workspace/third_party/Trident
export PYTHONPATH=$REPO HF_HUB_OFFLINE=1 CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH FLASHINFER_CUDA_ARCH_LIST=12.0a

bash reproduce.sh --from 10 --to 11      # causal clips, then the final evaluation
```

Stage 11 expands to:

```bash
$PY -m tools.zero_shot_eval.samples                                                         # A  frame lists
(cd $TRI && PYTHONPATH=$REPO:$TRI $TPY $REPO/tools/zero_shot_eval/teacher/extract_tokens.py \
      --dataset {semantickitti,occ3d} --shard g --shards 4 --device cuda:g)                 # B  Trident-H tokens
freeze.freeze('protocol'); freeze.freeze('evaluators')                                      # (kept if present)
$PY -c "from tools.zero_shot_eval.prediction.pipeline import run_shard; run_shard(ds, dev, g, 4)"   # C–I
$PY -c "from tools.zero_shot_eval.run_eval import evaluate_dataset; evaluate_dataset(ds)"         # J
$PY -c "from tools.zero_shot_eval import analysis; analysis.analyse()"                            # K
$PY -c "from tools.zero_shot_eval import report; report.write_all()"                              # L
$PY -c "from tools.zero_shot_eval import verify; verify.run('cuda:0')"                            # L
```

Stage 10 (`tools/occany_comparison/cli/build_clips.py --dataset … --shard … --device …`) writes the clips that step C
reads. `build_moge.py` adds MoGe frames only for Occ3D frames past the stored 40-frame streams.

**Resumability changes what "running" means.** `run_shard`, `build_clips` and `extract_tokens.py` skip files that
already exist. On this machine, the entry command re-scores the stored predictions; it does not recompute them. To make
it recompute, point `ZERO_SHOT_EVAL_DATA_ROOT` and `OCCANY_COMPARISON_DATA_ROOT` at empty roots.

---

## 2. Data flow

```
image t-4..t (RGB) ──► [C1] LingBot-Map GCTStream, 5-frame local reset ──► depth/conf/pose/K (canonical units)
        │                                                    │
        ├──► MoGe-2 calibrated-FOV depth (cached) ──► [C2] per-frame log-scale ──► clip scale s (m/unit)
        │                                                    ▼
        │                                  [D] IncrementalMapper (hard map) ──► S, F on the prediction grid
        │                                                    ▼
        │                                  [E] CompletionUNet seed 0 ([S,0,F]) ──► O (occupied), trunk d1 [24,X,Y,Z]
        │                                                    │
        └──► [B] Trident-H tokens ──► [F] lift at accepted ray endpoints ∩ S ──► evidence (flat, 1024-d, count, frame bits)
                                                             ▼
                                   [G] B0: EDT nearest evidence      [H] B3: 16 nearest candidates → MLP → softmax mix
                                                             ▼
                          [I] native-grid reduction + text readout ──► CanonicalPrediction .npz (one per sample)
                                                             ▼
          official label (OccAny voxel_label) ──► [J] Track P / Track O confusions ──► [K] mIoU, SC IoU, bootstrap, decision
```

---

## 3. Step-by-step trace (module → function → tensors → configuration)

Notation: `N` is the number of voxels on a grid, and the flat index is C-order `ravel_multi_index((x, y, z))`.

| grid | dims | voxel | origin (m) | N |
|---|---|---|---|---|
| SemanticKITTI = SSCBench-KITTI-360 prediction & evaluation grid | 256×256×32 | 0.2 m | (0, −25.6, −2), velodyne frame of the target | 2,097,152 |
| Occ3D prediction grid ("canonical") | 400×400×32 | 0.2 m | (−40, −40, −1), ego frame | 5,120,000 |
| Occ3D evaluation grid (native) | 200×200×16 | 0.4 m | (−40, −40, −1), ego frame | 640,000 |

### A. Sample enumeration — `tools/zero_shot_eval/samples.py`

- **`evaluated_samples(ds)`** = `tools.occany_comparison.samples.build_samples(ds)`, filtered by
  `Sample.protocol_ok(PROTOCOLS["causal"])`. The causal protocol is offsets (−4, −3, −2, −1, 0) with `anchor_slot` 4.
- **SemanticKITTI:** `semantickitti_samples`.
  - Reads `core.datasets.sources.segments("semantickitti")`, then `_val_segments`, `core.datasets.streams.build`,
    `core.datasets.frames.read_manifest` (manifest `artifacts/scale_gate/manifests/val.jsonl`).
  - Uses the stride-5 stream of 815 frames, with key `08_<raw id>` and position = raw id / 5.
  - `T_cam_to_grid` comes from `_sk_calib`, i.e. `core.datasets.occ_datasets.SemanticKittiOccSpec.build(...).cam_to_velo`.
  - Samples are the OccAny output directories under
    `/media/SSD1/MINH_DATASETS/occany_out/ssc_voxel_pred/OccAny_5frames_kitti512_rot60_vpi10_fwd3_sTrans2`
    (asserted 805), leaving 801 with a causal window.
- **Occ3D:** `occ3d_samples`.
  - Reads `core.datasets.nuscenes.load_annotations(OCC3D_ROOT)` and `scene_frames(ann, scene, "CAM_FRONT")`.
  - `T = T_camera_to_ego` of the target frame, and `K_native` per frame.
  - The prefix is asserted equal to the frozen stream (manifest `manifests/occ3d_zeroshot/val.jsonl`). For official
    anchors, the transform is asserted against `/media/SSD1/MINH_DATASETS/lingbot_occ3d_zeroshot/cache_lingbot`.
  - OccAny output root: `OccAny_5frames_nuscenes512_rot60_vpi10_fwd3_sTrans2` (asserted 4,819), leaving 4,219.
- **`write_frame_list`** writes `frames_<ds>.json`: the unique history frames, 805 for SemanticKITTI and 4,819 for Occ3D.

### B. Teacher tokens — `tools/zero_shot_eval/teacher/extract_tokens.py`

This step runs in `trident_env`, launched from the Trident root.

- **Teacher:** `gates.gate6.trident_adapter.TridentTeacher(vocab.load(ds).phrases, sam_ckpt=…/sam_vit_h_4b8939.pth)`.
  - Configuration `OFFICIAL_CITYSCAPES`: CLIP ViT-H-14 laion2b plus SAM ViT-H, and the frame is split into two halves.
- **Hooks:** they capture the output of `model.clip.encode_image`, asserted to be exactly one call per frame. The tokens are
  L2-normalised. `token_grid` derives (h, w) from the two SAM calls.
- **Tokens stored per frame:** `tokens [h·w, 1024] fp16`, `grid_hw`, `native_hw`.

  | dataset | token grid | native image |
  |---|---|---|
  | SemanticKITTI | 39×128 | 370×1226 |
  | Occ3D | 64×112 | 900×1600 |
  | KITTI-360 (source domain, `lingbot_feature_first_3d`) | 34×128 | 376×1408 |

- **Proof per frame:** the code recomputes Trident's coarse probabilities from the tokens as
  `softmax(40 · bilinear(tokens @ textᵀ))` and compares them with Trident's own. The minimum argmax agreement is
  0.998 (SemanticKITTI) and 0.997 (Occ3D).
- **Text embeddings** (`text_embeddings_<ds>.npz`): `text [C, 1024]` = `Trident.query_features`, plus `labels [C]`.
  - SemanticKITTI: C = 19, labels 1..19.
  - Occ3D: C = 17, labels 0..16.
  - Phrases come from `core/datasets/vocab.py`: official names, eight explicit rewrites, then `-`/`_` → space. The empty
    class is never included.

### C. Causal clip — stage 10, `tools/occany_comparison/clips.py::infer_clip`

1. **Load images:** `gates.gate7b.replay.load_images` calls `lingbot_map.utils.load_fn.load_and_preprocess_images(mode="crop", image_size=518, patch_size=14)`.

   | dataset | lattice (processed image) |
   |---|---|
   | SemanticKITTI | 154×518 |
   | Occ3D | 294×518 |
   | KITTI-360 | 140×518 |

   The lattice is a full-frame resize with no crop.

2. **Build the model:** `replay.build_model` constructs `GCTStream(img_size=518, patch_size=14, enable_3d_rope=True, max_frame_num=1024, kv_cache_sliding_window=64, kv_cache_scale_frames=5, kv_cache_cross_frame_special=True, kv_cache_include_scale_frames=True, use_sdpa=False, camera_num_iterations=4)`.
   - Weights: `checkpoints/lingbot-map/204754b/lingbot-map.pt`, `strict=False`.
   - All parameters are frozen.
   - `gates/scale_gate/config.py` supplies `REPO_ROOT`.
3. **Run the clip:** `model.clean_kv_cache()`, then `replay.replay_segment(model, images, keyframe_interval=1)`.
   - That calls `GCTStream.inference_streaming(num_scale_frames=5)` under bf16 autocast. With S = 5, **all five frames form
     one scale block** (bidirectional among t−4..t, still causal with respect to the future). `rope_slots_used` is 0.
   - Executed path: `gct_base.forward` → `aggregator/base.py::_embed_images` (DINOv2 `vision_transformer.vit_large`) →
     `aggregator/stream.py::_process_causal_stream` / `_process_global_attention` ×24, with
     `layers/flashinfer_cache.py::FlashInferKVCacheManager` and `layers/rope.py::WanRotaryPosEmbed` → `heads/camera_head.py`
     (4 iterations) → `heads/dpt_head.py` (depth).
   - `pose_encoding_to_extri_intri` gives `pred_pose_c2w [5, 4, 4]` (canonical units, frame 0 = origin) and
     `pred_K [5, 3, 3]` (lattice intrinsics).
   - Also output: `pred_depth [5, H, W] fp16` (optical-axis z, canonical units) and `pred_depth_conf [5, H, W] fp16`.
4. **Scale:** for each slot, `moge_extra.resolve_moge_path` (frozen Gate 7B cache `/media/SSD1/MINH_DATASETS/lingbot_gate7b/moge_b/<ds>/<key>.npz`,
   else `lingbot_occany_comparison/moge_b`) feeds `gates.gate7b.scale.frame_candidate`.
   - `log_s` = median of `log D_moge − log D_lingbot` over the jointly valid pixels: MoGe mask, 1 < MoGe depth < 60 m,
     conf ≥ 1.5, at least 1000 pixels, 3×MAD clip.
   - Clip scale `s = exp(nanmedian(log_s))` metres per unit.
5. **Output:** `clips/<ds>/causal/<sample>.npz`.

### D. Hard map — `tools/occany_comparison/geometry.py`

- **`WindowClip.load`**: the clip arrays cast to float32/float64, with `anchor_slot` asserted to be 4.
- **`build_mapper`**: `core.mapping.mapper.IncrementalMapper(sem_into=None, n_teacher=0)`, then
  `scale_state.force(s)`, then `step(FrameInput)` ×5.
  - `frame_inputs` re-bases each pose as `inv(P_4) @ P_q`.
  - **`_integrate`, accepted rays:** conf ≥ 1.5 and 1 m < s·z < 60 m (`core/mapping/depth.py`); translation × s.
  - **`_integrate`, occupied band:** ±0.2 m along the ray (`rays.BAND_HALF_M`), 0.2 m world keys (`pack(floor(p/0.2))`).
  - **`_integrate`, free carve:** stride 4 pixels, from 1.0 m to 0.2 m before the surface.
  - **`_integrate`, log-odds:** +0.85 occupied / −0.40 free, clamp ±4 (`core/mapping/voxmap.py`).
  - Stored in `VoxelTable`, a sorted hash.
- **`hard_channels(mapper, PREDICTION_GRID[ds], T)`**: `mapper.query(grid, inv(T_cam_to_grid))` looks up each grid centre's
  world key.
  - `S` = log-odds > 0 (and not MoGe-only, which never applies here).
  - `F` = log-odds ≤ 0 and `w_free` > 0.
  - Both are flat bool `[N_pred]`.

### E. Geometry — `tools/semantic_completion/frozen_geometry.py`

- **`load_frozen_geometry(0)`**: sha256 check of `artifacts/baseline_reconstruction/checkpoints/geo_seed0_step06000.pt`
  against `code_freeze.json`, then `core.model.net.load_checkpoint` → `CompletionUNet(cin=3, width=24)`.
  - 3-level U-Net with 24/48/96 channels and GroupNorm.
  - Asserted 967,322 parameters.
  - Eval mode, no gradients.
- **Input:** `x = [S, 0, F]` as `[1, 3, X, Y, Z]` float. It is built inline in `pipeline.predict_sample`, the same as
  `geometry_input`.
- **`occupancy_and_trunk`**: bf16 autocast, `net(x, return_trunk=True)`.
  - `O = apply_residual(0, residual) > 0`, i.e. `residual > 0`. The threshold is 0, and `LOCK_LOGODDS` never applies
    because the base is 0.
  - Outputs: `O [N_pred]` bool and `trunk d1 [1, 24, X, Y, Z]` fp32.
  - The semantic head output `head_sem` is computed but discarded.

### F. Evidence — `tools/zero_shot_eval/prediction/evidence.py::lift_evidence`

1. `tools.semantic_completion.evidence.lift.ray_endpoints(clip)` (which uses `baseline_reconstruction.current.geometry.frame_inputs`)
   applies the same acceptance test and scaled pose as the mapper. Each accepted ray's endpoint `p = dir · s·z` gets a world key.
2. For each slot q:
   - `trident.load_token_map` gives `[1024, h, w]`.
   - `sample_at_lattice` does a bilinear `grid_sample` (`align_corners=False`) at lattice pixel centres and renormalises.
   - The result is indexed by the accepted pixels.
3. Per unique world key:
   - the summed feature uses `index_add_`;
   - `count` = number of rays;
   - `frame_bits` bit q = slot q contributed.
4. Grid lookup: `grid.world_keys` (voxel centres → world key). A voxel is evidence if its key was hit **and** it is in `S`.
5. Output `Evidence`:
   - `flat [M]` int64, ascending
   - `feature [M, 1024]` unit
   - `count [M]` int32
   - `frame_bits [M]` uint8

   Typical sizes: SemanticKITTI `08_000020` has M = 44,157; KITTI-360 0006 / t = 4 has M = 18,536.

### G. B0 — `prediction/semantics.py::nearest_evidence`

`scipy.ndimage.distance_transform_edt(~E, return_indices=True)` gives the nearest evidence flat index `[N_pred]` int64 and
the distance in metres `[N_pred]` float32. A voxel's B0 feature is its nearest evidence feature.

### H. B3 — `prediction/semantics.py::b3_features` with the head from `delayed_semantic.gate16.evaluate.load_head`

**Head:** `EvidenceAssociation(ModelConfig(k_candidates=16, evidence_proj=16, hidden=64, path_samples=8, range_norm_m=6.0))`,
25,489 parameters, loaded from `artifacts/gate16_candidate_selection/checkpoints/B3_seed{0,1,2}_step03000.pt`.

**Queries:** the fine subvoxels of the query voxels (§ I) that are not evidence, processed in chunks of 16,384.

- **Candidates:** `nearest_candidates` uses `torch.cdist` and `topk(16)` over evidence centres, giving `[Q, 16]`.
- **Edge features** (48 per candidate) = `[trunk(e) 24, Linear(1024→16)·f(e) 16, (e−q)/6 m 3, |e−q|/6 m 1, log1p(count)/5, observed_in_t, n_frames/5, path_fractions in (F, S, O) 3]`.
  `path_fractions` samples 8 points along q→e.
- **Query features** (25) = `[trunk(q) 24, (z − origin_z)/(Z·voxel)]`.
- **Scorer:** an MLP (73→64→64→1, GELU), then softmax over the 16 candidates.
- **Output:** `normalize(Σ w_k f(e_k))` `[Q, 1024]`. Evidence voxels keep their own feature.

### I. Native reduction and readout — `prediction/pipeline.py::predict_sample`

1. Query list (native grid):
   - `label = scoring.load_occany(ds, pkl).label` (OccAny's stored `voxel_label`, 255 = ignore).
   - `q_native = valid & (any-subvoxel(O) | GT occupied)`.
   - The label is used **only to choose which voxels to label**; each voxel's prediction is independent of the list.
2. `subvoxels(q_native)` gives `[M, r³]` (r = 1 SemanticKITTI, 2 Occ3D). `members` = the predicted-occupied subvoxels, or all
   eight if none is occupied.
3. Per method, the member features are summed per native voxel and passed to `readout`: argmax of the cosine to `text`,
   mapped through `label_of_channel` to an official label id.
4. **`CanonicalPrediction`** (`canonical/prediction.py`), saved as compressed npz:

   | field | shape / dtype | rule |
   |---|---|---|
   | `occupied` | `[N_eval]` bool, bit-packed | `reduce_any(O)` |
   | `observed` | `[N_eval]` bool, bit-packed | `reduce_any(S\|F)` |
   | `evidence` | `[N_eval]` bool, bit-packed | `reduce_any(E)` |
   | `query_index` | `[M]` int32 | |
   | `evidence_distance` | `[M]` fp16 | `reduce_min` |
   | `label__{B0,B3_s0,B3_s1,B3_s2}` | `[M]` uint8 | |
   | `T_cam_to_grid`, `K`, `image_width`, `meta` | | |

   Sizes for SemanticKITTI `08_000020`: 389,448 occupied, 323,326 observed, M = 221,550.

### J. Scoring — `tools/zero_shot_eval/run_eval.py::evaluate_dataset`

- **Guards:** `freeze.check('protocol')` and `freeze.check('evaluators')` hash the predeclaration, `protocol.py`,
  canonical/*, `masks.py`, both evaluators and `analysis.py`. Any change makes the run refuse.
- **Per sample**, the code loads `CanonicalPrediction` and `load_occany` (label and OccAny's own `pred`).
- **Track P** — `practical/evaluator.py::score_sample` with `practical/masks.py::track_p_masks`:
  - `valid` = label ≠ 255.
  - `region` = valid ∩ horizontal FOV ∩ range ≤ 40 m. The FOV test is z_cam > 0 and 0 ≤ fx·x/z + cx ≤ W, with W = 1226
    (SemanticKITTI) or 1600 (Occ3D). Range = √(x² + z²) in camera coordinates.
  - Range bins: 0–10, 10–20, 20–30 and 30–40 m.
  - Occupancy TP/FP/FN over P and full, each split all / observed / unobserved.
  - Confusions (rows = GT, columns = prediction) for:
    - `ssc` (label where predicted occupied, else empty);
    - `ssc_no_evidence`;
    - the populations `gt_occupied`, `observed_evidence`, `no_evidence`, **`completed_no_evidence` = G ∩ O ∩ ¬E (the
      central metric)** and `geometrically_unobserved` = G ∩ O ∩ ¬(S ∪ F).
  - Pooled confusions by evidence distance (1/2/4/8 m) and by range.
- **Track O** — `official/evaluator.py`:
  - `score_sample_official`: SC TP/FP/FN over valid voxels, plus the SSC confusion over valid voxels.
  - `score_occupancy_official` for OccAny's raw and pooled geometry. Pooled = `gates/gate8b/pooling.geometry_dilation`, a
    3×3×3 max-pool applied before masking.
- **Output:** `artifacts/final_zero_shot/scores/scores_<ds>.npz`. Bootstrap blocks are runs of 25 samples (SemanticKITTI)
  or nuScenes scenes (Occ3D).

### K. Metrics and decision — `tools/zero_shot_eval/analysis.py::analyse`

- **SC IoU** = TP / (TP + FP + FN) from pooled counts (`canonical/metrics.occupancy_scores`).
- **Track P mIoU** (`track_p_miou` / `semantic_summary`): the mean IoU over semantic classes with GT support in the
  population. Pooled over samples, never averaged per sample.
- **Track O mIoU** (`occany_ssc_miou`, OccAny's `SSCMetrics.get_stats` definition): IoU = TP / (TP + FP + FN + 1e-5) over every
  non-empty class, absent classes counting as 0.
  - SemanticKITTI: empty class 0. "Other" = class 20, which does not exist, so all 19 classes are averaged.
  - Occ3D: empty class 17; class 0 "others" is included.
  - A superclass mIoU variant uses OccAny's mapping.
- **B3 − B0:** paired block bootstrap, 2,000 resamples, seed 0. The seed-pooled B3 confusion is compared with 3 × B0.
- **Decision:**
  - PAPER C unless the central Δ has CI > 0 and every seed > 0 on both datasets.
  - Otherwise PAPER A requires Δ ≥ 1.0 and Track O B3 ≥ OccAny's published 7.28 / 6.66.
  - Otherwise PAPER B.
- **Outputs:** `analysis.json`; then `report.write_all()` writes `results_numbers.md`, tables and figures, and `verify.run()`
  writes `verification.md` (9/9).

---

## 4. Executed functions (runtime trace, SemanticKITTI `08_000020`)

The table lists repo functions that actually ran, by stage. A `<module>` entry is only recorded when the module was first
imported inside that stage, so some files import other files without showing a call.

| stage | files → functions |
|---|---|
| samples | `zero_shot_eval/samples.py` evaluated_samples · `occany_comparison/samples.py` build_samples, semantickitti_samples, _occany_dirs, window, protocol_ok · `core/datasets/sources.py` segments, _val_segments, _sk_calib, anchors · `core/datasets/frames.py` read_manifest, frame_key, image_root · `core/datasets/streams.py` build, _segment_of, _order_of · `core/datasets/occ_datasets.py` SemanticKittiOccSpec.build, read_kitti_calib, cam_to_velo · `core/datasets/kitti_odometry.py` parse_calibration · `core/datasets/occupancy.py` VoxelGrid |
| build model (stage 10) | `gates/gate7b/replay.py` build_model · `lingbot_map/models/gct_stream.py` GCTStream.__init__, _build_aggregator, _build_camera_head · `gct_base.py` _build_depth_head · `aggregator/stream.py` _build_blocks, _setup_special_tokens, _init_kv_cache, _init_3d_rope · `aggregator/base.py` _build_patch_embed · `layers/vision_transformer.py` vit_large, init_weights · `layers/{block,attention,mlp,layer_scale,patch_embed,rope,swiglu_ffn,drop_path}.py` constructors · `heads/{camera_head,dpt_head,head_act,utils}.py` · `gates/scale_gate/config.py` |
| infer clip (stage 10) | `occany_comparison/clips.py` infer_clip · `replay.py` load_images, replay_segment, frame_index_trace · `utils/load_fn.py` load_and_preprocess_images, _load_one ×5 · `gct_stream.py` inference_streaming, clean_kv_cache, _aggregate_features · `gct_base.py` forward, _normalize_input, _predict_camera, _predict_depth, _predict_points, _predict_local_points · `aggregator/base.py` forward, _embed_images, _process_frame_attention ×24 · `aggregator/stream.py` _process_global_attention ×24, _process_causal_stream ×24, _get_3d_positions_streaming, _get_flashinfer_manager · `layers/flashinfer_cache.py` FlashInferKVCacheManager: append_frame ×120, _write_patch_page, _write_special_tokens, evict_frames, reset · `layers/rope.py` WanRotaryPosEmbed.forward, apply_rotary_emb ×48 · `layers/attention.py` forward, _apply_kv_cache_eviction_causal · `heads/camera_head.py` forward, trunk_fn · `heads/dpt_head.py` forward, _forward_impl, scratch_forward · `heads/head_act.py` activate_pose, activate_head · `utils/pose_enc.py` pose_encoding_to_extri_intri · `utils/rotation.py` quat_to_mat · `moge_extra.py` resolve_moge_path ×5 · `core/mapping/depth.py` moge_frame_path, unpack_mask · `gates/gate7b/scale.py` frame_candidate ×5 |
| load models | `zero_shot_eval/prediction/pipeline.py` load_models · `zero_shot_eval/config.py` b3_checkpoint · `semantic_completion/frozen_geometry.py` load_frozen_geometry, sha256_file · `semantic_completion/config.py` Paths.geometry_checkpoint · `core/model/net.py` load_checkpoint, CompletionUNet.__init__, _block, n_params · `delayed_semantic/gate16/evaluate.py` load_head ×3 · `delayed_semantic/model.py` EvidenceAssociation.__init__ ×3 |
| predict sample | `pipeline.py` predict_sample, readout ×4, camera_intrinsics · `prediction/grid.py` GridSpec.of, world_keys, subvoxels, reduce_any, reduce_min, voxel_centres, ijk_of · `occany_comparison/geometry.py` WindowClip.load, build_mapper, frame_inputs, hard_channels · `core/mapping/mapper.py` IncrementalMapper.__init__/step ×5/_integrate ×5/query, ScaleState.force/observe, VoxelTable.merge/lookup/_grow/occupied/free, pack · `core/mapping/rays.py` pixel_rays · `frozen_geometry.py` occupancy_and_trunk · `core/model/net.py` CompletionUNet.forward, apply_residual · `prediction/evidence.py` lift_evidence, Evidence.dense, scorer_inputs · `semantic_completion/evidence/lift.py` ray_endpoints · `baseline_reconstruction/current/geometry.py` frame_inputs · `semantic_completion/evidence/trident.py` load_token_map ×5, sample_at_lattice ×5 · `prediction/semantics.py` nearest_evidence, b3_features ×3, association_forward ×39, path_fractions ×39 · `delayed_semantic/model.py` nearest_candidates ×39 · `occany_comparison/scoring.py` load_occany · `core/datasets/sources.py` _sk_calib |
| score sample | `occany_comparison/scoring.py` load_occany, pool · `practical/evaluator.py` score_sample, occupancy_block, region_mask, score_occany_geometry · `practical/masks.py` track_p_masks, camera_coordinates, voxel_centres_grid, field_of_view, populations · `canonical/metrics.py` confusion ×96, OccupancyCounts.of · `canonical/prediction.py` dense_labels · `official/evaluator.py` score_sample_official, score_occupancy_official · `gates/gate8b/pooling.py` geometry_dilation |

These were not traced; they are read from source in § 3:

- `analysis.analyse`: `load_scores`, `semantic_tables`, `occupancy_tables`, `official_tables`, `bins`, `consistency`,
  `block_bootstrap`, `track_p_miou`, `canonical/metrics.{semantic_summary, occany_ssc_miou, superclass_confusion, occupancy_scores}`.
- `report.write_all` and `verify.run`.
- `freeze.{freeze, check}`.
- The Trident-env teacher.

Also not executed: `IncrementalMapper`'s semantic path (`sem_into=None`), `Completer.raw`, `head_sem`'s output,
`gates/gate8c1/occany_eval.official_sc_counts` (the OccAny-checkout scorer; it is imported but not called in stage 11), and
`EvidenceAssociation.forward` (`association_forward` is a grid-generic copy of it, tested equal on KITTI-360).

---

## 5. Tensor representations at the boundaries

| artifact | where | content |
|---|---|---|
| frame list | `$ZS/frames_<ds>.json` | `{key, path}` of history frames |
| Trident tokens | `$ZS/trident_tokens/<ds>/<key>.npz` | `tokens [h·w, 1024] fp16` unit, `grid_hw`, `native_hw`, coarse-reproduction stats |
| text embeddings | `$ZS/text_embeddings_<ds>.npz` | `text [C, 1024]`, `labels [C]`, `phrases`, `logit_scale = 40` |
| causal clip | `$OC/clips/<ds>/causal/<sample>.npz` | `pred_depth [5, H, W] fp16` canonical z, `pred_depth_conf [5, H, W] fp16`, `pred_pose_c2w [5, 4, 4] f32` canonical c2w (frame 0 = origin), `pred_K [5, 3, 3]` lattice, `proc_hw`, `log_s_frames [5]`, `scale` m/unit, `anchor_slot = 4`, `keys`, `positions` |
| MoGe frame | `lingbot_gate7b/moge_b/<ds>/<key>.npz` (or `$OC/moge_b`) | `moge_depth [H, W] fp16` metric z, `moge_mask` packed, `fov_x_deg` |
| S, F | in memory | `[N_pred]` bool |
| O, trunk | in memory | `[N_pred]` bool; `[24, X, Y, Z]` fp32 (bf16 forward) |
| evidence | in memory | `flat [M]`, `feature [M, 1024]`, `count [M]`, `frame_bits [M]` |
| canonical prediction | `$ZS/predictions/<ds>/<sample>.npz` | § I table (SemanticKITTI N = 2,097,152; Occ3D N = 640,000) |
| OccAny output (labels + geometry) | `occany_out/ssc_voxel_pred/<run>/<sample>/voxel_predictions.pkl` | `voxel_label` (255 = ignore), `render_recon_gen_recon{2.5,1.1}_gen{2.5,1.1}`, `T_cam_to_voxel` |
| scores | `artifacts/final_zero_shot/scores/scores_<ds>.npz` | `ps__occ__{P,full}__{all,observed,unobserved}` `[n, 3]`, `ps__conf__{P,full}__<pop>__<method>` `[n, C, C]`, `ps__official_*`, `ps__occany_{raw,pooled}__*`, `pool__{dist,range}__<method>`, `sample_id`, `segment`, `block` |

`$ZS = /media/SSD1/MINH_DATASETS/lingbot_zero_shot_eval` (`ZERO_SHOT_EVAL_DATA_ROOT`),
`$OC = /media/SSD1/MINH_DATASETS/lingbot_occany_comparison` (`OCCANY_COMPARISON_DATA_ROOT`).

---

## 6. Configuration dependencies

**Frozen hyperparameters on the path** (none are tuned on target data):

| where | value |
|---|---|
| `gates/gate7b/replay.py` | resolution 518, patch 14, 5 scale frames, KV window 64, `max_frame_num` 1024, bf16, keyframe interval 1, FlashInfer (`use_sdpa=False`) |
| `core/mapping/depth.py` | `CONF_THRESHOLD` 1.5, depth 1–60 m |
| `gates/gate7b/scale.py` | MAD clip 3, ≥ 1000 valid pixels; clip scale = `exp(nanmedian)` over the 5 frames |
| `core/mapping/{mapper,rays,voxmap}.py` | 0.2 m keys, band ±0.2 m, carve stride 4 from 1.0 m, L_occ +0.85, L_free −0.40, clamp ±4, occupied at > 0 |
| `core/model/net.py`, `baseline_reconstruction/config.py` | `CompletionUNet` cin 3, width 24; `OCCUPANCY_THRESHOLD` 0 |
| `delayed_semantic/config.py::ModelConfig` (in checkpoint) | K 16, projection 16, hidden 64, path samples 8, range 6 m |
| `zero_shot_eval/prediction/semantics.py` | `CHUNK` 16,384 |
| `zero_shot_eval/config.py` | `GEOMETRY_SEED` 0, `B3_SEEDS` (0, 1, 2) |
| `zero_shot_eval/practical/protocol.py` (hash-frozen) | 40 m, FOV by native width, range bins, evidence-distance bins, class sets, OccAny "other" handling, bootstrap 2,000 / seed 0 / 25-sample or per-scene blocks, decision thresholds |
| `occany_comparison/config.py` | causal offsets (−4..0), OccAny run directories and prediction keys, expected sample counts |
| `core/datasets/{occupancy,occ3d_grid,kitti360,grids}.py` | grids, the Occ3D any-subvoxel reduction |
| `core/datasets/vocab.py` | class lists and phrases |

**Freeze records the code checks at run time:**

- `artifacts/final_zero_shot/frozen_{protocol,evaluators}.json`
- `artifacts/baseline_reconstruction/code_freeze.json` (geometry checkpoint sha256)
- `artifacts/gate16_candidate_selection/code_freeze.json` (B3 checkpoints; `verify` check 2)

**Environment:**

- Variables: `PYTHONPATH=$REPO`, `HF_HUB_OFFLINE=1`, `CUDA_HOME=/usr/local/cuda-12.8`, `FLASHINFER_CUDA_ARCH_LIST=12.0a`.
  FlashInfer JIT-compiles kernels; sm_120 needs CUDA 12.8 plus this arch string.
- Optional data-root overrides: `ZERO_SHOT_EVAL_DATA_ROOT`, `OCCANY_COMPARISON_DATA_ROOT`, `GATE6_SAM_CKPT`.

**External code and weights:**

- `third_party/Trident` + `trident_env` + `third_party/checkpoints/sam_vit_h_4b8939.pth`, for step B only.
- `checkpoints/lingbot-map/204754b/lingbot-map.pt`.
- MoGe-2 at revision `39c4d5e9…` (`moge_gauge/`, `configs/gate5/moge_metric_gauge.yaml`), needed only when `build_moge.py`
  must fill frames. Stage 11 reads cached MoGe.
- OccAny's checkout and outputs: only its stored predictions and labels are read in stage 11. `independent_label` (in
  `verify`) reads the learning map at `/home/minh/workspace/OccAny/occany/datasets/semantic_kitti.yaml`.

**Datasets:**

- `data/kitti/dataset/sequences/08` (`image_2`, `calib.txt`, `voxels/`)
- `/media/SSD1/MINH_DATASETS/nuscenes` (`CAM_FRONT`, `occ3d_gt/Occupancy3D-nuScenes-trainval/annotations.json`)
- Manifests: `artifacts/scale_gate/manifests/val.jsonl`, `manifests/occ3d_zeroshot/val.jsonl`

---

## 7. Minimal working baseline — the files that constitute it

Sources: the runtime trace (§ 4) plus the import closure of every stage-10/11 entry module (131 repo files, measured with
`sys.modules`), plus the teacher script. All 133 are tracked in git.

**(a) Code executed on the path (67 files).**

| location | files |
|---|---|
| `tools/zero_shot_eval/` | `config.py`, `samples.py`, `prediction/{pipeline,grid,evidence,semantics}.py`, `canonical/{prediction,metrics}.py`, `practical/{protocol,masks,evaluator}.py`, `official/evaluator.py`, `run_eval.py`, `analysis.py`, `freeze.py`, `report.py`, `verify.py`, `teacher/extract_tokens.py` |
| `tools/occany_comparison/` | `config.py`, `samples.py`, `clips.py`, `geometry.py`, `moge_extra.py`, `scoring.py`, `cli/build_clips.py` |
| `tools/semantic_completion/` | `config.py`, `frozen_geometry.py`, `evidence/lift.py`, `evidence/trident.py` |
| `tools/delayed_semantic/` | `model.py`, `config.py`, `gate16/evaluate.py` (`load_head` only) |
| `tools/baseline_reconstruction/` | `config.py`, `current/geometry.py` (`frame_inputs` only) |
| `core/` | `mapping/{mapper,rays,depth,voxmap,scale}.py`, `model/net.py`, `datasets/{sources,frames,streams,occ_datasets,occupancy,kitti_odometry,nuscenes,grids,occ3d_grid,kitti360,vocab}.py` |
| `gates/` | `gate7b/{replay,scale,depth}.py`, `gate8b/pooling.py`, `scale_gate/config.py`, `gate6/trident_adapter.py` (teacher env) |
| `lingbot_map/` | `models/{gct_stream,gct_base}.py`, `aggregator/{base,stream}.py`, `layers/{attention,block,flashinfer_cache,rope,vision_transformer,patch_embed,mlp,layer_scale,swiglu_ffn,drop_path}.py`, `heads/{camera_head,dpt_head,head_act,utils}.py`, `utils/{load_fn,pose_enc,rotation,geometry}.py` |

**(b) Import-only.** These are needed for the imports to resolve but none of their functions run in stage 11:

- `core/supervision/targets.py` (`N_INPUT_CHANNELS` for `net.py`), `core/datasets/union_vocab.py` (`V8.U`)
- `core/datasets/{audit,kitti360_folds,kitti360_splits,kitti360_transforms}.py`, `core/mapping/feed.py`,
  `core/evaluation/official_targets.py`
- `gates/gate8c1/occany_eval.py`
- `tools/delayed_semantic/{memory,regions,stream}.py`, `tools/delayed_semantic/gate16/{config,context,candidates,candidate_metrics,candidate_regions,candidate_targets}.py`
- `tools/semantic_completion/{dataset,model/propagation_head}.py`, `tools/semantic_completion/evaluation/{baseline_nearest,evaluate,regions,semantic_metrics}.py`,
  `tools/semantic_completion/evidence/build.py`
- the `__init__.py` files

The `gate16/evaluate.py` import of `load_head` alone pulls in most of the `delayed_semantic` and `semantic_completion`
evaluation stack. `verify.py` pulls in more.

**(c) Frozen weights and records:**

- `checkpoints/lingbot-map/204754b/lingbot-map.pt`
- `artifacts/baseline_reconstruction/checkpoints/geo_seed0_step06000.pt` and `code_freeze.json`
- `artifacts/gate16_candidate_selection/checkpoints/B3_seed{0,1,2}_step03000.pt` and `code_freeze.json`
- `artifacts/final_zero_shot/{practical_evaluation_predeclaration.md, frozen_protocol.json, frozen_evaluators.json}`

**(d) Data caches** (for a re-score; a from-scratch rebuild regenerates the first three):

| cache | size / count |
|---|---|
| `$ZS/{trident_tokens, text_embeddings_*, frames_*}` | |
| `$OC/clips/<ds>/causal` | 787 MB SemanticKITTI, 6.3 GB Occ3D |
| `lingbot_gate7b/moge_b` + `$OC/moge_b` | |
| `$ZS/predictions` | 621 MB, 801 + 4,219 files |
| OccAny `ssc_voxel_pred` runs (labels + geometry) | |
| the manifests and datasets in § 6 | |

`$ZS/forward` (8.5 GB) is not read by the current system.

**Not part of the working baseline, only provenance:**

- stages 1–9: KITTI-360 caches, targets, geometry training (`tools/baseline_reconstruction/cli/train_current.py`),
  Trident evidence for training, delayed targets, Gate 16 training (`tools/delayed_semantic/gate16/{train,selection_loss,…}.py`),
  OccAny inference
- the gate packages under `gates/` other than those listed above
- `demo*.py`, `demo_render/`, `semantic/`

---

## 8. Quirks worth knowing before a deterministic trace

- **Evidence summation** uses CUDA `index_add_` on float32 (atomic adds, order not guaranteed). The
  KITTI-360 test therefore compares features with a 1e-3 tolerance. In every rerun observed (SemanticKITTI
  `08_000020`, KITTI-360 0006 / t = 4, and `verify` check 8 on four samples), the stored outputs and labels came out
  bit-identical. Treat label-level identity as the guarantee and feature-level identity as empirical.
- **The official label volume is read inside prediction** (step I), to choose the query voxels. It is not leakage, since
  each voxel's label is computed independently, but the prediction files are not label-free artifacts.
- **Track P mIoU vs Track O mIoU.** Track P averages only classes with GT support; Track O includes absent classes as 0.
  Track O SSC labels a voxel only where it is predicted occupied, so the same canonical file yields different SSC numbers
  in the two tracks.
- **Duck-typed clip.** `ray_endpoints` calls the baseline's `frame_inputs` with a `WindowClip`. That works because the
  baseline function anchors on slot −1, and `anchor_slot` is asserted to be 4.
- **B3's height feature** uses the target grid's own origin and extent: SemanticKITTI origin z −2, Occ3D −1, both with a
  6.4 m extent.
- **Two OccAny paths.** `OCCANY_SK_LEARNING_MAP` (`/home/minh/workspace/OccAny/…`) is not the `../third_party/OccAny` that
  `reproduce.sh` names. Only `verify`'s independent-label check reads it.

---

## 9. KITTI-360 anchor for a deterministic end-to-end trace

**Recommendation: drive `2013_05_28_drive_0006_sync`, anchor stream index `t = 4` (native frame `0000000137`).**

**Why this anchor:**

- **Development drive.** 0006 is the dev / evaluation drive: never trained on, human-labelled via SSCBench, and in Gate 15/16
  evaluation (590 anchors, stride 3).
- **First anchor.** Its window `t−4..t` is stream indices 0..4, native frames 117, 122, 127, 132, 137. No earlier frame
  exists, so the history is unambiguous.
- **Pinned by a test.** It is the anchor of `tests/zero_shot_eval/test_kitti360_reproduction.py`, which asserts that the
  grid-generic code (`zero_shot_eval.prediction.*`, the exact code that produced the target-domain numbers) reproduces the
  frozen Gate 15 evidence, Gate 16 provenance, B0 and **B3_s0 labels bit for bit**. That test passed today.
- **Every cache it needs exists:**

  | cache | path |
  |---|---|
  | raw PNGs | `/media/welf/MINH/datasets/kitti360/KITTI-360/data_2d_raw/2013_05_28_drive_0006_sync/image_00/data_rect/0000000{117..137}.png` |
  | clip | `/media/SSD1/MINH_DATASETS/lingbot_local_registration_audit/local_lingbot/2013_05_28_drive_0006_sync/00004.npz` |
  | hard map | `/media/SSD1/MINH_DATASETS/lingbot_gate12_suf/inputs/2013_05_28_drive_0006_sync/00004.npz` |
  | tokens | `/media/SSD1/MINH_DATASETS/lingbot_feature_first_3d/trident_tokens/kitti360/0000000{117,122,127,132,137}.npz` |
  | evidence | `/media/SSD1/MINH_DATASETS/lingbot_gate15_semantic/evidence/2013_05_28_drive_0006_sync/00004.npz` |
  | provenance | Gate 16 provenance file |
  | evaluation mask | `/media/SSD1/MINH_DATASETS/lingbot_k360_7drives/eval_masks/2013_05_28_drive_0006_sync/00004.npz` (`has_label = True`) |
  | Gate 16 prediction record | `{O, label_B0, label_B3_s{0,1,2}, …}` |

**Rebuilt from raw images in this session** (scratchpad script, nothing written to the repo or the data roots):

| stage | result |
|---|---|
| LingBot clip (`replay.replay_segment`, 5 frames, lattice 140×518) | `pred_depth`, `pred_depth_conf`, `pred_pose_c2w`, `pred_K`, `proc_hw` **identical** to the cached clip; `rope_slots_used` 0 |
| MoGe clip scale | 15.516587677725116 m/unit, **identical** (per-frame `log_s` 2.734–2.751) |
| hard map (`build_mapper` + `hard_channels`, `T = rect_cam_to_velo`) | `S` and `F` **identical** to the cached `S_H` / `F_H` |
| geometry seed 0 | `O` **identical** to the Gate 16 record |

**Anchor counts:**

| quantity | count |
|---|---:|
| S | 27,608 |
| F | 86,415 |
| O | 318,417 |
| evidence voxels | 18,536 |
| GT semantic-occupied | 8,483 (12 classes) |
| O ∩ GT | 4,746 |
| O ∩ GT ∩ ¬evidence | 4,174 |

**Anchor accuracy** on O ∩ GT ∩ ¬evidence: B0 21.6 %, B3 23.6 / 24.9 / 23.1 % (seeds 0 / 1 / 2). This is the same
qualitative picture as the headline results.

**Caveat:** this anchor has about half the labelled voxels of typical 0006 anchors (8.5 k vs about 15 k at t = 7, 10 and 889),
and sidewalk dominates them (3,816). If a trace needs richer class coverage, t = 7 also has every cache. It is not pinned by
a test, though.

**How to trace it with the current system's code:** `tests/zero_shot_eval/test_kitti360_reproduction.py` is the
ready-made harness. It needs `CUDA_VISIBLE_DEVICES` set to a free GPU, `PYTHONPATH=$REPO`, and
`python -m pytest tests/zero_shot_eval -q`. Its steps:

1. `LocalClip.load`, then `load_hard_map` (or `build_mapper` / `hard_channels` from the clip).
2. `lift_evidence`, `nearest_evidence`, `occupancy_and_trunk`.
3. `b3_features` with `B3_seed0_step03000.pt`.
4. Compare with the frozen record.

For the step before that (raw PNGs → clip), use `gates.gate7b.replay.build_model` / `load_images` / `replay_segment`, plus
`gates.gate7b.scale.frame_candidate` on `gates.gate7b.depth.load_moge_frame("kitti360", key, moge_frame_index(...))`.
That is what `tools/local_registration_audit/run_local_lingbot.py` does.
