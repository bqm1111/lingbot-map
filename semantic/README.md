# Open-vocabulary semantic extension

Turns a LingBot-MAP reconstruction into a **queryable 3D map**: fuse CLIP-aligned
features from every view into a voxel field, then ask it for things in free text
("a car", "stairs", "a lamp post") and get back 3D locations and per-pixel overlays.

No training and no labelled data — it reuses the geometry the model already predicts.

## How it works

1. **Dense CLIP features** (`dense_clip.py`). A stock CLIP ViT only produces one
   text-aligned vector per image. `DenseCLIP` rewrites the final attention block
   MaskCLIP-style — the value projection is applied per token and pushed straight
   through `ln_post` and the visual projection — yielding a **per-patch** feature in
   the same space as `encode_text`.
2. **Lifting** (`feature_field.py`). Each pixel's feature is unprojected to world
   space using the predicted depth + pose from the saved NPZs, and accumulated into
   a sparse voxel hash (packed int64 keys, running mean per voxel).
3. **Compression**. Storing 512-d float32 per voxel would cost gigabytes. Features
   are projected onto a 64-d **orthonormal, uncentered** basis fitted on a sample of
   frames before fusion. Because the basis is orthonormal, norms are preserved and
   cosine similarity survives the projection: `f·t ≈ (Pf)·(Pt)`. A text query is
   embedded once, projected once, and compared directly against stored voxels.
4. **Querying** (`query_field.py`). Raw CLIP cosines sit in a narrow band (~0.2) and
   are not comparable across prompts, so the default score is LERF-style
   **relevancy**: how much better a voxel matches the query than it matches generic
   background prompts ("object", "things", "stuff", …).

## Usage

Build a field from an existing `--save_predictions` run:

```bash
python -m semantic.build_field \
  --predictions output/kitti_seq08_render/image_2 \
  --output output/kitti_seq08_semantic/field.npz \
  --sky_mask_dir output/kitti_seq08_sky_masks \
  --pixel_stride 2 --grid_resolution 4096 --min_count 2
```

Query it — point clouds and/or overlay videos:

```bash
# Open-vocabulary relevancy, one video per query
python -m semantic.query_field \
  --field output/kitti_seq08_semantic/field.npz \
  --queries "a car" "a traffic sign" \
  --output_dir output/kitti_seq08_semantic \
  --predictions output/kitti_seq08_render/image_2 \
  --render_video --export_ply

# Closed-set segmentation (softmax across the given labels)
python -m semantic.query_field \
  --field output/kitti_seq08_semantic/field.npz \
  --queries "the road" "a car" "vegetation" "a building" \
  --mode softmax --temperature 0.05 \
  --output_dir output/kitti_seq08_semantic \
  --predictions output/kitti_seq08_render/image_2 --render_video
```

Building the field is cheap because inference is already done: **~50 s for the full
4071-frame KITTI sequence**, and querying/rendering runs at ~220 fps.

## Tuning that actually matters

| Knob | Why you'd touch it |
|---|---|
| `--grid_resolution` | Voxels across the scene's longest axis. Too low and a single voxel merges road + car; on KITTI seq 08, 1024 gave ~3 m voxels (useless) and 4096 gave ~0.8 m (good). |
| `--pixel_stride` | **Controls render coverage, not just speed.** Voxels only exist where fused samples landed, so a sparse fusion leaves most pixels with no voxel to look up. Stride 4 gave a 27 % per-pixel hit rate and a speckled overlay; stride 1–2 gives 90 %+. |
| `--display_lo/--display_hi` | Relevancy is heavily skewed — usually only the top few percent of voxels match. These percentiles stretch the visible band; a fixed threshold either shows nothing or everything. |
| `--min_count` | Drops voxels seen in fewer than N frames, which removes most moving-object smear and depth noise. |

## Evaluation

`eval_semantickitti.py` scores the field against SemanticKITTI ground truth
(sequence 08 is the official validation split):

```bash
python -m semantic.eval_semantickitti \
  --field output/kitti_seq08_semantic/field.npz \
  --predictions output/kitti_seq08_render/image_2 \
  --kitti_root data/kitti/dataset --sequence 08 \
  --frame_stride 5 --report_geometry --baseline_2d \
  --output output/kitti_seq08_semantic/eval.json
```

**Protocol is image-space, deliberately.** Aligning a monocular map to the LiDAR
frame with a Sim(3) fit would make global scale error and kilometre-scale drift
contaminate the semantic score. Instead each labelled LiDAR point is projected
into its own camera with the true KITTI calibration and the field is read at that
pixel, so drift cancels and what is measured is semantics. (The projection is
verified against the dataset's own depth maps: median 0.16 m agreement.)

### Results — sequence 08, 815 frames, 19 classes, zero-shot

| Metric | Value |
|---|---|
| **mIoU** | **15.99 %** |
| mIoU, raw per-frame 2D CLIP (same points) | 16.97 % |
| mIoU, restricted to depth-accurate pixels (<10 % rel. error) | 17.99 % |
| Point accuracy | 38.41 % |
| Coverage | 80.90 % |

Best classes: car 63.3, vegetation 49.2, building 45.9, road 39.5. Worst: person
0.3, bicyclist 0.02, other-ground 0.3, parking 1.5 — rare, thin, or not visually
separable from their surroundings by CLIP (sidewalk vs. road, parking vs. road).

### Semantic Scene Completion (`eval_ssc.py`)

The standard SemanticKITTI SSC benchmark: a 256×256×32 grid at 0.2 m in each
frame's velodyne coordinates, evaluated where `.invalid == 0`. Ground-truth
occupancy is `label > 0` — the `.bin` file is the sparse single-scan *input*, not
the target.

```bash
python -m semantic.eval_ssc \
  --field output/kitti_seq08_semantic/field_calib.npz \
  --predictions output/kitti_seq08_render/image_2 \
  --kitti_root data/kitti/dataset --sequence 08 \
  --frame_stride 20 --scale_mode per_frame
```

| | SC IoU | precision | recall | SSC mIoU |
|---|---|---|---|---|
| predicted intrinsic | 17.15 % | 37.5 % | 24.0 % | 3.67 % |
| **true calibration** | **17.84 %** | 38.4 % | 25.0 % | **4.02 %** |

**These are not comparable to MonoScene or VoxFormer, and shouldn't be quoted as
if they were.** SSC asks a method to hallucinate occupancy it cannot see; this
field only stores observed surfaces, so the recall term (25 %) is largely
measuring a capability the system does not have and was never built to have. The
numbers are a lower bound and a diagnostic.

Getting even this far required fixing two geometric issues that dominated the
metric, both worth knowing about:

- **Monocular scale drift is severe.** Fitting one global depth scale gave SC IoU
  7.2 %. Fitting per frame against that frame's LiDAR — the standard monocular
  protocol — more than doubled it to 17.2 %. The fitted scale ranges from **16.1
  to 27.5** across the sequence, a 1.7× drift, so no single scale can place the
  map correctly along a 3 km run.
- **The predicted focal length is ~5 % short** of the true calibration (fx 283 vs
  299), which stretches the reconstruction laterally. Measured per-axis, metric
  scale came out 21.2 lateral vs 22.8 in depth — an anisotropic distortion, not a
  pure scale. Rebuilding with `--intrinsic` from `calib.txt` fixes it and is
  worth ~0.7 SC IoU, but note it *lowers* coverage (81 % → 68 %), because a
  geometrically tighter map concentrates into fewer voxels.

### The finding that matters

**3D fusion does not beat the 2D baseline — it costs ~1 mIoU point.** Two
follow-ups locate the cause:

- *Not voxel quantization.* Rebuilding 4× finer (0.32 m instead of 0.78 m
  voxels, 37 M voxels) left mIoU flat at 15.91 % while coverage collapsed from
  81 % to 57 %, and the deficit versus 2D widened to −1.75.
- *Partly depth error.* Gating to geometrically accurate pixels recovers ~2
  points, so predicted-depth error accounts for roughly that much.

What remains is the fusion rule itself: averaging L2-normalized CLIP features
over many viewpoints (varying distance, angle, occlusion) is less discriminative
than one good view. Weighted or best-view fusion — weighting by depth
confidence, viewing distance, or surface obliquity instead of a uniform mean —
is the obvious next experiment.

So the honest case for the 3D field is *not* per-frame accuracy. It is that you
get a persistent, queryable map with one label per surface location, which
per-frame 2D inference does not give you at any accuracy. The metric that would
show that advantage is temporal label consistency, which this mIoU does not
measure.

### Component checks

- **Text alignment.** Dense "sky" similarity scored 0.837 mean ROC-AUC against
  the pipeline's independent ONNX sky masks.
- **Geometry.** GPU unprojection matches `depth_to_world_coords_points` to 1e-6;
  voxel hashing round-trips exactly.
- **Spatial sanity.** Projected back into image space, top relevancy pixels land
  correctly by height: road 0.81, car 0.72, vegetation 0.54, building 0.35
  (0 = image top, 1 = bottom).

## Limitations

- Features are per-patch (16 px), so boundaries are coarser than the depth map;
  small or distant objects blur into their surroundings.
- CLIP relevancy is uncalibrated across prompts. Rare classes ("a traffic sign")
  score low, and vague prompts ("a person") can fire broadly. Compare within a
  query, not across queries, unless using `--mode softmax`.
- Fusion is a running mean, so **moving objects smear** along their path.
  `--min_count` suppresses the worst of it.
- The field is built offline from saved NPZs. The model's streaming/causal
  property is not exploited yet — see the "native semantic head" option for the
  version that would be.
