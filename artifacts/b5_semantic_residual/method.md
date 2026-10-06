# B5 semantic residual: what the model receives, what it changes, and where the code is

Written before any training. Every claim here was checked against the code or the caches on this machine; the checks
are named so they can be repeated.

## 1. The causal state at time t

One anchor is a stream index `t` on a KITTI-360 drive. **One stream frame is 0.520 s**, not 0.1 s: the LingBot stream
subsamples KITTI-360's 10 Hz by about 5x. Measured, not assumed — the modal gap between distinct `first_dt_s` values in
the frozen delayed-target caches is 0.520 s on all four drives (0003, 0007, 0010, 0006), and the minimum is also 0.520 s.
Anything that converts frames to seconds at the native rate is wrong by 5x.

The student may see only frames `t-4..t` (`tools/delayed_semantic/stream.py::CausalHistory`, which raises `FutureAccess`
for any index above `t`).

| tensor | shape | frame | produced by |
|---|---|---|---|
| hard map `S`, free map `F` | `[256,256,32]` bool | velodyne frame of `t`, 0.2 m voxels | `core/mapping/mapper.py` via LingBot depth + MoGe-2 clip scale |
| occupancy `O` | `[256,256,32]` bool | same | frozen CompletionUNet, geometry seed 0 |
| trunk feature | `[24,256,256,32]` | same | frozen CompletionUNet decoder `d1` (`tools/semantic_completion/frozen_geometry.py`) |
| evidence `flat_index`, `feature`, `count` | `[E]`, `[E,1024]` fp16, `[E]` | same | Gate 15 lift of Trident-H tokens, cached at `lingbot_gate15_semantic/evidence/` |

`feature` is a **unit vector in the raw Trident-H space** — the same space the 18 frozen text embeddings live in.
Verified from the cache: `feature` is `float16 [E, 1024]`.

## 2. B0, the thing being corrected

B0 assigns every query voxel the feature of its **nearest evidence voxel** by 3D Euclidean distance transform
(`tools/zero_shot_eval/prediction/semantics.py::nearest_evidence`), then reads out

    label = argmax( normalize(feature) @ text.T )        # tools/zero_shot_eval/prediction/pipeline.py::readout

If there is no evidence anywhere in the anchor, no identity can be transferred and every method returns the empty class
(`tools/like_for_like/predict.py`, the `len(ev.flat) == 0` branch). B5 preserves that fallback by construction (§4).

## 3. B4 does **not** already do this

The execution brief asks this to be checked before anything is named B5. It was, and it does not.

`tools/semantic_field/model/semantic_field_head.py` predicts the feature outright. Its own docstring:

> The semantic feature of any voxel is `normalize(Linear(32 -> 1024)(hidden[v]))` in the RAW teacher space:
> **a free vector, not a combination of input features.**

- `semantic_head` is a plain `nn.Linear`, default initialisation — **not** zero-initialised.
- Its `ResidualBlock`s are internal convolutional residuals (`x + body(x)` inside the UNet). They give depth, not a
  path from B0 to the output. The brief explicitly warns that these do not count, and they do not.
- B0's nearest-evidence feature is never given to the network at a query voxel at all: evidence enters as a 32-d
  projection **only at evidence voxels**, zero elsewhere.

So at a voxel with no evidence, B4 must synthesise a 1024-d feature from scratch. B5 changes exactly that.

## 4. What B5 changes

    corrected = b0_feature + valid * residual_head(context)
    prediction = normalize(corrected) @ text.T           # frozen readout, unchanged

`tools/semantic_residual/model.py`. Properties, each asserted in `tests/semantic_residual/test_model.py`:

- **Starts exactly at B0.** `out.weight` and `out.bias` are zero-initialised, so the initial residual is zero and
  `normalize(b0 + 0) == b0` (b0 is already unit). Measured max deviation at initialisation: **1.5e-8**.
- **Only the output projection is zeroed.** The body keeps normal initialisation. A known consequence: on step 0 only
  `out` receives gradient; after one step every parameter trains, and the loss on a fixed batch decreases.
- **The `valid` gate** means a voxel with no nearest evidence keeps B0's behaviour even after training, so the
  no-evidence fallback survives.
- **Geometry and occupancy are untouched.** B5 changes only the feature at query voxels.

Context, all causal, per query voxel (`QueryContext`): the B0 feature (projected to 64-d, the feature itself stays
full rank), the frozen trunk feature (24), `S/F/O` (3), and nearest-evidence geometry — normalised displacement (3),
distance (1), validity (1), `log1p(count)/5` (1).

**Where spatial context enters.** The head is a per-voxel MLP; it mixes nothing across voxels. All spatial information
arrives through `trunk`, whose receptive field is the frozen CompletionUNet's (three resolutions over the
256x256x32 grid). This is deliberate for a first pilot: no attention and no new geometry backbone, so a positive
result cannot be ambiguous about where the information came from. 226,752 parameters.

## 5. Supervision, and why only +1 and +5

Targets are later observations of the same surface, attached only where source geometry and visibility establish the
match. This pilot admits future offsets **+1 and +5 stream frames = 0.52 s and 2.60 s**, and nothing longer.

That restriction is not a guess. The frozen registration gate (G1: point-to-plane residual of a future sweep moved into
the anchor grid by GT poses; thresholds median <= 0.10 m, p90 <= 0.20 m) was measured at offsets 1, 5, 10, 20 in
`artifacts/delayed_semantic_pilot/reliability/reliability.json`:

| offset | 0003 | 0007 | 0010 | 0006 (dev) | verdict |
|---|---|---|---|---|---|
| +1 (0.52 s) | 0.028 / 0.071 | 0.030 / 0.079 | 0.026 / 0.069 | 0.025 / 0.072 | **pass everywhere** |
| +5 (2.60 s) | 0.043 / 0.124 | 0.041 / 0.123 | 0.065 / 0.143 | 0.061 / 0.156 | **pass everywhere** |
| +10 (5.2 s) | 0.086 / 0.180 | 0.045 / 0.137 | 0.076 / **0.207** | 0.090 / **0.217** | fails per drive |

(median / p90 in metres.) +10 fails on 0010 and on the development drive even though the training-pooled value, 0.184 m,
passes — the pooled reading hides it. **This restriction is this pilot's own.** It does not repair or supersede the
BLOCKED B4 scaling study, and it licenses no claim about longer reveal delays.

The existing tier rule needs returns spanning >= 1 s (`persistence_min_span_s`). At 0.52 s per frame, +1 and +5 span
2.08 s, so the rule is satisfiable inside this window — which it would not be at the native 10 Hz. This is the concrete
reason the cadence had to be measured rather than assumed.

## 6. One complete sample

1. At `t`, freeze the inputs: `S`, `F`, `O`, trunk, evidence (`flat_index`, `feature`, `count`).
2. Query voxels := predicted occupied (`O`) **and** not an evidence voxel. Defined before any future data is touched.
3. For each query voxel, B0 looks up its nearest evidence voxel and takes that feature.
4. `ResidualHead` adds a correction conditioned on the context in §4. Nothing in that context comes from after `t`.
5. Later, frames `t+1` and `t+5` observe some of those voxels. Where the match is valid, their frozen Trident-H
   feature becomes the target. **Future data selects loss locations and supplies targets; it never enters the input.**
6. Loss: cosine between the normalised prediction and the detached future feature.
7. At evaluation the same frozen readout turns the corrected feature into one of 18 classes.

## 7. Status of the target cache

The frozen delayed targets store `prob [18] fp16` — **class distributions, not teacher features** (verified: the keys
are `flat_index, prob, n_views, first_dt_s, second_dt_s, view0_top1, view1_top1, tier, in_evidence, hist_seen,
hist_top1, n_future_sweeps`). A cosine feature loss needs the 1024-d token, so B5 builds its own target cache. Class
probabilities are never turned back into feature vectors.

Building it re-runs the frozen projection (LiDAR sweep of the future frame -> future camera -> bilinear Trident-H
token), which needs KITTI-360 `data_3d_raw` and `data_2d_raw`. See `BLOCKED.md` for the current availability problem.
