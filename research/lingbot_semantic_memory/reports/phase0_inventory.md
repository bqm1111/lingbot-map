# Phase 0 — inventory, capture points and smoke test

Everything below was measured on this machine, not read off documentation. The
reproducing command is in §7.

## 1. Repository and environment

| | |
|---|---|
| repo | `/home/minh/workspace/lingbot-map_fork`, branch `dev`, HEAD `4071d1e` |
| working tree at Phase 0 | clean (`git status --porcelain` empty) |
| python | 3.10, conda env `cu128` at `/home/minh/anaconda3/envs/cu128/bin/python` |
| torch | 2.7.1+cu128, CUDA 12.8 |
| GPUs | 4 × NVIDIA RTX PRO 6000 Blackwell Max-Q, 97 887 MiB each (GPU 1 was busy; this study uses GPU 0) |
| required env | `CUDA_HOME=/usr/local/cuda-12.8`, `FLASHINFER_CUDA_ARCH_LIST="12.0a"`, `PYTHONPATH=<repo root>` |

`PYTHONPATH` is not optional: the editable install of `lingbot-map` points at a
directory that no longer exists, so imports only resolve with the repo root on
`sys.path`.

## 2. Checkpoints

| artefact | path | note |
|---|---|---|
| LingBot-MAP | `checkpoints/lingbot-map/204754b/lingbot-map.pt` | loads with **0 missing / 0 unexpected** keys; 1 157 943 540 parameters, **0 trainable** after freezing |
| variants (unused) | `lingbot-map-stage1.pt`, `lingbot-map-long.pt` | present, not used |
| external DINO teacher | `~/.cache/torch/hub/checkpoints/dinov2_vitb14_reg4_pretrain.pth` | DINOv2 ViT-B/14 **with registers**, 86 583 552 parameters, loads offline via `torch.hub.load(..., source="local")` with `<All keys matched successfully>` |
| other local DINO weights | `dinov2_vitl14_pretrain.pth`, `dinov2_vitb14_pretrain.pth`, `dino_vitbase16_pretrain.pth`, `timm/vit_base_patch16_dinov3.lvd1689m` | available; ViT-B/14-reg chosen because patch size 14 makes the teacher grid identical to LingBot's |

**No text-to-DINO bridge exists in this repository.** Every hit for
`dino`/`dinov2` is LingBot's own patch embedder. The only text bridge present is the
MaskCLIP path in `semantic/dense_clip.py`, which targets CLIP space, not DINO space.
Text-query and semantic-mIoU evaluation is therefore recorded as **unavailable**, and
the fidelity arm of the go/no-go criterion applies instead. No unrelated bridge was
substituted.

## 3. Where each quantity is produced

| quantity | exact location | shape on a KITTI chunk |
|---|---|---|
| frame-independent encoder tokens (**pre-GCT**) | `lingbot_map/aggregator/base.py:378`, `patch_tokens = self.patch_embed(images)` → `x_norm_patchtokens`. Module is `DinoVisionTransformer` (`dinov2_vitl14_reg`) | `[S, 407, 1024]` |
| GCT block outputs | `lingbot_map/aggregator/base.py:601-604`, `cat([frame_intermediates[i], global_intermediates[i]], -1)`; emitted for `selected_idx=[4, 11, 17, 23]` set in `lingbot_map/models/gct_stream.py:299` | `[1, S, 413, 2048]`, patch tokens at `[..., 6:, :]` → `[S, 407, 2048]` |
| predicted depth | `gct_base._predict_depth` → `heads/dpt_head.py:246` | `depth [S, 154, 518, 1]` |
| depth confidence | same call, second return value | `depth_conf [S, 154, 518]` |
| predicted camera pose | `gct_base._predict_camera` → `pose_enc`; converted by `lingbot_map/utils/pose_enc.pose_encoding_to_extri_intri` | `pose_enc [S, 9]` → `extrinsic [S, 3, 4]`, `intrinsic [S, 3, 3]` |

`aa_block_size=1` and `depth=24`, so block group *b* is exactly `frame_blocks[b]`
followed by `global_blocks[b]`, and `aa_block_num = 24`. `patch_start_idx = 6`
(1 camera + 4 register + 1 scale token).

Representations are captured with **forward hooks** on `aggregator.patch_embed` and on
`aggregator` itself. Reading the aggregator's own return value (rather than
re-deriving block outputs) guarantees the captured tokens are byte-identical to the
ones the depth and camera heads consume.

## 4. Direction of the "confidence" quantity — checked, not assumed

`DPTHead` is constructed with `conf_activation="expp1"`
(`lingbot_map/heads/dpt_head.py:49`), and `head_act.activate_head` implements that as

```python
conf_out = 1 + conf.exp()          # lingbot_map/heads/head_act.py:103-104
```

so the quantity is bounded below by 1 and unbounded above. Measured on 8 KITTI
frames: `min = 1.0000`, `median = 3.0277`, `max = 37.0698`.

**It is a confidence / precision — larger means more certain. It is not an
uncertainty and not an inverse uncertainty in the variance sense** (there is no
`1/σ²` normalisation and no upper bound). The name matches the behaviour here, but the
direction was established from `head_act.py` and the measured range, not from the name.
`tests/test_hooks_gpu.py::test_depth_confidence_is_a_precision_not_an_uncertainty`
pins this. Downstream, "confidence filtering" therefore **keeps** patches *above* a
percentile threshold.

## 5. Data

| | |
|---|---|
| root | `data/kitti/dataset` → `/media/welf/SSD2/MINH_DATASETS/kitti/` (symlink; treated read-only) |
| sequences with labels | 00–10 (`labels/` present and complete) |
| sequences without labels | 13–16 |
| derived metric depth | `depth/sequences/<seq>/<frame>.npy`, float32 `[370, 1226]`, dense, measured range 3.0–78.0 m |
| GT poses | `sequences/<seq>/poses.txt`, cam0-to-world 3×4 |
| calibration | `sequences/<seq>/calib.txt`, `P2` and `Tr` |
| resolution | seqs **00–02**: 1241×376, **03**: 1242×375, **04–10**: 1226×370 |

Only sequences 04–10 are used, so every chunk shares one 1226×370 resolution and hence
one identical patch lattice. Training draws from 04, 05, 06, 07, 09, 10; validation is
sequence **08**, which appears in no training chunk.

**Preprocessing is a pure resize.** `load_and_preprocess_images(mode="crop",
image_size=518, patch_size=14)` maps 1226×370 → **518×154** (`new_height =
round(370·518/1226/14)·14 = 154`), and since 154 < 518 the crop branch never fires. So
oracle intrinsics are `P2[:3,:3]` scaled by `(518/1226, 154/370)` with **no principal
point offset**. `dataset_adapter.assert_pure_resize` fails loudly if this ever stops
holding. Patch grid: **11 × 37 = 407 patches**, plus 6 special tokens = 413.

## 6. Smoke test result

8 frames of sequence 08, GPU 0:

```
[model] params=1,157,943,540 trainable=0
[model] depth=24 aa_order=['frame','global'] aa_block_size=1 aa_block_num=24
        embed_dim=1024 patch_start_idx=6
[data]  images (8, 3, 154, 518), range [0.000, 1.000]
[infer] 8 frames in 1.32 s
```

| output / hook | shape | finite |
|---|---|---|
| `pose_enc` | `(1, 8, 9)` | yes |
| `depth` | `(1, 8, 154, 518, 1)` | yes |
| `depth_conf` | `(1, 8, 154, 518)` | yes |
| `patch_embed` | `(8, 407, 1024)` | yes |
| `frame_blocks[b]`, b ∈ {0,4,11,17,23} | `(8, 413, 1024)` | yes |
| `global_blocks[b]`, b ∈ {0,4,11,17,23} | `(1, 3304, 1024)` = 8×413 flattened | yes |
| DINOv2 ViT-B/14-reg teacher | `(8, 407, 768)` | yes |

Depth range 0.1328–7.7819, median 0.7651 — **monocular, arbitrary scale**, as expected;
this study never needs metric scale because predicted depth and predicted poses share
one consistent scale.

**Determinism**: repeated inference on identical input is **bit-exact** —
`max|Δdepth| = 0.000e+00`, `max|Δpose| = 0.000e+00`, `max|Δtokens| = 0.000e+00`.

**Hooks are non-invasive**: with the hooks registered and after removing them,
`torch.equal` holds for `depth`, `depth_conf` and `pose_enc`.

**Peak GPU memory**: 10.07 GiB for an 8-frame chunk.

Two traps found and worked around, both recorded here because they cost time:

* `kv_cache_sliding_window=-1` crashes in `flashinfer_cache.evict_frames` with
  `IndexError: pop from an empty deque`. The value used throughout is **64**, matching
  `semantic_sidecar`.
* the default `image_size=512, patch_size=16` of `load_and_preprocess_images` produces a
  height that is not a multiple of 14 and asserts inside `patch_embed`. The demo's
  `mode="crop", image_size=518, patch_size=14` is required.

## 7. Reproducing the smoke test

```bash
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH=$PWD CUDA_HOME=/usr/local/cuda-12.8 FLASHINFER_CUDA_ARCH_LIST="12.0a"
/home/minh/anaconda3/envs/cu128/bin/python research/lingbot_semantic_memory/_phase0_smoke.py \
    --device cuda:0 --num-frames 8 \
    --image-folder data/kitti/dataset/sequences/08/image_2
```

The same facts are asserted as tests:

```bash
/home/minh/anaconda3/envs/cu128/bin/python -m pytest research/lingbot_semantic_memory/tests -q
```

## 8. Blockers

**None.** The checkpoint loads, RGB inference runs, depth and pose are finite,
token hooks return the expected shapes, and inference is bit-exact reproducible.

One capability is genuinely absent rather than blocking: there is **no text-to-DINO
bridge**, so open-vocabulary text-query evaluation cannot be run in Phase 1 and is
reported as unavailable.
