# Phase 1 — verdict

> **Replicated on Replica.** The same protocol on an indoor, rotation-dominated domain
> reaches +7.0 % against this study's +7.2 %, with the same verdict. See
> [`phase1_replica_verdict.md`](phase1_replica_verdict.md). Two anomalies flagged in §8
> below were resolved there: confidence filtering is domain-dependent (helps indoors,
> hurts outdoors), and `SPARSE_PASS` is met on Replica.

## Verdict: `FAIL_CONSISTENCY`

The best GCT representation improves cross-view consistency by **+7.2 %** over the
frame-independent encoder baseline. The pre-registered gate required **+15 %**. The
result is not marginal and not a threshold artefact: under *oracle* geometry — which
doubles the number of valid correspondences and nearly triples retrieval accuracy — the
same gain **shrinks to +3.0 %**, so better geometry makes the effect smaller, not larger.

**Recommendation: stop.** Do not build the semantic-memory system on the premise that
post-GCT tokens are a more viewpoint-stable representation. §8 states what the evidence
does support.

---

## 1. Hypothesis

> Intermediate or post-GCT LingBot-MAP tokens contain geometrically contextualized
> features that are more consistent across viewpoints than ordinary frame-independent
> visual features, while retaining enough semantic information for a lightweight
> sidecar to decode DINO-space features.

## 2. Data and exact frame splits

| | |
|---|---|
| dataset | SemanticKITTI odometry, `data/kitti/dataset` (read-only) |
| training frames | **384**, six 64-frame chunks: `seq04:128-192`, `seq05:1369-1433`, `seq06:518-582`, `seq07:506-570`, `seq09:738-802`, `seq10:552-616` |
| held-out frames | **192**, three 64-frame chunks of **sequence 08**: `683-747`, `2021-2085`, `3368-3432` |
| sequence overlap | none — sequence 08 appears in no training chunk |
| why 04–10 | those sequences all measure 1226×370, so every chunk lands on one identical 11×37 patch lattice |
| preprocessing | `load_and_preprocess_images(mode="crop", image_size=518, patch_size=14)` → 518×154; a **pure resize**, no crop (asserted at load time) |
| seed | 0; chunk starts, teacher subsets, frame pairs and batch order are all functions of it |
| labels | used **only** for the boundary metric and figures; never in any loss |

Sequence 08 is a held-out *sequence*, not held-out frames from a training sequence, so
the "evaluate on non-teacher frames" requirement is satisfied by construction at every
teacher budget.

## 3. Checkpoints

| | |
|---|---|
| LingBot-MAP | `checkpoints/lingbot-map/204754b/lingbot-map.pt`, SHA-256 `ee665103348e07e6…`, 1 157 943 540 params, **0 trainable** |
| external teacher | DINOv2 **ViT-B/14 with registers**, `~/.cache/torch/hub/checkpoints/dinov2_vitb14_reg4_pretrain.pth`, 86 583 552 params, **0 trainable** |
| text-to-DINO bridge | **none exists in this repository** — see §7 |
| git | `4071d1e`, tree dirty only by this new `research/` directory |
| environment | python 3.10.19, torch 2.7.1+cu128, CUDA 12.8, RTX PRO 6000 Blackwell (GPU 0) |

## 4. Token layers

| name | exact source | width |
|---|---|---|
| `lingbot_encoder` | `aggregator.patch_embed` → `x_norm_patchtokens` (`aggregator/base.py:378`). This is DINOv2 ViT-L/14-reg, **before any cross-frame attention** | 1024 |
| `lingbot_gct_b04` | `cat(frame_blocks[4], global_blocks[4])`, read from the aggregator's own return value | 2048 |
| `lingbot_gct_mid` | block 11 | 2048 |
| `lingbot_gct_b17` | block 17 | 2048 |
| `lingbot_gct_final` | block 23 — the last tokens before the depth head | 2048 |

Blocks 4/11/17/23 are exactly the `selected_idx` that `GCTStream._aggregate_features`
passes to the camera, depth and point heads, so these are the tokens the model itself
uses. Capture is by forward hook and is **bit-exact non-invasive**: `depth`,
`depth_conf` and `pose_enc` satisfy `torch.equal` with hooks registered and removed
(`tests/test_hooks_gpu.py`).

## 5. Probe architecture and parameter count

`LayerNorm → Linear → GELU → Linear → L2-normalise`, identical for every
representation, trained only with `L = 1 − cos(pred, teacher)`.

Capacity is matched **by parameter count, not hidden width** — a fixed hidden width
would give the 2048-d GCT probes ~60 % more parameters than the 1024-d encoder probe
and would confound exactly the comparison being made:

| representation | input width | hidden | **trainable params** |
|---|---:|---:|---:|
| `lingbot_encoder` | 1024 | 1113 | **1 998 425** |
| every GCT block | 2048 | 708 | **1 999 300** |

A 0.044 % spread, both far under the 10 M cap. Identical optimiser (AdamW, lr 1e-3,
wd 0.01, 200-step warm-up + cosine, grad-clip 1.0), identical 4000 steps, identical
batch (8 frames × 407 tokens), **no augmentation for any representation**, identical
feature resolution (teacher and LingBot both use patch 14, so the token grids coincide
exactly and distillation needs no resize or interpolation).

## 6. Exact commands executed

```bash
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH=$PWD CUDA_HOME=/usr/local/cuda-12.8 FLASHINFER_CUDA_ARCH_LIST="12.0a"
PY=/home/minh/anaconda3/envs/cu128/bin/python

# Phase 0 smoke test
$PY research/lingbot_semantic_memory/_phase0_smoke.py --device cuda:0 --num-frames 8 \
    --image-folder data/kitti/dataset/sequences/08/image_2

# tests (41 passed)
$PY -m pytest research/lingbot_semantic_memory/tests -q

# one-command smoke test
$PY -m research.lingbot_semantic_memory.run_phase1 \
    --config research/lingbot_semantic_memory/configs/phase1.yaml --smoke

# one-command full experiment  (this report)
$PY -m research.lingbot_semantic_memory.run_phase1 \
    --config research/lingbot_semantic_memory/configs/phase1.yaml --device cuda:0 --seed 0

# tables
$PY -m research.lingbot_semantic_memory.report_tables \
    --metrics research/lingbot_semantic_memory/outputs/phase1/metrics.json \
    --output-dir research/lingbot_semantic_memory/outputs/phase1
```

Artefacts: `outputs/phase1/metrics.json`, `outputs/phase1/results.csv`,
`outputs/phase1/tables.md`, `outputs/phase1/figures/*.png`, log `outputs/phase1_full.log`.

## 7. Results

288 held-out frame pairs, 18 602 valid correspondences under predicted geometry.
Every representation is scored on the **same** correspondence set.

### 7.1 Main comparison — 100 % teacher, predicted geometry

| representation | params | teacher cos | cos (centred) | diversity | **cross-view cos** | vs encoder | margin | recall@1 | boundary margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lingbot_encoder` | 1 998 425 | 0.7087 | 0.6169 | 0.6680 | 0.7528 | — | 0.3515 | 0.1689 | 0.1177 |
| `lingbot_gct_b04` | 1 999 300 | 0.7078 | 0.6161 | 0.6732 | 0.7597 | **+0.9 %** | 0.3671 | 0.1736 | 0.1238 |
| `lingbot_gct_mid` | 1 999 300 | 0.7013 | 0.6054 | 0.6582 | 0.7653 | **+1.7 %** | 0.3618 | 0.1719 | 0.1179 |
| `lingbot_gct_b17` | 1 999 300 | 0.5960 | 0.4732 | 0.6731 | 0.7895 | **+4.9 %** | 0.3534 | 0.1682 | 0.1026 |
| `lingbot_gct_final` | 1 999 300 | 0.5969 | 0.4718 | 0.6544 | 0.8072 | **+7.2 %** | 0.3552 | 0.1686 | 0.1043 |
| `external_dino_teacher` (reference) | 0 | 1.0000 | 1.0000 | 0.8027 | 0.6299 | — | 0.3494 | 0.1633 | 0.1295 |

**Gate arithmetic.** Required: consistency ≥ 1.15 × and fidelity ≥ 0.95 ×. Achieved:

| representation | consistency ratio | fidelity ratio | criterion 1 | criterion 2 |
|---|---:|---:|:--:|:--:|
| `lingbot_gct_b04` | 1.009 | 0.999 | ✗ | ✓ |
| `lingbot_gct_mid` | 1.017 | 0.990 | ✗ | ✓ |
| `lingbot_gct_b17` | 1.049 | 0.841 | ✗ | ✗ |
| `lingbot_gct_final` | **1.072** | 0.842 | ✗ | ✗ |

No representation passes criterion 1. The two that *keep* their semantics (b04, mid)
barely move consistency at all (+0.9 %, +1.7 %); the two that move consistency most
(b17, final) lose ~16 % of teacher fidelity, failing criterion 2 as well. The trade-off
runs the wrong way at every depth.

### 7.2 The raw-token result, and why it does not rescue the hypothesis

Measured directly on the frozen tokens with no probe:

| representation | cross-view cos | vs encoder | negative control | **margin** | vs encoder | recall@1 | diversity |
|---|---:|---:|---:|---:|---:|---:|---:|
| `lingbot_encoder` | 0.8035 | — | 0.5950 | **0.2085** | — | 0.1713 | 0.4490 |
| `lingbot_gct_b04` | 0.8492 | +5.7 % | 0.6942 | 0.1550 | −25.7 % | 0.1721 | 0.3402 |
| `lingbot_gct_mid` | 0.8995 | +12.0 % | 0.8301 | 0.0694 | −66.7 % | 0.1539 | 0.1918 |
| `lingbot_gct_b17` | 0.9407 | +17.1 % | 0.9002 | 0.0405 | −80.6 % | 0.1421 | 0.1412 |
| `lingbot_gct_final` | 0.9849 | **+22.6 %** | 0.9735 | **0.0114** | **−94.6 %** | 0.1419 | 0.0371 |
| `external_dino_teacher` | 0.6299 | — | 0.2805 | 0.3494 | — | 0.1633 | 0.8026 |

**On raw tokens the naive metric passes the gate — and it is an artefact.**
`lingbot_gct_final` scores +22.6 %, comfortably over +15 %. But its cosine between
*non-corresponding* patches rises just as fast (0.595 → 0.974), its margin collapses by
94.6 %, its diversity falls 12-fold (0.449 → 0.037) and its retrieval accuracy *drops*.
Deep GCT tokens are not more viewpoint-stable; they are **more anisotropic** — nearly
every token points in nearly the same direction, so any two of them look similar
whether or not they are the same 3D point.

This is precisely the failure mode the pre-registration named in advance (README,
"Anti-degeneracy guard"), and the same one the earlier study in this repository hit
(`docs/semantic_sidecar_feasibility_report.md` §2). The guard was written before the
numbers existed; it is not a post-hoc reinterpretation. The primary metric was
pre-registered on **probe outputs** for exactly this reason — the probe maps every
representation into one shared, comparably-scaled teacher space, which removes most of
the anisotropy inflation and leaves the +7.2 % that §7.1 reports.

The figures show the same thing directly. In
`figures/similarity_gap08.png`, the encoder's similarity peak sits **exactly on** the
geometric match (peak 0.966, at-match 0.966), while `gct_final` peaks somewhere else
(peak 0.973, at-match 0.914): its similarity map is brighter overall but less
selective.

### 7.3 Predicted vs oracle geometry

| representation | predicted cos | oracle cos | Δ | predicted recall@1 | oracle recall@1 |
|---|---:|---:|---:|---:|---:|
| `lingbot_encoder` | 0.7528 | 0.8658 | +0.1130 | 0.1689 | 0.4516 |
| `lingbot_gct_b04` | 0.7597 | 0.8748 | +0.1151 | 0.1736 | 0.4662 |
| `lingbot_gct_mid` | 0.7653 | 0.8795 | +0.1142 | 0.1719 | 0.4687 |
| `lingbot_gct_b17` | 0.7895 | 0.8904 | +0.1009 | 0.1682 | 0.4564 |
| `lingbot_gct_final` | 0.8072 | 0.8920 | +0.0848 | 0.1686 | 0.4492 |

| correspondence set | matches | valid fraction |
|---|---:|---:|
| predicted | 18 602 | 0.160 |
| predicted + confidence filter | 14 142 | 0.125 |
| **oracle** | **37 324** | **0.318** |
| oracle + confidence filter | 24 248 | 0.209 |

Predicted geometry is materially worse than oracle: it yields **half** the valid
correspondences (16.0 % vs 31.8 % of patches) and **a third** the retrieval accuracy
(0.169 vs 0.452). That is a real limitation of monocular predicted depth/pose at these
baselines, and it is reported as such.

**It does not change the verdict, and this is the key robustness check.** Under oracle
geometry the GCT gains *shrink*: `gct_final` goes from +7.2 % to **+3.0 %**, and the
ordering of representations is unchanged. If poor geometry were masking a real GCT
advantage, oracle geometry would reveal it; instead it removes most of what was there.
So the verdict is `FAIL_CONSISTENCY`, **not** `FAIL_GEOMETRY` — the test was valid and
the hypothesis lost it.

### 7.4 Teacher budget

Teacher targets on 38 / 96 / 384 training frames; all evaluation is on held-out
sequence 08.

| representation | @10 % | @25 % | @100 % | retention 10 %/100 % |
|---|---:|---:|---:|---:|
| `lingbot_encoder` | 0.6137 | 0.6591 | 0.7087 | 86.6 % |
| `lingbot_gct_b04` | 0.6157 | 0.6634 | 0.7078 | 87.0 % |
| `lingbot_gct_mid` | 0.6143 | 0.6605 | 0.7013 | 87.6 % |
| `lingbot_gct_b17` | 0.5081 | 0.5589 | 0.5960 | 85.3 % |
| `lingbot_gct_final` | 0.5244 | 0.5729 | 0.5969 | **87.9 %** |

**`SPARSE_PASS`: not met** — 87.9 % against the 90 % threshold. The margin is small and
all five representations sit in a tight 85–88 % band, so sparse supervision degrades
every representation about equally. Nothing here distinguishes GCT tokens, and the
threshold is reported as pre-registered rather than relaxed to fit.

### 7.5 Temporal band

| representation | short gaps (1–3) | long gaps (8–24) | gap 1 | gap 8 | gap 24 |
|---|---:|---:|---:|---:|---:|
| `lingbot_encoder` | 0.7871 | 0.5939 | 0.8324 | 0.6333 | 0.5205 |
| `lingbot_gct_mid` | 0.7991 | 0.6090 | 0.8422 | 0.6521 | 0.5304 |
| `lingbot_gct_final` | 0.8404 | 0.6537 | 0.8735 | 0.7079 | 0.5545 |

The GCT advantage is slightly larger at long gaps (+10.1 %) than short (+6.8 %), which
is the right *direction* for the hypothesis — but both are far below +15 %, and both
shrink under oracle geometry.

### 7.6 Depth-confidence filtering

| representation | no filter | keep top 75 % confidence | Δ |
|---|---:|---:|---:|
| `lingbot_encoder` | 0.7528 | 0.7431 | −0.0097 |
| `lingbot_gct_mid` | 0.7653 | 0.7553 | −0.0100 |
| `lingbot_gct_final` | 0.8072 | 0.7994 | −0.0078 |

Filtering to high-confidence patches slightly **lowers** cross-view consistency for
every representation and discards 24 % of correspondences. `depth_conf` is a
precision (`1 + exp(raw)`, higher = better; §4 of the Phase-0 report), and the filter
was applied in that direction — the effect is small, uniformly negative, and does not
reorder anything. Predicted depth confidence is not a useful correspondence filter here.

### 7.7 Boundary preservation

Within-region vs across-boundary cosine over 4-neighbour patch pairs, regions from
projected SemanticKITTI labels (evaluation only):

`lingbot_gct_b04` 0.1238 > `lingbot_encoder` 0.1177 ≈ `lingbot_gct_mid` 0.1179 >
`lingbot_gct_final` 0.1043 > `lingbot_gct_b17` 0.1026; teacher reference **0.1295**.

Deep GCT blocks blur semantic boundaries relative to both the encoder and the teacher,
consistent with the diffuse similarity maps in §7.2.

### 7.8 Efficiency

| quantity | value |
|---|---|
| probe parameters | 1 998 425 (encoder) / 1 999 300 (GCT) |
| LingBot inference | **44.1 ms/frame** (64-frame chunk, streaming, bf16) |
| teacher inference | **0.73 ms/frame** |
| probe inference | **0.52 ms/frame** (mean over 15 probes) |
| probe training | 13–17 s per probe; **229 s** for all 15 |
| cache | 4.76 GiB for 576 frames (**8.46 MiB/frame**) |
| peak GPU memory | 10.14 GiB |

LingBot dominates cost by 60×; the probe is free by comparison. Caching means probe
training and evaluation never re-run either frozen model.

### 7.9 Text-query / semantic mIoU

**Unavailable.** No text-to-DINO bridge exists in this repository (§3 and Phase-0 §2).
The `semantic/` package contains a MaskCLIP → CLIP-space bridge, which targets a
different embedding space; substituting it would have measured a different thing, so it
was not used. The fidelity arm of criterion 2 (≤ 5 % relative teacher-cosine loss)
applies instead, as the protocol specifies.

## 8. Limitations and anomalies

1. **The encoder baseline is the teacher's sibling.** LingBot's pre-GCT embedder *is*
   DINOv2 ViT-L/14-reg; the teacher is DINOv2 ViT-B/14-reg. Same family, different
   weights and width. This makes `lingbot_encoder` an unusually strong fidelity
   baseline, so criterion 2 is a hard test — and it means the fidelity column should
   not be read as an absolute measure of "how semantic" a layer is. It does **not**
   weaken the consistency finding, which is the one that decided the gate: cross-view
   consistency is measured between two views of the *same* representation, and the
   teacher plays no part in it.
2. **One dataset, one domain** — *addressed*. This study covers outdoor driving only,
   192 held-out frames, mostly forward motion. The companion Replica run covers indoor,
   rotation-dominated motion and reproduces the verdict to within 0.2 percentage points.
   ScanNet is still not on this machine.
3. **Single seed.** Each configuration was trained once. The +7.2 % headline is far
   enough from +15 % that seed noise is very unlikely to bridge it, but the small
   contrasts (b04 vs mid, ±0.9 %) carry no variance estimate and should not be read.
4. **Absolute recall@1 is low under predicted geometry** (0.169 vs 0.452 oracle). This
   is a genuine weakness of the monocular geometry at these baselines, quantified in
   §7.3, and it bounds how strong any claim about predicted-geometry correspondence can
   be. Chance is 1/407 = 0.0025, so even 0.169 is 68× chance.
5. **Anomaly — `gct_b17` and `gct_final` have nearly identical fidelity** (0.5960 /
   0.5969) after a sharp drop from `gct_mid` (0.7013). The semantic loss is not gradual
   across depth; it happens between blocks 11 and 17 and then plateaus. Not explained
   here, and worth knowing for anyone selecting a layer.
6. **Anomaly — confidence filtering is uniformly unhelpful** (§7.6). A filter that
   keeps geometrically confident patches ought to improve correspondence quality; it
   does the opposite by a small margin. **Resolved by the Replica run**, where the same
   filter uniformly *helps* (+0.013): LingBot's depth confidence carries signal indoors,
   where the scene is within a few metres, but not on driving frames dominated by sky
   and far structure. The filter is domain-dependent and reorders nothing either way.
7. **Patch-granularity correspondences.** Matching is at 14-px patch resolution with a
   patch-centre depth test, which is the right granularity for patch features but
   coarser than pixel-level matching would be.

## 9. Verdict and recommendation

**`FAIL_CONSISTENCY`.**

No GCT representation improved cross-view consistency by the pre-registered 15 % over
the frame-independent encoder. Best observed: **+7.2 %** (`lingbot_gct_final`), which
falls to **+3.0 %** under oracle geometry. The representations that came closest to the
consistency threshold also failed the semantic criterion, retaining only 84 % of the
encoder's teacher fidelity. `SPARSE_PASS` was likewise not met (87.9 % vs 90 %).

What the experiment did establish, and what it costs the original plan:

* GCT depth buys **anisotropy, not viewpoint stability**. Raw deep-GCT tokens look
  dramatically more "consistent" (+22.6 %) only because they also look more consistent
  between *unrelated* patches (negative control 0.595 → 0.974). Any future work here
  must report a negative control or margin alongside any consistency number, or it will
  measure collapse and call it success.
* The premise that a GCT layer is the natural feature for a semantic memory is not
  supported. Blocks 4 and 11 preserve semantics essentially perfectly (99.9 %, 99.0 %
  of encoder fidelity) but add almost no consistency; blocks 17 and 23 add a little
  consistency and destroy 16 % of the semantics.
* The **largest** measured effect in this study is not the representation at all — it is
  **geometry**. Moving from predicted to oracle depth/pose doubles valid correspondences
  (16.0 % → 31.8 %) and raises retrieval accuracy 2.7× (0.169 → 0.452), dwarfing every
  representation-level difference in the table.

**Per the gate: stop.** Do not proceed to geometry-aware frame selection, persistent
semantic memory, 3D target retrieval, or paper writing.

**The single next action justified by this evidence:** if the streaming
open-vocabulary mapping goal is still worth pursuing, the binding constraint to attack
is **the quality of predicted depth and pose for cross-view association** — not which
transformer layer the features are read from. Concretely, re-run §7.3 as its own small
study over a range of baselines to find where predicted geometry's correspondence yield
collapses, because that curve, not the layer choice, sets the ceiling for any
LingBot-based semantic memory. Read features from `lingbot_encoder` or
`lingbot_gct_b04` if that work proceeds: they are the only options that cost nothing in
semantics, and blocks 17/23 have no measured advantage worth 16 % of teacher fidelity.
