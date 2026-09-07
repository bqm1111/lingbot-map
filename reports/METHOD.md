# The method, end to end

*A handbook for owning and presenting this project. Written 2026-09-04, current through
Gate 8C-1. Every number here is read from a file in `artifacts/`; the pointer is given so
you can check any of them yourself.*

---

## 0. The one-paragraph version

We take three large models that we did not train and never fine-tune — a streaming
reconstruction model (LingBot-Map), a monocular metric-depth model (MoGe-2) and an
open-vocabulary segmentation teacher (Trident-H) — and fuse their per-frame output into a
persistent 3D voxel map as a car drives. That map is accurate where the camera has looked
and empty everywhere else. We then train **one small network, 0.99 M parameters**, to fill
in the parts the camera has not seen yet, using as its training signal what the *future*
frames of the same drive will reveal. At inference the future is gone: the module sees only
the map built from past frames. The claim we are testing is that this completion module,
trained on one city, transfers to other datasets without ever seeing them.

---

## 1. Motivation — why this, and why it is interesting

**The problem.** A camera-only vehicle needs a 3D occupancy map: which cubes of space are
solid, which are free. A single image gives you surfaces you can see. Everything behind
them — the far side of a car, the road under a truck, the wall behind a hedge — is
*unknown*, and a planner cannot use unknown.

**The usual answer** is to train a large model end to end on the target benchmark. That
works, but it means one model per dataset, and the model must learn geometry, semantics and
completion all at once from that dataset's labels.

**Our angle** is a separation of concerns:

| responsibility | who does it | trained by us? |
|---|---|---|
| per-frame geometry and camera pose | LingBot-Map (frozen) | no |
| metric scale | MoGe-2 (frozen) + a five-frame estimator | no |
| what things are called | Trident-H (frozen, open-vocabulary) | no |
| accumulating evidence over time | a hand-written incremental map | no — it has no parameters |
| **guessing what has not been seen** | **a 0.99 M residual U-Net** | **yes — this is the only trained part** |

If that split works, the expensive parts are reusable and only a tiny, cheap module needs
to be learned — and because it consumes a *geometric* map rather than images, it has a
chance of transferring across sensors and cities.

**Two properties we insist on**, because they are what make the result worth anything:

1. **Causality.** At inference the module may use only frames at or before the current
   time. No future observation, ever. The future is used *only* to build training targets.
2. **Target-domain freedom.** The module is trained and selected on one dataset and then
   evaluated on others, with the evaluation dataset provably untouched until the model and
   its threshold are frozen and hashed.

---

## 2. The system, component by component

The pipeline for one frame at time *t*:

```
image_t ──► LingBot-Map  ──► depth_t (canonical units), pose_t (camera→world), K_t
        └─► MoGe-2       ──► metric depth_t  ──┐
                                               ├─► scale candidate s_t  (first 5 frames only)
                                               │
                            ScaleState ────────┘   → one scalar s, then FROZEN

        └─► Trident-H    ──► per-pixel open-vocabulary class probabilities

   depth_t × s, pose_t (translation × s), K_t, Trident probs
                    │
                    ▼
        IncrementalMapper.step(frame_t)      ← integrates each frame exactly ONCE
                    │
                    ▼
        persistent voxel map  (log-odds occupancy + semantic evidence)
                    │
     query on the benchmark grid → 32-channel dense tensor
                    │
                    ▼
        CompletionUNet  ──► occupancy residual + semantic logits    ← THE TRAINED PART
                    │
                    ▼
        final occupancy = base_logodds + residual·(gate)   → threshold → prediction
```

### 2.1 LingBot-Map (frozen)

A streaming Geometric Context Transformer. Given a sequence of images it produces, per
frame, a dense depth map, a camera-to-world pose and intrinsics — all in an arbitrary
**canonical** scale (the whole reconstruction is correct up to one unknown global factor).
Checkpoint `checkpoints/lingbot-map/204754b/lingbot-map.pt`.

*Caveat you must state:* LingBot-Map's published training mixture **includes KITTI-360**.
That is why we never call the system "zero-shot" (§8).

### 2.2 MoGe-2 + the five-frame scale anchor (frozen)

LingBot's output has no metric scale. MoGe-2 predicts *metric* depth from a single image
with a calibrated horizontal field of view. For each of the **first five frames** of a
sequence we compute a scale candidate as the weighted median of
`log(D_moge) − log(D_lingbot)` over valid pixels. `ScaleState` takes the median of those
five, **freezes it, and ignores every later frame** (`gate8/mapper.py::ScaleState`).

Two rules that matter and are enforced by tests:

* the scale is fixed **once**, from the first five frames only;
* the *same* scalar multiplies the depth **and** the pose translation
  (`d_m = s · depth`, `T[:3,3] *= s`) — rotation is untouched. Scaling one but not the
  other would silently warp the map.

### 2.3 The incremental map (frozen, zero parameters)

`gate8/mapper.py::IncrementalMapper`. A sorted hash table of voxels at 0.2 m, keyed by an
integer world lattice. For each new frame it casts the accepted depth rays and accumulates
**log-odds**:

* voxels within ±0.2 m of the measured surface get **+0.85** (occupied evidence);
* voxels the ray passes through, from 1 m up to 0.2 m before the surface, get **−0.40**
  (free evidence), sub-sampled 4×4 in pixels;
* **nothing at all is written beyond the surface** — the space behind an obstacle stays
  *unknown*, which is the whole point: "unknown" and "free" must stay distinguishable;
* log-odds are clamped to ±4.0.

It also carries, per voxel: free-evidence weight, observation count, first/last time seen,
and an accumulated Trident probability vector with its weight.

The map is **persistent and incremental**: every frame is integrated exactly once, and a
query is a lookup, never a rebuild. This is what makes the system streaming rather than
batch.

### 2.4 The completion module — **the only trained component**

`gate8/net.py::CompletionUNet`. A three-level dense 3D U-Net.

| | |
|---|---|
| **Trainable parameters** | **986,114 (0.99 M)** |
| widths | 24 / 48 / 96 |
| input | 32 channels on a dense voxel crop |
| outputs | 1 occupancy-residual channel + 25 semantic logits |
| per-level params | enc1 36 k, enc2 125 k, enc3 498 k, dec2 187 k, dec1 47 k, heads 650 |

**The 32 input channels**, in order (`gate8/targets.py::unpack_sample`):

| # | channel | normalisation |
|---|---|---|
| 1 | occupancy log-odds | ÷ 4 |
| 2 | free-evidence weight | clamp 20, ÷ 20 |
| 3 | observed (has any evidence) | 0/1 |
| 4 | **un**observed | 1 − observed |
| 5 | observation count | clamp 50, ÷ 50 |
| 6 | age since last seen | clamp, ÷ 100 |
| 7 | semantic evidence weight | clamp 20, ÷ 20 |
| 8–32 | mean Trident probability per union class (25 classes) | already a probability |

Channels 3 and 4 are deliberately redundant: the module must be able to condition on
"nothing has been seen here", which is exactly where completion is required.

**The residual rule** — the single most important design decision:

```python
gate  = (|base_logodds| < 2.0)
final = base_logodds + residual · gate          # gate8/net.py::apply_residual
```

A voxel that already carries ≥ 2.0 of accumulated log-odds (roughly three agreeing rays) is
**locked**: the network cannot change it. The module may therefore only act where the map
is unknown or weakly supported. It cannot "improve" the benchmark score by overwriting
measurements it did not make. The region where it *is* allowed to act is called the
**editable region** and is reported separately in every evaluation.

Semantics follow the same spirit: a voxel that already has fused teacher evidence keeps it;
the network's semantic head only names voxels it added.

---

## 3. What is trained, and what is not — read this before you present

> **Exactly one component is trained: the 0.99 M-parameter `CompletionUNet`.
> Everything else is frozen and was never fine-tuned by us.**

| component | parameters | trained by us | how used |
|---|---|---|---|
| **CompletionUNet** | **986,114** | **YES — the whole contribution** | predicts the occupancy residual + semantics |
| LingBot-Map | (large) | no — frozen checkpoint | per-frame depth, pose, intrinsics |
| MoGe-2 | (large) | no — frozen checkpoint | metric scale, first 5 frames only |
| Trident-H (CLIP ViT-H + SAM ViT-H) | 1,712,998,705 | no — frozen | per-pixel open-vocabulary probabilities |
| IncrementalMapper, ScaleState | **0** | n/a — hand-written rules | evidence accumulation |

Total inference footprint ≈ **1.71 B frozen + 0.99 M trained**. Training the trained part
costs **0.72 GPU-hours for all three seeds** (14.3 min per seed, 4.84 GiB peak).

If your supervisor asks one question about this project, it will probably be *"so what did
you actually train?"* — the answer is the table above.

---

## 4. Supervision — where the training signal comes from

This is the second idea in the project, after the frozen/trained split.

### 4.1 Privileged future observation

At an anchor time *t*:

* the **input** is the causal map, built from frames ≤ *t*;
* the **target** is built from frames *t+1 … t+20* — twenty stream frames of *future*
  observation, which the module will never see at inference.

So the module is trained to answer "what will the next ten seconds of driving reveal about
the space I cannot see right now?" This is *privileged learning*: the teacher has
information the student does not, and the student must learn to predict it.

The 20-frame horizon is the right one: Gate 8C-0 measured that after 11–20 stream frames the
sensor has physically left the 51.2 m box, so a longer window adds nothing.

### 4.2 Geometry targets — rebuilt from raw LiDAR (Gate 8C-1)

Originally we used the benchmark's own completion labels. Gate 8C-0 proved the KITTI-360
ones are broken (§7), so Gate 8C-1 rebuilds them from raw Velodyne:

* transform sweeps *t … t+20* into the anchor frame with ground-truth poses;
* ray **endpoints → occupied**;
* ray **interiors → free** (stopping 0.2 m short of the endpoint);
* nothing beyond an endpoint;
* a voxel a sweep measured is never carved free **by that same sweep** (otherwise grazing
  ground rays make ~73 % of surface voxels self-contradictory);
* a voxel that one sweep calls occupied and *another* calls free → **unknown**, excluded
  from the loss. This is real disagreement (moving cars, thin structures) and we do not
  adjudicate it. It is ~8–9 % of touched voxels.

Result: **100 % of supervised LiDAR endpoints are labelled OCCUPIED and 0 % FREE** — the
target agrees with the sensor by construction. Validation in
`artifacts/gate8c1/target_validation.json`; all five checks pass, including the ±1-voxel
shift scan that the old KITTI-360 label failed.

Ground-truth LiDAR and ground-truth poses appear **only** in target construction. A test
asserts they are unreachable from the inference path.

### 4.3 Semantic targets

Frozen Trident-H probabilities fused over *t+1 … t+20*, kept only where the geometry is
supported and where the teacher is confident. **No human semantic label is used anywhere**
in training or selection — not in the loss, not in checkpoint choice, not in the threshold.
The 25-class "union vocabulary" is a fixed superset that maps onto each benchmark's own
class list at evaluation time only.

### 4.4 The loss

```
L = 1.0 · focal_BCE(final_logits, occupied)      γ = 2, pos_weight ≤ 8, 2× on unobserved
  + 1.0 · soft_Dice(final_logits, occupied)
  + 0.5 · KL(teacher ‖ student)                  on voxels with a valid future-teacher target
```

The occupancy terms are computed on the **final** log-odds (after the residual is added to
the frozen map), so the network is optimised for the quantity that is actually evaluated.

---

## 5. Training strategy

| setting | value |
|---|---|
| training data | KITTI-360 drives **0003, 0007, 0010** — 1 358 anchors |
| source validation | KITTI-360 drive **0006** — disjoint, 590 anchors |
| target datasets | SemanticKITTI 08, Occ3D-nuScenes val — **never touched until frozen** |
| crop | 128 × 128 × 32 voxels, sampled **uniformly** over the valid volume |
| batch | 4 |
| optimiser | AdamW, lr 1e-3, weight decay 0.01, cosine schedule, grad clip 1.0 |
| steps | 6 000 |
| seeds | 0, 1, 2 — identical configuration except the seed (asserted by test) |
| cost | 14.3 min and 4.84 GiB per seed |

**The dataset firewall.** Before any target dataset may be read, three things must exist:
per-seed checkpoint, occupancy threshold, and a signed manifest. Training and selection ran
inside a *file-access audit* that intercepts `open`, `numpy.load` and `numpy.fromfile` and
raises on any SemanticKITTI / Occ3D / SSCBench-label path. **Zero violations** across all
runs (`artifacts/gate8c1/frozen_manifest.json → firewall_proof`). Five tests check both the
source text and the import closure of every pre-firewall tool.

**Selection, on KITTI-360 drive 0006 only:**

* **checkpoint**: highest occupancy **AP** (a threshold-free ranking metric);
* **occupancy threshold**: the one global final-logit threshold maximising source IoU
  (seed 0: −0.125, seed 1: −0.094, seed 2: −0.156);
* **semantic confidence threshold**: chosen by *teacher self-consistency* — agreement
  between the teacher's opinion from the first and second half of the future window. No
  human label involved.

All three are hashed into `frozen_manifest.json` **before** the evaluator can run. The
target evaluator accepts no checkpoint and no threshold argument; it reads the manifest and
refuses to start without it.

---

## 6. Evaluation protocol

### 6.1 Grids and masks (frozen since Gate 6)

| dataset | prediction grid | evaluation grid | frame |
|---|---|---|---|
| SemanticKITTI 08 | 256×256×32 @ 0.2 m | same | velodyne of the anchor |
| KITTI-360 | 256×256×32 @ 0.2 m | same | velodyne of the anchor |
| Occ3D-nuScenes | 400×400×32 @ 0.2 m | 200×200×16 @ 0.4 m | ego of the anchor |

Occ3D reduces 0.2 m → 0.4 m by the frozen "any sub-voxel occupied" rule. Masks: camera mask
applied, LiDAR mask not, and for the single-camera setting the rear half (`x < 0`) is
excluded — identical to OccAny's own code.

### 6.2 The three protocols

| name | input | causal? | what it is for |
|---|---|---|---|
| **A · past5** | 5 frames **ending** at *t* | **yes** | the primary, deployed setting |
| **A · stream** | all past frames, one persistent map | **yes** | shows the benefit of long memory |
| **B · occany_fwd** | offsets 0,+2,+4,+6,+8 — target first, 4 frames of **future** | **no** | only to match OccAny's input budget |

Protocol B is labelled non-causal everywhere and is never presented as a deployment result.

### 6.3 Metrics

* **SC IoU / precision / recall** — the headline occupancy numbers.
* **AP, AP/prevalence, AUROC** — threshold-free. These separate "the model ranks voxels
  well" from "the threshold is right", which turned out to matter enormously.
* **predicted / true occupied volume** — catches a model that wins by predicting more.
* **SSC mIoU, TP-conditioned naming accuracy** — semantics, reported as diagnostics.
* **Paired scene-level bootstrap 95 % CIs** — 10 000 draws, scenes as units.

### 6.4 Baselines, on identical clips and masks

frozen five-frame G51-B (raw and +0.4 m dilation) · incremental mapper alone · mapper +
0.4 m dilation · editable-region fill · matched-density random · all-valid-occupied ·
released OccAny. The last three exist specifically to catch a model that "wins" by
inflating volume.

---

## 7. What we learned along the way — the gate history

You will be asked *"how do you know the result is real?"*. This is the answer: each gate was
designed to falsify the previous one.

| gate | question | answer |
|---|---|---|
| **8** | does privileged completion help at all? | Yes on paper — but it lost to "declare everything occupied" on held-out KITTI-360. |
| **8A** | is that a sampling or calibration artefact? | **Calibration, not sampling.** Crop sampling changed the prior by < 1 pp; the loss changed ECE 0.15 → 0.04 and over-prediction 2.9× → 0.8×. But held-out AUROC was **0.493** — chance. |
| **8B** | does it fail everywhere, or just on KITTI-360? | Mixed: Occ3D **29.55 %** IoU and AUROC 0.705 (passes), SemanticKITTI 7.22 % / 0.536, KITTI-360 18.73 % / 0.493. |
| **8C-0** | why is KITTI-360 at chance even when trained on it? | **The published label is broken.** A ground-truth LiDAR return lands on a voxel SSCBench's own label calls *free* **71 %** of the time; on SemanticKITTI, under identical code, **1.7 %**. Their own voxel input disagrees with their own label by one voxel in z. Our chain reproduces their input at 99.90 % recall. Also: the model **can** memorise a fixed KITTI-360 batch to AP 0.997, so it is not an optimisation failure. |
| **8C-1** | with clean supervision, does one-source training transfer? | Trained on KITTI-360 alone with rebuilt raw-LiDAR targets: **Occ3D 30.89 %**, **SemanticKITTI 12.74 %**. |

The Gate 8C-0 finding is the one to lead with if anyone asks about KITTI-360 numbers in
earlier reports: **they were measuring a defective target and should be withdrawn.**

---

## 8. Results as they stand

### 8.1 Headline (Protocol A, causal 5 past frames, median of 3 seeds)

| | SemanticKITTI 08 | Occ3D-nuScenes val |
|---|---|---|
| **ours, raw** | **12.74 %** ± 1.03 | **30.89 %** ± 0.29 |
| ours + OccAny's post-processing | 11.85 %† | **42.04 %** |
| map + 0.4 m dilation | 15.41 % | 21.64 % |
| frozen 5-frame + dilation | 16.00 % | 20.94 % |
| fill every valid voxel | 7.82 % | 22.95 % |
| **matched-density random fill** | **7.45 %** | **13.33 %** |
| **map only, no completion** | **8.90 %** | **9.99 %** |
| OccAny (published, post-processed) | 25.91 % | 23.55 % — **reproduced here: 23.56 %** |
| OccAny (raw, no post-processing — measured here) | — | **20.67 %** |
| precision / recall | 15.9 / 39.1 | 67.1 / 36.4 |
| predicted ÷ true volume | **2.38×** | **0.54×** |
| AP / prevalence · AUROC | 2.54 · 0.651 | 2.31 · 0.763 |
| SSC mIoU | 2.65 % | 4.13 % |

† Occ3D uses OccAny's own pool-then-mask order (re-measured 2026-09-04); the SemanticKITTI
cell still carries the gate's mask-then-pool order and was not re-measured — it is the
stricter of the two on us, so it does not flatter the result.

**What the training actually bought** (the two bold rows are the before-states):

| | SemanticKITTI | Occ3D |
|---|---|---|
| map only, no completion | 8.90 (P 33.3 / R 10.8) | 9.99 (P 65.1 / R 10.6) |
| random fill of the editable region, same output volume as ours | 7.45 | 13.33 |
| **trained completion** | **12.74** (P 15.9 / R 39.2) | **30.89** (P 67.1 / R 36.4) |

The map alone is the same object on both benchmarks: high precision, ~10 % recall, a third
or a sixth of the true volume. It marks what the camera saw and nothing else.

The two datasets then diverge in *how* the module buys recall. On **Occ3D** recall goes
10.6 → 36.4 while precision goes 65.1 → **67.1** — it more than triples coverage at no
precision cost at all, and beats every baseline including a density-matched random fill by
17.6 points. On **SemanticKITTI** recall goes 10.8 → 39.2 but precision collapses 33.3 →
15.9: it pays for coverage, and §8.2.1 shows where that payment goes — 71 % of it into a
slab above 2 m. Training still beats both the raw map (+3.8) and matched random (+5.3)
there, but not by enough to beat a 0.4 m dilation of the map (15.41).

The random row matters because it is volume-matched: it rules out "the module just predicts
more voxels". At identical output volume, chance placement scores 13.33 on Occ3D and the
network scores 30.89.

### 8.1.1 Two ablations that change how §8.1 must be read

Full record: `artifacts/gate8c1/scale_and_prior_ablation.md`.

**(a) MoGe-2's scale is load-bearing, and no constant replaces it.** The LingBot canonical
unit is ~1/25 of a metre and the factor varies **5.9× across the 150 Occ3D scenes**
(9.37 → 55.28, median 24.48).

| Occ3D condition | map IoU | completion IoU |
|---|---|---|
| `s = 1` — LingBot-Map alone | **0.00** | 24.39 * |
| `s = 24.48` — one constant everywhere | 9.24 | 27.33 * |
| per-scene MoGe — **deployed** | **10.43** | **29.77** |

\* out of distribution for the network. Without MoGe there is no metric map at all — 0.00,
not a reduced score.

**(b) Most of the Occ3D score is a scene-independent prior — and the observations hurt.**
Row (a)'s first line is the tell: the map scores 0.00 yet the completion scores 24.4. Hand
the network a map in which *every cell is unobserved*:

| Occ3D, 40 anchors | IoU | precision | recall |
|---|---|---|---|
| **blind prior — no observations at all** | **44.43** | 84.45 | 48.39 |
| deployed (real map) | 31.41 | 68.37 | 36.75 |
| fill every valid voxel | 23.78 | 23.78 | 100.00 |

The blank-input prediction is **byte-identical on 40/40 anchors** — one constant volume that
fills the three ground layers (z = −0.8, −0.4, 0.0 m at 100/100/98 %) plus the top layer
(z = +5.2 m at 99.5 %, the same slab as §8.2.1). Since 53 % of Occ3D ground truth inside the
camera mask *is* the road band, blindly filling the ground scores 44.4.

Why conditioning hurts: the residual lock freezes 13.4 % of cells (2.9 % occupied,
10.5 % free), and of the cells locked *free* the ground truth says occupied for **26.6 %**.
But that only accounts for ~4.3 IoU; the other ~8.7 is the network's own response to map
evidence in the open region.

**What this obliges you to say.** The Occ3D result is *not* evidence that the completion
module transfers — a ground-plane prior from the same checkpoint beats it, beats OccAny's
20.7, and beats every baseline. What transfers is a prior, which is cheap and is not the
claim. Any Occ3D number, ours or anyone's, should be quoted next to the blind-prior row.
SemanticKITTI does not show this: 12.26 deployed vs 12.63 blind, i.e. the observations are
roughly neutral there rather than harmful.

**Read it as:** on Occ3D the module is precise and conservative and beats every baseline
with paired CIs excluding zero. On SemanticKITTI it over-predicts 2.4× and loses to a plain
0.4 m dilation. Ranking is genuinely above chance on both (AUROC 0.65 / 0.76) — so on
SemanticKITTI it is the *decision boundary*, not the ordering, that is wrong.

### 8.2 What the pictures show

Figures live in `artifacts/gate8c1/`. Four of them, each answering a different question.

**`fig_qualitative_semantickitti.png` / `fig_qualitative_occ3d.png`** — one row per moment:
**camera → causal map → our completion → ground truth → error from above → error from the
side**; green = correct, red = hallucinated, blue = missed.

* **SemanticKITTI**: the side view shows a solid **red slab across the top of every scene**,
  3–4 m up, running the full 50 m. The module predicts solid matter in mid-air. One artefact
  explains the whole number. It also misses building fronts (blue) — it fills open space and
  leaves real walls empty.
* **Occ3D**: no slab of that severity. Green at ground level, blue above. The best row
  reaches IoU 0.647 alone. A fog-and-rain row is the honest counter-example: the map is
  nearly empty and the module **declines to invent geometry** — its error is almost all blue
  (missed), not red. That is the failure mode you want.

**`fig_3d_gallery_semantickitti.png`** — six input frames from SemanticKITTI 08 in 3D, the
same visual language. No OccAny column: their released nuScenes checkpoint does not cover
this benchmark, and their KITTI checkpoint has not been run here. What replaces it is more
useful anyway — **the causal map is shown next to the completion**, so panel 3 against panel
4 is the trained module's entire contribution: green in 4 that is not blue in 3 was *filled
in* by the 0.99 M network, not observed.

| panel | what it is |
|---|---|
| 1 | camera at t |
| 2 | ground truth, neutral grey |
| 3 | **causal map — the module's input** (blue = observed in the 5 past frames) |
| 4 | completion: only what it recovered |
| 5 | + what it invented, in red |
| 6 | panel 5 again **from ground level** |

**Panel 6 is the point of the figure.** The SemanticKITTI failure is a *height* failure, and
an oblique view from above hides it almost perfectly. From the side it is unmistakable: a
solid red sheet floating over every scene, in all six frames.

Measured over those frames (`slab_profile_semantickitti.json`):

* **71 % of all false positives sit above 2 m**, where only **1.8 %** of the real geometry is.
* The four layers from +3.7 to +4.3 m carry **64 %** of the false-positive mass and
  **0.0 %** of the ground truth.
* Per frame the completion produces ~30–67 k correct voxels against ~187–312 k red ones.

### 8.2.1 What the slab costs — a diagnosis, not a result

`slab_diagnosis_semantickitti.json`, 24 frames. **Every row but the first is post-hoc**: the
cut height was chosen by *looking at* SemanticKITTI, which is precisely the target-domain
tuning the gate forbids. It is reported to size the artefact, never as a score.

| predictions dropped above | SC IoU | precision | recall |
|---|---|---|---|
| **nothing (the frozen result)** | **12.26** | 15.87 | 35.06 |
| 4.0 m | 18.39 | 27.90 | 35.05 |
| **3.5 m** | **21.42** | 35.52 | **35.04** |
| 2.5 m | 22.37 | 38.31 | 34.96 |
| 2.0 m | 22.60 | 39.10 | 34.88 |

**Read the recall column.** Removing everything above 3.5 m costs **0.02 points of recall**
and gains **9.2 points of IoU**. The slab recovers nothing whatsoever — it is pure false
positive. So the SemanticKITTI failure is not "the module does not transfer"; it is one
localised artefact sitting on top of a module whose actual predictions rank correctly
(AUROC 0.651) and whose recall is unaffected by removing it.

That reframes the next gate entirely: find why the module asserts occupancy in a horizontal
sheet near the top of the grid, rather than redesigning the module.

**`fig_occany_3d.png`** — six panels per scene, one viewpoint throughout:

| panel | what it is |
|---|---|
| 1 | ground truth in **neutral grey** — the shape to be reproduced, nothing more |
| 2 | OccAny: **only what it recovered** — green over grey ghosts, no red at all |
| 3 | ours: only what it recovered |
| 4 | OccAny: the same, with its **hallucinations added back** in red |
| 5 | ours: the same |
| 6 | **who recovers what** — blue only-ours, orange only-OccAny, grey both. Voxels neither recovers are dropped. |

Columns 2 and 3 are the pair to compare first: with red removed the eye compares *coverage*
directly instead of netting two colours against each other. Columns 4 and 5 then show what
each method paid for that coverage. The figure shows **eight distinct scenes**, one anchor
each, spread across the split — we win IoU in five, OccAny in three, which is an honest
spread rather than a curated one.

**One caveat on panel 6, worth stating before a supervisor does.** It counts *recovered*
voxels only, so it rewards recall and is blind to hallucination. On several scenes OccAny's
only-OccAny count exceeds ours while its IoU is lower — it recovers more and invents far
more. Panel 6 is the fastest read; the IoU in columns 4 and 5 is the fair one.

Misses are drawn at 30 % edge length. They are *not* hidden — hiding them would flatter both
methods — but at full size they occlude everything that distinguishes the two.

**How to read red — it is not "nothing is there".** Red means *the ground truth marks this
voxel free*, which is a weaker statement. Audited over 60 samples
(`occany_fp_composition.json`):

| | ours | OccAny |
|---|---|---|
| false-positive voxels | 130 630 | 266 107 |
| within 1.2 m of real occupied geometry | **39 %** | 43 % |
| nearest real voxel is `manmade` | 30.5 % (GT share 19.5) | 19.7 % (19.5) |
| nearest real voxel is `vegetation` | 27.5 % (19.3) | 19.3 % (19.3) |
| movable-class enrichment (car/truck/pedestrian/…) | **1.27×** | **1.60×** |

Three readings, in order of size:

1. **~40 % of red is displacement, not invention** — it sits within three voxels of real
   structure, put back slightly thick or slightly offset.
2. **Our red concentrates on facades and canopy** (`manmade` and `vegetation` enriched
   ~1.5× over their ground-truth share, `driveable_surface` *de*-enriched to 0.6×). We
   over-thicken vertical structure; we do not spray the road.
3. **Movable objects are genuinely enriched next to red** (1.27× ours, 1.60× OccAny), which
   is the expected signature of a **moving object smearing across the five frames the map
   accumulates while the target is a single instant**. So red landing squarely on a car is a
   real effect — but it accounts for well under a tenth of the false-positive mass, and it
   is *larger* for OccAny than for us.

**`fig_occany_why_missed.png`** — the answer to "why is so much missed by *both*". It is a
property of the benchmark, not of either method:

* Occ3D ground truth is **accumulated LiDAR over the whole 40 m box**. **53 % of it is the
  road-surface band** (−0.8 to +0.6 m); the rest is facade and tree canopy that a single
  forward-facing camera never observes at all.
* Recall *and* precision are plotted per height, because recall alone at a height is
  misleading — a method that floods a slab scores well on it.

| height | GT share | ours R / P | OccAny R / P |
|---|---|---|---|
| −0.8 m | 6.0 % | **99.6 / 84.3** | 11.9 / 70.6 |
| −0.4 m | 14.1 % | **95.8 / 86.8** | 24.9 / 77.6 |
| +0.0 m | 22.8 % | 30.4 / **92.9** | **48.6** / 82.9 |
| +0.4 m | 10.5 % | 14.9 / **60.9** | **46.1** / 47.2 |
| +0.8 … +2.0 m | 19.3 % | ~10 / ~28 | ~20 / ~17 |
| +5.2 m | 2.9 % | 94.2 / **37.7** | 13.8 / 50.5 |

**Three things to take from that table.** (1) We own the road surface — near-total recall at
high precision where half the ground truth lives, which is where the IoU advantage comes
from. (2) OccAny is genuinely better in the 0 to +1 m band, and saying so costs nothing.
(3) The **+5.2 m row is our SemanticKITTI artefact showing up on Occ3D too** — 94 % recall
at 38 % precision in the top slice is the same mid-air slab, just confined to one layer
holding 2.9 % of the ground truth instead of dominating the volume. That is a lead, and it
is why the SemanticKITTI failure and this one are probably a single bug.

Numbers behind the table: `occany_height_profile.json` (60 validation samples).

**`fig_occany_2d.png`** — the same comparison in bird's-eye, with per-row IoU/P/R and
pred÷GT. Useful for the road-extent story; the 3D figure is better for structure.

### 8.3 The OccAny comparison — now a real head-to-head

**The released OccAny checkpoint was downloaded and run on this machine** over all 4 819
Occ3D-nuScenes validation samples with its published settings, and scored by **its own**
`compute_metrics_from_saved_voxels.py`:

| OccAny, full official val (4 819 samples) | precision | recall | SC IoU |
|---|---|---|---|
| with majority pooling (their default) | 36.11 | 40.39 | **23.56** |
| **published (paper / README)** | 36.09 | 40.39 | **23.55** |
| raw, no pooling | 42.67 | 28.61 | **20.67** |

**We reproduce their published number to 0.01.** Two things follow: the pipeline is
faithful, and the published 23.55 % is definitively the **pooled** number — their raw
number is 20.67 %.

Three independent checks make the comparison airtight:

* their `SSCMetrics` on our predictions returns **byte-identical** TP/FP/FN to our counting
  (42.67 / 28.61 / 20.67 from both, on all 4 819);
* their stored ground-truth `voxel_label` is **byte-identical** to the target our evaluator
  builds, on every matched sample;
* grid, masks, split, camera and aggregation all match their code line for line.

**Head-to-head on 882 samples both methods evaluate**, everything scored by OccAny's own
evaluator and both sides using **OccAny's own pool-then-mask order**:

| method | SC IoU | precision | recall | pred/GT |
|---|---|---|---|---|
| OccAny, raw | 20.64 | 42.55 | 28.61 | 0.67 |
| OccAny, pooled *(= their published protocol)* | 23.48 | 35.96 | 40.36 | 1.12 |
| **ours, matched forward input, raw** | **27.56** | 63.83 | 32.66 | 0.51 |
| **ours, matched forward input, pooled** | **40.47** | 55.75 | 59.62 | 1.07 |
| ours, causal 5 past frames, raw | 30.43 | 65.16 | 36.34 | 0.56 |
| ours, causal 5 past frames, pooled | 41.13 | 55.48 | 61.39 | 1.11 |

OccAny scores 20.64 / 23.48 on this subset against 20.67 / 23.56 on all 4 819, so the subset
is representative.

**The two rows to quote** are the matched-input ones, where OccAny and we consume the
*same five frames*: **27.56 vs 20.64 raw (+6.9)** and **40.47 vs 23.48 pooled (+17.0)**.

*A correction to our own earlier number.* `gate8c1_results.json` reports our pooled Occ3D
result as 33.43 %. That applied the valid mask **before** pooling; OccAny pools the full
prediction and lets the evaluator mask afterwards. Measured on the same 250 samples the two
orders give **42.04** (their order) versus **33.26** (ours) — so our previously reported
pooled figure was *stricter on us* than OccAny's own convention. The head-to-head table
above uses their order for both sides. Raw numbers are unaffected.

**What still favours OccAny**, and must be said: they trained on nuScenes and we did not,
and they tune the confidence threshold per dataset (1.1 nuScenes, 2.5 KITTI) while ours was
frozen on KITTI-360 before Occ3D was opened. In the causal rows our input is additionally
strictly harder — no future observation at all.

Full record: `artifacts/gate8c1/occany_headtohead_verified.json`.

## 9. Honest limitations — say these before you are asked

1. **Not zero-shot.** The completion module never sees the target dataset, but LingBot-Map's
   published training mixture includes KITTI-360, and MoGe-2's and Trident-H's training data
   have not been audited. The defensible phrase is **"target-domain-free transfer of the
   completion module"**.
2. **SemanticKITTI fails.** 12.74 % against a 16.00 % dilation baseline, with 2.4×
   over-prediction. Under the Gate 8C-1 decision rule this is a **FAIL** overall.
3. **Semantics do not work.** 2.65 % / 4.13 % SSC mIoU, below the plain dilation baseline on
   both. Only the geometry half transfers.
4. **KITTI-360 cannot currently be evaluated** — its published completion label is defective
   (§7). Every KITTI-360 number in Gates 5.2–8B should be withdrawn, not repaired.
5. **The evaluation subsample** is 1 182 of ~4 800 possible Occ3D anchors (a uniform 1-in-5
   stride). Checked representative, but it is a subsample.
6. **One-source training.** Only KITTI-360. Whether a second clean source helps is untested.

---

## 10. Where everything lives

```
gate8/          mapper.py (map + ScaleState) · net.py (CompletionUNet, residual rule)
                targets.py (32-channel export, privileged targets) · losses.py · vocab.py
gate8a/         regions.py (editable region) · scores.py (threshold-free metrics) · boot.py
gate8b/         pooling.py (OccAny's post-processing, verified bit-for-bit) · clips.py
gate8c0/        transforms.py (the KITTI-360 chain) · oracle.py · checks.py
gate8c1/        rawtarget.py (raw-LiDAR supervision) · sources.py (the firewall)
                occany_eval.py (their metric) · render3d.py (3D voxel rendering)
tools/gate8c1/  build_targets · validate_targets · build_samples · train · selection
                freeze_manifest · eval_target · aggregate · report · visualize · compare_occany
artifacts/gate8c1/  report.md · frozen_manifest.json · source_target_audit.md
                    occany_reproduction.md · occany_comparability.json · fig_*.png
reports/        gate8a/, gate8b/, gate8c0/ reports · METHOD.md (this file)
```

Reproduce (GPUs 1–3, conda env `cu128`):

```bash
tools/gate8c1/run_targets.sh          # rebuild raw-LiDAR supervision
python tools/gate8c1/validate_targets.py
tools/gate8c1/run_samples.sh
tools/gate8c1/run_train.sh            # seeds 0,1,2 concurrently — 0.72 GPU-h total
python tools/gate8c1/selection.py     # KITTI-360 drive 0006 only
python tools/gate8c1/freeze_manifest.py    # <- the firewall lifts here
python tools/gate8c1/eval_target.py --dataset occ3d --mode past5 --seed 0
python tools/gate8c1/aggregate.py && python tools/gate8c1/report.py --write
python tools/gate8c1/visualize.py --dataset occ3d
python -m pytest tests/gate8c0 tests/gate8c1 tests/gate8 tests/gate8a tests/gate8b -q   # 120 pass
```

---

## 11. How to present it

**The claim, in one sentence:** *a 0.99 M-parameter causal completion module, trained on one
dataset with no human labels and no target-domain data, reaches 30.4 % scene-completion IoU
on Occ3D-nuScenes against 20.6 % for a 651 M-parameter model that trained on that dataset
and sees four seconds of future — while using only past frames.*

**The three slides that carry it:**

1. *The split* — the frozen/trained table in §3. One trained component, 0.99 M parameters,
   0.72 GPU-hours.
2. *The pictures* — panel 4 of `fig_occany_3d.png` ("who recovers what") is the single
   strongest slide: blue is ours, orange is theirs, and no interpretation is required.
   Follow it with `fig_occany_why_missed.png` so nobody has to ask why both methods miss so
   much — it is 53 % road band and 47 % facade-and-canopy a forward camera never sees.
3. *The honesty slide* — §9, and it is now the strongest slide you have. Present the
   SemanticKITTI failure yourself with panel 6 of `fig_3d_gallery_semantickitti.png`: a red
   sheet floating over all six frames. Then the two numbers that turn a failure into a
   lead: **71 % of the false positives sit above 2 m where 1.8 % of the geometry is**, and
   **removing everything above 3.5 m costs 0.02 points of recall and gains 9.2 points of
   IoU** (§8.2.1 — say plainly that this is a diagnosis, not a result). Close by noting the
   +5.2 m row of the Occ3D height table shows the same artefact in miniature, so it is one
   bug rather than two datasets disagreeing.

**Questions you should expect, with the answers:**

* *"What did you train?"* → §3.
* *"Is this zero-shot?"* → No. §9.1. Use "target-domain-free transfer of the completion
  module".
* *"Is the Occ3D result real?"* → Partly, and say so first: a blind prior from the same
  checkpoint scores 44.4 there against our 31.4 (§8.1.1b). Lead with that, not with 30.9.
* *"You beat a 1.5 B model with 1 M parameters?"* → On Occ3D geometry, yes, under matched
  protocol (30.51 vs 23.55). But they saw nuScenes in training, tune per dataset, and we
  fail on SemanticKITTI. The tiny module is only doing completion — the heavy lifting is in
  the frozen models beneath it.
* *"Why is SemanticKITTI so bad?"* → One artefact, now measured: a horizontal slab of
  predicted occupancy near the top of the grid. 71 % of false positives above 2 m; cutting
  it costs no recall at all and would move 12.3 → 21.4 IoU (a diagnosis, not a claim).
  AUROC is 0.65, so the ranking is fine and the decision boundary is not. §8.2.1.
* *"Why no KITTI-360 result?"* → Its published label contradicts its own LiDAR 71 % of the
  time (§7). We measured it against SemanticKITTI as a control, and against SSCBench's own
  voxel input. Withdrawn rather than repaired.

**What to propose next:** the SemanticKITTI height artefact is the highest-value target —
it is specific, visible and probably cheap to fix. Second, a controlled test of whether a
second clean training source helps. Do **not** propose scaling the network; nothing in the
evidence points there.
