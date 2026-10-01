# Gate 21 — plan and method

A learning experiment, not an audit. One hypothesis, one pre-registered gate.

## 1. The hypothesis, stated so it can fail

Gate 20 measured that 27.21 % of correctly-completed-but-never-observed geometry becomes camera-visible
again inside the frozen 10 s privileged window, and that where it does, the future observation beats B0 by
+10.56 mIoU. 27 % is enough to train on and not enough to deploy on. So the paper claim cannot be *"future
observations label hidden space"*; it has to be:

> semantic supervision from naturally revealed hidden voxels teaches a scene prior that generalises to
> hidden voxels for which no future semantic observation exists.

Train on `U^R`, decide on `U^N`. If the gain lives only on `U^R`, the model has fitted the distribution its
own pseudo-labels came from and the claim is empty.

## 2. What is frozen

Everything upstream, loaded read-only and asserted: RGB causal clip → LingBot (local reset, 5 frames
t−4..t) → MoGe-2 clip scale → hard mapper (S/F) → CompletionUNet geometry seed 0 step 6000 → completed
occupancy. Also Trident-H and its 18 KITTI-360 phrases, B0's nearest-evidence rule, the voxel grid, the
poses, the occupancy threshold, the evaluation masks, the class mapping, Gate 20's `visible` reveal
definition and Gate 20's `mean_probability` target.

Geometry seed is **fixed at 0 for every control and seed**, so B0 is literally the same array in every
comparison. `validate.py` check 7 asserts the network's state digest is unchanged by every run and that no
gradient ever reached it; check 8 asserts `E0` is still bit-exactly the frozen region
`completed_never_observed`, rebuilt from GT rather than read from a cache.

## 3. Populations

| | 0003/0007/0010 | 0006 |
|---|---|---|
| revealed | `U^R_train` — the training loss, 2,280,255 voxels | `E1`, 876,254 |
| never revealed | `U^N_train` (unused) | **`E2`, 1,866,958 — the decision** |

`E1 ∪ E2 = E0 = ` Gate 20's `P0` = 2,743,212 voxels. Future visibility splits the evaluation population
and does nothing else.

## 4. Supervision

`U^R` = Gate 20's `P0` ∧ future-visible. Gate 20's `visible` level is used unchanged: a future LiDAR return
in the voxel that passes the frozen 4 px z-buffer test (tolerance max(0.5 m, 0.05 z)) and carries a
bilinear Trident-H token. Because that level guarantees at least one view with a token, "future-visible"
and "a valid target exists" are the same condition — there is no second, looser rule anywhere.

The target is Gate 20's `mean_probability` **vector**, not its argmax. Gate 20 stored only the argmax, so
`reveal_record` gained one optional output (`want_mean_prob`) that writes the vector it was already
computing. The build asserts the vector's argmax equals Gate 20's stored class on every voxel; a test
asserts the option changes none of Gate 20's other outputs. Storage is float16: over all 2,280,255 voxels
282 argmaxes (0.0124 %) differ from the fp32 readout, and the largest top-2 probability gap among them is
exactly 0.0 — every one is an exact tie, so rounding changed tie-breaking and nothing else. The check
downstream is "no non-tie disagreement", which is stronger than a rate tolerance.

Loss: `−Σ_c q_c log softmax(logits)_c`, soft, never hardened.

## 5. The model

Gate 15's `PropagationHead` body, imported unchanged: four residual blocks of dilated 3×3×3 convolutions
at width 32, dilations 1/2/4/8, receptive-field radius 30 voxels = 6 m. Output changed to 18 class logits,
because section 8 of the brief rules out regressing CLIP/DINO/Trident features — B4 and B5 already tried
that and the result was ambiguous about where the information came from.

Input channels, fixed before training, in **two disjoint groups** so that C2/C3/C4 is a clean factorial
rather than three different networks:

| group | channels | width |
|---|---|---|
| geometry | `trunk` (frozen CompletionUNet decoder `d1`) | 24 |
| | `hard_S`, `hard_F` (observed surface / carved free) | 2 |
| | `occupancy` (frozen completed occupancy `O`) | 1 |
| | `height`, `range` (voxel-centre position; pure functions of the index) | 2 |
| semantics | `evidence_mask` | 1 |
| | `evidence_dist` (B0's own EDT, log-normalised) | 1 |
| | `evidence_proj` (learned 1024 → 32 of the unit teacher feature) | 32 |

`evidence_dist` is grouped with **semantics**, not geometry: it is a function of where Trident-H evidence
landed, so a "geometry-only" model using it would not be geometry-only.

| control | channels | parameters |
|---|---|---|
| `C1_prior` | none — one learned 18-vector | 18 |
| `C2_geometry` | geometry only | 223,506 |
| `C3_full` | both — **the primary model** | 257,394 |
| `C4_semantics` | semantics + `occupancy` | 256,498 |

All well under the 1 M ceiling. `C1`'s optimum under this loss is the mean training target, which
`validate.py` check 11 also computes in closed form and compares.

## 6. Training

No crop. The head runs on the full 256×256×32 grid at training time, exactly as at evaluation time, so no
result can be an artefact of a crop boundary. 3,000 steps (the pilot's schedule), 2 anchors per step,
AdamW lr 1e-3 / wd 0.01, cosine schedule, grad clip 1.0 — the Gate 15 / B5 / pilot convention, unchanged.
Three seeds. **No learning-rate sweep and no architecture search.** The reported checkpoint is the final
step; nothing is selected on a dev metric.

## 7. Leakage control

The brief's section 11 is the hardest constraint, so it is enforced four independent ways:

1. **By construction** — no input channel is derived from a future frame, and `validate.py` check 3 tests
   the channel names against a whitelist.
2. **By file audit** — building the head input runs under `FileAudit(INPUT_FORBIDDEN)`, which forbids both
   the Gate 20 and Gate 21 reveal caches as well as LiDAR, GT poses and human labels. Check 4 records that
   nothing from either cache was opened.
3. **By corruption** — check 5 shuffles the reveal column and then sets it to all-hidden, and requires the
   prediction to be bit-identical. This is the operational content of "the prediction function is identical
   for E1 and E2".
4. **By fingerprint** — every training step takes an exact float64 fingerprint of the head input and of the
   whole causal state before the target file is opened, and again after the backward pass, and asserts both
   unchanged.

Separately, the **training process never opens a file of the development drive**: `drive_0006` is on the
training firewall's forbidden pattern list. That is why the training curves are swept over saved
checkpoints after training rather than measured inside the loop — evaluating 0006 between optimiser steps
would have required opening 0006's ground truth and weakened the claim to nothing.

## 8. Selection bias

Gate 20 showed revealability is not random. `selection_bias.py` tabulates four cells — revealed/never
revealed × train/dev — over class, distance-to-evidence, height, range and thing/stuff, each with its
total-variation distance from `U^R_train`. Nothing is corrected: section 12 asks for the measurement, and
an inverse-frequency weighting would be a separate ablation, not a silent change. The numbers exist so a
failure can be attributed.

## 9. Known limitations, stated up front

- **`U^R` is built with GT occupancy.** Section 4 of the brief specifies "GT occupied for training-population
  construction", so this is as designed, but it means the training population is not fully deployable. The
  semantic *target* uses no GT; only membership does.
- **One drive decides.** 0006, 590 anchors. The bootstrap covers anchor sampling within it, not drive-level
  variation.
- **10 s is the whole future.** "Never revealed" means never inside the frozen privileged window, not never
  in the universe. A longer window would move some of `E2` into `E1`; Gate 20 showed it would move very
  little, since the median reveal is 0.53 s and the p90 is 1.58 s.
- **The teacher caps everything.** Gate 20's future target itself reaches only 19.01 mIoU where it exists,
  and Gate 18's current-evidence teacher 16.91. No amount of transfer can exceed what Trident-H can say.

## 10. Repository notes

`tools/delayed_semantic/targets.py` carries one additive field from Gate 20 (`CameraView.in_image_index`,
used by the frustum diagnostic). That package is untracked at HEAD, as every gate tool since the pilot is,
so the field is not committed on its own; committing one file of an untracked package would leave a broken
partial package. Gate 21's own additive change — `want_mean_prob` on `reveal_record` and `want_trunk` on
`build_targets.causal_state` — is in `tools/future_revealability/`, which **is** tracked, and is committed
with this gate. Both are covered by tests that assert Gate 20's existing outputs are unchanged.

## 11. The survey section 7 asks for, and what it concluded

Existing implementations were read before anything new was written:

| implementation | what it is | why it was or was not reused |
|---|---|---|
| `tools/semantic_completion/model/propagation_head.py` (Gate 15, B1) | 1×1 stem + 4 dilated residual blocks, width 32, → 1024-d unit feature, 290 k params | **body reused verbatim** (`DilatedResidualBlock` is imported, not copied). It is the only tested module in the repository that mixes information across voxels at the right scale and small size. Only the output head changed, to 18 logits. |
| `tools/semantic_residual/model.py` (B5) | per-voxel MLP, width 128, depth 2, additive correction to B0's 1024-d feature | **not reused as the model.** Its own docstring states "the head itself is a per-voxel MLP: it mixes nothing across voxels" — all spatial context arrives through `trunk`. A semantic-completion prior has to combine neighbouring context, so a per-voxel MLP cannot answer Gate 21's question. Its *experimental* design — query sampling, three seeds with geometry fixed at seed 0, pre-registered pass conditions — is what Gate 21 follows. |
| `tools/semantic_field/model/semantic_field_head.py` (B4) | multi-resolution residual 3D UNet (1, ½, ¼), → 1024-d feature | **not reused.** It regresses the foundation feature, which section 8 of the brief rules out, and it is the larger model that section 7 rules out for a first proof of mechanism. Its *input* layout — trunk 24 + geometry 3 + evidence 32 + mask 1 + count 1 = 61 channels — is almost exactly the set chosen here, arrived at independently, which is some corroboration that the channel list is the natural one. |
| `tools/delayed_semantic` (pilot B1/B2) | learned evidence association over k nearest evidence voxels | **not reused as the model**: it is a propagation mechanism by construction (it predicts from candidate evidence features), which is the hypothesis Gate 21 is trying to move away from. Its `ViewCache`, stream, target config and region definitions **are** reused, through Gate 20. |
| `core/model/net.py` `CompletionUNet` | the frozen geometry network | loaded read-only; its decoder feature `d1` is channel group 1 of the input. Never retrained. |

One channel B4 and B5 both used was deliberately left out: `log1p(ray count)` at evidence voxels. It is a
legitimate causal signal, but the channel set was fixed and written down before training as section 9
requires, and adding a channel afterwards would be a tuning decision made after seeing a result. It is
noted here as an available, untried input rather than quietly added.
