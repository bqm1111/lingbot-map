# Fair evaluation protocol — matched input and scoring conditions

**Written 2026-10-20.** This fixes the conditions under which any number in this experiment
directory may be compared, before results are quoted. It records what *is* matched, what
*cannot* be matched, and which comparisons are therefore inadmissible.

---

## 1. The single most important protocol fact

**OccAny's released checkpoint has no causal five-frame mode.** Verified from
`tools/observability_split/runs.py`, which records every locally produced OccAny run, and from
the checkpoint audit:

| OccAny run | window | frames consumed | status |
|---|---|---|---|
| `occany_1f_geom` | **causal** | 1 (the anchor) | reproduces their published 24.03 monocular IoU (we measured **24.02**) |
| `occany_1f_sem` | causal | 1 | constructed here; OccAny publishes no such number |
| `occany_5f_geom` | **forward** | 5, spanning **t … t+40** | their published 5-frame geometry |
| `occany_5f_sem` | forward | 5, t … t+40 | reproduces their published 7.28 / 13.53 |

Their 5-frame mode walks *forward* in time. Their 1-frame mode is causal. There is no released
configuration that consumes five past frames.

Consequently:

* The brief's "primary setting: five causal frames from one forward-facing camera" **cannot be
  given to OccAny** without modifying their inference. One attempt to do so by monkeypatching
  the window produced SC IoU 14.62, which was **wrong and has been retracted** — a causal
  5-frame window contains their 1-frame input plus four more, so it cannot score below their
  own 24.03. Running their unmodified 1-frame preset reproduced 24.02, isolating the fault to
  the patch. That episode is the reason this document exists.
* The admissible causal comparator is therefore **OccAny 1-frame causal vs ours 5-frame
  causal**. This gives OccAny strictly less temporal context, and that asymmetry must be
  stated every time the comparison is used. It is an asymmetry **in their favour to remove**,
  not one we may quietly benefit from.
* The measured value of the forward window to OccAny is **+1.9 SC IoU** (1-frame 24.02 →
  5-frame forward 25.93). That bounds what their missing causal 5-frame mode could plausibly
  be worth, and it is small — which is what makes the 1-frame comparison usable at all.

---

## 2. What is matched

Both systems are scored by **one scorer, on one set of anchors, against one stored ground
truth**:

* **Anchors**: the intersection of samples both systems produced, 801 on SemanticKITTI
  sequence 08.
* **Ground truth**: read from OccAny's own stored `voxel_label`, and asserted identical across
  all OccAny runs per sample. We score against *their* copy of the labels, not ours.
* **Grid**: 256×256×32 at 0.2 m, the anchor's velodyne frame; OccAny's own placement.
* **Occupancy definition**: `label != 255` (valid) and `label != empty` (occupied) — the
  **Track O** convention, which is OccAny's.
* **Scoring code**: the same functions for both; no method-specific branch.
* **Postprocess**: reported twice, `raw` and `pooled` (the geometry dilation), because pooling
  is not neutral — it adds about +15 SC IoU to ours on Occ3D and about +3 to theirs. Both
  numbers are always given.

---

## 3. What cannot be matched, and is declared instead

| asymmetry | direction | handling |
|---|---|---|
| temporal context: ours 5 causal frames, theirs 1 | **favours us** | stated at every use; their forward window is worth only +1.9 SC IoU, which bounds the gap |
| OccAny selected its checkpoint on a **10-view** KITTI validation set (`min/max_memory_num_views=10`, `recon_view_idx=[0,2,4,6,8]`, `ray_map_idx=[1,3,5,7]`) | favours them at 10 views, not at 1 | a 1-frame number is off-setting for that checkpoint and is labelled so |
| training corpora differ entirely (5 datasets + MUSt3R + a prior OccAny run vs KITTI-360 from scratch) | unknown | this is the whole point of the experiment; it is why no §4 number can attribute anything to architecture |
| occupancy thresholding | method-specific | ours: the frozen threshold, never tuned on target labels. Theirs: `volume != empty & != 255` on their own stored output. Neither was fitted to the benchmark. |
| semantics | favours them | ours 4.45 SSC mIoU vs their 7.28 on SemanticKITTI forward; out of scope here, geometry is the priority |

---

## 4. Three convention traps, named once

Every SC IoU in this repository belongs to exactly one of these. Mixing them has already
caused confusion and retraction.

1. **Track P** — `valid & horizontal FOV & range ≤ 40 m`. Our SemanticKITTI SC IoU is
   **32.65** under this.
2. **Track O** — every `label != 255` voxel, OccAny's convention. Our SemanticKITTI SC IoU is
   **26.40** under this. **All comparisons in this directory use Track O.**
3. **Window** — our causal 26.32 raw against our forward 29.82 raw on the same scorer. Forward
   rows are **not our system** and may not be quoted as it.

---

## 5. Required metrics and their exact populations

Per brief §4. Occupancy IoU, precision and recall on each population below, for every method,
both raw and pooled.

| population | definition | source |
|---|---|---|
| full volume | `label != 255` | — |
| observed region | anchor-frame GT raycast reaches it | `observability_split` |
| inside frustum | voxel centre in front of the anchor camera and inside its image rectangle | `tools/waymo_arm/cli/frustum_split.py` |
| outside frustum | complement | same |
| **anchor-observed** | GT raycast from the anchor reaches it | `tools/method_vs_data/observed_strict.py` |
| **earlier-only** | not anchor-observed, reached from one of the four earlier permitted frames | same |
| **never-observed** | reached from none of the five permitted frames | same |

The last three are the brief's stricter classification and are the ones the method claim
depends on. They are built from **ground-truth occupancy and ground-truth poses** of the five
permitted frames — not from either method's reconstruction or pose estimate — so they are a
property of the permitted input and are a legitimate common denominator. The pose chain is
gated per sample: the anchor's camera→grid transform derived from GT poses must equal
`cam_to_velo` and must match the transform stored in the canonical prediction.

### Two definitional warnings, carried from the brief

* **Outside the anchor frustum is not unobserved.** A voxel behind the car at time *t* may have
  been seen cleanly at *t−4*. The frustum split cannot tell the difference; the strict split
  can.
* **Previously observed geometry is not completion.** Only `never_observed` supports a claim
  about predicting unobserved space. Any aggregate advantage must be decomposed before it is
  described as completion.

---

## 6. Statistics

Paired block bootstrap, the established repository procedure: **25 contiguous anchors per block
on SemanticKITTI, one block per scene on Occ3D, 2,000 resamples, bootstrap seed 0**, from
`tools/zero_shot_eval/practical/protocol.Statistics`. Both methods are always resampled on the
same anchors, so each interval is an interval on the **difference**.

Our models carry three training seeds; OccAny is a single released checkpoint. **Variation
across data blocks and variation across model seeds are different quantities** and are reported
separately — a block-bootstrap CI says nothing about seed variance, and our seed spread on
SemanticKITTI SC IoU is ±0.32.

A confidence interval containing zero is **not** evidence of equivalence.

---

## 7. Provenance tagging

Every number carries one of these tags, with no exceptions:

* **[published]** — from the OccAny paper or README.
* **[reproduced]** — produced on this machine from the released checkpoint, with the
  reproduction gate stated. Gates passed: KITTI 5-frame semantic 7.27/13.52 vs published
  7.28/13.53; KITTI 1-frame geometry 24.02 vs 24.03; nuScenes surround semantic 6.78/10.43 vs
  6.66/10.32 on a 377-sample shard. The OccAny checkout was never modified; runs go through a
  symlink shadow.
* **[ours]** — our system, with window (causal/forward) and track (P/O) named.
* **[off-setting]** — a number produced under conditions the system was not designed or
  selected for, such as OccAny at one frame.

---

## 8. Inadmissible comparisons

Stated so they are not made by accident later:

* Our 5-frame causal result against OccAny's **published surround-view** numbers. Their
  nuScenes 6.66 semantic figure is the 6-camera surround setting at `video_length=1`.
* Our causal result against their **forward** 5-frame result, presented as if the input
  conditions matched.
* Our **forward** rows presented as our system.
* Track P against Track O.
* Any aggregate SC IoU advantage described as better *completion* before the §5 decomposition
  has been applied.
* A **DDAD-trained** arm of ours against OccAny, as evidence about architecture — see
  `ddad_feasibility.md`; that arm is not authorized and could not be built comparably.
