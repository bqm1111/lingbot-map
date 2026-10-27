# Which numbers to put in the paper — the OccAny comparison under a causal claim

Written 2026-10-15. Every figure below is traceable to a file named beside it. Nothing here is recalled
from memory; where a number was measured in this session the log path is given.

**The governing principle.** The paper's claim is that the system is *causal* — it consumes the current
and past frames only. The headline comparison must therefore be a **causal row**. Reporting a
forward-window number while claiming causality is the one inconsistency a reviewer is certain to catch,
and it is also the easiest to avoid, because the causal rows are good.

---

## 1. Use these

### Geometry, SC IoU. OccAny's own samples, labels, placement and scorer; our causal frames

| | ours (causal) | OccAny (forward) | Δ, paired 95 % |
|---|---:|---:|---|
| SemanticKITTI 08, raw | 26.32 | 25.21 | **+1.11** [+0.30, +1.93] |
| SemanticKITTI 08, pooled | 25.89 | 25.93 | **−0.04** [−0.77, +0.70] |
| Occ3D-nuScenes val, raw | 31.64 | 20.66 | **+10.98** [+9.01, +12.97] |
| Occ3D-nuScenes val, pooled | 40.91 | 23.55 | **+17.36** [+16.04, +18.66] |

Source: `artifacts/occany_comparison/report.md` (rows 21 and 30; deployed-setting lines 41 and 47).

**Report raw and pooled side by side — this is not optional.** OccAny's majority pooling adds **+14.96**
to our Occ3D number and **+2.89** to theirs (`report.md` line 94), so the two columns tell materially
different stories. Raw is the conservative floor. State in the text that pooling is the default of
OccAny's released metric script (`sh/compute_metric.sh`: `USE_MAJORITY_POOLING=1`,
`POOLING_MODE=separate`) and is not described in their paper.

### The extra causal-vs-causal row (measured 2026-10-15)

OccAny's `1frame` preset is a single image — causal by construction, no temporal window at all.

| SemanticKITTI 08, pooled, their metric | SC IoU |
|---|---:|
| ours, causal | **25.89** |
| OccAny, **1-frame (causal)** | **24.02** |
| OccAny, 5-frame forward | 25.93 |

Measured here: `/media/SSD1/occany_causal/logs/score_k1frame_geom.txt`, 814 samples,
P 45.62 / R 33.65 / IoU 24.02 against their published P 45.64 / R 33.66 / IoU 24.03 — reproduced to 0.01.
Caveat to state: 814 samples (theirs) vs 801 (ours, causal-eligible).

So against OccAny's causal configuration we are **+1.9**; against its future-using configuration we are
level. Both statements support the causality claim.

### Suggested wording

> Our system is causal: it consumes only the current and past frames. On OccAny's own samples, labels
> and evaluator it matches OccAny's five-frame *forward* configuration on SemanticKITTI (−0.04 pooled,
> +1.11 raw) and exceeds it on Occ3D-nuScenes by +10.98 raw / +17.36 under OccAny's pooling. Against
> OccAny's single-frame causal configuration on SemanticKITTI we are +1.9 SC IoU. OccAny's forward
> configuration observes four frames after the evaluated frame; ours observes none.

"Level with a system that sees the future" is itself a causality result. Lead with that framing instead
of searching for a larger number.

---

## 2. Do not use these

| number | why not |
|---|---|
| **32.65** SC IoU | Track P — scored on `valid ∩ horizontal FOV ∩ range ≤ 40 m` (`tools/zero_shot_eval/practical/masks.py`). A different region from OccAny's, which scores every `label != 255` voxel. Fine as our own ablation; never in a comparison table. |
| **29.82 / 28.66** as the headline | Those are *our forward-window* rows. They are our best SemanticKITTI figures precisely because they use future frames. Quoting them under a causality claim is self-defeating. |
| **+16.54** on its own | Pooling-inflated (see above). Always paired with the +4.46 raw row, or replaced by the causal pair. |
| "greatly outperforms on **both** datasets" | SemanticKITTI is a tie in our deployed setting (−0.04, interval spanning zero). Only Occ3D supports a strong claim. |

---

## 3. Disclose before a reviewer finds it

**Semantics are not our win.** SSC mIoU, Track O: ours **4.45** vs OccAny **7.28** (SemanticKITTI, their
forward, their pooling — `artifacts/like_for_like/results_semantickitti_semantic_frozencheck.json`,
`/media/SSD1/occany_causal/logs/score_forward.txt`); ours **4.09** vs OccAny **6.66** (Occ3D surround,
`artifacts/like_for_like/results_occ3d_surround_semantic.json`). If the contribution is causal
*geometry*, scope the claim to geometry and report semantics plainly.

**They beat us with six cameras.** OccAny's published surround geometry is **34.15**; ours is **32.62**
(`artifacts/like_for_like/results_occ3d_surround.json`). Our advantage is a *single-camera* advantage —
write "single-camera" into every claim and acknowledge the surround result.

**The Occ3D 5-frame comparison is a front-half-label setting**, not the full 360°
(`use_surround_label=False` sets `voxel_label[:100] = 255`), so those rows describe a restricted region.

**Both systems are zero-shot on both benchmarks.** OccAny trains on DDAD, ONCE, PandaSet, VKITTI and
Waymo; we train on KITTI-360. Neither trains on SemanticKITTI or nuScenes. One sentence; it is what
makes the comparison fair.

---

## 4. Reproduction gates passed in this session

| check | measured | published | source |
|---|---:|---:|---|
| OccAny KITTI 5-frame semantic | 7.27 / 13.52 | 7.28 / 13.53 | `/media/SSD1/occany_causal/logs/score_forward.txt` |
| OccAny KITTI 1-frame geometry | 24.02 | 24.03 | `logs/score_k1frame_geom.txt` |
| OccAny nuScenes surround semantic (377-sample shard) | 6.78 / 10.43 | 6.66 / 10.32 | `logs/score_ns_surround_gate.txt` |
| our scorer vs our own frozen numbers | 4.32 / 4.43 | 4.32 / 4.43 | `artifacts/like_for_like/results_semantickitti_frozencheck.json` |

The OccAny checkout was never modified by this work: the runs go through a shadow `PROJECT` of symlinks
(`/media/SSD1/occany_causal/occany_shadow`). Four pre-existing local patches dated 2026-09-04 were
already in that checkout before this session.

---

## 5. Retracted — do not cite these from earlier notes

Two claims made in this session were wrong and are withdrawn:

- **"Removing the future costs OccAny 2.69 mIoU / 10.6 SC IoU."** Produced by a causal-window monkeypatch
  that is discredited: it returned SC IoU 14.62, which is 9.4 points *below* OccAny's own single-frame
  24.03 — impossible, since a causal 5-frame window contains the single frame plus four more. The
  1-frame diagnostic then reproduced 24.02, isolating the fault to the patch. The real value of the
  future window to OccAny is **+1.9 SC IoU** (24.02 → 25.93).
- **"Our geometry leads OccAny by +11.8 SC IoU."** Came from comparing our 26.40 against that broken
  14.62. The true leads are the table in § 1.

The causal-vs-forward question on SemanticKITTI did not need re-deriving at all: it was already measured
on 2026-09-14 and is in `artifacts/occany_comparison/report.md` line 41 (−0.04, not distinguishable).

---

## 6. Still being measured

OccAny **1-frame semantic** on SemanticKITTI (their causal monocular setting, for which they publish no
semantic number). This is the fair semantic comparison against our 4.45 and is the only missing cell in
the causal table. Preset constructed in `/media/SSD1/occany_causal/code/presets_kitti.sh`
(`K_1FRAME_SEM`), derived from their own geometry→semantic argument delta, with each choice documented.
