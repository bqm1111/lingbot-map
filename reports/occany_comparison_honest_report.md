# Our causal system vs OccAny — honest results report

**Compiled 2026-10-16.** Every number traces to a file named in §8. Nothing is recalled from memory.
Companion files: `reports/occany_comparison_results.tex` (tables for a supervisor), 
`reports/paper_numbers_occany_comparison.md` (which numbers to put where).

**Status in one line.** SemanticKITTI is complete and the comparison is fair, including causal-vs-causal.
Occ3D-nuScenes is **incomplete**: no OccAny causal number exists there, so every Occ3D "causal" row is
*our causal* against *their forward*. Treat Occ3D as provisional.

---

## 1. What a reviewer will accept

| claim | evidence | strength |
|---|---|---|
| The system is causal (frames $t{-}4..t$ only) | by construction; protocol frozen and hash-checked | solid |
| Causal occupancy **matches** OccAny's forward configuration on SemanticKITTI | $-0.04$, CI $[-0.77,+0.70]$ | solid |
| Causal occupancy **beats** OccAny's own causal configuration on SemanticKITTI | $25.89$ vs $24.02$ | solid, frame budgets differ |
| Causal semantics **beat** OccAny's causal configuration on SemanticKITTI | $4.45$ vs $3.97$ | solid, frame budgets differ |
| OccAny's semantic lead over us is **entirely** its future window | $3.97 \to 7.31$ ($+3.34$) | solid |
| Causal occupancy beats OccAny on Occ3D single-camera | $+10.98$ raw, 92 % of blocks | **provisional** — their side is forward |
| We lead on semantics overall | — | **false**, do not claim |
| We lead in the 6-camera setting | their $34.15$ vs our $32.62$ | **false**, do not claim |

---

## 2. SemanticKITTI 08 — complete

### 2.1 Occupancy (SC IoU), OccAny's samples, labels, placement, scorer

| our window | pooling | ours | OccAny (forward) | Δ, paired 95 % CI |
|---|---|---:|---:|---|
| **causal** | none | 26.32 | 25.21 | **+1.11** [+0.30, +1.93] |
| **causal** | geometry | 25.89 | 25.93 | **−0.04** [−0.77, +0.70] |
| forward | none | 29.82 | 25.20 | +4.61 [+3.73, +5.50] |
| forward | geometry | 28.66 | 25.92 | +2.74 [+2.05, +3.43] |

805 samples (801 causal-eligible). The forward rows are *our model fed future frames* — they match
OccAny's protocol but are not a causal result and must not be the headline of a causality paper.

**Cost of causality, our own system, identical targets:** 28.66 → 25.89 = **−2.77 SC IoU**.

### 2.2 Causal vs causal

| geometry pooling | SC IoU | frames |
|---|---:|---|
| **ours** | **25.89** | 5 past |
| OccAny 1-frame (causal) | 24.02 | 1 |
| OccAny 5-frame (forward) | 25.93 | 5 future |

### 2.3 Semantics (Track O, semantic pooling both sides)

| system | window | SSC mIoU | superclass | SC IoU | n |
|---|---|---:|---:|---:|---:|
| ours, B0 (nearest evidence) | causal | 4.39 | 8.34 | 26.40 | 801 |
| **ours, B3 (3-seed mean)** | causal | **4.45** | 8.39 | 26.40 | 801 |
| **OccAny, 1-frame** | **causal** | **3.97** | 8.15 | 20.33 | 814 |
| OccAny, 5-frame | forward | 7.31 | 13.58 | 25.25 | 796 |

### 2.4 What the future window buys OccAny

| | gain from 4 future frames |
|---|---:|
| SC IoU | +1.91 (24.02 → 25.93) |
| **SSC mIoU** | **+3.34** (3.97 → 7.31) |
| superclass mIoU | +5.43 (8.15 → 13.58) |

The future helps their *semantics* almost twice as much as their geometry. This is the single most
useful finding for the paper: OccAny's semantic advantage is not a method advantage, it is a protocol
advantage.

---

## 3. Occ3D-nuScenes val — incomplete, provisional
<!--  -->
### 3.1 Occupancy — their column is **forward** in every row

| our window | pooling | ours | OccAny | Δ, paired 95 % CI |
|---|---|---:|---:|---|
| causal | none | 31.64 | 20.66 *(forward)* | +10.98 [+9.01, +12.97] |
| causal | geometry | 40.91 | 23.55 *(forward)* | +17.36 [+16.04, +18.66] |
| forward | none | 25.13 | 20.67 | +4.46 [+2.30, +6.62] |
| forward | geometry | 40.09 | 23.56 | +16.54 [+15.05, +18.03] |

Single camera (`CAM_FRONT`), front-half label (`use_surround_label=False` sets `voxel_label[:100]=255`),
4,819 samples (4,219 causal-eligible).

**Why this is not a causal comparison.** There is no OccAny causal number on Occ3D. On SemanticKITTI we
closed that with their 1-frame preset; the Occ3D equivalent was cancelled and never re-run.

**Pooling distortion is severe here.** Geometry pooling adds **+14.96** to us and **+2.89** to them,
because one-voxel dilation rewards a conservative high-precision prediction, and ours is exactly that
(raw precision 79.07, recall 26.91). **Quote +4.46 raw, never +17.36 alone.**

### 3.2 Semantics — one like-for-like pair, and OccAny wins it

| system | setting | SSC mIoU | superclass | SC IoU |
|---|---|---:|---:|---:|
| ours, B3 | 6 cameras, 1 timestep | 4.09 | 7.24 | 32.62 |
| **OccAny** | 6 cameras, 1 timestep | **6.66** | 10.32 | — |
| ours, B3 | 1 camera, 5 causal frames † | 3.80 | 6.76 | 31.94 |
| ours, B3 | 1 camera, 5 forward frames, front-half label | 2.62 | 5.08 | 25.45 |
| OccAny | 1 camera, 5 frames | *not published; not measured* | | |

† scored against the frozen protocol label, **not** OccAny's front-half label — not comparable to any
OccAny number.

OccAny's published nuScenes semantics exist **only** for the surround setting, which is a *single
timestep* (`video_length=1`) and therefore already causal in time. There is no future window to remove
from 6.66, and no causal semantic comparison on Occ3D exists at all.

### 3.3 Also true on Occ3D

OccAny's published surround **geometry is 34.15 against our 32.62**. In its six-camera setting OccAny
is ahead on occupancy. Our advantage is a **single-camera** advantage and must be written that way.

---

## 4. Reproduction gates — all passed

| check | measured | published | agreement |
|---|---:|---:|---|
| OccAny KITTI 5-frame semantic (805) | 7.27 / 13.52 | 7.28 / 13.53 | 0.01 |
| OccAny KITTI 1-frame geometry (814) | 24.02 (P 45.62 / R 33.65) | 24.03 (P 45.64 / R 33.66) | 0.01 |
| OccAny nuScenes surround semantic (377-sample shard) | 6.78 / 10.43 | 6.66 / 10.32 | 0.12 |
| our scorer vs our own frozen numbers (801) | 4.32 / 4.43 | 4.32 / 4.43 | exact |

OccAny was rebuilt and re-run here from scratch, including a GroundingDINO box cache that did not
previously exist (814 KITTI frames; 36,114 nuScenes sample-camera pairs). The OccAny checkout was never
modified — runs go through a symlink shadow project. Four local patches dated 2026-09-04 were already
in that checkout before this work.

---

## 5. Weaknesses to state in the paper, not hide

1. **Our learned semantic head is not shown to help.** The frozen decision is `PAPER C`:
   B3 − B0 = **+0.25** on SemanticKITTI with the CI **containing zero**, +0.54 on Occ3D. The semantic
   contribution over trivial nearest-evidence copying is not statistically demonstrated.
2. **Semantics are bounded by the 2D teacher**, which is 63.9 % top-1 correct and covers 12.3 % of
   hidden occupied voxels. Measured attempts to move that bound: oracle instance grouping **+0.09**
   (CI [−0.27, +0.34]); a VLM recognizer on the same regions **+0.88** (CI touching zero); bounded
   region extension **±0.02**. Re-grouping and re-naming do not work.
3. **Frame budgets differ in the causal comparison**: our 5 past frames vs OccAny's 1 frame. Their
   5-frame causal number is unmeasured (§7).
4. **Absolute numbers are below supervised methods** trained on these benchmarks. The claim must be
   *zero-shot and causal*, not absolute SOTA.
5. **Occ3D causal rows are provisional** (§3.1).
6. Both systems are zero-shot on both benchmarks — OccAny trains on DDAD, ONCE, PandaSet, VKITTI,
   Waymo; we train on KITTI-360. Neither trains on SemanticKITTI or nuScenes. This symmetry is what
   makes the comparison fair and deserves one sentence.

---

## 6. Suggested text

> Our system is causal: it consumes only the current and past frames. On OccAny's own samples, labels
> and evaluator, it matches OccAny's five-frame *forward* configuration on SemanticKITTI occupancy
> (−0.04 SC IoU, CI [−0.77, +0.70]) and exceeds OccAny's own *causal* configuration on both occupancy
> (25.89 vs 24.02) and semantics (4.45 vs 3.97 SSC mIoU). Being causal costs our system 2.77 SC IoU on
> identical targets, and OccAny gains +1.91 SC IoU and +3.34 SSC mIoU from its future window — so its
> semantic advantage is a protocol advantage rather than a method advantage. On Occ3D-nuScenes our
> causal system exceeds OccAny's published single-camera configuration by +4.46 SC IoU raw; we note that
> OccAny provides no causal configuration on that benchmark and that OccAny leads in its six-camera
> surround setting (34.15 vs 32.62).

---

## 7. Retracted — do not cite from earlier notes

| retracted (15 Oct 2026) | corrected |
|---|---|
| "the future window costs OccAny 2.69 mIoU / 10.6 SC IoU" | **+1.91 SC IoU, +3.34 SSC mIoU** |
| "our geometry leads OccAny by +11.8 SC IoU" | **−0.04 to +2.74** (SK), **+4.46 to +17.36** (Occ3D) |

Cause: a monkeypatch reversing OccAny's input window returned SC IoU 14.62 — 9.4 points *below*
OccAny's own single-frame 24.03, which is impossible because a causal 5-frame window contains that
single frame plus four more. Running OccAny's unmodified 1-frame preset reproduced 24.02, isolating the
fault to the patch. The SemanticKITTI causal-vs-forward question had in fact already been measured on
2026-09-14 (`artifacts/occany_comparison/report.md` line 41) and did not need re-deriving.

---

## 8. Open items, with cost

| item | why it matters | cost |
|---|---|---|
| OccAny 1-camera **1-frame geometry** on Occ3D | turns §3.1 from provisional into a real causal-vs-causal row | ~2.5 h |
| OccAny 1-camera **1-frame semantic** on Occ3D | the only route to a causal semantic comparison on Occ3D | ~14 h |
| OccAny 5-frame **causal** (both datasets) | removes the frame-budget asymmetry in §2.2 / §2.3 | needs a correct window implementation; one attempt already failed |

The first needs no window patch: `eval_helper.py`'s nuScenes `else` branch already gives
1 camera / `video_length=1` / front-half label, and the metric side has the same fallback. The box cache
is built and covers it.

---

## 9. Provenance

- **Occupancy tables** — `artifacts/occany_comparison/report.md`, rows 21–23 and 29–33, deployed-setting
  lines 41 and 47.
- **Semantic tables** — `artifacts/like_for_like/results_*_semantic*.json`; OccAny side
  `/media/SSD1/occany_causal/logs/{paired_forward,score_k1frame_sem,score_ns_surround_gate}.txt`.
- **Gates** — `/media/SSD1/occany_causal/logs/score_*.txt`;
  `artifacts/like_for_like/results_semantickitti_frozencheck.json`.
- **`PAPER C` decision** — `artifacts/final_zero_shot/analysis.json`.
- **Teacher bound and grouping results** — `artifacts/gate17_instance_oracle/analysis.json`;
  `/media/SSD1/region_naming_pilot/report.md`; memory note `waymo-supervision-pilot`.
- **Protocol definitions** — `tools/zero_shot_eval/practical/masks.py`,
  `tools/zero_shot_eval/official/evaluator.py`, `tools/zero_shot_eval/canonical/metrics.py`.
- **Run scripts** — `/media/SSD1/occany_causal/code/`.
