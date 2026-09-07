# Gate 8C-1 — the ceiling bug, its root cause, and one model that fixes both benchmarks

Asked for: find and fix the ceiling occupancy problem; produce a **single** KITTI-360-trained
checkpoint that does well on SemanticKITTI *and* Occ3D-nuScenes, beating OccAny on both.

Delivered: the bug is fixed and one model improves both benchmarks substantially. It beats
OccAny on Occ3D by a wide margin. **It does not beat OccAny on SemanticKITTI**, and the last
section shows why that is not a threshold or a bug.

## Root cause — two defects, both in how the grid boundary is handled

**1. Zero padding manufactures an impossible input state.** Input channels 2 and 3 are
`observed` and `1 - observed`; they sum to 1 in every voxel that can occur.
`nn.Conv3d(padding=1)` pads with zeros, so at the array boundary *both* are 0 — a state
absent from training. The response to it leaks inward, and the grid is only 32 voxels deep
in z. Signature: the median final log-odds rose monotonically toward the boundary
(−1.06 at +1.9 m → **+0.13** at +4.3 m) exactly where ground-truth occupancy fell to 0.01 %.

**2. The ceiling was never supervised.** `gate8c1/rawtarget.py` marked every unobserved voxel
UNKNOWN and dropped it from the loss. A Velodyne ray never climbs above its topmost ring, so
the sky was unobserved — but it is not *occluded*: nothing stands between it and the sensor
and the column below it was measured to a surface. Only **1.12 %** of the top layer carried
supervision, so nothing constrained the model there while both benchmarks label that region
EMPTY and score it.

## The fix

* `gate8/net.py`: `padding_mode="replicate"` — the truthful "outside the grid looks like its
  edge" instead of an impossible vector. Configurable, stored in the checkpoint, default
  still `"zeros"` so earlier gates stay bit-identical.
* `gate8c1/rawtarget.py`: **open sky is FREE, not UNKNOWN**. In a column whose returns reached
  a surface, voxels more than `SKY_MARGIN_VOX` above the topmost return are marked free. The
  rule is derived from KITTI-360 LiDAR geometry alone; no target dataset is consulted.
  Top-layer supervision **1.12 % → 21.9 %**, with occupancy among supervised voxels 0.39 %.
* Targets and samples rebuilt (`G8C1_DATA_ROOT` redirect, so the frozen set survives), three
  seeds retrained from scratch, checkpoint and threshold re-selected by the gate's own rule
  on KITTI-360 drive 0006. **Firewall violations: 0** at every stage.

An inference-only version of fix 1 was tried and rejected: the model had been *trained* with
the defect, so patching at test time is a train/test mismatch and cost source IoU
(11.33 → 8.99). The fix requires the retrain it got.

## The ceiling is gone

SemanticKITTI, voxels above 2 m (16 anchors):

| | before | after |
|---|---|---|
| predicted occupied | 22.24 % | **1.80 %** |
| ground truth occupied | 0.27 % | 0.43 % |
| median final log-odds | −0.42 | **−1.56** |
| top layer (+4.3 m) predicted | 95.1 % | **0.62 %** |

Source-domain quality improved too: AUROC **0.62 → 0.68**, AP/prevalence **1.63 → 2.04**.

## One model, both benchmarks

Same checkpoint per seed, KITTI-360 only, scored by OccAny's own `SSCMetrics`.

| | before | **after** | OccAny |
|---|---|---|---|
| **Occ3D-nuScenes** (882 samples, causal) | 30.43 | **33.28** | 20.64 |
| Occ3D, matched forward input | 27.56 | **30.70** | 20.64 |
| Occ3D, pooled | 41.13 | **43.15** | 23.48 |
| **SemanticKITTI 08** (161 frames) | 12.39 | **22.61** | 25.28 |
| SemanticKITTI, seeds 0/1/2 | — | 22.61 / 23.79 / 22.83 | — |

SemanticKITTI improves by **+10.4 IoU (84 % relative)**; precision 15.7 → 32.7 and
over-prediction 2.81× → 1.30×. Occ3D improves as well, so this is not a source/target trade.

## Why SemanticKITTI still loses, and why it is not calibration

Sweeping the threshold **on SemanticKITTI itself** — target-domain tuning the gate forbids,
done here only to size the gap:

| tau | IoU | precision | recall | pred/GT |
|---|---|---|---|---|
| −0.500 | 23.54 | 29.64 | 53.32 | 1.80 |
| **−0.375 (best possible)** | **24.31** | 31.78 | 50.85 | 1.60 |
| **−0.125 (source-selected)** | **23.69** | 33.57 | 45.95 | 1.37 |
| 0.000 | 23.14 | 37.30 | 37.87 | 1.02 |
| +0.500 | 19.19 | 41.02 | 26.51 | 0.65 |

**The oracle threshold reaches 24.31 — still below OccAny's 25.28.** The source-selected
threshold is within 0.6 of the oracle, so our calibration is fine. The gap is capability:
our precision saturates near 44 % even at very high thresholds, while OccAny holds 45.5 %
precision *at* 36 % recall.

Where the remaining error sits (16 anchors): **50.7 % of false positives are in the five
layers at or below −1.0 m** — the road band, where the model over-thickens the ground. But
those layers also carry **75.5 % of true positives**, so they cannot simply be suppressed.
That is the next target, and unlike the ceiling it is not a boundary artefact: it is genuine
surface-thickness error, and closing it needs a change in capacity or supervision density,
not a bug fix.

## Files

`gate8/net.py`, `gate8c1/rawtarget.py`, `gate8c1/sources.py`,
`configs/gate8c1_sky/seed{0,1,2}.yaml`,
`artifacts/gate8c1/checkpoints/sky_seed{0,1,2}_{best,last}.pt`,
`artifacts/gate8c1/frozen_manifest_sky.json`, `selection_sky.json`,
`occany_kitti_headtohead.json`, `occany_headtohead.json`.
Data: `/media/SSD1/MINH_DATASETS/lingbot_gate8c1_sky`. Every frozen gate artifact untouched;
evaluation tools pick the variant up through `G8C1_MANIFEST`.

---

# Addendum — the residual ceiling, and a fix that did not work

Reported after the first fix: *"there are still many red cells on the ceiling."* True. Two
separate things were happening.

## 1. The elevation panel exaggerates a thin scattering into a sheet

Measured on the six frames the gallery draws: above 2 m the predicted-occupied density is
**1.8 %**, but the side view looks along 256 voxels of depth, so it integrates. A non-empty
line of sight crosses a **median of 8** false-positive voxels (p90 39, max 116). Anything
above about two reads as opaque. The panel is not lying about *presence*, but it makes a
few-percent scattering look like a solid roof.

## 2. The residue is real, and worth about the whole remaining gap

**14 % of all false positives sit above 2 m.** Removing every one of them would move
SemanticKITTI from 23.7 to **25.3** — parity with OccAny's 25.28. So this is the right thing
to chase.

Where it lives, over 12 frames:

| | share of the high false positives |
|---|---|
| columns with **no** ground-truth geometry at all (43 % of the grid) | 32.5 % |
| columns that do have geometry — i.e. *below* their own roof | 67.5 % |

## The attempted fix, and why it was rejected

The first group looked reachable: the column-wise sky rule needs a return to measure a roof
from, so a column with none stays unsupervised. The extension took the **maximum column top
over a 15×15 window**, letting an empty column borrow a roof from its neighbours
(`SKY_NEIGHBOURHOOD_VOX = 7`, conservative by construction — a window with no return frees
nothing). Rebuilt, retrained, re-selected, all firewall-clean.

Source-domain quality improved markedly: **AUROC 0.68 → 0.74**, AP/prevalence **2.04 → 2.78**.

The target results did not follow:

| | plain column rule | neighbourhood rule |
|---|---|---|
| SemanticKITTI, seeds 0/1/2 | 22.61 / 23.79 / 22.83 | 24.19 / 22.26 / 22.93 |
| SemanticKITTI median | **22.83** | 22.93 |
| **Occ3D-nuScenes** (seed 0) | **33.28** | 28.12 |

A wash on SemanticKITTI and a **5.2-point regression on Occ3D**: the rule over-carves, so
recall falls (38.2 → 31.0) faster than precision rises (72.3 → 75.0). **Not adopted.** The
plain column rule stays the model, and the shipped figures are its output.

That result is worth keeping for its own sake: source AUROC rose while target IoU fell, so
source-domain ranking quality is not a reliable proxy for target transfer here.

## What is actually left

Two thirds of the residual high false positives are *below* their own column's roof — the
occluded space between road and canopy, or behind a facade. No column-height rule can reach
it, because it is genuinely unobserved rather than merely above the sensor's rings.
Supervising it needs real multi-frame visibility reasoning, not another geometric shortcut.
