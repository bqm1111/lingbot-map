# Why the completion module hallucinates occupancy — root cause

*Written 2026-09-05. Every number here was measured on this machine; the script that
produces each one is named beside it. This explains one failure mode, its two instances,
what fixing the first bought, and why the second is still open.*

---

## 0. The answer in one paragraph

Every voxel the camera has not observed arrives at the network carrying **the same 32
numbers**. The network therefore cannot tell one unobserved voxel from another, and its
output across such a region collapses to a single learned constant. What that constant
*should* be is decided entirely by the training loss — and our KITTI-360 target excluded
unobserved voxels from the loss, while both evaluation benchmarks label those same voxels
and score us on them. A region the loss never saw is a region the benchmark grades. The
negative decision threshold then turns the network's silence into a positive claim: with
τ = −0.125, an output of "no opinion" (≈ 0) is classified **occupied**. The result is a
solid slab of predicted occupancy wherever supervision was absent.

---

## 1. The mechanism, one step at a time

### 1.1 Unobserved voxels are indistinguishable to the network

The input is 32 channels (`gate8/net.py::Completer.raw`, `gate8/targets.py::unpack_sample`).
In a voxel with no evidence they take fixed values:

| channel | value in an unobserved voxel |
|---|---|
| 1 occupancy log-odds ÷ 4 | 0 |
| 2 free-evidence weight | 0 |
| 3 `observed` | 0 |
| 4 `1 − observed` | 1 |
| 5 observation count | 0 |
| 6 age | 0 |
| 7 semantic weight | 0 |
| 8–32 mean class probability | 0 |

Two unobserved voxels are byte-identical inputs. The only things that can differentiate
their outputs are (a) observed voxels within the convolutional receptive field, and (b)
distance to the array boundary.

**Verified, not argued** (`tools/gate8c1/prior_ablation.py`, `tools/gate8c1/diag_verify_prior.py`):
feeding the network a map in which *every* cell is unobserved produces a prediction that is
**byte-identical on 40 of 40 anchors** — literally one constant volume. On Occ3D it fills
160,581 of 640,000 evaluated cells (25.1 %): the three ground layers at 100 / 100 / 98 % and
the top layer at 99.5 %.

That constant volume scores **44.43 IoU** on Occ3D — higher than the deployed system's
31.41 and higher than OccAny's 23.56. A large part of what these benchmarks reward is
reproducible with no camera information at all.

### 1.2 The loss never constrained most of that region

`gate8c1/rawtarget.py` built supervision from raw Velodyne: endpoints occupied, ray
interiors free, **everything unobserved marked UNKNOWN and dropped from the loss**. The loss
masking itself is correct — `gate8/losses.py::focal_bce` weights by `gt_valid` and
normalises by `weight.sum()`, so unsupervised voxels contribute exactly zero gradient. That
is the problem, not a bug in the masking: the region receives no gradient at all.

Supervision coverage by height in the original KITTI-360 target
(`tools/gate8c1/diag_fp_layer.py`-style scan over 120 targets):

| z index | 0 | 8 | 16 | 24 | 28 | **31 (top)** |
|---|---|---|---|---|---|---|
| supervised | 8.9 % | 55.2 % | 43.7 % | 8.6 % | 2.9 % | **1.12 %** |

**1.12 % of the top layer carried any supervision.** The network's behaviour there was an
accident of initialisation, boundary effects and whatever generalised inward.

### 1.3 The threshold converts silence into a positive claim

`final = base_logodds + residual · gate`, where `gate = (|base_logodds| < 2)`. In an
unobserved voxel `base = 0`, so the gate is open and `final = residual`. The occupancy
threshold selected on KITTI-360 is **τ = −0.125**, i.e. *negative*. A residual of ≈ 0 —
exactly what an unconstrained network emits — is therefore called **occupied**.

### 1.4 The benchmarks grade the region the loss ignored

SemanticKITTI marks sky voxels **valid and empty** and scores them: at z = +4.3 m, 1,037,398
voxels are evaluated per 16 anchors and 0.00 % are occupied. Occ3D does the same. So the
unsupervised region is not merely unconstrained — it is a scored region where the answer is
known to be "empty" and we said "occupied".

---

## 2. A contributing amplifier: zero-padded convolutions

Channels 3 and 4 are `observed` and `1 − observed`; **they sum to 1 in every voxel that can
occur**. `nn.Conv3d(..., 3, padding=1)` pads with zeros, so at the array boundary it
manufactures `observed = 0` **and** `unobserved = 0` together — a state absent from the
training distribution. The response to it propagates inward, and the grid is only 32 voxels
deep in z.

Signature (`tools/gate8c1/diag_slab_bug.py`, 12 anchors, original model):

| z (m) | +1.9 | +2.5 | +3.1 | +3.5 | +3.9 | +4.3 |
|---|---|---|---|---|---|---|
| median final log-odds | −1.06 | −0.78 | −0.54 | −0.30 | −0.14 | **+0.13** |
| ground-truth occupied | 1.98 % | 0.88 % | 0.20 % | 0.06 % | 0.02 % | **0.01 %** |

Confidence rises monotonically toward the boundary exactly where evidence falls to nothing,
and it does the same at the floor. No scene prior has that shape; a zero-padded convolution
does. This is an amplifier, not the root cause: it decides *where within* the unsupervised
region the slab is strongest.

**Fixed** in `gate8/net.py` with `padding_mode="replicate"` — the truthful "outside the grid
looks like its edge". Configurable, stored in the checkpoint, default still `"zeros"` so
every earlier gate stays bit-identical.

*An inference-only version was tried and rejected*: the model had been **trained** with the
defect and adapted to it, so patching at test time is a train/test mismatch and cost source
IoU (11.33 → 8.99). The fix requires the retrain it got.

---

## 3. Instance A — the sky. Fixed.

A Velodyne ray never climbs above its topmost ring, so the sky is unobserved. But it is not
**occluded**: nothing stands between it and the sensor, and the column beneath it was
measured all the way to a surface. Calling it UNKNOWN was the mistake.

**The rule** (`gate8c1/rawtarget.py`, `SKY_MARGIN_VOX = 1`): in a column whose returns
reached a surface, voxels more than one cell above the topmost return are **FREE**. Derived
from KITTI-360 LiDAR geometry alone; no target dataset consulted.

Top-layer supervision **1.12 % → 21.9 %**, with occupancy among supervised voxels 0.39 %.

### What it bought

Targets and samples rebuilt, three seeds retrained, checkpoint and threshold re-selected by
the gate's own rule on KITTI-360 drive 0006. **Firewall violations: 0** at every stage.

| | before | after |
|---|---|---|
| predicted occupied above 2 m | 22.24 % | **1.80 %** |
| top layer (+4.3 m) predicted | 95.1 % | **0.62 %** |
| median final log-odds above 2 m | −0.42 | **−1.56** |
| share of all false positives above 2 m | 71 % | **14 %** |
| source AUROC | 0.62 | **0.68** |
| source AP / prevalence | 1.63 | **2.04** |

| benchmark | before | **after** | OccAny |
|---|---|---|---|
| Occ3D-nuScenes (causal, 882 samples) | 30.43 | **33.28** | 20.64 |
| Occ3D, matched forward input | 27.56 | **30.70** | 20.64 |
| SemanticKITTI 08 (161 frames) | 12.39 | **22.61** | 25.28 |
| SemanticKITTI, seeds 0/1/2 | — | 22.61 / 23.79 / 22.83 | — |

SemanticKITTI **+10.4 IoU (+84 % relative)**; precision 15.7 → 32.7, over-prediction
2.81× → 1.30×. Occ3D improved as well, so it is not a source/target trade.

---

## 4. Instance B — below the road surface. Still open. This is what you are seeing.

Hallucination rate by height for the shipped model, against training supervision at the same
height (`tools/gate8c1/diag_rootcause.py`, 12 SemanticKITTI anchors + 80 KITTI-360 targets):

| z (m) | +3.5 | +2.7 | +1.5 | +0.3 | −0.5 | −0.9 | **−1.3** | **−1.7** |
|---|---|---|---|---|---|---|---|---|
| unsupervised in training | 76.9 % | 73.1 % | 57.5 % | 44.5 % | 52.0 % | 59.8 % | 72.8 % | 87.6 % |
| **hallucination rate** | 1.0 % | 3.1 % | 3.5 % | 4.9 % | 10.5 % | 23.0 % | **90.1 %** | **99.9 %** |

The road band (z ≤ −1.1 m) carries **51 % of all false positives**; everything above 2 m
carries **14 %**. The dominant red is **3.6× more ground than ceiling**.

### Why it looks like a ceiling in the figures

Two rendering effects, both real and both worth knowing:

1. **The oblique panel views from +30 m looking down.** A solid red plane at *ground* level
   covers the whole scene and reads as a lid over it. The gallery's own printed statistic
   already said so — *"worst layer z = −1.3 m"*.
2. **The elevation panel integrates along 256 voxels of depth.** A non-empty line of sight
   crosses a **median of 8** false-positive voxels (p90 39, max 116), so even a
   few-percent-dense layer renders opaque.

### Why the sky trick does not transfer to the floor

Below the road the network's prior is "solid" — which is **physically correct**; that is
earth. But SSC benchmarks define *occupied* as **"contains a measured surface"**, not
"contains matter", so they label sub-surface voxels empty. Our target leaves that region
UNKNOWN, the network fills it with the physically sensible answer, and the benchmark scores
it wrong.

### The consistent fix (not yet run)

Make the target state what the task means: **a voxel more than `BAND_HALF_M` behind the
measured surface along its ray contains no surface, so it is not occupied** — the mirror of
the sky rule, derivable from KITTI-360 geometry alone rather than borrowed from the
benchmarks. Cost: rebuild targets and samples, retrain three seeds, re-select, re-evaluate —
about 40 minutes end to end. The road band holds half the remaining error, so this is where
the headroom is.

---

## 5. The evidence that isolates the cause

The correlation between "unsupervised in training" and "hallucinates at inference"
**flips sign**:

| region | Pearson r |
|---|---|
| above 1 m (where the sky rule now teaches "free") | **−0.78** |
| all 32 heights (dominated by the untaught floor) | **+0.45** |

Above 1 m, *more* unsupervised means *less* hallucination — because supervision was added
and it taught the right answer. Across the whole grid, more unsupervised means more
hallucination — because below the road nothing taught anything. **Same unsupervised
condition, opposite behaviour, decided purely by whether the loss had an opinion.** That is
the cause, isolated.

---

## 6. Three things ruled out

**It is not the threshold.** Sweeping τ *on SemanticKITTI itself* — the target-domain tuning
the gate forbids, done only to size the gap:

| tau | IoU | precision | recall | pred/GT |
|---|---|---|---|---|
| −0.500 | 23.54 | 29.64 | 53.32 | 1.80 |
| **−0.375 (best possible)** | **24.31** | 31.78 | 50.85 | 1.60 |
| **−0.125 (source-selected)** | **23.69** | 33.57 | 45.95 | 1.37 |
| 0.000 | 23.14 | 37.30 | 37.87 | 1.02 |
| +0.500 | 19.19 | 41.02 | 26.51 | 0.65 |

The oracle threshold reaches 24.31, still below OccAny's 25.28, and the source-selected
threshold is within 0.6 of the oracle. Calibration is fine.

**It is not fixable by carving more sky.** The column rule cannot reach a column with no
return of its own — 43 % of the grid, holding 32.5 % of the high false positives. Extending
it to take the maximum column top over a 15×15 window (`SKY_NEIGHBOURHOOD_VOX = 7`) was
built, retrained and evaluated. Source quality rose sharply (**AUROC 0.68 → 0.74**,
AP/prevalence **2.04 → 2.78**) but the targets did not follow:

| | plain column rule | neighbourhood rule |
|---|---|---|
| SemanticKITTI median | **22.83** | 22.93 |
| Occ3D-nuScenes | **33.28** | 28.12 |

A wash on one benchmark and a **5.2-point regression** on the other: it over-carves, so
recall falls (38.2 → 31.0) faster than precision rises (72.3 → 75.0). **Not adopted.**
Worth keeping for its own sake — source AUROC rose while target IoU fell, so source-domain
ranking is not a reliable proxy for transfer here.

**It is not padding alone.** Replicate padding without the supervision fix cleared the layers
from +3.1 to +3.9 m but left the outermost two flooded at ~74 %, and under the gate's own
checkpoint rule it gave no reliable IoU gain (seed 2 regressed 14.32 → 12.83). Padding
decides *where* the artefact concentrates; supervision decides *whether* there is one.

---

## 7. Where each number comes from

| number | script |
|---|---|
| constant-volume prior, 40/40 identical | `tools/gate8c1/prior_ablation.py` |
| log-odds ramp toward the boundary | `tools/gate8c1/diag_slab_bug.py` |
| supervision coverage by height | scan of `…/targets/*/*.npz` |
| padding fix, per-layer effect | `tools/gate8c1/padding_fix.py` |
| threshold re-selection (source only) | `tools/gate8c1/reselect_threshold.py`, `selection.py --prefix` |
| head-to-head vs OccAny | `tools/gate8c1/compare_occany{,_kitti}.py` |
| false positives by layer | `tools/gate8c1/diag_fp_layer.py` |
| supervision ↔ hallucination correlation | `tools/gate8c1/diag_rootcause.py` |
| side-view integration depth | `tools/gate8c1/diag_where_ceiling.py` |

Models and manifests: `artifacts/gate8c1/frozen_manifest_sky.json` (shipped),
`frozen_manifest_sky2.json` (rejected extension), `frozen_manifest_padfix.json`
(padding only). Every frozen gate artifact is untouched; evaluation tools pick a variant up
through the `G8C1_MANIFEST` environment variable. Narrative record:
`artifacts/gate8c1/ceiling_fix.md`, `padding_bug.md`.
