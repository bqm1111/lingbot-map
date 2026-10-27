# Region naming for semantic completion — pilot report

**Decision: STOP.** On 24 evaluation anchors, replacing the teacher's name for a region with Qwen's moves the primary population by **+0.88 mIoU** (CI [-0.03, +1.80]) at region accuracy 58.47 % against pooled Trident's 56.98 %, and the bounded region extension moves it by **-0.02** under Trident naming and **+0.01** under Qwen naming.

## 1. What was run

One question: with completed occupancy fixed, can a vision-language model name objects and regions more accurately than the deployed Trident evidence, and can those names be carried to occupied voxels that have no causal semantic evidence?

- **Scope**: KITTI-360 `2013_05_28_drive_0006_sync` only, 24 evaluation anchors (one per temporal block out of 28), 4 calibration anchors, the golden anchor `t=4`. No SemanticKITTI, no Occ3D, no target evaluation.

- **Everything frozen** was reused unchanged: the 5-frame causal clip, the hard map, `CompletionUNet` seed 0 occupancy, the Trident-H evidence, the nearest-evidence rule B0 and the repository's own scorer. Legacy commit `148c46e93`.

- **New**: SAM region proposals, pooled-Trident region naming (P), Qwen/Qwen3-VL-8B-Instruct region naming (Q), and one bounded geodesic extension over the fixed occupancy.

- **Isolation**: the experiment itself is entirely under `/media/SSD1/region_naming_pilot` -- all code, caches, the model checkpoint and every output. Nothing was added to, imported from or changed in `/home/minh/workspace/StreamingSSC`.

- **One exception, on explicit instruction**: a read-only copy of the deliverables (this report, `results.csv`, the manifest, the figures, the small artifacts and the pilot's own code) was placed at `/home/minh/workspace/lingbot-map_fork/reports/region_naming_pilot` so the write-up sits beside the repository's other reports. It is a record, not an integration: nothing under `reports/` is importable by the system, and no module, config or pipeline of the legacy repository references it.

## 2. Adapter alignment — the baseline is reproduced exactly

B0 recomputed through this experiment's adapter over all 590 anchors of the drive, against the numbers the legacy Gate 17 report published:

| population | published mIoU | reproduced | voxels | |
|---|---:|---:|---:|---|
| `completed_unobserved` | 8.796376 | 8.796376 | 2,743,212 | match |
| `ssc_no_evidence` | 1.392765 | 1.392765 | 553,275,690 | match |
| `oracle_unobserved` | 8.392475 | 8.392475 | 7,834,276 | match |

The grid, the transform, the occupancy threshold, the teacher readout and the metric are therefore the frozen ones.

## 3. Region proposals

Frozen SAM ViT-H automatic mask generation on each of the five causal frames, 465 frame masks and **267 cross-frame regions per anchor** on average.

- Lifted support is **exactly** the frozen evidence set: a mask pixel becomes 3D support only if it is a ray the frozen mapper accepts, so the region decomposition partitions the existing evidence and adds nothing. 98.4 % of evidence voxels fall inside some mask; the rest keep the original per-voxel baseline label.

- **Fragmentation is the dominant failure of the proposer**: 75 % of regions are seen in a single frame, 15.0 regions per anchor merge two masks of the same frame, and 12.0 span non-contiguous frames.

- 57.0 regions per anchor have RGB support but no Trident evidence at all. They are counted here and never mixed into the matched comparison: neither recognizer can seed from them.

## 4. Did Qwen name the same regions better than pooled Trident?

Region-level, on the 939 queried regions that have a human label (of 1857 queried; 918 have none and are *unknown*, never counted wrong):

| | pooled Trident (P) | Qwen (Q) |
|---|---:|---:|
| modal-class accuracy | 56.98 % | 58.47 % |
| accuracy weighted by evidence support | 64.52 % | 66.87 % |

- P and Q agree on 73.67 % of queried regions.
- Qwen outcomes: {'ok': 1857, 'abstain': 0, 'out_of_vocabulary': 0, 'malformed': 0}. Abstentions and out-of-vocabulary answers fall back to the same region's pooled Trident label, so a refusal never removes a seed.
- On 149 multi-frame regions asked twice, a single view instead of three changes the answer 18.12 % of the time — the recognizer is not view-stable, and its self-reported wording is not treated as a probability.
- Mean region purity against human labels is 90.11 %; 675 regions are at least 90 % one class and 94 are below 60 % (mixed proposals, computed after prediction).

Per modal human class (regions, P correct, Q correct):

| class | regions | P | Q |
|---|---:|---:|---:|
| vegetation | 477 | 401 | 404 |
| building | 97 | 55 | 59 |
| sidewalk | 75 | 20 | 21 |
| other structure | 64 | 3 | 11 |
| terrain | 54 | 0 | 0 |
| road | 50 | 26 | 24 |
| parking | 48 | 2 | 0 |
| car | 28 | 22 | 23 |
| other ground | 23 | 3 | 2 |
| other vehicle | 14 | 0 | 1 |
| fence | 8 | 3 | 4 |
| other object | 1 | 0 | 0 |

## 5. Voxel results — the matched two-by-two, plus the original baseline

Pooled over the 24 evaluation anchors; one confusion matrix per population, the metric computed once from the pooled matrix. Occupancy is one volume shared by every method: SC IoU **12.86** (TP 241,735 FP 1,091,258 FN 546,113) for all five.

| population | voxels | B0 | P_N | P_R | Q_N | Q_R |
|---|---:|---:|---:|---:|---:|---:|
| GT-occupied & causal evidence (teacher copy quality) | 14,173 | 17.85 | 17.57 | 17.59 | 19.52 | 19.73 |
| GT-occupied & no causal evidence (oracle occupancy) | 351,485 | 9.33 | 9.04 | 9.03 | 8.33 | 8.32 |
| GT-occupied & O & causal evidence | 8,749 | 17.96 | 17.87 | 17.91 | 19.30 | 19.56 |
| PRIMARY  GT-occupied & O & no causal evidence | 110,913 | 10.14 | 9.57 | 9.55 | 10.45 | 10.45 |
| deployable SSC, every labelled voxel | 23,709,740 | 1.57 | 1.52 | 1.51 | 1.57 | 1.57 |
| deployable SSC, every labelled voxel without causal evidence | 23,490,098 | 1.47 | 1.40 | 1.39 | 1.44 | 1.43 |
| PRIMARY, clip geometrically observed the voxel | 18,341 | 11.07 | 10.41 | 10.42 | 10.84 | 10.87 |
| PRIMARY, clip never observed the voxel (S|F empty) | 92,572 | 8.90 | 8.47 | 8.44 | 9.46 | 9.46 |
| PRIMARY, object classes | 1,435 | 14.83 | 14.18 | 14.25 | 11.30 | 11.46 |
| PRIMARY, broad surface classes | 109,478 | 15.78 | 15.15 | 15.10 | 16.55 | 16.54 |

mIoU in points. The same table by overall accuracy:

| population | B0 | P_N | P_R | Q_N | Q_R |
|---|---:|---:|---:|---:|---:|
| PRIMARY  GT-occupied & O & no causal evidence | 48.16 | 47.31 | 47.40 | 49.00 | 49.05 |
| deployable SSC, every labelled voxel without causal evidence | 15.20 | 14.93 | 14.96 | 15.46 | 15.48 |
| deployable SSC, every labelled voxel | 15.92 | 15.66 | 15.70 | 16.20 | 16.22 |
| PRIMARY, object classes | 27.74 | 28.01 | 28.29 | 30.52 | 31.22 |
| PRIMARY, broad surface classes | 48.42 | 47.56 | 47.66 | 49.24 | 49.29 |

### The five comparisons the design asks for

| comparison | what it isolates | Δ mIoU on the primary population | 95 % block CI |
|---|---|---:|---|
| `P_N-B0` | pooling per-voxel teacher argmax into one label per region | -0.57 | [-1.06, -0.00] |
| `Q_N-P_N` | recognition, identical nearest extension | +0.88 | [-0.03, +1.80] |
| `P_R-P_N` | extension, Trident naming | -0.02 | [-0.09, +0.05] |
| `Q_R-Q_N` | extension, Qwen naming | +0.01 | [-0.08, +0.11] |
| `Q_R-P_R` | recognition, identical region extension | +0.90 | [+0.01, +1.79] |

Blocks are the 24 evaluation anchors, one per temporal block, 2,000 paired resamples, seed 0. This is an exploratory pilot: the interval describes anchor-to-anchor variation, not a significance claim about individual correlated voxels.

### Corrections and damage on the primary population, against B0

| method | voxels B0 got wrong and this got right | voxels B0 got right and this got wrong | net |
|---|---:|---:|---:|
| P_N | 2,219 | 3,161 | -942 |
| Q_N | 4,261 | 3,328 | +933 |
| P_R | 2,559 | 3,393 | -834 |
| Q_R | 4,629 | 3,634 | +995 |

## 6. Did the extension reach the no-evidence population?

- The membership map is computed once per anchor from occupancy, observed free space and adjacency, with `max_hops = 6` (1.2 m), selected on the calibration anchors alone.

- It assigns **10.6 %** of predicted-occupied voxels; **90.4 %** fall back to the matching nearest-evidence prediction. Growth adds 29595 voxels per anchor beyond 16791 seeds, and 3669 are contested by two regions and left unassigned.

- Of the primary population, 18,341 voxels were geometrically observed by the clip (17 %) and 92,572 were not. This is a geometry split, not an occlusion split: the frozen representation cannot separate "occluded in all five frames" from "never resolved", so **physical-occlusion attribution is unavailable** and is not claimed.

## 7. Calibration trace

| max_hops | mIoU on the primary population | accuracy | occupied coverage |
|---|---:|---:|---:|
| P_N (no region extension) | 11.08 | 40.73 | 0.000 |
| 2 | 11.09 | 40.74 | 0.048 |
| 4 | 11.14 | 40.87 | 0.075 |
| 6 | 11.15 | 40.99 | 0.102 |
| 8 | 11.07 | 41.07 | 0.129 |
| 12 | 10.61 | 41.00 | 0.180 |
| 16 | 10.62 | 41.14 | 0.227 |

Measured on anchors [40, 481, 922, 1363], pooled-Trident naming, before any evaluation prediction was written. The association thresholds were **not** searched: {'assoc_iou_min': 0.15, 'assoc_rgb_max': 0.22}.

## 8. Answers

**Did Qwen name the same regions better than pooled Trident?**  
At region level, 58.47 % against 56.98 % (Qwen ahead); weighted by evidence support, 66.87 % against 64.52 %. At voxel level under the identical nearest extension, `Q_N - P_N` = +0.88 mIoU, CI [-0.03, +1.80].

**Did region extension improve over nearest extension at fixed naming?**  
With Trident naming `P_R - P_N` = -0.02 mIoU, CI [-0.09, +0.05]; with Qwen naming `Q_R - Q_N` = +0.01 mIoU, CI [-0.08, +0.11]. The membership map is identical in both, so this isolates the extension rule.

**Did improvements reach the no-evidence population and full SSC?**  
The primary population *is* the no-evidence population. On the deployable SSC read over every labelled voxel without causal evidence, B0 1.47, P_N 1.40, P_R 1.39, Q_N 1.44, Q_R 1.43; over every labelled voxel, B0 1.57, P_N 1.52, P_R 1.51, Q_N 1.57, Q_R 1.57.

**How much was fallback, and where did errors spread?**  
90.4 % of predicted-occupied voxels fall back to nearest evidence, plus Qwen's 0 non-answers of 1857 queries, which fall back to the pooled Trident label. Against B0 on the primary population, Q_R fixes 4,629 voxels and breaks 3,634; P_R fixes 2,559 and breaks 3,393.

**Were improvements confined to visible evidence repair rather than occlusion?**  
Split by whether the clip geometrically observed the voxel: on the 18,341 observed voxels B0 11.07, P_N 10.41, P_R 10.42, Q_N 10.84, Q_R 10.87; on the 92,572 never-observed voxels B0 8.90, P_N 8.47, P_R 8.44, Q_N 9.46, Q_R 9.46. Physical occlusion cannot be separated from unresolved visibility in the frozen representation, so that attribution is marked unavailable rather than guessed.

**What remained unchanged, and were any external concurrent changes observed?**  
Occupancy, the evidence set, the grid and the scorer are identical across all five methods (the shared occupancy volume gives one SC IoU for all of them). Concurrent external activity is reported in `artifacts/protected_state_after.txt`.

**Should we continue this idea, fix a specific problem, or stop?**  
STOP. CONTINUE if Qwen beats pooled Trident at region level AND its mIoU gain on the primary population has a bootstrap CI above zero AND an extension gain has a CI above zero; FIX ONE COMPONENT if exactly one holds; STOP otherwise.

## 9. Limitations

- 24 evaluation anchors on one drive of one city: a pilot, not a benchmark. Paired per-anchor variation is reported; no claim is projected onto SemanticKITTI or Occ3D, which were never opened.

- The modal-class target for recognition comes from human voxel labels inside each region. Regions with no labelled voxel are unknown, and region purity is far from 1, so region accuracy is a coarse measure of naming, not a clean instance benchmark.

- The proposer is one frozen setting of SAM, not a tuned one. Its fragmentation bounds everything downstream.

- Qwen saw tight crops and a highlighted context frame; its answer about a region is a statement about the visible surface, never evidence about hidden geometry.

- Only one constant (`max_hops`) was selected, on calibration anchors. No hyperparameter search ran.

## 10. What was not touched

- `/home/minh/workspace/StreamingSSC`: never imported, never read as a dependency, never written. Its git state was recorded before and after. It *was* changed during this run by another session (`src/lingssc/training/geometry_training.py`, `.pytest_cache`, one `.pyc`); that is reported, not reverted, and none of it came from here.

- `/home/minh/workspace/lingbot-map_fork`: imported read-only with bytecode writes disabled; no legacy script executed. Its tree was byte-identical through the whole measurement and the only later change is the deliverable copy under `reports/` described in section 1.

- No retraining or fine-tuning of anything. No dataset permission, lockfile or shared environment was changed; the Qwen checkpoint and all caches live inside the experiment directory.


## 11. One next action
Stop this line. The measured bottleneck is not the name: recognition at region level is 58.47 % for Qwen against 56.98 % for the teacher, yet the primary population moves +0.88 mIoU. The single next action worth taking is to attack the proposer, not the recognizer: measure how much of the primary population a *perfect* region decomposition could reach at all, by scoring the same extension rule seeded from connected components of the human labels — the oracle that bounds every proposal-based method here — before any further VLM work.


---
Generated by `code/report.py` from `artifacts/analysis.json`, `artifacts/calibration.json`, `artifacts/verify_b0.json` and `logs/`. Runtime: proposals 2 min, prediction 0 min.
