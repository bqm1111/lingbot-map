# Waymo supervision pilot — does a small Waymo subset give hidden-region semantic supervision our KITTI-360 recipe lacks?

`waymo_pilot.1` · commit `f8203f2` · audit only, no training launched, no target benchmark opened.

## Verdict: **BLOCKED** on the Waymo arm — and one measured finding that should change what we do when it unblocks

The question cannot be answered on this machine: **there is no Waymo data here and the SDK is not installed.** The pilot therefore delivers the repository-grounded design, a runnable entry point that demands real input, and the full KITTI-360 reference measurement the Waymo arm would have been compared against.

That reference measurement is not a null result. On the population the method has to serve, the **existing privileged target reaches 63.9 % top-1 correctness and 21.36 mIoU against human labels**, and it covers only 12.3 % of the hidden occupied voxels it is supposed to supervise. Before asking whether a second dataset adds diversity, it is worth knowing that the supervision we would mix it into is itself about one third wrong wherever it exists.

This says nothing about whether heterogeneous training works. It was not tested.

## 1. The question and the exact data used

> Does a small Waymo subset give us useful hidden-region semantic supervision that our existing KITTI-360 recipe lacks?

**Waymo — not available.** Probed, not assumed:

```json
{
 "root": "/media/SSD1/MINH_DATASETS/waymo_perception_v1_4_2",
 "root_exists": false,
 "n_tfrecords": 0,
 "sdk_importable": false,
 "tensorflow_importable": false,
 "sdk_error": "ModuleNotFoundError: No module named 'waymo_open_dataset'",
 "expected_version": "perception v1.4.2 (individual_files/training/*.tfrecord)",
 "n_segments": 0,
 "ok": false,
 "missing": [
  "dataset root /media/SSD1/MINH_DATASETS/waymo_perception_v1_4_2 does not exist",
  "tensorflow is not importable in this environment",
  "waymo-open-dataset-tf-2-12-0 is not importable (ModuleNotFoundError: No module named 'waymo_open_dataset')"
 ]
}
```

**KITTI-360 — available and measured.** The causal contract was verified rather than inherited: the cached recipe's own frame table gives a causal clip of **5 frames spanning 2.088 s** at a stream step of **0.522 s**, ending at the anchor, from one camera (`image_00`) — the historical contract. The privileged window is the existing recipe's: 19 future stream frames out to 9.92 s.

Sample: **35 anchors from 3 drives** (digest `29f7009386b9bcdd`), selected before any yield was examined, spread with a minimum separation of 40 stream frames so no two anchors share a causal clip or a privileged window. Drive `0006` is development data and is asserted absent.

A cap on anchors is not the same as having that many independent examples. The measured ceiling matters more than the cap:

| drive | eligible anchors | temporally independent |
|---|---:|---:|
| 0003 | 193 | 5 |
| 0007 | 569 | 15 |
| 0010 | 596 | 15 |
| **total** | **1,358** | **35** |

The frozen training pool is 1,358 anchors, but at a separation that guarantees no shared frame (~21 s) only **35 of them are temporally independent**, across 3 routes through one city. That is the quantity a second dataset would be adding to.

## 2. Measured comparison

`k360_reference` is the three drives the current recipe trains on. `k360_7drives` is the same recipe, same teacher, same grid, on every KITTI-360 training drive whose caches exist — it varies **scene count** with everything else fixed. It is the closest available stand-in for the diversity question, and it is **not** a substitute for Waymo: same city, same sensor, same camera count.

| quantity | k360_reference | k360_7drives |
|---|---:|---:|
| scenes | 3 | 7 |
| anchors | 35 | 92 |
| usable pairs | 35 | 92 |
| privileged occupied voxels | 1,005,087 | 2,864,594 |
| causal semantic evidence voxels | 702,287 | 1,780,039 |
| **hidden (O_priv & ~E_causal)** | **972,983** | **2,767,707** |
| hidden per anchor | 27,800 | 30,084 |
| **U (hidden & target valid)** | **119,641** | **391,659** |
| U / hidden | 12.3 % | 14.2 % |
| U per anchor | 3,418 | 4,257 |
| sub-cases below: anchors / hidden voxels | 35 / 972,983 | 32 / 877,551 |
| — surface observed, teacher silent | 20,433 (2 %) | 20,092 (2 %) |
| — outside every causal frustum | 16,893 (2 %) | 7,076 (1 %) |
| — occluded throughout | 363,130 (37 %) | 302,642 (34 %) |
| — unresolved visibility | 572,527 (59 %) | 547,741 (62 %) |
|    of which causal map carved it free | 67,916 (7 %) | 58,877 (7 %) |
| newly supported / all targets | 69.0 % | 73.0 % |
| single-observation targets | 56.0 % | 48.0 % |
| two-view agreement (consistency) | 74.5 % | 75.4 % |
| top class share of U | 47.2 % (vegetation) | 48.8 % (vegetation) |
| classes > 1 % of U | 11/18 | 9/18 |
| thing share of U | 6.1 % | 5.7 % |
| causal clip span (s) | 2.088 | 2.091 |
| stream dt (s) | 0.522 | 0.523 |


Definitions, all label-free: `O_priv` = privileged LiDAR says occupied; `E_causal` = the causal clip has lifted teacher evidence there; `hidden` = `O_priv & ~E_causal`; `U` = `hidden` and a reliable privileged semantic target exists. Every fraction above names its denominator.

**Doubling the scene count barely moves the supervision.** 3 drives → 7 drives changes yield from 12.3 % to 14.2 % of hidden voxels, the dominant class stays `vegetation` at 47.2 % → 48.8 %, and the thing share stays at 6.1 % → 5.7 %. Within KITTI-360, scene count is not what limits this supervision's class balance.

### What the recipe discards before semantics are even considered

| stage | voxels | of future-hit |
|---|---:|---:|
| privileged future sweeps hit the voxel | 2,666,340 | 100 % |
| excluded: occupied/free conflict between sweeps (dynamic) | 2,027,825 | 76.1 % |
| excluded: failed the ≥1 s static-support rule | 2,341,577 | 87.8 % |
| kept as a target (tiers A+B, ≥1 camera-visible view) | 123,261 | 4.6 % |

The motion exclusions are the existing recipe's own, and they are the right default — ego-motion compensation does not align moving objects. They are also most of the loss: the recipe keeps about 5 % of what the future sweeps touch.

## 3. Is the supervision correct? (annotation-assisted diagnostic)

Teacher agreement measures whether two views of the frozen teacher say the same thing. It is **not** correctness. Measured separately, on the labelled source subset only, with human labels read nowhere else in this pilot:

| quantity | value | denominator |
|---|---:|---|
| two-view **consistency** | 74.5 % | targets in `U` with ≥2 views |
| top-1 **correctness** | 63.9 % | 78,195 labelled target voxels in `U` |
| top-1 correctness on two-view-**agreeing** voxels | 73.0 % | 26,973 voxels |
| target mIoU, 18 classes | 21.36 | 18/18 classes present |

Agreement overstates correctness by roughly 10 points, and even where two independent views of the teacher agree, **about a quarter of the targets are still wrong**. The mIoU is the quality of the *supervision*, not of any model.

Split by the recipe's own reliability tier — tier A is occupancy at `t` confirmed by a causal-window sweep, tier B is occupancy at `t` uncertain and carried by two or more future sweeps:

| tier | target voxels | top-1 correctness |
|---|---:|---:|
| A (occupied at `t` confirmed) | 27,619 | 62.4 % |
| B (occupancy at `t` uncertain) | 50,576 | 64.8 % |

**Tier is not the lever.** Tier B carries about two thirds of all targets and is, if anything, marginally *more* accurate than tier A, so dropping it would shrink the supervision without improving it. The one thing that does separate right from wrong targets is two-view agreement — but that gap was measured with human labels, so it must not be turned into a filtering threshold here.

Per-class target IoU — the supervision is usable for a handful of large stuff classes and close to useless for the rest:

| class | target IoU | | class | target IoU |
|---|---:|---|---|---:|
| vegetation | 73.7 | | parking | 11.6 |
| building | 56.2 | | traffic-sign | 8.9 |
| road | 50.2 | | sidewalk | 8.2 |
| truck | 48.7 | | bicycle | 7.2 |
| car | 44.5 | | pole | 5.1 |
| motorcycle | 20.0 | | other-object | 1.5 |
| person | 18.4 | | other-ground | 1.1 |
| fence | 14.8 | | terrain | 0.5 |
| other-structure | 14.0 | | other-vehicle | 0.0 |

On the seven-drive sample the same diagnostic gives 66.4 % top-1 and 24.15 mIoU, but only 30 of its 92 anchors carry labels, so it is a different and smaller subset — not a comparison.

## 4. Inspectable examples

Selection rule: fixed quantiles (0.0, 0.25, 0.5, 0.75, 1.0) of per-anchor U/hidden yield, then unrepresented scenes. Worst and best cases are included by construction.

| scene | anchor | yield `U`/`hidden` | targets | figure |
|---|---:|---:|---:|---|
| 0010 | 414 | 3.7 % | 850 | `artifacts/waymo_supervision_pilot/examples/k360_reference/0010_00414.png` |
| 0007 | 328 | 9.7 % | 3,302 | `artifacts/waymo_supervision_pilot/examples/k360_reference/0007_00328.png` |
| 0007 | 368 | 12.5 % | 4,288 | `artifacts/waymo_supervision_pilot/examples/k360_reference/0007_00368.png` |
| 0007 | 206 | 15.5 % | 3,039 | `artifacts/waymo_supervision_pilot/examples/k360_reference/0007_00206.png` |
| 0007 | 571 | 26.7 % | 3,332 | `artifacts/waymo_supervision_pilot/examples/k360_reference/0007_00571.png` |
| 0003 | 124 | 6.9 % | 2,150 | `artifacts/waymo_supervision_pilot/examples/k360_reference/0003_00124.png` |

Each figure shows the five causal views, the causal semantic evidence in BEV, the privileged view that first reveals the targets, the hidden population before a target is required, and the target itself coloured by readout — with provenance printed underneath. The JSON sidecar beside each figure carries the frame indices, timestamps and image paths.

## 5. Implementation and checks

New, and only what the pilot needed:

| file | what it is |
|---|---|
| `tools/waymo_pilot/config.py` | single source of truth: caps, bounds, sources, firewall |
| `tools/waymo_pilot/populations.py` | the §4 decomposition, stated without human labels |
| `tools/waymo_pilot/provenance.py` | per-target source, time, camera, count, agreement; overlaps retained |
| `tools/waymo_pilot/sample.py` | reproducible selection + the independence ceiling |
| `tools/waymo_pilot/k360.py` | KITTI-360 arm over the existing caches and the existing recipe |
| `tools/waymo_pilot/waymo.py` | Waymo reader; SDK-verified fields, axis convention, availability probe |
| `tools/waymo_pilot/stats.py` | aggregation; every fraction carries its denominator |
| `tools/waymo_pilot/examples.py` | deterministic inspectable figures |
| `tools/waymo_pilot/checks.py` | the four focused checks |
| `tools/waymo_pilot/diagnostic.py` | ANNOTATION-ASSISTED correctness audit, quarantined |
| `tools/waymo_pilot/report.py` | this report, generated from the artifacts |
| `tools/waymo_pilot/cli/` | waymo_extract · run_k360_reference · run_checks · run_examples · run_diagnostic |

Reused unchanged: the causal clip and hard mapper (`tools/baseline_reconstruction`), the frozen geometry and evidence lifting (`tools/semantic_completion`), the privileged target recipe (`tools/delayed_semantic/targets.py`), the raw-LiDAR occupancy target (`core/supervision/rawtarget.py`), the frozen Trident-H teacher and its 18 phrases.

**Checks: all pass** (`0007` anchor 9).

| check | result | evidence |
|---|---|---|
| preprocessing matches cached fixture | PASS | 17,566 evidence voxels identical, max feature diff 2.53e-04 ≤ fp16 9.77e-04, geometry 967,322 params |
| privileged inputs cannot change causal tensors | PASS | causal digest `630cde00b5c1c578` unchanged after perturbation; causal state opened 6 paths, none privileged |
| known projection / alignment / visibility case | PASS | voxel binning exact; 25 m return behind a 10 m surface rejected at 0.5 m tolerance; Waymo forward point → principal point [960.0, 640.0] |
| invalid supervision excluded, not assigned a class | PASS | 28,204 unsupervised voxels, 0 given a class; 0 unknown voxels called occupied |

The whole measurement ran inside a file audit forbidding `data/kitti/dataset`, `occ3d_gt`, `nuscenes`, `lingbot_occ3d_zeroshot`, `scale_gate/cache`, `semantickitti`: 385 paths opened, 0 violations.

## 6. Runtime and storage

| item | value |
|---|---:|
| KITTI-360 reference, 35 anchors | 0.3 min (0.004 GPU-h) |
| KITTI-360 seven drives, 92 anchors | 0.6 min (0.010 GPU-h) |
| GPU-hour cap | 2.0 h |
| new derived artifacts | 3.1MB |
| derived-artifact cap | 20.0 GiB |
| Waymo derived cache | none — arm blocked |

The pilot reuses every existing cache and adds essentially nothing: both caps are untouched. A Waymo run would be the first real cost, and its size should be estimated from the smoke test before expanding, as §2 requires.

## 7. Limitations

- **The headline question is untested.** No Waymo data was read, so nothing here supports or refutes the claim that a second dataset helps.
- `k360_7drives` varies scene count inside one city with one camera. It is evidence about KITTI-360, not about heterogeneity.
- Sub-cases of `hidden` are measured on the 32 anchors that still have a causal clip; the four extra drives keep coverage caches but their clips were not retained, so their frusta cannot be rebuilt without re-running LingBot. Those anchors contribute coverage and are excluded from the sub-case sums.
- **Privileged sensors are not omniscient.** `hidden` is bounded by what the LiDAR sweeps resolved; genuinely hidden occupied space that no sweep ever touched is in none of these counts. Coverage of all hidden space is not established.
- Correctness is measured on the labelled source subset only, and on one teacher. A different teacher could change it.
- Anchors are independent in time, not in environment: three drives is three routes.
- Two-view consistency is reported wherever it is measured, and is never called mIoU.

## 8. What should happen next

### The prerequisite, not a command

The Waymo arm is blocked on a step a human must take:

```
1. accept the Waymo Open Dataset licence at https://waymo.com/open/licensing/
   (a human must do this; it is not automatable and this pilot does not do it)
2. install the gcloud SDK, then `gcloud auth login`
3. copy 20 segments only (~22 GiB), not the full split:
   gsutil -m cp $(gsutil ls gs://waymo_open_dataset_v_1_4_2/individual_files/training/ \
       | head -20) /media/SSD1/MINH_DATASETS/waymo_perception_v1_4_2/
4. create the reader environment (kept apart from cu128; TF 2.12 pins numpy<2):
   conda create -y -n waymo python=3.10 && conda run -n waymo pip install \
       waymo-open-dataset-tf-2-12-0==1.6.7
5. WAYMO_ROOT=/media/SSD1/MINH_DATASETS/waymo_perception_v1_4_2 conda run -n waymo python \
       tools/waymo_pilot/cli/waymo_extract.py --segments 2 --frames-per-segment 40
```

`tools/waymo_pilot/cli/waymo_extract.py` is written and runnable; today it exits 2 and prints exactly the above. It is not presented as executable.

### The comparison this pilot was meant to justify

Kept here so it is not re-derived later, and **not launched**:

- **A**: KITTI-360, current target recipe.
- **B**: KITTI-360 + Waymo, same recipe. Tests adding a source at fixed compute; it does not isolate diversity from dataset size.
- **C**: the same mixed data and anchors as B, with W-extended targets. Tests the supervision recipe alone.

Architecture, initialisation, geometry predictions, optimiser, step budget and source-only checkpoint selection stay fixed; B and C must match sampling exactly and report their change in unique scenes and target counts. Evaluation scenes stay disjoint from training scenes. A compatible learner already exists and does not need inventing: `tools/future_semantic_prior/model.SemanticCompletionHead` (`C3_full`, 257,394 parameters) consumes geometry plus causal evidence and is trained by soft distillation against exactly this kind of privileged target. B0 nearest-evidence stays as the non-learning reference; it cannot learn a new prior.

### One verified command

Two internal levers were tested in the course of this pilot and **both are closed**: scene count inside KITTI-360 does not change the supervision profile (§2), and the recipe's reliability tier does not separate correct targets from wrong ones (§3). There is no cheap repair left to run on KITTI-360, which is precisely why the Waymo question stays open rather than being answered by proxy.

The next supported action is therefore the first step of the Waymo arm, and it is gated on the human prerequisite above. The command that runs today, and that returns 0 the moment the prerequisite is met, is the readiness probe:

```bash
python tools/waymo_pilot/cli/waymo_extract.py --probe-only
```

It opens nothing and downloads nothing: it reports the dataset root, the number of `.tfrecord` files, and whether TensorFlow and the SDK import, then exits 0 if the arm can run and 2 otherwise. **Cost: under a second, no GPU, no storage.** It is verified — it is the command that produced the availability block in §1, and it exits 2 today.

When it exits 0, the next command is the smoke test the brief asks for — `python tools/waymo_pilot/cli/waymo_extract.py --segments 2 --frames-per-segment 40` — whose runtime and storage must be measured from that run before any expansion, rather than estimated here. The pilot's caps (20.0 GiB derived, 2.0 GPU-h) stop the expansion if the smoke test says it would exceed them.

