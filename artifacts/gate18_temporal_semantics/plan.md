# Gate 18 — temporal semantic evidence scaling: audit and plan

No training. The question: as more *causal* past observations accumulate, does frozen semantic evidence become more
reliable, and does that reach the completed-unobserved region?

## 1. Audit

| component | what it is | consequence for this gate |
|---|---|---|
| `LocalClip` (`baseline_reconstruction/current/geometry.py`) | one clip = exactly `t-4..t`, hard-asserted; poses are **canonical with clip frame 0 as origin**, plus a per-clip MoGe-2 metric scale | each clip has its **own arbitrary origin and scale**; there is no pose relating two clips |
| `lift.ray_endpoints(clip)` | per frame, the mapper's accepted ray endpoints as world keys + lattice pixels | lets me lift **any subset of a clip's 5 frames** — this is how K<5 is built |
| `lift.lift_clip_evidence` | fuses the 5 frames: count-weighted mean of unit teacher features at hard-map-occupied voxels | the frozen evidence rule; reused unchanged for every K |
| `memory.EvidenceMemory` | accumulates observations, `frame_bits` is **uint8 over the 5 history slots** | the repo's evidence memory is structurally capped at 5 frames |
| evidence caches | 590 anchors on 0006, at **anchor step 3** (t = 4, 7, ... 1771) | consecutive anchors overlap by 2 frames; fusing N anchors spans `5 + 3(N-1)` frames |
| Trident token caches | present for **all 1,777** frames of 0006 | no token is missing for any history length |
| GT poses | `sscbench_kitti360/data_poses/<drive>/poses.txt` | present |

**Historical frames beyond the five-frame clip are available. A deployable pose linking them is not.**

## 2. The one honest deviation, stated up front

To place evidence from an earlier clip into anchor `t`'s grid, something must relate the two clips. The repository
has no deployable cross-clip pose — each clip is locally reset with its own origin and scale. The only available
association is **GT poses**, which the frozen target builder already uses (and whose accuracy the frozen G1 gate
measured: median ≤ 0.066 m, p90 ≤ 0.156 m out to +5 frames).

So every row with K > 5 uses **GT-pose registration** and is an **upper bound on temporal accumulation**, labelled as
such throughout — the same status Gate 17's oracle grouping had. This is the right way round for a go/no-go gate: if
even perfectly registered accumulation does not help, the answer is decisive. If it does help, the follow-up question
becomes whether deployable registration is good enough, and G1 already bounds that.

Rows with K ≤ 5 need no registration at all and are fully deployable.

## 3. How each history length is built

| K | construction | registration |
|---|---|---|
| 1 | clip(t), slot 4 only (frame t) | none |
| 3 | clip(t), slots 2-4 (`t-2..t`) | none |
| 5 | clip(t), all slots — **must reproduce the frozen evidence cache exactly** | none |
| >5 | clip(t) all slots, plus earlier anchors' frozen evidence caches moved into `t`'s grid by GT pose | GT pose (oracle) |
| all-past | every earlier anchor, fused into a persistent world voxel map, pruned to `t`'s grid | GT pose (oracle) |

Frames spanned and seconds spanned are reported beside every row, because "more anchors" and "seeing the same place
again" are different things (one stream frame = 0.520 s, measured in Gate B5).

## 4. Aggregation rules (deterministic, no training)

A `latest` · B `mean_feature` (the frozen rule: normalise the count-weighted mean of unit features) ·
C `mean_probability` (average the per-observation text readout distributions) · D `count_weighted` — using the ray
count the evidence cache already stores. No confidence measure is invented: the teacher exposes none beyond count.

## 5. Populations

- **T1** directly observed evidence voxels with a GT label — isolates teacher reliability. Needs no registration.
- **T2** mapped/observed voxels — does fusion improve the semantic map.
- **T3** completed-unobserved (`completed_never_observed`), via the **existing B0 rule** on the accumulated evidence.
  No new propagation head.

Breakdowns: observation-count bins (1, 2, 3-5, 6-10, >10), distance-to-evidence bins (frozen edges), thing vs stuff,
per class. Metric and masks are the repository's frozen ones (`semantic_metrics.summarize`, classes 1..18).

## 6. Order of work

§2 of the brief (raw teacher quality) comes first: it needs no accumulation, is fully deployable, and is exactly the
measurement Gate 17 recommended as the next step. Then the K ladder.
