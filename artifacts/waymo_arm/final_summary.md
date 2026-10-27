# Waymo training experiment — final summary, Q1–Q5

**2026-10-20.** Phases 0–6 complete. Nothing frozen was modified; no KITTI-360 checkpoint,
preprocessing output or historical result file was overwritten. All new work sits under
`artifacts/waymo_arm/`, `tools/waymo_arm/` and the four reports named below.

## Deliverables

| path | contents |
|---|---|
| `artifacts/waymo_arm/predeclaration.md` | prediction and decision rule, fixed before training |
| `artifacts/waymo_arm/phase1_gates.md` | transform, visual and integrity gates |
| `artifacts/waymo_arm/stage_a_audit.md` | prior-evidence audit |
| `artifacts/waymo_arm/waymo_vs_kitti360_report.md` | Phase 3 geometry + §9 frustum addendum |
| `reports/waymo_semantic_impact.md` | Phase 4 |
| `reports/semantic_bottleneck_diagnosis.md` | Phase 5 |
| `reports/privileged_semantic_supervision_feasibility.md` | Phase 6 |
| `artifacts/waymo_arm/results/` | `waymo_vs_kitti360.json`, `semantic_impact_B0.json`, `frustum_split.json`, `bottleneck_B0.json` |
| `artifacts/waymo_arm/logs/` | gate logs, run logs, clip timing census, semseg probe |

## Q1 — Does Waymo-only geometry training outperform KITTI-360 zero-shot?

**No, on neither benchmark.** SemanticKITTI Δ SC IoU **−3.03** [−3.74, −2.48]; Occ3D
**−8.36** [−9.31, −7.42]. Three seeds per arm, identical evaluation, `map_only` control
bit-identical across arms. Verdict **REJECT** under the preregistered rule.

The predeclared direction (*SemanticKITTI down, Occ3D up*) was **supported on SemanticKITTI
and falsified on Occ3D**, and is not revised. Input-distribution similarity did not predict
transfer: Waymo lost most on the benchmark its input statistics most resembled.

Not undertraining — the Waymo models fit their own held-out domain better than the KITTI-360
models fit theirs (val IoU 0.181/0.198/0.197 against 0.147/0.142/0.143).

The frustum addendum rules out the narrow-camera explanation in the form first proposed (the
loss is not confined inside the frustum; on SemanticKITTI it is slightly larger outside) and
gives direct support to a less aggressive completion prior: the Waymo arm's recall rises in
observed regions and drops everywhere it must complete. That is a confirmed description, not
a demonstrated cause.

## Q2 — Does better geometry translate into better semantic mIoU?

**No, and the decoupling is structural.** With SC IoU down 2.84 and 9.09 points, SSC mIoU
moved **+0.58** [+0.44, +0.71] and **+0.04** [−0.24, +0.28].

On SemanticKITTI the B0 label is **provably invariant** to the geometry checkpoint: the
prediction grid is the evaluation grid, so each native voxel has one subvoxel selected
identically whether or not it is occupied. Measured: **0 label mismatches in 1,921,828 shared
query voxels** against 1,522,698 occupancy mismatches. On the fixed `shared_correct` set the
accuracy is bit-identical (42.371 % both arms).

The apparent +5.92 pp on `correct_occupied` is a selection effect — the Waymo arm's
true-positive set is 9.9 % smaller and easier to label. Held fixed, the difference is zero.

**H1 (source data is the bottleneck) is now falsified for both geometry and semantics.**

## Q3 — What is the dominant remaining bottleneck?

**Geometric completion, by a wide margin**; then evidence coverage coupled to
unobserved-region inference; teacher accuracy last.

On SemanticKITTI, KITTI-360 arm: 52.8 % of GT surface never predicted, precision 37.5 %,
semantic evidence reaching 5.6 % of GT-occupied voxels. Oracles, each substituting GT labels
into one bucket with the occupancy mask untouched: perfect teacher where it speaks **+1.63**;
perfect labels on non-evidence occupied voxels **+17.50**; perfect semantics on all correct
geometry **23.88 mIoU out of 100**. Roughly **76 points of the metric belong to occupancy.**

Two bounds worth carrying forward: the +17.50 uses GT labels and is **not reachable** by a
better extrapolator fed by this teacher, which is only 56.2 % correct where it speaks; and
about two thirds of semantic errors cross OccAny superclass boundaries, so at most a third
are candidates for genuine ambiguity.

## Q4 — Does privileged 3D semantic supervision offer a credible path beyond B0/B3?

**No. Recommend closing the branch.** Three measured reasons:

* **Coverage** — 68.1 % of hidden occupied voxels are never revealed at any horizon (dev
  drive, held out).
* **Reliability** — where revealed, the privileged label is 53.3 % top-1, capping a perfect
  learner at **+4.16** mIoU against the **+15.96** that correct labels in the same voxels
  would give. Label noise destroys **73.9 %** of the headroom before any learning problem.
* **Learnability** — Gate 21 already trained on this: three seeds, four variants, every one
  **below** B0; on the never-revealed population **−3.99 mIoU, 95 % CI [−5.03, −3.13]**.
  That is a positive finding of harm, not an inconclusive result.

Real Waymo 3D semantic labels do exist (13.4 % of frames in three probed segments, TOP laser
only), so that variant is not blocked by data — it is blocked by what it would cost the
claim: it converts a label-free open-vocabulary method into a supervised one.

## Q5 — The single most promising next experiment

**Not a semantic method.** Rebuild the semantic evidence field from **all five Waymo cameras**
instead of one, and measure hidden-region evidence coverage plus `shared_correct` semantic
accuracy at fixed occupancy.

Why this one:

* It tests the only untested cause of the coverage ceiling, and it is the step Gate 20 itself
  recommended. That gate measured **−59.96 pp** of its revealability loss at the step "and
  that return lands inside a future image" — camera field of view — and concluded in writing:
  *"Extending the horizon is not the lever; widening the field of view would be."* Gate 18
  independently showed the ceiling cannot be moved by more history: evidence voxels rise
  2.2 % from 5 to 20 frames while mIoU moves +0.05. Field of view is the live lever and the
  one never varied.
* The data is already on disk. The per-frame cache carries `image_FRONT`, `image_FRONT_LEFT`,
  `image_FRONT_RIGHT`, `image_SIDE_LEFT`, `image_SIDE_RIGHT` with per-camera poses and
  calibration — five 50° cameras, about 250° of coverage, no download.
* It changes no architecture, no loss and no training. Only the evidence lift's camera set.
* It has a clean null and a clean falsification. If coverage rises and `shared_correct`
  accuracy does not, the semantic ceiling is the teacher, not the geometry of observation, and
  the semantic programme closes with a measured reason. If both rise, the project has a real
  mechanism and the right comparison becomes OccAny's own surround-view numbers — which also
  resolves the comparability problem flagged in `reports/occany_observed_vs_unobserved.md`.

**Preconditions, both known.** Phase 1's T2 gate must be closed first: `waymo.project` applies
no lens distortion (p99 residual 0.237 m against a 0.20 m one-voxel criterion), which is
tolerable on the occupancy path it was scoped to but not on a semantic-evidence path. The
verified `_project_distorted` (p99 0.193 m, 99.3 % within a voxel) must be adopted. And a
multi-camera arm is a **different setting** from the monocular protocol, so it must be
reported as an additional arm rather than substituted into the frozen tables.

**What should be written up regardless of that experiment.** Gates 20 and 21, Phase 5's oracle
ladder and `reports/occany_observed_vs_unobserved.md` together form a complete preregistered
account of why hidden-region semantic completion is capped in a monocular causal system, and
why our geometric lead over OccAny is entirely out-of-frustum — we win by remembering, not by
completing. That is a publishable finding on its own, and it is better supported than any
positive semantic result this project has produced.
