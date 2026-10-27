# A2 (DDAD-only) training report — BLOCKED, not executed

**2026-10-20.** This is a stage report, not a results report. Brief §12 lists this file as a
deliverable for "execution results and reproducibility information"; the honest content is
that **no DDAD training run was executed**, why, and what was executed instead.

---

## 1. Stage status

| stage | status |
|---|---|
| repository state recorded | **done** — HEAD `148c46e935858457887b17944f5c5427147ee964`, branch `feature-completion-study`, 33 uncommitted entries, `require_source_freeze()` passing on 54 files |
| OccAny audit (brief §3) | **done** — `occany_training_audit.md` |
| fair evaluation protocol (brief §4) | **done** — `fair_evaluation_protocol.md` |
| DDAD data audit (brief §6) | **done** — `ddad_feasibility.md` |
| DDAD adapter implementation (brief §6) | **not started** — blocked by the audit |
| correctness gates (brief §6) | **gates 6 and 7 FAIL**; see below |
| A2 training (brief §7) | **NOT LAUNCHED** |
| A2 zero-shot evaluation (brief §8) | **not applicable** |
| observed/completed decomposition (brief §9) | **done** — `observed_vs_completed_analysis.md` |
| matched-data control feasibility (brief §10) | **done** — `matched_data_control_feasibility.md` |

Nothing was overwritten. No existing checkpoint, cache, manifest, frozen configuration or
historical result file was modified. The only writes outside this directory are the two new
probe tools under `tools/method_vs_data/`.

---

## 2. Why A2 was not launched

Brief §6 says: *"If the data cannot satisfy these requirements, do not launch training."* Two
of its ten gates fail.

**Gate 6 (correct occupied/free/unknown assignment) and Gate 7 (comparable target
statistics).** DDAD publishes one aggregated LiDAR cloud per sample, merged from four
Luminar-H2 sensors, in a vehicle frame whose origin sits on the ground below the rear axle,
with per-point channels X/Y/Z/INTENSITY only. Our occupancy supervision carves free space
along rays **from the sensor origin**; that origin is not recoverable from the release.

Measured on one complete real DDAD scene with the frozen builder called unchanged:

| | valid fraction | occupied prevalence | conflict/touched |
|---|---:|---:|---:|
| DDAD, published origin | 0.583 – 0.616 | **0.0042 – 0.0094** | 0.062 – 0.097 |
| KITTI-360 (reference) | 0.5703 | 0.0276 | 0.0865 |
| Waymo (reference) | 0.6616 | 0.0222 | 0.0679 |

Occupied prevalence is 3–6× out of family, and varying the assumed sensor mount across
plausible values moves the valid fraction by ~20 percentage points and prevalence by ~2×. An
arm trained under those conditions would confound DDAD with a constant we invented, so it
could not serve as the control A2 exists to be.

Full evidence, including the azimuth and elevation tests showing the merge is not invertible,
is in `ddad_feasibility.md`.

---

## 3. What was executed, and how to reproduce it

### 3.1 DDAD data probe

No 256 GiB download. `DDAD.tar` (275,266,242,560 bytes, public, HTTP 200) honours range
requests and tar is sequential, so two ranged reads totalling ≈1.2 GiB recovered one complete
scene — metadata, calibration and all 100 LiDAR sweeps.

```bash
# window 1: archive head
curl -r 0-209715199      -o head200.tar "https://tri-ml-public.s3.amazonaws.com/github/DDAD/datasets/DDAD.tar"
# window 2: the scene's calibration, scene JSON and point clouds (512-byte aligned start)
curl -r 1000000000-2800000000 -o mid.bin "https://tri-ml-public.s3.amazonaws.com/github/DDAD/datasets/DDAD.tar"
# member extraction, then the frozen carve
PYTHONPATH=$PWD python tools/method_vs_data/ddad_probe.py \
    --scene-dir <extracted> --device cuda:2 --anchors 0,20,40
```

Outputs: `artifacts/method_vs_data/results/ddad_probe.json`. Scene used:
`ddad_train_val/000020` (100 samples, 10 s, 38.6 m path, 229,726 points/sweep).

The downloaded `.tfrecord`-equivalent archive slices were deleted after extraction; the
derived JSON is kept.

### 3.2 Strict observed/completed decomposition

```bash
PYTHONPATH=$PWD python tools/method_vs_data/observed_strict.py --device cuda:2
```

Outputs `artifacts/method_vs_data/results/observed_strict.json`, log at
`artifacts/method_vs_data/logs/observed_strict_raw.log`. 801 anchors, ≈0.64 s each.

---

## 4. Reproducibility notes for whoever runs the substitute arm

* The frozen builder is `core/supervision/rawtarget.py`; the multisensor variant
  `core/supervision/rawtarget_multisensor.py` is already validated bit-for-bit against it and
  is the right entry point for a dataset with per-point sensor identity (PandaSet).
* `tools/method_vs_data/ddad_probe.py` duck-types the three members `rawtarget.build` touches
  (`cam0_to_world`, `read_velodyne`, `velo_to_velo`). Any new dataset can be gated the same way
  in well under an hour, before any adapter is written. **Run that probe first on ONCE or
  PandaSet** — it is what turned a multi-day adapter task into a one-afternoon audit here.
* The supervision comparability gate and sensor-origin gate that any substitute must pass are
  predeclared in `predeclaration.md` §6, before any substitute data was inspected.
