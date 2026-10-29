# What the three trained checkpoints were actually supervised with

**2026-10-27.** Provenance audit of the supervision behind the three CompletionUNet training
arms whose zero-shot results this directory compares. Code:
`tools/method_vs_data/supervision_provenance.py`. Raw:
`artifacts/method_vs_data/waymo_canonical_rescore/supervision_provenance.json`.

Repository state: `db465c2ce2a692817d9ce4a141df7c24c4fa2d13`, branch
`feature-completion-study`, working tree carrying one unrelated modification
(`M lingbot_map/heads/dpt_head.py`) plus this task's new files. No training was launched, no
target was rebuilt, no threshold was changed.

> ## The headline, because it changes how two results read
>
> **The KITTI-360 reference checkpoint and the Waymo checkpoint were supervised over ~10 s.
> The ONCE checkpoint was supervised over 2 s.** Measured from the target files those three
> trainings actually read:
>
> | arm | frames | sweeps | span | spacing | direction |
> |---|---:|---:|---:|---:|---|
> | KITTI-360 | **21** | **21** | **10.44 s** | 0.522 s | future only |
> | Waymo | **21** | **105** (21 × 5 lasers) | **10.00 s** | 0.500 s | future only |
> | ONCE | **5** | **5** | **2.00 s** | 0.500 s | future only |
>
> So the answer to the question this audit was given — *did KITTI-360 train with 21 sweeps over
> ~10.46 s while ONCE trained with ~5 sweeps over 2 s?* — is **yes, confirmed from the
> caches, not inferred from a statistics probe.**
>
> The consequence is asymmetric and it has not been stated anywhere before:
>
> * the **Waymo** arm is a corpus substitution that **preserved** the frozen supervision
>   window;
> * the **ONCE** arm changed the corpus **and** shortened the supervision window by 5.2×.
>
> Treating the two as two interchangeable draws from "OccAny's sources" is therefore wrong.
> Only one of the two failing arms is a clean source-only intervention.

---

## 1. Method, and why a statistics probe would not have answered it

This audit opens the target `.npz` files the three trainings consumed and reads what the
builder wrote into them — `frames_used`, `future_natives`, `n_sweeps`, per-sweep timestamps
where the corpus publishes them — then recomputes the occupancy statistics from the stored
`valid_packed` / `occ_packed` volumes so all three arms are summarised the same way.

That distinction is load-bearing. `once_supervision_validation.md` §5 measured KITTI-360 under
a **rebuilt** 5-sweep / 2.1 s window to decide whether a predeclared band could adjudicate the
ONCE protocol. That probe answered its question, but it never trained anything, and it is not
evidence about what the KITTI-360 *checkpoint* saw. The only way to know that is to read the
cache the trainer read, which is what this does.

Sampling: even stride, no RNG. KITTI-360 3 train drives × 40 anchors = 120; Waymo 60 of 791
segments × 5 = 300; ONCE 6 sequences × 40 = 240.

## 2. The provenance table

| | **KITTI-360** (reference) | **Waymo** | **ONCE** |
|---|---|---|---|
| **status** | frozen baseline | exploratory arm | **exploratory, prevalence gate FAILED** |
| checkpoint (seed 0) | `artifacts/baseline_reconstruction/checkpoints/geo_seed0_step06000.pt` | `artifacts/waymo_arm/checkpoints/geo_seed0_step06000.pt` | `artifacts/method_vs_data/once_arm/checkpoints/geo_seed0_step06000.pt` |
| sha256 | `7f5e853dd05b…` | `e40fb93467e4…` | `03fc0d26fc83…` |
| target cache read | `/media/SSD1/MINH_DATASETS/lingbot_k360_7drives/targets` | `/media/minh/MINH1/lingbot_waymo_arm/targets` | `/media/minh/MINH1/lingbot_once_arm/targets` |
| **training scenes** | 3 drives (0003, 0007, 0010) | **751 segments** | **5 sequences** (000076/80/92/104/113) |
| val scene, disjoint | drive 0006 | 40 held-out segments | sequence 000121 |
| **training anchors** | **1,358** | **3,755** (5 per segment) | **1,259** |
| val anchors used | 118 | 40 | 65 |
| **independent scenes** | **3** | **751** | **6** |
| eligible anchors | every eligible stream frame | 13 per segment eligible | 1,771 |
| **input frames** | 5, `t−4…t`, stream stride 5 native | 5, `t−4…t`, stream stride 5 native | 5, `t−4…t`, native stride 1 |
| input direction | **causal, 0 future frames** | **causal, 0 future frames** | **causal, 0 future frames** |
| input span | ≈2.09 s | 2.00 s | 2.00 s |
| camera | `image_00`, **103.9°** HFOV | FRONT, **50.0°** HFOV | `cam01`, **90.1°** HFOV |
| **target sweep selection** | anchor + **20 future stream frames** (`FUTURE_FRAMES = 20`) | anchor + **20 future stream frames** (same constant) | anchor + **4 future native frames** (2.1 s duration cap) |
| **frames used** (min/p50/max) | **21 / 21 / 21** | **21 / 21 / 21** | **5 / 5 / 5** |
| **sweeps** (a sweep = frame × laser) | **21** | **105** | **5** |
| lasers | 1 (HDL-64E, 64 beams) | **5** (per-point laser id from a sidecar) | 1 (40-beam, ~200 m roof) |
| **target span, measured** | **10.44 s** (min 10.41, max 48.99*) | **10.00 s** | **2.00 s** (min 1.998, max 2.002) |
| span source | KITTI-360 `timestamps.txt` | native index span ÷ 10 Hz nominal | per-sweep timestamps stored in the target file |
| implied sweep spacing | **0.522 s** | 0.500 s | 0.500 s |
| **temporal context of the target** | **future only** | **future only** | **future only** |
| sensor origin | Velodyne, published extrinsics | 5 laser origins, per-(frame, laser) carve | roof LiDAR; origin **confirmed from data** (road at −1.73…−1.76 m, elevation p1 −24.97°) |
| grid / frame | `(256,256,32)`, 0.2 m, anchor velodyne frame | same grid, anchor-vehicle frame, z aligned to measured road | same grid, asserted at import |
| carve rule | `core/supervision/rawtarget.py` | same rule via `rawtarget_multisensor.py`, gated bit-identical to the single-sensor path | same rule, single-sensor path |
| carve constants | near 1.0 m, band half 0.2 m, decimation 4, sky margin 1 / neighbourhood 7 | identical | identical |
| conflict rule | occupied in one sweep + free in **another** → **unknown**, never voted | identical | identical |
| **occupied prevalence in valid** (p50, recomputed) | **0.0227** | **0.0200** | **0.0131** |
| prevalence denominator | valid (known) voxels of that anchor's target | identical | identical |
| prevalence sampling population | 120 anchors, 3 drives | 300 anchors, 60 segments | 240 anchors, 6 sequences |
| valid fraction (p50) | 0.633 | 0.671 | 0.579 |
| conflict of touched (p50) | **not recorded** in the frozen cache | 0.062 | 0.017 |
| occupied voxels (p50) | 29,253 | 28,102 | **13,493** |
| **source validation** | source-domain eval, logged, never used for selection | same | same (final val IoU 0.147/0.156/0.161) |
| **checkpoint selection** | **final step (6,000), no selection** | identical | identical |
| recipe file | `artifacts/baseline_reconstruction/training_config.yaml` | same file | same file |
| recipe sha256 | `4ef021b6…` | `4ef021b6…` | `4ef021b6…` |
| steps / batch / crop | 6,000 / 4 / 128×128×32 | identical | identical |
| loss | focal + Dice, `w_focal` 1.0, `w_dice` 1.0, γ 2.0, `pos_weight_cap` 8.0, `w_unknown` 2.0 | identical | identical |
| optimizer | AdamW, lr 1e-3, wd 0.01, clip 1.0, cosine | identical | identical |
| parameters | 967,322 | 967,322 | 967,322 |
| seeds trained | 0, 1, 2 | 0, 1, 2 | 0, 1, 2 |
| init state digest, seed 0 | `1956770cb219…` | `1956770cb219…` | `1956770cb219…` |

`*` one sampled KITTI-360 anchor spans 48.99 s: the drive's own timestamps contain a long gap
inside its 21-frame window. It is a property of the frozen cache, reported rather than
filtered; the median is unaffected.

**Native sensor frequency against pipeline-selected frequency** — the distinction the brief
asks for explicitly:

| corpus | native sensor rate | rate the pipeline selects | selection |
|---|---:|---:|---|
| KITTI-360 | **9.56 Hz** (0.1046 s velodyne period) | **1.91 Hz** | every 5th native frame |
| Waymo | 10 Hz | 2.0 Hz | every 5th native frame |
| ONCE | **2 Hz** | **2.0 Hz** | every native frame |

All three arms therefore supervise at ~2 Hz. ONCE's 2 FPS release was never the obstacle, and
the retraction of that claim in `once_vs_occany.md` Addendum 1 is confirmed here from the
caches. What differs is not the cadence but **how many of those ~2 Hz sweeps the window holds**:
21 against 5.

The same three initialization digests being identical confirms the three arms start from the
same weights per seed, so nothing in the comparison is an initialization difference.

## 3. Does a duration-matched KITTI-360 training control exist?

**No.** Every KITTI-360 completion checkpoint in this repository was trained on the frozen
21-sweep / 10.44 s cache. Three KITTI-360 5-sweep artefacts exist and **none of them is a
trained control**:

| artefact | what it is | window | trained? |
|---|---|---|---|
| `k360_duration_matched.json`, `span2.1` rows | target-statistics probe | **5 sweeps over 2.1 s** — matches ONCE on both axes | **no** |
| `k360_sweep_control.json`, `dec5` rows | target-statistics probe | 5 sweeps over the **full 10.46 s** — sweep-count matched, duration *un*matched | **no** |
| `k360_beam_control.json` / `k360_point_budget_control.json` | sweep-thinning probes | **the frozen 21-sweep window throughout** | **no** |

What such a control would isolate, stated without launching it: retrain the frozen recipe on
KITTI-360 with the supervision window truncated to 2.0–2.1 s and everything else — drives,
anchors, input clips, grid, loss, schedule, seeds — held fixed. That isolates **supervision
window length** from corpus identity, and it is the only intervention that can tell whether
ONCE's 8.2-point SemanticKITTI deficit against KITTI-360 is a property of ONCE's scenes or a
property of training a completion network on 2 s of accumulated target instead of 10 s. It is
the one experiment that would make the ONCE arm interpretable. **Not launched; not authorized
by this task.**

Note also what the measured statistics already say about the window: at 2.1 s KITTI-360 drops
from 32 occupied z-levels to **21**, which is a change in the *shape* of the supervision and
not only its amount. A truncated-window KITTI-360 control would therefore differ from the
reference arm in a structured way, not a uniform one.

## 4. The scope of the sparsity experiments, stated honestly

`once_supervision_validation.md` §7 concluded that sparsity is "falsified" as the explanation
of ONCE's statistics. That conclusion is correct **for what it tested** and is narrower than it
reads:

* every thinning condition was built at the **frozen 21-sweep window** (`k360_beam_control.py`
  states this at its own call site: *"poses, calibration, the 21-sweep window and the grid are
  the frozen ones"*);
* the quantity measured was **target prevalence**, not any trained model's transfer;
* so it shows that removing up to 85 % of returns does not move prevalence out of band **when
  21 sweeps are accumulated**. Accumulation over 10 s is exactly the mechanism that makes
  prevalence insensitive to per-sweep density, so the result is close to a property of the long
  window.

It therefore does **not** rule out:

1. a density effect under **5-sweep** supervision, where there is 4× less accumulation to hide
   behind — untested in any condition;
2. a density effect on **learned transfer** rather than on target statistics — no thinned
   KITTI-360 arm was ever trained, so no thinning experiment in this directory speaks to model
   quality at all.

And on the proposed content-matched alternative: `method_vs_data_updated_conclusions.md`
Addendum 2 §5 suggests selecting more open KITTI-360 anchors until prevalence reaches 0.0131.
That is **not a pure density intervention** — selecting by openness changes which scenes are
trained on, so it varies scene content and supervision density together. It is a different
experiment from the duration-matched control in §3, and neither substitutes for the other.

## 5. Differences that remain uncontrolled, per arm

Against the KITTI-360 reference, under one identical recipe:

| difference | Waymo | ONCE |
|---|---|---|
| training corpus | ✗ changed | ✗ changed |
| **supervision window length** | **✓ preserved (21 frames, 10.0 s)** | **✗ changed (5 frames, 2.0 s)** |
| sweeps per window | ✗ 105 vs 21 (5 lasers) | ✗ 5 vs 21 |
| camera HFOV | ✗ 50.0° vs 103.9° | ✗ 90.1° vs 103.9° |
| independent scenes | ✗ 751 vs 3 | ✗ 6 vs 3 |
| target prevalence | ✗ 0.0200 vs 0.0227 | ✗ **0.0131** vs 0.0227 |
| monocular metric-scale bias | not measured on this diagnostic | ✗ 1.35× vs KITTI-360's 1.20× |
| input frame count / direction | ✓ identical (5, causal) | ✓ identical (5, causal) |
| target temporal direction | ✓ identical (future only) | ✓ identical (future only) |
| carve rule, conflict rule, grid | ✓ identical | ✓ identical |
| recipe, loss, optimizer, steps, batch, crop | ✓ identical (`4ef021b6…`) | ✓ identical (`4ef021b6…`) |
| initialization per seed | ✓ identical digest | ✓ identical digest |

**The Waymo arm is the cleaner of the two source interventions.** It changes corpus, sensor
count and camera FOV while preserving the supervision window. The ONCE arm changes all of that
*plus* the window, and its window change is the largest single departure from the frozen recipe
anywhere in this directory.

## 6. Reproduction

```bash
cd /home/minh/workspace/lingbot-map_fork
PYTHONPATH=$PWD python tools/method_vs_data/supervision_provenance.py \
    --per-scene 40 --max-scenes 60
```

## 7. Not claimed

* No causal attribution of either arm's deficit to the supervision window, the corpus, the
  camera, the prevalence or the scale bias. Those move together and none is isolated.
* No statement that the ONCE window is incorrect. It was authorized, it is built from real
  native sweeps, nothing is interpolated or duplicated, and every correctness gate passed.
  It is simply **not the frozen recipe**, and that was not previously recorded in the arm
  comparison tables.
* The ONCE prevalence gate remains **failed** at 0.0131 against the [0.015, 0.040] floor. No
  threshold was altered and no retrospective pass is declared.
