# Like-for-like evaluation of our semantics against OccAny

Our frozen heads, scored under the settings OccAny actually publishes. Nothing is trained, tuned or selected here: the
method, checkpoints, teacher, grids and metric are the frozen ones, and only the **sample set** and the **input window**
change.

## Which OccAny setting is which

`occany/datasets/eval_helper.py` defines a different input and a different ground truth per setting, and OccAny publishes
each dataset's semantics in **one** of them:

| dataset | OccAny setting | cameras | window | ground truth | published semantics |
|---|---|---|---|---|---|
| SemanticKITTI | `5frames` | 1 | 5 frames, interval 5 | official label | mIoU 7.28 / superclass 13.53 |
| nuScenes | `5frames` | 1 (`CAM_FRONT`) | 5 frames, interval 2 | `use_surround_label=False` → `voxel_label[:100] = 255` | none |
| nuScenes | `surround` | 6 | 1 timestep | `use_surround_label=True` → full volume | mIoU 6.66 / superclass 10.32 |

The saved OccAny outputs on this machine are the two `5frames` runs, and they store **occupancy only** — which is why the
geometry comparison could re-score their predictions with our evaluator but the semantic column has to be their published
number. Their stored nuScenes label confirms the setting: no valid voxel at x < 100.

**Consequence:** our forward protocol is comparable to OccAny on SemanticKITTI, and only our *surround* setting is
comparable on Occ3D.

## Settings we evaluate

`evaluate.SETTINGS` maps each to (prediction source, ground truth):

| setting | input | ground truth | what it is for |
|---|---|---|---|
| `forward` | 5 forward frames, 1 camera | OccAny's stored label | matches OccAny `5frames`; the SemanticKITTI comparison |
| `forward_surround_label` | same predictions | 360° label | isolates the rear-coverage penalty from the camera count |
| `surround` | 6 cameras, 1 timestep | 360° label | matches OccAny `surround`; the Occ3D comparison |

## The surround setting

Six cameras of one timestep in **ring order** (`CAM_FRONT, CAM_FRONT_LEFT, CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT,
CAM_FRONT_RIGHT`), so consecutive slots overlap — our model is sequential and binds views through its KV cache, unlike
OccAny's shared decoder memory, which takes them in its own order. **No rig calibration is fed to either model.** The
only calibration used is the placement of the finished map in the benchmark grid, `T_cam_to_voxel`, which is
`CAM_FRONT`-to-ego and is exactly what OccAny uses in both of its settings (checked to 1.8e-16 by
`surround_predict.check_placement`).

The clip scale is the unchanged frozen rule: a MoGe-2 candidate per view, median over the clip.

## Running it

```bash
bash artifacts/like_for_like/run_surround_tokens.sh    # 1. Trident-H tokens for the 5 non-front cameras (~12 GPU-h)
bash artifacts/like_for_like/run_surround_moge.sh      # 2. MoGe-2 depth for the same frames   (~1.5 GPU-h)
bash artifacts/like_for_like/run_surround.sh           # 3-6. clips, predictions, scoring, report
```

Every stage skips what already exists, so any script can be rerun to resume.

## Verification

| claim | check |
|---|---|
| the prediction body is the frozen pipeline | `python -m tools.like_for_like.verify` → `all_equal: true` against the frozen stored predictions |
| the surround MoGe recipe is the frozen one | `surround_moge.reproduction_check` re-runs cached frames: 0.0 depth difference, identical masks |
| the 360° label is OccAny's surround label | `tests/like_for_like/test_surround.py` — identical to their stored label on the front half, valid on the rear |
| the placement needs no rig calibration | `surround_predict.check_placement` |
| six-slot windows are handled | `tests/like_for_like/test_surround.py` — `n_frames` counts six, five-slot reading unchanged |
