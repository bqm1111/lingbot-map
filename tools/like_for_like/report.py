"""Markdown report: our semantics under OccAny's settings, beside the frozen causal numbers and OccAny's published ones.

Reads ``artifacts/like_for_like/results_<dataset>[_<setting>][_semantic].json`` (this package) and the frozen causal
numbers. Computes nothing new; a setting whose results file does not exist is simply left out.

The settings matter, because OccAny publishes each dataset's semantics in one setting only:
SemanticKITTI in its five-frame single-camera setting (which our forward protocol matches) and nuScenes in its
six-camera surround setting (which our surround setting matches). Rows are grouped so only comparable ones sit together.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from tools.like_for_like import config as LC
from tools.like_for_like.evaluate import OCCANY_PUBLISHED
from tools.zero_shot_eval import config as ZC

FROZEN = {"semantickitti": {"n": 801, "sc_iou": 26.40, "b0": 4.32, "b3": 4.43, "b0_sc": 8.33, "b3_sc": 8.42},
          "occ3d": {"n": 4219, "sc_iou": 31.94, "b0": 3.71, "b3": 3.80, "b0_sc": 6.63, "b3_sc": 6.76}}
NAMES = {"semantickitti": "SemanticKITTI 08", "occ3d": "Occ3D-nuScenes val"}
#: (setting, window shown in the table, ground truth shown in the table)
SETTING_ROWS = {"semantickitti": [("forward", "5 forward frames, 1 camera", "OccAny's stored label")],
                "occ3d": [("forward", "5 forward frames, 1 camera", "front half (OccAny `5frames`)"),
                          ("forward_surround_label", "5 forward frames, 1 camera", "360 deg (OccAny `surround`)"),
                          ("surround", "6 cameras, 1 timestep", "360 deg (OccAny `surround`)")]}


def mean_b3(methods: Dict, key: str) -> float:
    seeds = [v[key] for m, v in methods.items() if m.startswith("B3_")]
    return sum(seeds) / len(seeds)


def load(dataset: str, setting: str, pooling: str) -> Optional[Dict]:
    tag = ("" if setting == "forward" else f"_{setting}") + ("" if pooling == "none" else f"_{pooling}")
    path = os.path.join(LC.ARTIFACT_DIR, f"results_{dataset}{tag}.json")
    return json.load(open(path)) if os.path.exists(path) else None


def _row(name: str, r: Dict, window: str, gt: str, post: str, bold: bool = False) -> str:
    b3, b3_sc = mean_b3(r["methods"], "official_ssc_miou"), mean_b3(r["methods"], "official_superclass_miou")
    miou = f"**{b3:.2f}**" if bold else f"{b3:.2f}"
    label = f"**{name}**" if bold else name
    return (f"| {label} | {r['n_samples']} | {window} | {gt} | {post} | {r['official_sc_iou']:.2f} | {miou} | {b3_sc:.2f} |")


def table(dataset: str) -> List[str]:
    f, pub = FROZEN[dataset], OCCANY_PUBLISHED[dataset]
    comparable = "forward" if dataset == "semantickitti" else "surround"
    rows = [f"### {NAMES[dataset]}", "",
            "| setting | samples | input | ground truth | post-processing | SC IoU | SSC mIoU | superclass mIoU |",
            "|---|---:|---|---|---|---:|---:|---:|"]
    for setting, window, gt in SETTING_ROWS[dataset]:
        for pooling, post in (("semantic", "majority pooling"), ("none", "raw")):
            r = load(dataset, setting, pooling)
            if r is None:
                continue
            is_primary = setting == comparable and pooling == "semantic"
            rows.append(_row(f"ours, B3 seed mean ({setting})", r, window, gt, post, bold=is_primary))
        r = load(dataset, setting, "none")
        if r is not None:
            b0 = r["methods"]["B0"]
            rows.append(f"| ours, B0 ({setting}) | {r['n_samples']} | {window} | {gt} | raw | {r['official_sc_iou']:.2f} | "
                        f"{b0['official_ssc_miou']:.2f} | {b0['official_superclass_miou']:.2f} |")
    rows += [f"| ours, deployed causal setting (B3, frozen) | {f['n']} | 5 causal frames, 1 camera | frozen protocol | raw | "
             f"{f['sc_iou']:.2f} | {f['b3']:.2f} | {f['b3_sc']:.2f} |",
             f"| **OccAny, published (Tab. 5/6)** | {'805' if dataset == 'semantickitti' else '4,819'} | {pub['setting']} | "
             f"its own | majority pooling | — | **{pub['ssc_miou']:.2f}** | {pub['superclass_miou']:.2f} |", "",
             f"The comparable pair is the bold one: OccAny publishes this dataset's semantics only for the "
             f"*{'five-frame single-camera' if dataset == 'semantickitti' else 'surround'}* setting.", ""]
    return rows


def write() -> str:
    lines = ["# Like-for-like semantic evaluation: our heads under OccAny's settings", "",
             "**What changed from the frozen evaluation.** Only the sample set and the input window. The method, heads, "
             "checkpoints, teacher, grids, labels and metric are the frozen ones.", "",
             "- **Samples:** every sample OccAny scored (805 / 4,819), not just those with a causal window (801 / 4,219).",
             "- **Input:** for SemanticKITTI, OccAny's own five frames starting at the target and running forward "
             "(non-causal); for Occ3D, both that window and the six surround cameras of one timestep, because OccAny "
             "publishes nuScenes semantics only in the surround setting.",
             "- **Placement:** the comparison's declared primary — SemanticKITTI one shared transform, Occ3D OccAny's own "
             "target-camera-to-voxel transform (which is `CAM_FRONT`-to-ego, the transform both OccAny settings use).",
             "- **Metric:** Track O, the official computation OccAny reports (`SSCMetrics` conventions, every valid voxel).", "",
             "**Which OccAny setting is which** (`occany/datasets/eval_helper.py`). nuScenes `5frames` = one `CAM_FRONT`, "
             "five frames at interval 2, `use_surround_label=False`, which sets `voxel_label[:100] = 255` and so ignores the "
             "rear half of the grid. nuScenes `surround` = six cameras at one timestep, `use_surround_label=True`, full "
             "volume. The saved OccAny outputs on this machine are the `5frames` run, and their stored labels confirm it "
             "(no valid voxel at x < 100). Their published nuScenes semantic number is the surround one.", "",
             "**Verification.** The pipeline reproduces the frozen predictions array for array when run with the causal "
             "protocol and placement (`pipeline_equivalence.json`, `all_equal: true`). The surround ground truth reproduces "
             "OccAny's stored label voxel for voxel on the front half and restores the rear "
             "(`tests/like_for_like/test_surround.py`). The surround MoGe-2 depths reproduce cached frozen frames exactly.", ""]
    for ds in ZC.DATASETS:
        lines += table(ds)
    lines += ["## What is still not like-for-like", "",
              "- **OccAny's semantics cannot be run here** (its GroundingDINO step fails on this GPU) and its saved outputs "
              "on this machine store occupancy only, so its column is the published number, not a re-scored prediction. "
              "Only the geometry column is a same-evaluator comparison.",
              "- **Surround view order.** OccAny feeds `CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_LEFT, "
              "CAM_BACK_RIGHT` to a model with shared memory across views; we feed a ring "
              "(`CAM_FRONT, CAM_FRONT_LEFT, CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT, CAM_FRONT_RIGHT`) because our model is "
              "sequential and needs overlap with the previous view. Neither model is given the rig calibration.",
              "- **Pooling:** OccAny's released metric script applies majority pooling by default (`sh/compute_metric.sh`: "
              "`USE_MAJORITY_POOLING=1`, `POOLING_MODE=separate`), and its README ties those runs to the published numbers. "
              "The semantic branch relabels occupied voxels by a 3x3x3 majority vote and leaves occupancy untouched, so our "
              "SC IoU is identical in both rows. It is not described in the paper.",
              "- **The forward window is not our deployed setting**: it sees four frames after the target. It exists here only "
              "to match OccAny; the causal rows remain the honest description of our system.", ""]
    out = os.path.join(LC.ARTIFACT_DIR, "report.md")
    open(out, "w").write("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    print(write())
