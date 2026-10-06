# OccAny comparison audit (recovered 2026-09-15 from the paper text, the official repository and files on this machine)

Provenance: paper text `/tmp/occany.txt` (extracted from the OccAny PDF); repository `/home/minh/workspace/OccAny` (commit
`04d4616`, README "Evaluation" section); released checkpoint and prior extraction logs under `/media/SSD1/MINH_DATASETS/occany_*`;
our earlier like-for-like geometry study `artifacts/occany_comparison/`. Nothing below is from memory.

| item | OccAny | source |
|---|---|---|
| training data | a mixture of five urban datasets — Waymo, DDAD, PandaSet, VKITTI2, ONCE — "using images from all cameras and projected LiDAR pointmap as ground truth" | paper §4 "Training" (text lines 354–357) |
| target datasets | SemanticKITTI (val seq 08) and Occ3D-nuScenes (val); described as out-of-domain / zero-shot ("never trained on SemanticKITTI") | paper lines 106–107, 296, 484 |
| target-domain supervision | none (no target 3D labels, no target training) | paper lines 354–357 |
| semantics supervision | Segmentation Forcing distils SAM2 features into the reconstruction model; test-time semantics by Grounded-SAM2 (GroundingDINO boxes from text prompts → SAM2 masks → features on voxels), several prompt variants per class | paper §3, Tab. 4; `occany/datasets/kitti.py` prompt lists |
| cameras | SemanticKITTI: 1 (left colour); Occ3D-nuScenes: 1 (5-frame front setting, geometry Tab. 1) or **6 surround cameras** (surround setting, geometry Tab. 3 and **all published nuScenes semantics**, Tab. 4) | README presets 2/3/5; `eval_helper.prepare_eval_setting` |
| temporal context | SemanticKITTI 5 frames `[0, 2, 4, 6, 8]` of a 10-frame clip at interval 5 **starting at the target** (target + ~4 s of future); nuScenes 5-frame: target + later keyframes; surround: one timestep | `eval_helper.py` (`recon_view_idx`, `video_length`, `frame_interval`) |
| causal / online | not causal in the published sequence settings (future frames of the target); surround single-timestep is causal but multi-camera | same |
| novel-view rendering | yes: renders extra views along/around the trajectory (`-rot 60 -vpi 10 -fwd 3`), up to 180 views | README presets, "SLURM workflow" |
| input resolution | 512×160 (KITTI), 512×288 (nuScenes) | paper Tab. 1/3/4 |
| grids | official: SemanticKITTI 256×256×32 @ 0.2 m; Occ3D 200×200×16 @ 0.4 m | stored `voxel_label` shapes |
| classes | SemanticKITTI 20 incl. empty (19 semantic); Occ3D 18 incl. `others` and `free` | `kitti.py`, `nuscenes.py` `CLASS_NAMES` |
| evaluation masks | label ≠ 255; Occ3D `mask_camera`; single-camera settings additionally ignore the rear half (x index < 100) | stored labels; `tools/occany_comparison/scoring.independent_label` (verified identical per sample) |
| metrics | `SSCMetrics`: SC P/R/IoU; mIoU = mean over non-empty classes of TP/(TP+FP+FN+1e-5), absent classes count as 0; KITTI ignores the (non-existent) `other` index 20; Occ3D includes `others`; superclass mIoU (`mIoU_sc`) with the repo's superclass maps | `occany/metrics/ssc.py`, `compute_metrics_from_saved_voxels.py` |
| post-processing | 3×3×3 majority pooling (geometry "separate" mode) | README `USE_MAJORITY_POOLING=1 POOLING_MODE=separate` |
| published SC IoU | SemanticKITTI 5-frame **25.91** (P 36.79, R 46.70); Occ3D 5-frame **23.55**; Occ3D surround **34.15** | paper Tab. 1/3; README expected metrics |
| published semantic mIoU / mIoU_sc | SemanticKITTI 5-frame **7.28 / 13.53**; Occ3D **surround** **6.66 / 10.32** (distill semantics); pretrained-SAM2 variant 7.67 / 13.75 and 7.42 / 10.78 | paper Tab. 4; README Tab. 5–7 rows |
| released checkpoints | `occany.pth` (MUSt3R-based), `sam2.1_hiera_large.pt`, `groundingdino_swinb_cogcoor.pth` — all on this machine | `/media/SSD1/MINH_DATASETS/occany_checkpoints` |
| model size | 651,129,550 parameters for encoder + decoder + raymap encoder + generative decoder (log of the released run); plus SAM2-L (~224 M) and GroundingDINO Swin-B (~172 M, approximate public figures) for semantics | `occany_out/logs_kitti/extract_pid0.log` |
| inference cost (geometry) | 0.79 s per emitted prediction (p10 0.63, p90 1.93, n = 1,215) on this machine, excluding semantics | `occany_out/cost.log` |
| released predictions here | geometry only (binary `render_recon_gen_*`), 805 SemanticKITTI and 4,819 Occ3D 5-frame front-camera samples; **no semantic predictions** | pickle keys |

## Comparison levels used

* **Level A, geometry:** the released predictions go through Track P and Track O unchanged (raw, and with OccAny's published pooling).
  Input differences remain: OccAny sees the target frame and four *later* frames (non-causal), rendering extra views; we see four
  *earlier* frames (causal).
* **Level A, semantics:** attempted with the official semantic presets (see `final_zero_shot_results.md` for the outcome).
* **Level B, semantics:** OccAny's published Table 4 numbers beside our Track O numbers. On Occ3D these come from the six-camera
  surround setting with the full camera mask; ours use one front camera and the front-half mask. Not like-for-like.
* **Level C:** the table above — training data, supervision, cameras, temporal direction, rendering, model size.
