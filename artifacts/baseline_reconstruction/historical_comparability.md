# Historical vs current baseline -- comparability audit

The classification is mechanical (`metrics/comparability.py`): evaluation-side differences decide *not comparable*; pipeline-side differences decide *partially* vs *fully* comparable. Magnitudes never enter the classification.

## Property table

| Property | Historical post-ceiling-fix baseline | Current geometry-only baseline | differs |
|---|---|---|---|
| LingBot mode | long-stream causal replay (persistent KV cache, stream depth/pose cache) | local-reset L, five frames t-4..t, KV cache cleared per clip | yes |
| depth | LB-M (LingBot canonical depth x metric scale) | LB-M (LingBot canonical depth x metric scale) | no |
| scale | stream scale: MoGe-2 vs LingBot, median of the FIRST five stream frames, frozen for the stream | clip scale: MoGe-2 vs LingBot, median over the clip's own five frames | yes |
| pose | long-stream LingBot pose | local LingBot pose (anchor = identity) | yes |
| geometry representation | log-odds map with w_free, n_obs, age, semantic evidence (H mapper + accumulators) | H: hard S / F from the same mapper, semantics off | yes |
| input channels | 32 | 3 | yes |
| semantics | Trident-H probabilities as input; semantic KL loss 0.5; semantic head | none | yes |
| residual lock | final = base + residual * [|base| < 2] | none (base log-odds 0: the network output is the map) | yes |
| completion architecture | CompletionUNet width 24, cin 32, 986,114 params, replicate padding | CompletionUNet width 24, cin 3, 967,322 params, replicate padding | yes |
| training data | KITTI-360 0003/0007/0010; targets: raw-LiDAR + plain-column open-sky rule (pre-647bcf3) | KITTI-360 0003/0007/0010; targets: raw-LiDAR + neighbourhood-7 open-sky rule + 647bcf3 fix | yes |
| validation drive / checkpoint rule | 0006; highest AP -> step 6000 for all seeds | 0006; final step 6000, no selection | yes |
| decision threshold | per seed, IoU-maximising on 0006: -0.125 / -0.4375 / -0.3125 | 0 for every seed | yes |
| training recipe | focal+Dice (gamma 2, cap 8, w_unknown 2) + 0.5 semantic KL; AdamW 1e-3 wd 0.01; cosine; clip 1; batch 4; 128x128x32; 6000 steps | focal+Dice (gamma 2, cap 8, w_unknown 2); AdamW 1e-3 wd 0.01; cosine; clip 1; batch 4; 128x128x32; 6000 steps | yes |
| seed statistic | median of 3 | mean of 3 (median reported) | yes |
| SemanticKITTI protocol | sequence 08, image_2, stride-5 stream of 815 unique frames; anchor = last frame of each official clip; clip = stream indices i-4..i (native frames 20 apart end to end); official voxels/<frame>.invalid == 0 (label != 255) | sequence 08, image_2, stride-5 stream of 815 unique frames; anchor = last frame of each official clip; clip = stream indices i-4..i (native frames 20 apart end to end); official voxels/<frame>.invalid == 0 (label != 255) | no |
| Occ3D protocol | 150 official validation scenes, CAM_FRONT samples in timestamp order, one stream per scene; anchor = every sample with a label; clip = stream indices i-4..i; mask_camera == 1, lidar mask off, ego x-index < 100 ignored (single front camera); native voxel occupied iff any of its 8 canonical subvoxels is occupied | 150 official validation scenes, CAM_FRONT samples in timestamp order, one stream per scene; anchor = every sample with a label; clip = stream indices i-4..i; mask_camera == 1, lidar mask off, ego x-index < 100 ignored (single front camera); native voxel occupied iff any of its 8 canonical subvoxels is occupied | no |
| metric | SC IoU = TP/(TP+FP+FN) inside the official mask | SC IoU = TP/(TP+FP+FN) inside the official mask | no |
| anchor set | 163 SemanticKITTI / 1182 Occ3D official anchors | 163 SemanticKITTI / 1182 Occ3D official anchors | no |


## Classification

| dataset | class | anchor set identical | evaluation differences | pipeline differences |
|---|---|---|---|---|
| semantickitti | **partially comparable** | True | none | LingBot mode, scale, pose, geometry representation, input channels, semantics, residual lock, completion architecture, training data, validation drive / checkpoint rule, decision threshold, training recipe, seed statistic |
| occ3d | **partially comparable** | True | none | LingBot mode, scale, pose, geometry representation, input channels, semantics, residual lock, completion architecture, training data, validation drive / checkpoint rule, decision threshold, training recipe, seed statistic |


## Numbers side by side

| dataset | current mean (median) | historical median (mean) | current map-only | historical map-only | class |
|---|---|---|---|---|---|
| KITTI-360 0006 (source, full region) | 13.51 () | n/a (historical source metric used a different target build and threshold) () |  |  | not comparable |
| semantickitti | 26.17 (26.25) | 22.71 (22.93) | 8.82 | 8.90 | partially comparable |
| occ3d | 31.22 (31.56) | 33.88 (34.13) | 10.04 | 9.99 | partially comparable |


**What the comparison can and cannot say.** Both targets are *partially comparable*: the anchors, masks, grids, cameras and the SC IoU computation are identical, so the numbers measure the same thing on the same voxels -- but the two systems differ in every pipeline row marked above. A difference in IoU therefore cannot be attributed to local-reset geometry alone; the closest fair control for the geometry question does not exist in the historical record (there is no long-stream geometry-only 3-channel model), and this experiment did not build one because its purpose is a clean forward baseline, not a retrospective ablation.

The map-only rows are the one like-for-like quantity: they use no network, so they isolate what the geometry mode + scale rule changed in the observed map itself (historical map-only comes from the sky-model eval JSONs, `baselines.mapper_native`).
