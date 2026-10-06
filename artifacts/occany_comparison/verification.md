# OccAny comparison — verification

**19/19 checks pass.** Recomputed from files by `tools/occany_comparison/verify.py`.

| check | result | detail |
|---|---|---|
| 1 freeze | PASS | source freeze 2026-09-14T12:21:03+02:00 verified (54 files); uncommitted edits under core, gates, tools/baseline_reconstruction, tools/gate8c1, lingbot_map: none |
| 2 order | PASS | predeclaration 1789400400 < first shard 1789401014 (unix s); 6 shards |
| 3 coverage [semantickitti] | PASS | 805 samples (expected 805); forward windows 805, causal windows 801; eligible windows left unscored 0; clips without a scale 0 |
| 3 coverage [occ3d] | PASS | 4819 samples (expected 4819); forward windows 4819, causal windows 4219; eligible windows left unscored 0; clips without a scale 0 |
| 4 labels [semantickitti] | PASS | OccAny label == independent official loader on 805/805 |
| 4 labels [occ3d] | PASS | OccAny label == independent official loader on 4819/4819 |
| 5 scorer [semantickitti] | PASS | GPU tally == OccAny SSCMetrics on all 14458 scored volumes |
| 5 scorer [occ3d] | PASS | GPU tally == OccAny SSCMetrics on all 154246 scored volumes |
| 6 OccAny [semantickitti] | PASS | pooled 25.92 vs published 25.91 (raw 25.20) |
| 6 OccAny [occ3d] | PASS | pooled 23.56 vs published 23.55 (raw 20.67) |
| 7 placement [semantickitti] | PASS | max |T_occany - T_ours| = 2.22e-16 |
| 7 placement [occ3d] | PASS | |T_occany - T_ours| max element: median 0.200, max 0.466 (OccAny = static extrinsic; ours = timestamp-compensated; both scored) |
| 8 MoGe | PASS | 109 extra Occ3D frames written to a separate root; recipe reproduced 3 frozen-cache frames exactly: True |
| 9 clips [semantickitti] | PASS | 161 official-anchor causal clips vs frozen: max |depth| 0.00e+00, |conf| 0.00e+00, |pose| 0.00e+00, scale rel 0.00e+00 |
| 9 clips [occ3d] | PASS | 900 official-anchor causal clips vs frozen: max |depth| 0.00e+00, |conf| 0.00e+00, |pose| 0.00e+00, scale rel 0.00e+00 |
| 10 end-to-end [semantickitti] | PASS | 483 (anchor, seed) pairs vs frozen per-anchor counts: exact 4, TP&FN equal 483, FP <= frozen 483, FP removed by the label fix 445,289 |
| 10 end-to-end [occ3d] | PASS | 2700 (anchor, seed) pairs vs frozen per-anchor counts: exact 2700, TP&FN equal 2700, FP <= frozen 2700, FP removed by the label fix 0 |
| 11 rescoring [semantickitti] | PASS | primary delta and CI recomputed from the shards equal analysis.json |
| 11 rescoring [occ3d] | PASS | primary delta and CI recomputed from the shards equal analysis.json |
