# Final zero-shot verification — 9/9 pass

| check | pass | detail |
|---|---|---|
| 1 protocol and evaluators frozen before any prediction or score; unchanged since | yes | `{"protocol": "2026-09-15T10:31:34+02:00", "evaluators": "2026-09-15T10:31:34+02:00"}` |
| 2 frozen B3 checkpoints (== Gate 16 code freeze) and geometry checkpoint (== baseline freeze); none trained after target access | yes | `{"0": true, "1": true, "2": true}` |
| 3 one canonical prediction per evaluated sample | yes | `{"semantickitti": [801, 801], "occ3d": [4219, 4219]}` |
| 4 occupancy (geometry seed 0, causal) identical per sample to the frozen OccAny-comparison evaluation | yes | `{"semantickitti": true, "occ3d": true}` |
| 5 Track O and Track P (full region) read the same predictions: confusions identical | yes | `{"semantickitti": {"B0": true, "B3_s0": true, "B3_s1": true, "B3_s2": true}, "occ3d": {"B0": true, "B3_s0": true, "B3_s1": true, "B3_s2": true}}` |
| 6 Trident tokens for every history frame; tokens reproduce Trident's coarse probabilities (argmax agreement >= 0.99) | yes | `{"semantickitti": {"frames": 805, "tokens": 805, "min_argmax_agree": 0.9980512261390686}, "occ3d": {"frames": 4819, "tokens": 4819, "min_argmax_agree": 0.9968500137329102}}` |
| 7 labels: OccAny stored labels == independent official loader on every evaluated sample | yes | `{"semantickitti": true, "occ3d": true}` |
| 8 predictions reproduce on re-run (deterministic) | yes | `{"semantickitti:08_000020": true, "semantickitti:08_002020": true, "occ3d:scene-0003_02651ca22b08484294d473d9c26cd9af": true, "occ3d:scene-0632_b39f4a387bfb416789997e070446663e": true}` |
| 9 tests (protocol, serialization, masks, metrics, KITTI-360 reproduction of the grid-generic code) | yes | `"9 passed, 2 warnings in 3.77s"` |
| note | yes | `"the first freeze command ran while one test failed because its expected value was wrong (class-1 FP count); the metric code was correct and no frozen file changed; the test was corrected and all tests pass"` |
