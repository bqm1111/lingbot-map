# Verification — 15/15 pass

| check | pass | detail |
|---|---|---|
| 1 frozen stages unchanged | yes | `{"predeclaration": "2026-09-14T19:56:10+02:00", "reliability": "2026-09-14T19:59:35+02:00", "addendum_A": "2026-09-14T20:07:43+02:00", "decision": "2026-09-14T20:11:28+02:00"}` |
| 2 freeze order | yes | `{"predeclaration < targets": true, "reliability freeze < targets": true, "addendum A < first checkpoint": true, "decision freeze < first evaluation": true, "no evaluation existed at decision freeze": true}` |
| 3 provenance replay == Gate 15 evidence (voxels and counts asserted per anchor; features sampled) | yes | `{"0003": {"files": 193, "anchors": 193, "feature_max_abs": 0.00026279687881469727, "class_agree_min": 0.9998666644096375}, "0007": {"files": 569, "anchors": 569, "feature_max_abs": 0.00026357173919677734, "class_agree_min": 0.9998438358306885}, "0010": {"files": 596, "anchors": 596, "feature_max_abs": 0.0002632737159729004, "class_agree_min": 0.9997971653938293}, "0006": {"files": 590, "anchors": ` |
| 4 no firewall violation | yes | `{"files": 22, "violations": 0}` |
| 5 reliability gate as declared (strict per-drive reading reported) | yes | `{"declared": true, "strict_per_drive_incl_dev": false, "tiers": [1, 2]}` |
| 6 B1/B2/B2s identical observed-term draws and steps; final checkpoints hashed | yes | `{"B1": {"digest": "99a9eec4bdae", "steps": 3000, "tiers": []}, "B2": {"digest": "99a9eec4bdae", "steps": 3000, "tiers": [1, 2]}, "B2s": {"digest": "99a9eec4bdae", "steps": 3000, "tiers": [1, 2]}}` |
| 7 evaluation covers every 0006 anchor once | yes | `{"anchors": 590}` |
| 8 evidence path identical across methods (K3) | yes | `{"anchors_equal": 590}` |
| 9 occupancy TP/FP/FN == Gate 15 seed 0 per anchor (K6) | yes | `{"TP_FP_FN": [6172202, 27026109, 12021922]}` |
| 10 B0 == Gate 15 B0 seed 0 per anchor (primary, ssc_full) | yes | `{}` |
| 11 predictions-at-t reproduce from the stored record; adding GT-occupied queries changes no prediction on O | yes | `{"4": {"stored_record_reproduced": true, "query_independent": true}, "754": {"stored_record_reproduced": true, "query_independent": true}, "1504": {"stored_record_reproduced": true, "query_independent": true}}` |
| 12 test suite | yes | `"11 passed, 2 warnings in 5.35s"` |
| 13 live LingBot: future images first leave the clip bit-identical; frozen mapper reproduces the cached hard map | yes | `{"t": 904, "clip_equal_after_future": true, "clip_equal_cache": true, "hard_map_equal": true, "max_abs_depth_after_vs_fresh": 0.0}` |
| 14 delayed targets rebuild (exact; fp16 distribution within 1e-4 with identical argmax) | yes | `{"0006:304": {"exact_except_prob": true, "prob_max_abs": 6.103515625e-05, "prob_argmax_equal": true}, "0010:204": {"exact_except_prob": true, "prob_max_abs": 0.0, "prob_argmax_equal": true}}` |
| 15 prediction/evaluation path does not import target construction | yes | `"False"` |
