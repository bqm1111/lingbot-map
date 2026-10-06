# Gate 16 verification — 15/15 pass

| check | pass | detail |
|---|---|---|
| 1 same geometry / evidence / target caches as the pilot (counts; none written after the Gate 16 predeclaration) | yes | `{"provenance": true, "delayed_targets": true, "train_cache": true, "files_newer_than_predeclaration": 0}` |
| 2 same occupancy checkpoint (sha256 == frozen baseline) | yes | `"artifacts/baseline_reconstruction/checkpoints/geo_seed0_step06000.pt"` |
| 3 same Trident evidence: evidence path identical, B0 per-anchor confusions == pilot | yes | `{"B0_observed_current_equal_pilot": true, "B0_observed_past_only_equal_pilot": true, "B0_never_observed_equal_pilot": true, "B0_completed_never_observed_equal_pilot": true, "B0_completed_geo_observed_equal_pilot": true, "B0_completed_geo_unobserved_equal_pilot": true, "B0_completed_later_revealed_eq` |
| 4 same candidate sets: C0 and oracle == frozen ceiling; B1/B2 per-anchor confusions == pilot (same candidates, same heads) | yes | `{"C0_equal_ceiling": true, "oracle_equal_ceiling": true}` |
| 5 same delayed targets (rebuild test) / 6 prefix-causal prediction (tests) / test suite | yes | `"20 passed, 6 warnings in 8.44s"` |
| 7 no GT labels in training (TRAIN_FORBIDDEN audit, no violation; eval masks forbidden) | yes | `{"runs": 6}` |
| 8 B0 reproduces the pilot | yes | `"per-anchor confusions, every region"` |
| 9 candidate ceiling reconstructed and frozen before any B3 score | yes | `{"ceiling_frozen": "2026-09-15T00:11:18+02:00"}` |
| 10 selection threshold frozen before evaluation (payload == ceiling file == analysis) | yes | `{"T": 0.5165905879676816, "code_frozen": "2026-09-15T00:14:56+02:00"}` |
| 11 no target dataset accessed (eval and train audits forbid them, no violations) | yes | `{"shards": 4}` |
| 12 occupancy bit-identical to the pilot / Gate 15 seed 0 per anchor | yes | `"TP/FP/FN per anchor"` |
| 13 candidate count fixed at 16 | yes | `{"B1": 16, "B2": 16, "B3_s0": 16, "B3_s1": 16, "B3_s2": 16, "B3geom_s0": 16, "B3geom_s1": 16, "B3geom_s2": 16}` |
| 14 no architecture-size increase (B3 == pilot scorer; B3geom smaller) | yes | `{"B1": 25489, "B2": 25489, "B3_s0": 25489, "B3_s1": 25489, "B3_s2": 25489, "B3geom_s0": 9089, "B3geom_s1": 9089, "B3geom_s2": 9089}` |
| 15 no candidate radius / search-rule tuning (one ModelConfig for all heads; candidates use model.nearest_candidates with K) | yes | `{"evidence_proj": 16, "hidden": 64, "k_candidates": 16, "path_samples": 8, "range_norm_m": 6.0, "trunk_channels": 24}` |
| 16 freeze order and matched draws | yes | `{"predeclaration < first B3 checkpoint": true, "predeclaration < ceiling < code < evaluation": true, "seed-0 B3 draws == pilot B2 draws": true}` |
