# OccAny comparison of the frozen current baseline

Answers one question: are the frozen baseline's zero-shot numbers (SemanticKITTI 26.25 % / Occ3D 31.56 %)
comparable to OccAny's published 25.91 % / 23.55 %? The quoted pairs differ in sample set, post-processing,
temporal direction and (Occ3D) benchmark placement at once. This tool scores both methods on **every sample
OccAny scored** (805 / 4,819), with OccAny's stored labels, OccAny's scorer and OccAny's pooling, and removes
the remaining differences one at a time. Result: [artifacts/occany_comparison/report.md](../../artifacts/occany_comparison/report.md).

Nothing is trained or tuned. Every GPU entry point verifies the baseline's source freeze; frozen code under
`core/`, `gates/` and `tools/baseline_reconstruction/` is imported, never edited.

| module | role |
|---|---|
| `config.py` | paths, OccAny runs, protocols (`forward` 0/2/4/6/8 target first; `causal` −4..0), placements, subsets, predeclared primary |
| `samples.py` | OccAny's sample directories → our streams (full Occ3D scenes rebuilt from `annotations.json`, prefix asserted equal to the frozen stream) |
| `moge_extra.py` | MoGe-B for the 109 Occ3D frames past our 40-frame streams (frozen recipe, separate root, reproduction-checked) |
| `clips.py` | five-frame local-reset LingBot clips for any window, frozen clip-scale rule |
| `geometry.py` | hard map `(S, F)` with the target at any slot; last slot == frozen function bit for bit |
| `scoring.py` | OccAny outputs, independent official-label check, OccAny pooling and `SSCMetrics` |
| `evaluate.py` | per-sample TP/FP/FN for every variant, sharded |
| `analysis.py` | pooled metrics, seed statistics, paired block bootstrap |
| `verify.py` | 11 checks recomputed from files (incl. exact reproduction of the frozen per-anchor counts) |
| `report.py` | results table, figures, `verification.md`, `report.md` |
| `figures_3d.py` | six-panel 3D maps ours vs OccAny (raw and pooled), samples at fixed Δ-rank quantiles, drawn volumes asserted equal to the evaluation |

Run order: see the *Files* section of the report. Tests: `pytest tests/occany_comparison`.
