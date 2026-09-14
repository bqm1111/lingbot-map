# SUPERSEDED — interrupted draft, not a reported result (2026-09-13)

This directory holds a partial "Gate 10 depth-source sufficiency" run (0006 audit + aggregation only;
no replay, verification or report) that was interrupted by the user and replaced by
`artifacts/depth_source_comparison/`.

Why it was superseded: its condition list assumed that Gate 9's PD was raw LingBot depth and
added a separate "LB-M" (LingBot x causal MoGe-2 clip scale). The scale audit
(`artifacts/depth_source_comparison/scale_audit.md`) shows Gate 9 PD *is* LB-M, so the draft's
LB / LB-M split duplicated Gate 9 and its "LB" row is raw canonical LingBot depth (not metric).
The draft also used no LingBot confidence gate on the shared mask and an oracle scalar named
"LB-LiDAR". Its numbers are kept unmodified for the record; do not cite them.
Code: `tools/gate10_depth_source/` (left as it ran; the never-run `scalar_check.py` was removed).
