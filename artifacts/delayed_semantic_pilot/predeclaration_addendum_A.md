# Predeclaration addendum A — 2026-09-14T20:06:18+02:00, after the reliability statistics, before any head was trained

1. **Disclosed implementation discrepancy.** The first run of `reliability.py` evaluated G1 on each drive
   separately *including development drive 0006*. The rule above says training drives pooled. That reading
   failed only at offset 10 (p90 0.207 m on 0010, 0.217 m on 0006; limit 0.20 m); G2–G4 passed. The gate is
   re-run exactly as declared (training drives pooled); the strict per-drive reading stays in
   `reliability.json` (`G1_strict_per_drive_incl_dev`) and the first output is kept as
   `reliability_first_run_per_drive_gate.json`. This correction was made after seeing the failure.
2. **Secondary sensitivity condition (not part of K1–K6).** Because the residual grows with the time offset,
   **B2s** = B2 trained on targets rebuilt with the same code but a 4 s / 8-frame window (every view and every
   persistence sweep within 4 s of t, where every drive meets G1 separately). Same draws and steps. Reported
   next to B2; if B2 passes and B2s does not, the result is flagged as dependent on long-offset targets.
3. The endpoint, thresholds and K1–K6 are unchanged. No model result exists at this time.
