# Resource accounting and study estimate

**Status:** updated 2026-09-15 ~20:45, after cache preparation and before any study training.

**Ledger:** `ledger.jsonl`. Device time is counted once per GPU, as the union of the busy intervals of that GPU's jobs, from each job's `/usr/bin/time` log. Jobs stopped by the memory guard are counted from log creation to last write (see `tools/semantic_field/scaling/resources.py`).

## Used so far

| category | GPU-h | contents |
|---|---:|---|
| readiness | ≈ 0.26 | loader equivalence ×2, single / concurrent / expanded-pool throughput benchmarks |
| preparation | ≈ 6.0 | tokens 8,350 frames, clips + inputs 7,708 anchors, evidence / provenance / delayed targets / B3 caches, killed and restarted attempts |
| **total** | **≈ 6.2 of 48** | run `python -m tools.semantic_field.scaling.resources summary` for the exact current value |

**Storage (new study caches):**

| tier | size | contents |
|---|---:|---|
| hot, NVMe | ≈ 319 GB (SSD2 ≈ 240, SSD1 ≈ 83) | inputs, evidence, provenance, delayed targets, B3 caches |
| cold, MINH1 | ≈ 76 GB | tokens, clips |
| **total** | **≈ 395 GB of 750** | |

**Relocated existing data (not new):** 110.8 GB, moved to MINH1 with checksum verification.

**Free space now:** SSD2 51 GiB, SSD1 63 GiB, MINH1 ≈ 3.9 TB.

## Measured preparation cost

Per-GPU wall times from the logs:
- **Tokens:** 1.69–1.88 s per frame.
- **Clips:** 0.16–0.17 s per anchor.
- **Inputs:** 0.16–0.22 s per anchor.
- **Evidence:** 0.21–0.28 s per anchor.
- **Provenance:** 0.18 s per anchor.
- **Delayed targets:** 0.26 s per anchor.
- **B3 cache:** 0.06 s per anchor.

**Measured hot footprint per anchor:**

| drive | MB / anchor | evidence |
|---|---:|---:|
| 0005 | 37.9 | 35.6 |
| 0000 | 46.9 | 44.4 |
| 0002 | 40.2 | 37.9 |
| 0004 | 40.9 | 38.6 |

## Measured training throughput (disk-backed, cold page cache)

| pool | B4 s / update (1 run) | B3 s / update (1 run) | B4 packed, 3 runs / GPU | B3 packed, 4 runs / GPU |
|---|---:|---:|---:|---:|
| current (1,358) | 0.387 | 0.424 | 3.71 updates/s per GPU | 8.36 updates/s per GPU |
| expanded (9,066) | 0.437 | 0.575 | ≈ 3.29 (scaled by the single-run ratio) | ≈ 6.16 (scaled) |

**Peak memory:** GPU 12.0 GB (B4) / 4.7 GB (B3) per run; host RSS ≈ 2.5 GB per run.

## Remaining study (estimate)

| part | calculation | GPU-h |
|---|---|---:|
| reliability gate | 7 drives + 0006, label-free statistics | ≈ 0.3 |
| B4 current, 3 × 40,056 | GPU 1, 3 runs packed: 120,168 / 3.71 / 3600 | ≈ 9.0 |
| B4 expanded, 3 × 40,056 | GPU 0, 3 runs packed: 120,168 / 3.29 / 3600 | ≈ 10.1 |
| B3 current + expanded, 6 × 20,028 | GPU 2, 6 runs packed; at worst the two 4-run rates | ≤ 4.7 |
| source evaluation | 2 pools × 590 anchors × ≈ 1.9 s (shardable) | ≈ 0.6 |
| **remaining** | | **≈ 24.7** |

**Uncertainty.**
- Disk and CPU contention with all 12 runs active is not measured. Allowing +20 % on training gives ≈ 29.5 GPU-h remaining.
- **Whole study ≈ 31–36 GPU-h, within the 48 GPU-h cap.**
- Wall time: ≈ 10–12 h for the B4 runs.

**Guardrails.**
- Every run writes a resume checkpoint every 1,000 updates, so an interruption costs at most ~15 min of one GPU.
- The page-cache trimmer runs throughout training.
