# How to rerun Gate 7B yourself

Everything below is a plain script; nothing is trained. Read the modules in the order of
§2 — each file's docstring says what it does and what it must never do.

## 0. Environment (every command)

```bash
cd /home/minh/workspace/lingbot-map_fork
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST="12.0a"      # sm_120 needs the suffixed arch
export HF_HUB_OFFLINE=1                        # MoGe weights are already cached
PY=/home/minh/anaconda3/envs/cu128/bin/python
```

## 1. The pipeline, stage by stage

| # | command | what it produces | cost (measured) |
|---|---|---|---|
| 0 | `$PY tools/gate7b/stage0.py` | `artifacts/gate7b/stage0_audit.json` — hashes of everything Gate 7B must not touch | seconds |
| 1 | `$PY tools/gate7b/precommit.py` | `configs/gate7b/streaming_replay_precommit.yaml` + pin — every choice, before any result | seconds |
| 2 | `$PY tools/gate7b/stream_lingbot.py --dataset {semantickitti,occ3d,kitti360} --device cuda:N` | `/media/SSD1/MINH_DATASETS/lingbot_gate7b/stream/<ds>/<segment>.npz` — depth, confidence, poses, K for the whole causal stream | 0.8 / 6.4 / 1.6 min |
| 3 | `$PY tools/gate7b/cache_moge_b.py --dataset {semantickitti,occ3d}` | `/media/SSD1/MINH_DATASETS/lingbot_gate7b/moge_b/<ds>/<frame>.npz` — calibrated-FOV MoGe depth per frame (KITTI-360 reuses the Gate-5.2 `moge/B` cache) | 0.8 / 7 min |
| 4 | `$PY tools/gate7b/scale_candidates.py --dataset <ds>` | `/media/SSD1/MINH_DATASETS/lingbot_gate7b/scale/<ds>/<segment>.npz` — one log-scale candidate per frame | < 1 min each |
| 5 | `$PY tools/gate7b/run_stream_eval.py --dataset <ds> --variant S1 --scale G-A --horizon all --device cuda:0` | `artifacts/gate7b/eval_<ds>_<tag>.json` + `counts_<ds>_<tag>.npz` — one configuration | SK 1 min · Occ3D 3.5–5 min · K360 ~19 min |
| 5' | `tools/gate7b/run_matrix.sh` | all 12 configurations × 3 benchmarks on 4 GPUs | ≈ 1.5 h wall |
| 6 | `$PY tools/gate7b/thickness.py --dataset <ds> --stride 10` | `artifacts/gate7b/thickness_<ds>.json` — map thickness / duplicate surfaces per gauge | minutes |
| 7 | `$PY tools/gate7b/recoverability.py --dataset <ds> --stride 4` | `artifacts/gate7b/recoverability_<ds>.json` — **diagnostic only**, looks at future frames | minutes |
| 7' | `tools/gate7b/run_diagnostics.sh` | 6 + 7 for all three, one GPU each | ≈ 30 min |
| 8 | `$PY tools/gate7b/aggregate.py` | `artifacts/gate7b/summary.json` — pooled numbers, paired bootstrap, decision rules | 1–2 min |
| 9 | `$PY tools/gate7b/figures.py` | `artifacts/gate7b/fig_*.png` | seconds |
| 10 | `$PY tools/gate7b/report_tables.py --write` | `reports/gate7b/streaming_metric_semantic_replay.md` | seconds |
| 11 | `$PY -m pytest tests/gate6 tests/gate7a tests/gate7b -q` | 131 prior + 56 new tests | ~3 min |

Stages 2–4 only need to be run once; 5–7 read their caches. To try a new configuration
you only ever re-run stage 5 (one command, one config), then 8–10.

Useful flags on `run_stream_eval.py`: `--limit-anchors N` (pilot), `--skip-anchors N`
(start mid-stream, where windows are long), `--horizon {1,5,20,50,all}`,
`--variant {S1,S2,S3,S4}`, `--scale {G-A,G-B,G-C}`, `--conf X` (S2 only), `--tag NAME`.

## 2. The skeleton — read in this order

```
gate7b/
  config.py     every predeclared constant; the precommit is generated from it
  streams.py    chronological, deduplicated frames; a boundary only where one is genuine
  replay.py     native direct-mode LingBot replay; the RoPE keyframe capacity rule
  depth.py      depth conventions (z, not ray length); frozen gate; per-frame MoGe index
  scale.py      ScaleState — G-A fixed anchor / G-B running median / G-C per-frame
  voxmap.py     EvidenceVolume — occupied / free / unknown / provisional, log-odds
  rays.py       cast one frame: free before the surface, occupied band, NOTHING behind
  evidence.py   S1–S4 acceptance rules and the geometry-based semantic weight
  fuse.py       causal window (<= t only) + geometric reach filter; rematerialisation
```

The one loop that does the work is `tools/gate7b/run_stream_eval.py`: for each official
timestamp `t`, pick the causal frames, ask `ScaleState` for `s(t)`, fuse them with
`fuse.fuse_window` into a fresh `EvidenceVolume` in the benchmark grid, read out occupancy
and the argmax semantic channel, and score with **Gate 6's own** `gate6.metrics.clip_counts`.

Things it reuses rather than reimplements: `gate6.metrics` (all metrics),
`gate6.targets` (official masks), `gate6.grids` (grid specs and the Occ3D 0.2 → 0.4 m
reduction), `gate7a.frustum` (in-frustum test), `gate7a.stats` (vectorised bootstrap),
`tools/gate6/analyze.unit_ids` (bootstrap units), `decompose_residual.scaled_relative_pose`
(the frozen pose scaling).

## 3. Where the time goes, and the cheap ways to cut it

* **Per-timestamp rebuild** (`fuse.fuse_window`): KITTI-360 fuses a median of 84 frames
  per timestamp, ~4 ms per frame. An incremental map would fuse each frame once.
* **Semantic tensors**: `[C, H, W]` float16 per frame from the Trident cache, kept in an
  LRU so consecutive timestamps share them.
* **Free-space carving** is already decimated 4 × 4 in pixels; occupied evidence is not.
* To iterate quickly: `--limit-anchors 20 --skip-anchors 800` on KITTI-360 gives
  representative long windows in ~15 s.

## 4. What must never change between runs

`configs/gate7b/streaming_replay_precommit.yaml` is SHA-256-pinned in
`artifacts/gate7b/precommit_pin.json`; `run_stream_eval.py` refuses to start if the file
no longer matches. Change a constant in `gate7b/config.py` → re-run stage 1 → the pin
changes, and that is the audit trail.
