#!/usr/bin/env python
"""Write and SHA-256-pin ``configs/gate7b/streaming_replay_precommit.yaml``.

Generated from the modules the experiment imports, so a changed constant changes the pin.

    python tools/gate7b/precommit.py
"""
from __future__ import annotations

import hashlib, json, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import yaml                                                              # noqa: E402
from gates.scale_gate.config import REPO_ROOT                                  # noqa: E402
from gates.gate6 import grids as G6G                                           # noqa: E402
from gates.gate7b import config as C, depth as D7, evidence as EV, rays as RY, \
    replay as RP, scale as SC, streams as ST, voxmap as VM               # noqa: E402

OUT = "configs/gate7b/streaming_replay_precommit.yaml"


def load_json(rel):
    p = os.path.join(REPO_ROOT, rel)
    return json.load(open(p)) if os.path.exists(p) else {"missing": rel}


def build() -> dict:
    st0 = load_json("artifacts/gate7b/stage0_audit.json")
    cfg = {}
    cfg["gate"] = {
        "name": "Gate 7B - native causal streaming replay, metric-gauge stability, "
                "complementary depth evidence",
        "nothing_is_trained": list(C.NOT_DONE),
        "repo_commit": st0.get("git", {}).get("commit"),
        "seeds": dict(C.SEEDS),
        "causality": list(C.CAUSALITY),
        "stop_conditions": list(C.STOP_CONDITIONS),
        "resource_limits": dict(C.RESOURCE_LIMITS),
        "semantic_caveat": C.SEMANTIC_CAVEAT,
    }
    cfg["streaming_interface"] = {
        "input_at_t": "one current RGB image",
        "past_carried_by": ["LingBot native anchor context (the 5 scale frames, pinned "
                            "in the KV cache)",
                            f"the pose-reference window (KV sliding window of "
                            f"{RP.KV_CACHE_SLIDING_WINDOW} blocks)",
                            "trajectory / camera special tokens, cross-frame",
                            "the persistent metric occupancy-semantic map"],
        "entry_point": "lingbot_map.models.gct_stream.GCTStream.inference_streaming",
        "reset_policy": "clean_kv_cache() exactly once per segment, at a genuine boundary",
        "rope": {"max_frame_num": RP.MAX_FRAME_NUM, "margin": RP.ROPE_MARGIN,
                 "keyframe_rule": ("the smallest keyframe_interval that fits the segment "
                                   "in the frozen RoPE table; a non-keyframe consumes no "
                                   "global frame index. Depends only on stream length "
                                   "and the frozen table size, never on a score.")},
        "inference": {"resolution": RP.INFERENCE_RESOLUTION,
                      "patch_size": RP.PATCH_SIZE,
                      "num_scale_frames": RP.NUM_SCALE_FRAMES,
                      "autocast": RP.AUTOCAST_DTYPE,
                      "checkpoint": RP.CHECKPOINT},
    }
    cfg["datasets"] = {}
    for ds in C.DATASETS:
        G = G6G.EVAL_GRID[ds]
        cfg["datasets"][ds] = {
            "boundary_rule": ST.BOUNDARY_RULE[ds],
            "evaluation_grid": G.name, "dims": list(G.dims),
            "voxel_size_m": G.voxel_size, "origin_m": list(G.origin), "frame": G.frame,
            "evaluated_timestamps": ("every official Gate-6 clip anchor, addressed as the "
                                     "causal prefix of the stream ending at that frame"),
        }
    cfg["horizons"] = {"causal": list(C.HORIZONS), "primary": C.PRIMARY_HORIZON,
                       "definition": "frames [t-H+1, t] of the deduplicated stream, "
                                     "intersected with a purely geometric reach "
                                     "prefilter that can remove no contributing frame"}
    cfg["scale_policies"] = {
        "primary": C.PRIMARY_SCALE,
        "estimator": ("per-frame median of log(D_moge) - log(D_lingbot) over jointly "
                      f"valid pixels after a {SC.MAD_CLIP} x MAD clip; at least "
                      f"{SC.MIN_VALID_PIXELS} valid pixels or the frame abstains"),
        "G-A": {"rule": f"median over the stream's first {SC.N_ANCHOR_FRAMES} anchor "
                        "frames, then frozen for the whole stream", "role": "primary"},
        "G-B": {"rule": "expanding causal median over every candidate with index <= t",
                "role": "comparison"},
        "G-C": {"rule": "each frame gauged by itself",
                "role": "ABLATION ONLY; can never be selected as primary"},
        "application": ("the same scalar multiplies LingBot depth and LingBot pose "
                        "translation; rotations are never scaled"),
        "rematerialisation": ("the metric crop at each evaluation timestamp is rebuilt "
                              "from canonical observations with the scale in force at "
                              "that timestamp, so a gauge change transforms old and new "
                              "observations together and cannot duplicate surfaces"),
        "moge_gauge": "G51-B: full lattice, calibrated horizontal FOV, frozen MoGe-2",
    }
    cfg["depth_evidence_variants"] = {
        "S0": "the frozen five-frame B-D prediction of Gate 6, read back unchanged",
        "S1": {"gate": f"conf >= {D7.CONF_THRESHOLD}, "
                       f"{D7.MIN_DEPTH_M} m < s*d < {D7.MAX_DEPTH_M} m"},
        "S2": {"confidence_sweep": list(EV.S2_CONF_SWEEP),
               "primary": EV.S2_PRIMARY_CONF,
               "range": "unchanged; the evaluation volume is 51.2 m across, so relaxing "
                        "the 60 m cap cannot add a scorable voxel and is not attempted",
               "single_global_threshold": True},
        "S3": {"rule": "LingBot where its frozen gate accepts; dense MoGe on the rays it "
                       "rejects, if finite and inside the physical range",
               "weight": VM.W_MOGE, "tracked_separately": True},
        "S4": {"rule": "S3 with a continuous analytic reliability weight and a "
                       "free-space veto",
               "tau_log": EV.S4_TAU_LOG, "free_veto_logodds": EV.S4_FREE_VETO_L,
               "min_weight": EV.S4_MIN_WEIGHT,
               "provisional": f"MoGe-only occupancy is provisional until "
                              f"{VM.MOGE_CONFIRMATIONS} independent frames support it"},
    }
    cfg["occupancy_update"] = {
        "representation": "occupied / free / unknown / provisional, per voxel",
        "rule": "fixed log-odds accumulation, published OctoMap defaults, never tuned",
        "l_occ": VM.L_OCC, "l_free": VM.L_FREE, "clamp": VM.L_CLAMP,
        "occupied_at": f"log-odds > {VM.L_OCCUPIED_AT}",
        "band_half_m": RY.BAND_HALF_M,
        "carve": {"near_m": RY.CARVE_NEAR_M, "pixel_stride": RY.CARVE_STRIDE,
                  "note": "free space is carved from the near plane to band_half_m "
                          "before the surface and NEVER beyond it; the region behind a "
                          "predicted surface stays unknown"},
        "source_weights": {"lingbot": VM.W_LINGBOT, "moge": VM.W_MOGE},
    }
    cfg["semantic_update"] = {
        "teacher": "the cached Gate-6 Trident probability vectors; Trident is NOT re-run",
        "rule": "weighted sum of complete probability vectors on the surface voxel, "
                "argmax at readout",
        "weight": ("geometry reliability and observation quality -- LingBot confidence "
                   "where LingBot supplied the ray, the MoGe reliability weight where "
                   "MoGe did. NEVER the teacher's own maximum probability, which Gate 7A "
                   "showed does not track propagation accuracy"),
        "separation": "semantic accumulators are separate from occupancy accumulators, "
                      "so a geometry update never erases semantic history",
        "caveat": C.SEMANTIC_CAVEAT,
    }
    cfg["ray_consistency"] = {
        "depth_convention": ("LingBot and MoGe both give optical-axis z on the processed "
                             "lattice; a voxel's distance along a ray is Euclidean and is "
                             "converted explicitly"),
        "lingbot_units": "canonical; metric only after multiplication by s(t)",
        "moge_units": "already metric",
        "disagreement": "|log(D_moge) - log(s * D_lingbot)|",
        "tau_log": EV.S4_TAU_LOG,
        "free_veto_logodds": EV.S4_FREE_VETO_L,
    }
    cfg["aggregation"] = {
        "primary": "pooled voxel counts over the whole benchmark (micro)",
        "metrics": "gate6.metrics.clip_counts / summarize, unchanged, so every number is "
                   "directly comparable with Gates 6 and 7A",
        "mean_per_clip": "reported only where labelled",
    }
    cfg["bootstrap"] = dict(C.BOOTSTRAP)
    cfg["paired_comparisons"] = [{"a": b, "b": a, "what": w}
                                 for a, b, w in C.PAIRED_COMPARISONS]
    cfg["temporal_recoverability"] = {
        "horizons": list(C.RECOVERY_HORIZONS),
        "status": "DIAGNOSTIC ONLY; future frames never enter a causal map",
        "isolation": "computed in a separate volume, written to a separate artifact",
    }
    cfg["range_bands_m"] = [list(b) for b in C.RANGE_BANDS]
    return cfg


def main() -> int:
    cfg = build()
    path = os.path.join(REPO_ROOT, OUT)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = yaml.safe_dump(cfg, sort_keys=False, width=100, allow_unicode=True)
    with open(path, "w") as fh:
        fh.write("# Gate 7B precommit. Written and SHA-256-pinned BEFORE the full "
                 "evaluation.\n# Generated by tools/gate7b/precommit.py from the modules "
                 "the experiment imports.\n")
        fh.write(text)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    with open(os.path.join(REPO_ROOT, "artifacts", "gate7b", "precommit_pin.json"),
              "w") as fh:
        json.dump({"path": OUT, "sha256": digest, "bytes": os.path.getsize(path)},
                  fh, indent=2)
    print(f"{OUT}\n  sha256 {digest}\n  bytes  {os.path.getsize(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
