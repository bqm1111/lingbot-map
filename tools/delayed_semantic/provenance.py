"""Per-anchor evidence provenance: replay the five history frames into ``EvidenceMemory`` one at a time.

For every anchor this recomputes the Gate 15 evidence set from the cached local-reset clip and hard
map (frame-by-frame insertion), asserts it equals the Gate 15 evidence voxels exactly, and stores the
frame bits and slot timestamps. On every ``feature_check_every``-th anchor it also inserts the Trident
features and compares the resulting evidence feature with the Gate 15 cache (stored fp16).
"""
from __future__ import annotations

import os
import time
from typing import Dict

import numpy as np
import torch

from core.datasets.audit import FileAudit
from tools.baseline_reconstruction.current.geometry import LocalClip
from tools.delayed_semantic import config as C
from tools.delayed_semantic.memory import EvidenceMemory
from tools.delayed_semantic.stream import DriveStream
from tools.semantic_completion.evidence import build as G15B, trident as TR
from tools.semantic_completion.evidence.lift import load_evidence, ray_endpoints
from tools.semantic_completion.frozen_geometry import load_hard_map


def replay_history(stream: DriveStream, t: int, device, with_features: bool, paths: C.Paths = C.Paths()) -> Dict:
    """Evidence of H_t built incrementally from the history accessor only."""
    hist = stream.history(t)
    clip = LocalClip.load(paths.g15.clip_path(stream.drive, t))
    assert list(clip.keys) == [f.key for f in hist.frames()], "clip frames are not H_t"
    S, _ = load_hard_map(stream.drive, t, device, paths.g15)
    mem = EvidenceMemory(t, device)
    lattice_hw = clip.depth_canonical.shape[1:]
    for slot, ((keys, pix), f) in enumerate(zip(ray_endpoints(clip, device), hist.frames())):
        feats = None
        if with_features:
            feats = TR.sample_at_lattice(TR.load_token_map(f.token_path, device), lattice_hw).reshape(-1, C.TEACHER_DIM)[pix]
        mem.observe(slot, f.index, f.time_s, keys, feats)
    return {"evidence": mem.snapshot(S, G15B.camera_to_grid()), "slot_time_s": np.asarray([f.time_s for f in hist.frames()], np.float64)}


def build_drive(drive: str, device: str, shard: int = 0, shards: int = 1, feature_check_every: int = 25, paths: C.Paths = C.Paths()) -> Dict:
    dev = torch.device(device); torch.cuda.set_device(dev)
    ids = G15B.anchor_ids(drive, paths.g15)[shard::shards]
    stream = DriveStream(drive)
    rows, checks, t0 = [], [], time.time()
    with FileAudit(patterns=C.PREDICT_FORBIDDEN, strict=True) as audit:
        for k, t in enumerate(ids):
            out = paths.provenance_path(drive, t)
            check = (G15B.anchor_ids(drive, paths.g15).index(t) % feature_check_every) == 0
            if os.path.exists(out) and not check:
                continue
            r = replay_history(stream, t, dev, with_features=check, paths=paths)
            ev, ref = r["evidence"], load_evidence(paths.g15.evidence_path(drive, t), dev)
            assert torch.equal(ev.flat_index, ref.flat_index), f"{drive}:{t} evidence voxels differ from Gate 15"
            assert torch.equal(ev.count, ref.count), f"{drive}:{t} evidence counts differ from Gate 15"
            if check:
                d = (ev.feature - ref.feature).abs().amax(1)
                text, _, _ = TR.load_text_embeddings(paths.g15.text_embeddings)
                checks.append({"t": t, "n": len(d), "max_abs": float(d.max()) if len(d) else 0.0,
                               "class_agree": float((TR.classify(ev.feature, text) == TR.classify(ref.feature, text)).float().mean()) if len(d) else 1.0})
            os.makedirs(os.path.dirname(out), exist_ok=True); tmp = out + ".tmp.npz"
            np.savez(tmp, flat_index=ev.flat_index.cpu().numpy().astype(np.int32), frame_bits=ev.frame_bits.cpu().numpy(),
                     slot_time_s=r["slot_time_s"], t=np.int64(t), native=np.int64(stream.frames[t].native))
            os.replace(tmp, out)
            cur = ev.current()
            rows.append({"t": t, "n": int(len(ev.flat_index)), "n_current": int(cur.sum()), "n_past_only": int((~cur).sum())})
            if (k + 1) % 100 == 0:
                print(f"  [prov {drive[17:21]} s{shard}] {k+1}/{len(ids)} {(time.time()-t0)/(k+1):.2f}s each", flush=True)
    return {"drive": drive, "shard": shard, "n": len(ids), "rows": rows, "feature_checks": checks, "seconds": time.time() - t0,
            "firewall": {"opened": len(audit.opened), "violations": audit.violations}}


__all__ = ["replay_history", "build_drive"]
