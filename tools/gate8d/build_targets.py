#!/usr/bin/env python
"""Gate 8D Phase 2: build the dense privileged supervision for one KITTI-360 drive.

Runs inside the file-access audit, so an attempt to open a target-benchmark path raises
rather than being noticed later. Refuses to run unless the protocol is frozen.

    python tools/gate8d/build_targets.py --drive 2013_05_28_drive_0003_sync --device cuda:1
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, default_device                               # noqa: E402
from gates.scale_gate.config import write_json                                         # noqa: E402
from gates.gate8c0 import transforms as TF                                             # noqa: E402
from gates.gate8d import densetarget as DT, futureframes as FF, protocol as P          # noqa: E402
from gates.gate8d import sources as SRC                                                # noqa: E402
from sscbench_kitti360.audit import FileAudit                                    # noqa: E402
from tools.gate8d.freeze_protocol import assert_unchanged                        # noqa: E402


def load_teacher(path):
    """Read one cached teacher frame, decimating the sky map to the protocol's ray stride.

    The sky teacher is cached at ``SKY_CACHE_STRIDE`` so the ray stride can be chosen
    without re-running the frozen model; decimation here is a plain subsample, never an
    interpolation, so no probability is invented.
    """
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as z:
        shape = tuple(int(x) for x in z["shape"])
        out = {"shape": shape, "stride": int(z["stride"])}
        if "depth_z" in z:
            out["depth_z"] = z["depth_z"].astype(np.float32)
            out["mask"] = np.unpackbits(z["mask"])[:int(np.prod(shape))].astype(bool)
        else:
            s_ = z["sky"].astype(np.float32)
            k = max(1, P.SKY_PIXEL_STRIDE // int(z["stride"]))
            s_ = s_[::k, ::k]
            out["sky"] = s_
            out["shape"] = s_.shape
            out["stride"] = int(z["stride"]) * k
        return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drive", required=True)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    assert_unchanged()
    margin = json.load(open(os.path.join(ART, "moge_margin.json")))
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    geo = TF.DriveGeometry(a.drive, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
    ancs = SRC.anchors(a.drive, REPO_ROOT)[a.shard::a.shards][:a.limit]
    rows, t0, n_skip = [], time.time(), 0
    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        for k, anc in enumerate(ancs):
            dst = SRC.target_path(a.drive, anc.stream_index)
            if os.path.exists(dst):
                n_skip += 1; continue
            dense = FF.native_window(anc.native_frame, anc.future_natives,
                                     geo.cam0_to_world)
            img = [f for f in dense
                   if (f - int(anc.native_frame)) % P.IMAGE_EVIDENCE_NATIVE_STRIDE == 0]
            moge = {f: load_teacher(SRC.moge_future_path(a.drive, f)) for f in img}
            sky = {f: load_teacher(SRC.sky_future_path(a.drive, f)) for f in img}
            t = DT.build(geo, anc.native_frame, dense, dev, image_natives=img,
                         moge={k_: v for k_, v in moge.items() if v},
                         sky={k_: v for k_, v in sky.items() if v}, margin=margin)
            DT.save(dst, t, TF.GRID.dims, anc.native_frame, anc.stream_index,
                    anc.future_stream_indices, anc.future_natives)
            cov = FF.coverage(anc.native_frame, anc.future_natives, geo.cam0_to_world)
            rows.append(dict(t.stats, stream_index=int(anc.stream_index), **cov))
            if len(rows) % 25 == 0:
                print(f"  [{a.drive[17:21]}] {len(rows)}/{len(ancs)} "
                      f"({(time.time()-t0)/len(rows):.2f}s each)", flush=True)
    agg = {}
    if rows:
        for k_ in rows[0]:
            v = [r[k_] for r in rows if isinstance(r[k_], (int, float))]
            if v:
                agg[k_] = float(np.mean(v))
    out = {"drive": a.drive, "n_built": len(rows), "n_skipped": n_skip,
           "shard": a.shard, "shards": a.shards, "seconds": time.time() - t0,
           "protocol_sha256": P.source_digest(), "mean": agg, "per_anchor": rows[:50],
           "firewall": audit.summary()}
    write_json(os.path.join(ART, f"targets_{a.drive[17:21]}_s{a.shard}.json"), out)
    print(f"[{a.drive[17:21]}] {len(rows)} built, {n_skip} present, "
          f"{(time.time()-t0)/60:.1f} min, firewall {len(out['firewall']['violations'])}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
