#!/usr/bin/env python
"""Case-C diagnosis: how repeatable is the future teacher target at voxel level?

On voxels where the causal map already carries a teacher vector *and* the future window
lifts one (both LiDAR-occupied), the two are independent projections of the same surface
from different frames. Their agreement is the ceiling any causal predictor of the future
vector can reach at that voxel. Reported overall, against a shuffled pairing, by the
amount of evidence on each side, and by the future teacher's class.

    python tools/future_semantic_completion/diagnose_targets.py --device cuda:0
"""
from __future__ import annotations
import argparse, glob, os, sys, time
import numpy as np, torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.future_semantic_completion import study as ST                          # noqa: E402
from tools.future_semantic_completion import head as H                            # noqa: E402
from core.datasets import kitti360_splits as SRC, union_vocab as V8               # noqa: E402
from core.datasets.audit import FileAudit                                         # noqa: E402
from core.supervision import targets as TG                                        # noqa: E402

W_EDGES = np.asarray([0, 1, 2, 4, 8, 16, 1e9]); W_NAMES = ["(0,1]", "(1,2]", "(2,4]", "(4,8]", "(8,16]", ">16"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--drive", default=SRC.VAL_DRIVE)
    ap.add_argument("--stride", type=int, default=2)
    a = ap.parse_args()
    ST.require_data_root()
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    files = sorted(glob.glob(os.path.dirname(SRC.sample_path(a.drive, 0)) + "/*.npz"))[::a.stride]
    by_w = np.zeros((len(W_NAMES), 3)); by_fw = np.zeros((len(W_NAMES), 3))
    by_cls = np.zeros((V8.U, 3)); tot = np.zeros(3); shuf = np.zeros(3)
    conf = np.zeros((V8.U, V8.U), np.int64)
    t0 = time.time()
    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        for f in files:
            with np.load(f) as z:
                d = TG.unpack_sample(z, dev)
                n = d["gt_occ"].numel()
                fw = torch.zeros(n, device=dev)
                fr = torch.from_numpy(z["fut_rows"].astype(np.int64)).to(dev)
                fw[fr] = torch.from_numpy(z["fut_w"].astype(np.float32)).to(dev)
            x = d["input"]; semw = x[6].reshape(-1) * 20.0
            both = d["observed"].reshape(-1) & (semw > 0) & d["fut_valid"].reshape(-1) & \
                ((d["gt_occ"] > 0) & d["gt_valid"]).reshape(-1)
            nb = int(both.sum())
            if nb < 2:
                continue
            cur = x[7:7 + V8.U].reshape(V8.U, -1).T[both]; fut = d["fut_p"].reshape(-1, V8.U)[both]
            ca, fa = cur.argmax(1), fut.argmax(1)
            ok = (ca == fa).float(); cs = H.cosine(cur, fut)
            tot += (nb, float(ok.sum()), float(cs.sum()))
            perm = torch.randperm(nb, generator=torch.Generator().manual_seed(nb)).to(dev)
            shuf += (nb, float((ca == fa[perm]).sum()), float(H.cosine(cur, fut[perm]).sum()))
            conf += np.bincount((fa * V8.U + ca).cpu().numpy(), minlength=V8.U * V8.U).reshape(V8.U, V8.U)
            for arr, w in ((by_w, semw[both]), (by_fw, fw[both])):
                b = np.clip(np.searchsorted(W_EDGES, w.cpu().numpy(), side="left") - 1, 0, len(W_NAMES) - 1)
                for k in range(len(W_NAMES)):
                    s = b == k
                    arr[k] += (s.sum(), float(ok[torch.from_numpy(s).to(dev)].sum()), float(cs[torch.from_numpy(s).to(dev)].sum()))
            fa_np = fa.cpu().numpy(); ok_np = ok.cpu().numpy(); cs_np = cs.cpu().numpy()
            for u in np.unique(fa_np):
                s = fa_np == u
                by_cls[u] += (s.sum(), ok_np[s].sum(), cs_np[s].sum())
    def tab(arr, names):
        return {nm: {"n": int(arr[k, 0]), "top1_agreement": float(arr[k, 1] / max(arr[k, 0], 1)),
                     "cosine": float(arr[k, 2] / max(arr[k, 0], 1))} for k, nm in enumerate(names) if arr[k, 0] > 0}
    out = {"drive": a.drive, "n_anchors": len(files), "stride": a.stride,
           "definition": "voxels LiDAR-occupied, map-observed with teacher evidence (sem_w > 0) and carrying a "
                         "future-window teacher vector; agreement between the causal-map vector and the future vector",
           "overall": {"n": int(tot[0]), "top1_agreement": float(tot[1] / max(tot[0], 1)), "cosine": float(tot[2] / max(tot[0], 1))},
           "shuffled_pairing": {"top1_agreement": float(shuf[1] / max(shuf[0], 1)), "cosine": float(shuf[2] / max(shuf[0], 1))},
           "by_current_map_weight": tab(by_w, W_NAMES), "by_future_weight": tab(by_fw, W_NAMES),
           "by_future_class": {V8.UNION[u]: {"n": int(by_cls[u, 0]), "top1_agreement": float(by_cls[u, 1] / by_cls[u, 0]),
                                             "cosine": float(by_cls[u, 2] / by_cls[u, 0])}
                               for u in range(V8.U) if by_cls[u, 0] > 0},
           "confusion_future_rows_current_cols": conf.tolist(), "union": V8.UNION,
           "seconds": time.time() - t0, "firewall": audit.summary()}
    ST.write_json(os.path.join(ST.ART, "target_diagnosis.json"), out)
    print(f"overall n={out['overall']['n']} top1 {out['overall']['top1_agreement']:.3f} cos {out['overall']['cosine']:.3f} "
          f"(shuffled {out['shuffled_pairing']['top1_agreement']:.3f} / {out['shuffled_pairing']['cosine']:.3f})")
    for k, v in out["by_current_map_weight"].items():
        print(f"  map weight {k:7s} n {v['n']:8d} top1 {v['top1_agreement']:.3f} cos {v['cosine']:.3f}")
    for k, v in out["by_future_weight"].items():
        print(f"  fut weight {k:7s} n {v['n']:8d} top1 {v['top1_agreement']:.3f} cos {v['cosine']:.3f}")
    for k, v in sorted(out["by_future_class"].items(), key=lambda x: -x[1]["n"])[:10]:
        print(f"  class {k:20s} n {v['n']:8d} top1 {v['top1_agreement']:.3f} cos {v['cosine']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
