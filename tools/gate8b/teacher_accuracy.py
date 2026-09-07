#!/usr/bin/env python
"""How good is the privileged semantic *target* itself? Future-Trident accuracy against the
benchmark's semantic ground truth on future-visible occupied voxels.

The completion is distilled from the frozen Trident teacher fused over frames t+1..t+20,
never from a human label. This measures that teacher against the ground truth on the
evaluation target -- an evaluation-only use of semantic labels -- so the report can say
how much of the semantic ceiling is the teacher's rather than the student's.

    python tools/gate8b/teacher_accuracy.py --dataset semantickitti --device cuda:1
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, default_device                               # noqa: E402
from gates.scale_gate.config import write_json                                         # noqa: E402
from gates.gate6 import grids as G6G, targets as G6T, vocab                            # noqa: E402
from gates.gate8 import sources as S, vocab as V8                                      # noqa: E402
from gates.gate8.feed import CachedFeed                                                # noqa: E402
from gates.gate8.mapper import IncrementalMapper                                       # noqa: E402
from gates.gate8.targets import future_volume, MIN_FUTURE                              # noqa: E402
from tools.gate8.evaluate import _grid_to_world                                  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=["semantickitti", "occ3d", "kitti360"])
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--anchor-stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    ds = a.dataset; v = vocab.load(ds); C = len(v)
    MAP, EVAL = G6G.PREDICTION_GRID[ds], G6G.EVAL_GRID[ds]
    labels = np.asarray(v.labels, np.int32)
    into = V8.into_matrix(ds); out_ch = np.asarray(V8.out_channel(ds))
    per_class = np.zeros((C, 2), np.int64)      # [gt class] -> (n, correct)
    n_all = n_cor = n_obs = n_obs_cor = 0
    n_anchor, t0 = 0, time.time()
    for seg in S.segments(ds, REPO_ROOT):
        feed = CachedFeed(seg, dev)
        m = IncrementalMapper(dev, sem_into=into, n_teacher=C)
        anchors = set(seg.anchors[::a.anchor_stride])
        for i in range(len(seg)):
            m.step(feed.frame(i))
            if i not in anchors or not m.scale_state.frozen or len(seg) - 1 - i < MIN_FUTURE:
                continue
            f = seg.frames[i]
            target, keep = G6T.semantic_target(ds, f.gt_ref, REPO_ROOT)
            if target is None:
                continue
            Tgw = _grid_to_world(m, f, feed)
            fut, _ = future_volume(feed, i, m.scale_state.scale, into, dev)
            qf = fut.query(MAP, Tgw)
            q = m.query(MAP, Tgw)
            semw = qf["sem_w"]; has = semw > 0
            obs_np = q["observed"].cpu().numpy()
            # teacher label per evaluation voxel, through the frozen reduction: on Occ3D the
            # union probability vectors of the evidence-bearing sub-voxels are averaged
            # (gate6.grids.reduce_occ3d_probs), exactly as every evaluated prediction is
            fl = has.nonzero(as_tuple=True)[0].cpu().numpy().astype(np.int64)
            pp = qf["sem"][has] / semw[has].unsqueeze(1)
            if G6G.NEEDS_REDUCTION[ds]:
                r = G6G.RATIO; d = MAP.dims
                obs_np = obs_np.reshape(d[0] // r, r, d[1] // r, r, d[2] // r, r) \
                    .transpose(0, 2, 4, 1, 3, 5).reshape(-1, r ** 3).any(axis=1)
                if len(fl):
                    fl, pp = G6G.reduce_occ3d_probs(fl, pp)
            n_eval = int(np.prod(EVAL.dims))
            pred_np = np.full(n_eval, int(v.empty_label), np.int32)
            has_np = np.zeros(n_eval, bool)
            if len(fl):
                ch_b = out_ch[pp.argmax(1).cpu().numpy()]
                pred_np[fl] = labels[ch_b]; has_np[fl] = True
            t = target.reshape(-1); k = keep.reshape(-1)
            gt_occ = (t != v.empty_label) & k
            sel = gt_occ & has_np
            cor = sel & (pred_np == t)
            n_all += int(sel.sum()); n_cor += int(cor.sum())
            so = sel & obs_np
            n_obs += int(so.sum()); n_obs_cor += int((cor & obs_np).sum())
            for j, l in enumerate(labels):
                mm = sel & (t == l)
                per_class[j, 0] += int(mm.sum()); per_class[j, 1] += int((mm & cor).sum())
            n_anchor += 1
            if a.limit and n_anchor >= a.limit:
                break
        if a.limit and n_anchor >= a.limit:
            break
    out = {"dataset": ds, "n_anchors": n_anchor, "future_frames": 20,
           "teacher": "frozen Trident-H fused over frames t+1..t+20 (the training target)",
           "n_future_visible_gt_occupied": n_all, "n_correct": n_cor,
           "accuracy_future_visible_occupied": n_cor / max(n_all, 1),
           "accuracy_on_causally_observed_subset": n_obs_cor / max(n_obs, 1),
           "n_causally_observed_subset": n_obs,
           "accuracy_on_newly_visible_subset": (n_cor - n_obs_cor) / max(n_all - n_obs, 1),
           "n_newly_visible_subset": n_all - n_obs,
           "per_class": {v.names[j]: {"n": int(per_class[j, 0]),
                                      "accuracy": float(per_class[j, 1] / max(per_class[j, 0], 1))}
                         for j in range(C)},
           "seconds": time.time() - t0}
    os.makedirs(ART, exist_ok=True)
    write_json(os.path.join(ART, f"teacher_accuracy_{ds}.json"), out)
    print(f"[{ds}] teacher accuracy on future-visible GT-occupied voxels: "
          f"{out['accuracy_future_visible_occupied']:.4f} (n={n_all:,}); causally observed "
          f"{out['accuracy_on_causally_observed_subset']:.4f}, newly visible "
          f"{out['accuracy_on_newly_visible_subset']:.4f}; {n_anchor} anchors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
