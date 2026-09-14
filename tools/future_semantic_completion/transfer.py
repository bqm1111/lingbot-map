#!/usr/bin/env python
"""Zero-shot semantic completion on a target benchmark, on oracle-occupied voxels.

Refuses to run until the study manifest is frozen. Inference is the gate's ``past5``
protocol -- a fresh causal map over the five stream frames ending at the anchor, the
frozen five-frame scale, no future frame -- and the frozen geometry, the trained head, B0
and B1 all read exactly that map. The benchmark's own semantic label is the target; it is
opened here and nowhere else.

Populations, on the benchmark's official evaluation grid and mask:

* ``unobserved`` -- GT-occupied, and the causal map reached none of the voxel
                    (for Occ3D: none of its eight 0.2 m subvoxels);
* ``observed``   -- GT-occupied, and the causal map reached it.

Each condition's 25-way union prediction is mapped to the benchmark's classes by the
declared one-to-one map (``union_vocab.out_channel``), as the gate's evaluator does.
Occ3D predictions are reduced to the 0.4 m grid by averaging the probability vectors of
the eight subvoxels -- occupancy is oracle here, so all eight contribute.

    python tools/future_semantic_completion/transfer.py --seed 0 --dataset semantickitti
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.future_semantic_completion import study as ST                          # noqa: E402
from tools.future_semantic_completion import head as H                            # noqa: E402
from tools.future_semantic_completion.evaluate_source import load_head            # noqa: E402

EXPECTED = {"semantickitti": 163, "occ3d": 1182}
POPS = ("unobserved", "observed")


def input_from_query(q, grid, dev):
    """The 32-channel causal input, exactly as ``core.model.net.Completer.raw`` stacks it."""
    from core.datasets import union_vocab as V8
    X, Y, Z = (int(d) for d in grid.dims); n = X * Y * Z
    sem = q["sem"]; semw = q["sem_w"]; obs = q["observed"].float()
    age = q.get("age", torch.full((n,), -1.0, device=dev)).float()
    base = q["logodds"].reshape(X, Y, Z)
    return torch.stack([base / 4.0, q["w_free"].reshape(X, Y, Z).clamp(max=20) / 20.0,
                        obs.reshape(X, Y, Z), 1.0 - obs.reshape(X, Y, Z),
                        q["n_obs"].float().reshape(X, Y, Z).clamp(max=50) / 50.0,
                        (age.clamp(min=0) / 100.0 * (age >= 0)).reshape(X, Y, Z),
                        semw.reshape(X, Y, Z).clamp(max=20) / 20.0] +
                       [(sem[:, u] / semw.clamp_min(1e-9)).reshape(X, Y, Z) for u in range(V8.U)], 0)


class LabelAccumulator:
    """Per-anchor correctness and a benchmark-class confusion on one population."""

    def __init__(self, C):
        self.C = C; self.conf = np.zeros((C, C), np.int64); self.anchors = []

    def add(self, cid, grp, pred_ch, gt_ch):
        n = len(gt_ch)
        ok = int((pred_ch == gt_ch).sum()) if n else 0
        self.anchors.append((cid, grp, n, ok))
        if n:
            self.conf += np.bincount(gt_ch * self.C + pred_ch, minlength=self.C * self.C).reshape(self.C, self.C)

    def summary(self, names):
        tp = np.diag(self.conf).astype(np.float64); sup = self.conf.sum(1); pr = self.conf.sum(0)
        den = sup + pr - tp
        iou = np.where(den > 0, tp / np.maximum(den, 1), np.nan)
        present = sup > 0
        n = max(self.conf.sum(), 1)
        return {"n_voxels": int(self.conf.sum()), "n_anchors": len(self.anchors),
                "top1_accuracy": float(tp.sum() / n),
                "miou_present_classes": float(np.nanmean(iou[present])) if present.any() else None,
                "per_class": {names[c]: {"support": int(sup[c]), "predicted": int(pr[c]),
                                         "iou": (None if np.isnan(iou[c]) else float(iou[c])),
                                         "recall": (float(tp[c] / sup[c]) if sup[c] else None)}
                              for c in range(self.C)}}

    def rows(self):
        return np.asarray([(a, b, n, ok, float(ok), 0.0) for a, b, n, ok in self.anchors],
                          dtype=[("id", "U64"), ("group", "U64"), ("n", "i8"), ("n_top1", "i8"),
                                 ("sum_cos", "f8"), ("sum_kl", "f8")])


@torch.no_grad()
def run(dataset, seed, device, limit=None):
    from core.run._common import REPO_ROOT
    from core.datasets import grids as G6G, vocab, sources as S, union_vocab as V8
    from core.evaluation import official_targets as G6T
    from core.mapping.feed import CachedFeed
    from core.run.eval_target import window_map, PAST5_OFFSETS
    dev = torch.device(device); torch.cuda.set_device(dev)
    man = json.load(open(ST.manifest_path()))
    fz = man["seeds"][str(seed)]
    geo = H.FrozenGeometry(os.path.join(REPO_ROOT, fz["frozen_checkpoint"]), dev)
    assert ST.sha256_file(geo.path) == fz["frozen_checkpoint_sha256"]
    head, ck = load_head(seed, dev)
    assert ST.sha256_file(ST.head_checkpoint(seed)) == fz["head_checkpoint_sha256"], "head differs from the frozen one"
    with np.load(os.path.join(ST.ART, "b0_mean.npz")) as z:
        b0 = torch.from_numpy(z["mean"]).to(dev)
    assert ST.sha256_json(b0.cpu().numpy().tolist()) == man["b0_sha256"]
    v = vocab.load(dataset); C = len(v)
    MAP, EVAL = G6G.PREDICTION_GRID[dataset], G6G.EVAL_GRID[dataset]
    into = V8.into_matrix(dataset); out_ch = torch.as_tensor(V8.out_channel(dataset), device=dev)
    chan_of = np.full(int(max(max(v.labels), v.empty_label)) + 2, -1, np.int64)
    chan_of[np.asarray(v.labels)] = np.arange(C)
    conds = ("B0", "B1", "B2", "B2ref_existing_head")
    acc = {(c, p): LabelAccumulator(C) for c in conds for p in POPS}
    n_done = n_skip = 0; t0 = time.time(); n_input_check = 0; input_check = []
    same_set = []
    for seg in S.segments(dataset, REPO_ROOT):
        feed = CachedFeed(seg, dev)
        with np.load(S.scale_path(dataset, seg.name)) as z:
            scale = float(np.exp(np.median(z["log_s"][:5])))
        anchors = sorted(seg.anchors)
        for i in range(len(seg)):
            if i not in anchors:
                continue
            f = seg.frames[i]
            target, keep = G6T.semantic_target(dataset, f.gt_ref, REPO_ROOT)
            if target is None or i + min(PAST5_OFFSETS) < 0:
                n_skip += 1; continue
            m, used = window_map(feed, seg, i, PAST5_OFFSETS, scale, into, C, dev)
            assert max(used) <= i, "future frame in the causal window"
            P = feed.pose[i].copy(); P[:3, 3] *= scale
            Tgw = P @ np.linalg.inv(np.asarray(f.T_cam_to_grid, np.float64))
            q = m.query(MAP, Tgw)
            q["age"] = torch.where(q["last_time"] >= 0, (i - q["last_time"]).float(),
                                   torch.full_like(q["last_time"], -1).float())
            x = input_from_query(q, MAP, dev)
            occ_res, sem_exist, trunk = geo.forward(x[None])
            if n_input_check < 3:            # the stacked input reproduces the gate's own path
                from core.model.net import apply_residual
                from core.model.net import Completer
                fin_ref, _ = Completer(geo.net, dev).raw(q, MAP)
                fin_here = apply_residual(q["logodds"].reshape(MAP.dims), occ_res[0, 0]).reshape(-1)
                input_check.append(float((fin_ref - fin_here).abs().max()))
                n_input_check += 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = head(trunk, x[None])
            n_map = x.shape[1] * x.shape[2] * x.shape[3]
            all_rows = torch.arange(n_map, device=dev)
            probs = {"B2": torch.softmax(lg[0].float().reshape(V8.U, -1).T, 1),
                     "B2ref_existing_head": torch.softmax(sem_exist[0].reshape(V8.U, -1).T, 1),
                     "B1": H.b1_predict(x, all_rows, b0, ST.B1_MAX_DIST_VOX)[0],
                     "B0": b0.expand(n_map, V8.U)}
            obs_map = q["observed"]
            tf = target.reshape(-1); kf = keep.reshape(-1)
            gt_occ = (tf != v.empty_label) & kf
            if G6G.NEEDS_REDUCTION[dataset]:
                # canonical 0.2 m -> native 0.4 m: mean of the eight subvoxels; observed if any is
                r = G6G.RATIO
                Xc, Yc, Zc = (int(d) for d in MAP.dims); Xn, Yn, Zn = (int(d) for d in EVAL.dims)
                def reduce_mean(t):
                    t = t.reshape(Xc, Yc, Zc, -1).reshape(Xn, r, Yn, r, Zn, r, -1)
                    return t.mean(dim=(1, 3, 5)).reshape(Xn * Yn * Zn, -1)
                probs = {k: reduce_mean(p_) for k, p_ in probs.items()}
                obs_eval = reduce_mean(obs_map.float().unsqueeze(1))[:, 0] > 0
            else:
                obs_eval = obs_map
            obs_np = obs_eval.cpu().numpy()
            pops = {"unobserved": gt_occ & ~obs_np, "observed": gt_occ & obs_np}
            cid, grp = str(f.gt_ref["clip_id"]), str(seg.name)
            same_set.append([int(pops["unobserved"].sum()), int(pops["observed"].sum())])
            for pn, mask in pops.items():
                rows = np.flatnonzero(mask)
                gt_ch = chan_of[np.clip(tf[rows], 0, len(chan_of) - 1)]
                assert (gt_ch >= 0).all()
                rt = torch.from_numpy(rows).to(dev)
                for c in conds:
                    pc = out_ch[probs[c][rt].argmax(1)].cpu().numpy()
                    acc[(c, pn)].add(cid, grp, pc, gt_ch)
            n_done += 1
            if n_done % 100 == 0:
                print(f"  {n_done} anchors, {(time.time()-t0)/n_done:.2f}s each", flush=True)
            if limit and n_done >= limit:
                break
        if limit and n_done >= limit:
            break
    out = {"dataset": dataset, "seed": seed, "mode": "past5", "n_clips": n_done, "n_skipped": n_skip,
           "frozen_checkpoint_sha256": fz["frozen_checkpoint_sha256"],
           "head_checkpoint_sha256": fz["head_checkpoint_sha256"],
           "populations": {pn: {c: acc[(c, pn)].summary(list(v.names)) for c in conds} for pn in POPS},
           "input_reproduces_gate_path_max_abs_diff": input_check,
           "same_voxel_set": {"per_anchor_counts_sha256": ST.sha256_json(same_set),
                              "n_unobserved": int(sum(s[0] for s in same_set)),
                              "n_observed": int(sum(s[1] for s in same_set))},
           "paired": {}}
    for pn in POPS:
        rows = {c: acc[(c, pn)].rows() for c in conds}
        for a_, b_ in (("B2", "B0"), ("B2", "B1"), ("B2", "B2ref_existing_head"), ("B1", "B0")):
            out["paired"][f"{pn}:{a_}-{b_}:n_top1"] = H.paired_bootstrap(rows[a_], rows[b_], "n_top1", by_group=True)
    out["seconds"] = time.time() - t0; out["env"] = ST.environment()
    os.makedirs(os.path.join(ST.ART, "transfer"), exist_ok=True)
    np.savez_compressed(os.path.join(ST.ART, "transfer", f"per_anchor_{dataset}_{ST.run_tag(seed)}.npz"),
                        **{f"{c}__{pn}": acc[(c, pn)].rows() for c in conds for pn in POPS},
                        **{f"conf_{c}__{pn}": acc[(c, pn)].conf for c in conds for pn in POPS},
                        names=np.asarray(list(v.names)))
    ST.write_json(os.path.join(ST.ART, "transfer", f"eval_{dataset}_{ST.run_tag(seed)}.json"), out)
    if n_done != EXPECTED[dataset]:
        raise SystemExit(f"population mismatch on {dataset}: scored {n_done}, expected {EXPECTED[dataset]}")
    for pn in POPS:
        print(f"  [{dataset} {pn}] n={out['populations'][pn]['B0']['n_voxels']}")
        for c in conds:
            s = out["populations"][pn][c]
            print(f"     {c:22s} top1 {s['top1_accuracy']:.4f} mIoU {s['miou_present_classes']:.4f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dataset", required=True, choices=list(EXPECTED))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    ST.require_data_root()
    if not os.path.exists(ST.manifest_path()):
        raise SystemExit(f"refusing to open {a.dataset}: {ST.manifest_path()} is not frozen")
    run(a.dataset, a.seed, a.device, a.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
