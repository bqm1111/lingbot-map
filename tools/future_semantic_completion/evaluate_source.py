#!/usr/bin/env python
"""B0 / B1 / B2 on KITTI-360 drive 0006, on oracle-occupied voxels.

Populations, all intersected with the rebuilt target's ``valid`` mask and requiring a
future teacher target:

* ``key``       -- the primary: map-unobserved at the anchor, **measured** occupied by a
                   later LiDAR sweep (the feature study's frozen measurement partition),
                   teacher target present;
* ``key_agree`` -- the subset of ``key`` where the two halves of the future window agree on
                   the teacher's class (the less noisy targets);
* ``observed``  -- the contrast: map-observed, LiDAR-occupied, teacher target present.

Every condition is scored on exactly the same voxels of every anchor (one mask per anchor,
one index array, asserted equal). Distance to the nearest map voxel carrying teacher
evidence is recorded per voxel so B1 and B2 can be compared as a function of how far the
prediction sits from anything the map already named.

    python tools/future_semantic_completion/evaluate_source.py --seed 0 --device cuda:0
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

POPS = ("key", "key_agree", "observed")
DIST_EDGES = np.asarray([0, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, np.inf])
DIST_NAMES = ["<=1", "1-2", "2-3", "3-5", "5-10", "10-15", ">15"]


def load_head(seed, dev):
    ck = torch.load(ST.head_checkpoint(seed), map_location="cpu", weights_only=False)
    head = H.SemanticCompletionHead(width=ck["head_width"], padding_mode=ck["padding_mode"]).to(dev)
    head.load_state_dict(ck["state_dict"]); head.eval()
    return head, ck


@torch.no_grad()
def predict_all(geo, head, x, b0, rows):
    """All conditions at ``rows`` of one full grid. Returns dict name -> [n, U] probs, and
    the B1 distance per voxel."""
    _, sem_exist, trunk = geo.forward(x[None])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg = head(trunk, x[None])
    p2 = torch.softmax(lg[0].float().reshape(V8.U, -1).T[rows], 1)
    pref = torch.softmax(sem_exist[0].reshape(V8.U, -1).T[rows], 1)
    p1, dist = H.b1_predict(x, rows, b0, ST.B1_MAX_DIST_VOX)
    p0 = b0.expand(len(rows), V8.U)
    return {"B0": p0, "B1": p1, "B2": p2, "B2ref_existing_head": pref}, dist


@torch.no_grad()
def run(seed, device, limit=None):
    dev = torch.device(device); torch.cuda.set_device(dev)
    geo = H.FrozenGeometry(ST.frozen_checkpoint(seed), dev)
    head, ck = load_head(seed, dev)
    assert ck["frozen_checkpoint_sha256"] == ST.sha256_file(geo.path), "head was trained on another frozen geometry"
    with np.load(os.path.join(ST.ART, "b0_mean.npz")) as z:
        b0 = torch.from_numpy(z["mean"]).to(dev)
    files = sorted(glob.glob(os.path.dirname(SRC.sample_path(SRC.VAL_DRIVE, 0)) + "/*.npz"))
    if limit:
        files = files[:limit]
    conds = ("B0", "B1", "B2", "B2ref_existing_head")
    acc = {(c, p): H.SetAccumulator(f"{c}/{p}") for c in conds for p in POPS}
    # distance-stratified counts on the key population: [cond][bin] -> (n, sum_cos, n_top1)
    strat = {c: np.zeros((len(DIST_NAMES), 3), np.float64) for c in conds}
    dist_hist = np.zeros(len(DIST_NAMES), np.int64)
    same_set = {"n_anchors": 0, "n_key_measured": 0, "n_key_sample_rule": 0, "n_symmetric_diff": 0,
                "n_missing_mask": 0, "index_digest": []}
    t0 = time.time()
    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        for p in files:
            idx = int(os.path.basename(p)[:-4]); cid = os.path.basename(p)[:-4]
            mp = ST.eval_mask_path(SRC.VAL_DRIVE, idx)
            if not os.path.exists(mp):
                same_set["n_missing_mask"] += 1; continue
            with np.load(p) as z:
                d = TG.unpack_sample(z, dev)
                n = d["gt_occ"].numel()
                agree = torch.from_numpy(np.unpackbits(z["fut_half_agree"])[:n].astype(bool)).to(dev)
            with np.load(mp) as z:
                um = lambda k: torch.from_numpy(np.unpackbits(z[k])[:n].astype(bool)).to(dev)
                later, mobs, mocc = um("later_measured"), um("map_observed"), um("measured_occ")
            valid = d["gt_valid"].reshape(-1); fv = d["fut_valid"].reshape(-1)
            key = valid & ~mobs & later & mocc & fv
            key_rule = H.key_mask(d).reshape(-1)
            same_set["n_key_measured"] += int(key.sum()); same_set["n_key_sample_rule"] += int(key_rule.sum())
            same_set["n_symmetric_diff"] += int((key ^ key_rule).sum())
            pops = {"key": key, "key_agree": key & agree, "observed": H.observed_mask(d).reshape(-1)}
            fut = d["fut_p"].reshape(-1, V8.U)
            for pname, mask in pops.items():
                rows = mask.nonzero(as_tuple=True)[0]
                preds, dist = predict_all(geo, head, d["input"], b0, rows)
                tgt = fut[rows]
                for c in conds:
                    assert preds[c].shape[0] == rows.numel()
                    acc[(c, pname)].add(cid, "0006", preds[c], tgt, b0)
                if pname == "key":
                    same_set["index_digest"].append(ST.sha256_json(rows.cpu().numpy().tolist())[:12])
                    b = np.clip(np.searchsorted(DIST_EDGES, dist.cpu().numpy(), side="right") - 1, 0, len(DIST_NAMES) - 1)
                    dist_hist += np.bincount(b, minlength=len(DIST_NAMES))
                    ta = tgt.argmax(1)
                    for c in conds:
                        cs = H.cosine(preds[c], tgt).cpu().numpy(); ok = (preds[c].argmax(1) == ta).cpu().numpy()
                        for k in range(len(DIST_NAMES)):
                            s = b == k
                            strat[c][k] += (s.sum(), cs[s].sum(), ok[s].sum())
            same_set["n_anchors"] += 1
            if same_set["n_anchors"] % 100 == 0:
                print(f"  {same_set['n_anchors']} anchors, {(time.time()-t0)/same_set['n_anchors']:.2f}s each", flush=True)
    out = {"seed": seed, "frozen_checkpoint": ck["frozen_checkpoint"],
           "frozen_checkpoint_sha256": ck["frozen_checkpoint_sha256"],
           "head_checkpoint_sha256": ST.sha256_file(ST.head_checkpoint(seed)),
           "b0_sha256": ST.sha256_json(b0.cpu().numpy().tolist()),
           "populations": {pn: {c: acc[(c, pn)].summary() for c in conds} for pn in POPS},
           "same_voxel_set": dict(same_set, index_digest=ST.sha256_json(same_set["index_digest"])),
           "b1_max_dist_vox": ST.B1_MAX_DIST_VOX,
           "distance_to_evidence": {"edges_vox": DIST_EDGES.tolist(), "names": DIST_NAMES,
                                    "key_hist": dist_hist.tolist(),
                                    "by_condition": {c: {"n": strat[c][:, 0].tolist(),
                                                         "cosine": (strat[c][:, 1] / np.maximum(strat[c][:, 0], 1)).tolist(),
                                                         "top1": (strat[c][:, 2] / np.maximum(strat[c][:, 0], 1)).tolist()}
                                                     for c in conds}},
           "paired": {}}
    for pn in POPS:
        rows = {c: acc[(c, pn)].per_anchor() for c in conds}
        for a_, b_ in (("B2", "B0"), ("B2", "B1"), ("B2", "B2ref_existing_head"), ("B1", "B0")):
            for f in ("sum_cos", "n_top1", "sum_kl"):
                out["paired"][f"{pn}:{a_}-{b_}:{f}"] = H.paired_bootstrap(rows[a_], rows[b_], f)
    np.savez_compressed(os.path.join(ST.ART, "source", f"per_anchor_{ST.run_tag(seed)}.npz"),
                        **{f"{c}__{pn}": acc[(c, pn)].per_anchor() for c in conds for pn in POPS})
    out["seconds"] = time.time() - t0; out["firewall"] = audit.summary(); out["env"] = ST.environment()
    ST.write_json(os.path.join(ST.ART, "source", f"source_{ST.run_tag(seed)}.json"), out)
    for pn in POPS:
        print(f"  [{pn}] n={out['populations'][pn]['B0']['n_voxels']}")
        for c in conds:
            s = out["populations"][pn][c]
            print(f"     {c:22s} cos {s['cosine_mean']:.4f} kl {s['kl_mean']:.4f} top1 {s['top1_agreement']:.4f} "
                  f"mIoU {s['miou_present_classes']:.4f} cos_b0 {s['cosine_to_b0_pred']:.4f} var {s['pred_variance']:.4f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    ST.require_data_root()
    os.makedirs(os.path.join(ST.ART, "source"), exist_ok=True)
    run(a.seed, a.device, a.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
