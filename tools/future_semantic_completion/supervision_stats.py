#!/usr/bin/env python
"""How much privileged semantic supervision exists, and what it looks like.

Walks every cached training and validation sample once and records, per drive: the number
of voxels in each population (unobserved -> future-observed -> occupied -> teacher target;
the observed contrast set; the voxels lost for lack of a teacher target), the argmax class
histogram and entropy of the supervision, the causal-validity counts (input frames <= t <
target frames on every file), and the teacher-projection consistency check on voxels the
causal map and the future mapper both carry evidence for. Writes the B0 vector -- the mean
teacher target over the *training* supervision set -- which every later stage reads.

    python tools/future_semantic_completion/supervision_stats.py --device cuda:0
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


def files_of(drive):
    return sorted(glob.glob(os.path.dirname(SRC.sample_path(drive, 0)) + "/*.npz"))


def entropy(h):
    p = h / max(h.sum(), 1); p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    ST.require_data_root()
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    drives = list(SRC.TRAIN_DRIVES) + [SRC.VAL_DRIVE]
    out = {"drives": {}, "causal": {"n_files": 0, "n_input_leq_t": 0, "n_targets_gt_t": 0,
                                    "n_target_window_20": 0}}
    b0_acc = np.zeros(V8.U, np.float64); b0_n = 0
    t0 = time.time()
    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        for d in drives:
            fs = files_of(d)[::a.stride]
            c = {"n_anchors": len(fs), "key": 0, "observed": 0, "key_no_target": 0,
                 "unobs_future_occ": 0, "fut_valid": 0, "gt_occupied": 0,
                 "n_zero_key": 0, "half_agree_key": 0, "unobserved_on_key": 0,
                 "occupied_on_key": 0, "both_evidence": 0, "both_top1_agree": 0,
                 "both_cos_sum": 0.0, "shuffled_top1_agree": 0, "shuffled_cos_sum": 0.0}
            hist = np.zeros(V8.U, np.int64); per_anchor = []
            for f in fs:
                with np.load(f) as z:
                    inf, tf, t = z["input_frames"], z["target_frames"], int(z["t"])
                    out["causal"]["n_files"] += 1
                    out["causal"]["n_input_leq_t"] += int(inf.max() <= t)
                    out["causal"]["n_targets_gt_t"] += int(len(tf) and tf.min() > t)
                    out["causal"]["n_target_window_20"] += int(len(tf) and tf.max() <= t + TG.FUTURE_FRAMES)
                    dd = TG.unpack_sample(z, dev)
                    ha = torch.from_numpy(np.unpackbits(z["fut_half_agree"])[:dd["gt_occ"].numel()]
                                          .astype(bool)).to(dev)
                k = H.key_mask(dd).reshape(-1); o = H.observed_mask(dd).reshape(-1)
                go = ((dd["gt_occ"] > 0) & dd["gt_valid"]).reshape(-1)
                ufo = ((~dd["observed"]) & dd["fut_observed"] & (dd["gt_occ"] > 0) & dd["gt_valid"]).reshape(-1)
                fp = dd["fut_p"].reshape(-1, V8.U)
                nk = int(k.sum())
                c["key"] += nk; c["observed"] += int(o.sum()); c["fut_valid"] += int(dd["fut_valid"].sum())
                c["gt_occupied"] += int(go.sum()); c["unobs_future_occ"] += int(ufo.sum())
                c["key_no_target"] += int((ufo & ~dd["fut_valid"].reshape(-1)).sum())
                c["n_zero_key"] += int(nk == 0); c["half_agree_key"] += int((k & ha).sum())
                c["unobserved_on_key"] += int((~dd["observed"].reshape(-1))[k].sum())
                c["occupied_on_key"] += int(go[k].sum())
                per_anchor.append(nk)
                if nk:
                    tk = fp[k]
                    hist += np.bincount(tk.argmax(1).cpu().numpy(), minlength=V8.U)
                    if d != SRC.VAL_DRIVE:
                        b0_acc += tk.double().sum(0).cpu().numpy(); b0_n += nk
                # projection consistency: causal-map teacher vector vs future teacher vector
                x = dd["input"]; semw = x[6].reshape(-1)
                both = dd["observed"].reshape(-1) & (semw > 0) & dd["fut_valid"].reshape(-1) & go
                nb = int(both.sum())
                if nb > 1:
                    cur = x[7:7 + V8.U].reshape(V8.U, -1).T[both]; fut = fp[both]
                    c["both_evidence"] += nb
                    c["both_top1_agree"] += int((cur.argmax(1) == fut.argmax(1)).sum())
                    c["both_cos_sum"] += float(H.cosine(cur, fut).sum())
                    g = torch.Generator(device="cpu").manual_seed(nb)
                    perm = torch.randperm(nb, generator=g).to(dev)
                    c["shuffled_top1_agree"] += int((cur.argmax(1) == fut[perm].argmax(1)).sum())
                    c["shuffled_cos_sum"] += float(H.cosine(cur, fut[perm]).sum())
            nb = max(c["both_evidence"], 1)
            share = hist / max(hist.sum(), 1)
            order = np.argsort(-share)
            c.update({"key_per_anchor_mean": float(np.mean(per_anchor)),
                      "key_per_anchor_median": float(np.median(per_anchor)),
                      "key_per_anchor_p10_p90": [float(np.percentile(per_anchor, 10)),
                                                 float(np.percentile(per_anchor, 90))],
                      "argmax_hist": {V8.UNION[u]: int(hist[u]) for u in range(V8.U)},
                      "argmax_top1_share": float(share[order[0]]),
                      "argmax_top2_share": float(share[order[:2]].sum()),
                      "argmax_entropy_nats": entropy(hist),
                      "argmax_entropy_max_nats": float(np.log(int((hist > 0).sum()) or 1)),
                      "n_classes_present": int((hist > 0).sum()),
                      "projection_consistency": {
                          "n_both_evidence": c["both_evidence"],
                          "top1_agreement": c["both_top1_agree"] / nb,
                          "cosine_mean": c["both_cos_sum"] / nb,
                          "shuffled_top1_agreement": c["shuffled_top1_agree"] / nb,
                          "shuffled_cosine_mean": c["shuffled_cos_sum"] / nb}})
            for kk in ("both_evidence", "both_top1_agree", "both_cos_sum", "shuffled_top1_agree",
                       "shuffled_cos_sum"):
                c.pop(kk)
            out["drives"][d] = c
            print(f"  {d[17:21]}: anchors {c['n_anchors']} key {c['key']} "
                  f"({c['key_per_anchor_mean']:.0f}/anchor) observed-contrast {c['observed']} "
                  f"lost-no-target {c['key_no_target']} top1 {c['argmax_top1_share']:.3f} "
                  f"top2 {c['argmax_top2_share']:.3f} H {c['argmax_entropy_nats']:.2f} "
                  f"proj-agree {c['projection_consistency']['top1_agreement']:.3f} "
                  f"(shuffled {c['projection_consistency']['shuffled_top1_agreement']:.3f})",
                  flush=True)
    b0 = (b0_acc / max(b0_n, 1)).astype(np.float32)
    os.makedirs(ST.ART, exist_ok=True)
    np.savez(os.path.join(ST.ART, "b0_mean.npz"), mean=b0, n_vectors=np.int64(b0_n),
             drives=np.asarray(list(SRC.TRAIN_DRIVES)))
    tr = [out["drives"][d] for d in SRC.TRAIN_DRIVES]
    out["train_total_key"] = int(sum(c["key"] for c in tr))
    out["train_total_anchors"] = int(sum(c["n_anchors"] for c in tr))
    out["b0"] = {"vector": b0.tolist(), "n_vectors": int(b0_n),
                 "argmax": V8.UNION[int(b0.argmax())], "sha256": ST.sha256_json(b0.tolist()),
                 "rule": "mean fut_p over the training drives' key set (unobserved -> future-observed -> occupied -> teacher target)"}
    out["definitions"] = {
        "key": "~observed & fut_observed & gt_occ & gt_valid & fut_valid (the supervision set)",
        "observed": "observed & gt_occ & gt_valid & fut_valid (contrast set)",
        "unobs_future_occ": "~observed & fut_observed & gt_occ & gt_valid, with or without a teacher target",
        "key_no_target": "unobs_future_occ voxels the future mapper lifted no teacher evidence into",
        "projection_consistency": "voxels with causal-map teacher evidence AND a future teacher target AND LiDAR-occupied: agreement between the two independently projected vectors, against a within-anchor shuffled pairing"}
    out["stride"] = a.stride; out["seconds"] = time.time() - t0
    out["firewall"] = audit.summary(); out["env"] = ST.environment()
    ST.write_json(os.path.join(ST.ART, "supervision.json"), out)
    print(f"B0 argmax {out['b0']['argmax']}  train key voxels {out['train_total_key']}  "
          f"causal {out['causal']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
