"""Where does the remaining SemanticKITTI error live, layer by layer?"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, "tools/gate8c1")
from _common import REPO_ROOT
from gates.gate6 import grids as G6G, targets as G6T, vocab
from gates.gate8 import sources as S, vocab as V8
from gates.gate8.feed import CachedFeed
from gates.gate8.net import load_checkpoint
from gates.gate8a.regions import to_eval_grid
from tools.gate8c1.eval_target import MANIFEST, PAST5_OFFSETS, window_map
from tools.gate8c1.figure_3d_gallery import anchors

ds = "semantickitti"; N = 16
fz = json.load(open(MANIFEST))["seeds"]["0"]; tau = float(fz["occupancy_threshold"])
dev = torch.device("cuda:2"); torch.cuda.set_device(dev)
C = len(vocab.load(ds)); MAP, EVAL = G6G.PREDICTION_GRID[ds], G6G.EVAL_GRID[ds]
edims = tuple(int(d) for d in EVAL.dims); into = V8.into_matrix(ds)
net = load_checkpoint(os.path.join(REPO_ROOT, fz["checkpoint"]), dev)
z0, vs = float(EVAL.origin[2]), float(EVAL.voxel_size)
zc = (np.arange(edims[2]) + .5) * vs + z0
TAUS = np.round(np.arange(-1.5, 1.001, 0.0625), 4)
ACC = np.zeros((len(TAUS), 3))
for seg, i in anchors(ds, N):
    f = seg.frames[i]
    target, keep = G6T.semantic_target(ds, f.gt_ref, REPO_ROOT)
    if target is None: continue
    feed = CachedFeed(seg, dev)
    with np.load(S.scale_path(ds, seg.name)) as z:
        sc = float(np.exp(np.median(z["log_s"][:5])))
    m, _ = window_map(feed, seg, i, PAST5_OFFSETS, sc, into, C, dev)
    P = feed.pose[i].copy(); P[:3, 3] *= sc
    Tgw = P @ np.linalg.inv(np.asarray(f.T_cam_to_grid, np.float64))
    q = m.query(MAP, Tgw)
    q["age"] = torch.where(q["last_time"] >= 0, (i - q["last_time"]).float(),
                           torch.full_like(q["last_time"], -1).float())
    fin, _ = net.raw(q, MAP)
    sc_ = to_eval_grid(fin, q["logodds"], ds)[0].cpu().numpy().reshape(edims)
    gt = ((target != EVAL.empty_class) & keep).reshape(edims); val = keep.reshape(edims)
    sv, gv = sc_[val], gt[val]
    for j, t in enumerate(TAUS):
        pv = sv >= t
        ACC[j, 0] += (pv & gv).sum(); ACC[j, 1] += (pv & ~gv).sum(); ACC[j, 2] += (~pv & gv).sum()
    del m; torch.cuda.empty_cache()

iou = ACC[:, 0] / np.maximum(ACC.sum(1), 1)
j = int(np.argmax(iou))
print("DIAGNOSTIC ONLY -- choosing tau by looking at SemanticKITTI is the target-domain")
print("tuning the gate forbids. Reported to size the gap, never as a result.\n")
print(f"{'tau':>8} {'IoU':>7} {'prec':>7} {'recall':>7} {'pred/GT':>8}")
for k in range(len(TAUS)):
    if k % 4 and k != j: continue
    tp, fp, fn = ACC[k]
    mark = "  <== best" if k == j else ("  <== source-selected" if abs(TAUS[k] - tau) < 1e-9 else "")
    print(f"{TAUS[k]:+8.4f} {100*tp/max(tp+fp+fn,1):7.2f} {100*tp/max(tp+fp,1):7.2f} "
          f"{100*tp/max(tp+fn,1):7.2f} {(tp+fp)/max(tp+fn,1):8.2f}{mark}")
jf = int(np.argmin(np.abs(TAUS - tau)))
tp, fp, fn = ACC[jf]
print(f"\nat the source-selected tau {tau:+.4f}: IoU {100*tp/(tp+fp+fn):.2f}")
tp, fp, fn = ACC[j]
print(f"at the best-possible tau {TAUS[j]:+.4f}: IoU {100*tp/(tp+fp+fn):.2f}  "
      f"(OccAny raw = 25.28)")
