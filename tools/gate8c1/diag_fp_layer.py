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
TP = np.zeros(edims[2]); FP = np.zeros(edims[2]); FN = np.zeros(edims[2])
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
    pr = (to_eval_grid(fin, q["logodds"], ds)[0] >= tau).cpu().numpy().reshape(edims)
    gt = ((target != EVAL.empty_class) & keep).reshape(edims); val = keep.reshape(edims)
    p, g = pr & val, gt & val
    TP += (p & g).sum(axis=(0,1)); FP += (p & ~g).sum(axis=(0,1)); FN += (~p & g).sum(axis=(0,1))
    del m; torch.cuda.empty_cache()
tot_fp = FP.sum()
print(f"total TP {TP.sum():,.0f}  FP {tot_fp:,.0f}  FN {FN.sum():,.0f}  "
      f"IoU {100*TP.sum()/(TP.sum()+tot_fp+FN.sum()):.2f}")
print(f"\n{'z (m)':>7} {'TP':>10} {'FP':>10} {'FN':>10} {'% of all FP':>12}")
for k in range(edims[2]):
    print(f"{zc[k]:+7.1f} {TP[k]:10,.0f} {FP[k]:10,.0f} {FN[k]:10,.0f} "
          f"{100*FP[k]/tot_fp:12.2f}")
lo = zc <= -1.0
print(f"\nlayers at or below -1.0 m carry {100*FP[lo].sum()/tot_fp:.1f}% of all false positives")
print(f"removing them entirely would give IoU "
      f"{100*TP[~lo].sum()/(TP[~lo].sum()+FP[~lo].sum()+FN.sum()):.2f} "
      f"(TP there is {100*TP[lo].sum()/TP.sum():.1f}% of all TP, so this is NOT a free win)")
