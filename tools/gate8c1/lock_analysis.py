"""Why does feeding the real map make Occ3D worse? Test the residual-lock hypothesis."""
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
from tools.gate8c1.prior_ablation import blank
from tools.gate8c1.scale_ablation import moge_scale, counts

ds = "occ3d"; N = 20
fz = json.load(open(MANIFEST))["seeds"]["0"]; tau = float(fz["occupancy_threshold"])
dev = torch.device("cuda:2"); torch.cuda.set_device(dev)
C = len(vocab.load(ds)); MAP, EVAL = G6G.PREDICTION_GRID[ds], G6G.EVAL_GRID[ds]
edims = tuple(int(d) for d in EVAL.dims); into = V8.into_matrix(ds)
net = load_checkpoint(os.path.join(REPO_ROOT, fz["checkpoint"]), dev)
GATE = 2.0
tot = dict(valid=0, lock_occ=0, lock_free=0, open=0,
           prior_on_lockfree=0, gt_on_lockfree=0)
agg = {k: [0, 0, 0] for k in ("prior", "deployed", "prior where map is silent")}
n = 0
for seg, i in anchors(ds, N):
    f = seg.frames[i]
    target, keep = G6T.semantic_target(ds, f.gt_ref, REPO_ROOT)
    if target is None: continue
    s = moge_scale(ds, seg.name); feed = CachedFeed(seg, dev)
    m, _ = window_map(feed, seg, i, PAST5_OFFSETS, s, into, C, dev)
    P = feed.pose[i].copy(); P[:3, 3] *= s
    Tgw = P @ np.linalg.inv(np.asarray(f.T_cam_to_grid, np.float64))
    q = m.query(MAP, Tgw)
    q["age"] = torch.where(q["last_time"] >= 0, (i - q["last_time"]).float(),
                           torch.full_like(q["last_time"], -1).float())
    base = q["logodds"]
    qb = blank(q)
    fp_, _ = net.raw(qb, MAP)
    fd_, _ = net.raw(q, MAP)
    R = lambda t: t.cpu().numpy().reshape(edims)
    pri = R(to_eval_grid(fp_, qb["logodds"], ds)[0] >= tau)
    dep = R(to_eval_grid(fd_, base, ds)[0] >= tau)
    # the lock lives on the prediction grid; carry it to the eval grid the same way
    lock_o = R(to_eval_grid((base >= GATE).float(), base, ds)[0] > 0.5)
    lock_f = R(to_eval_grid((base <= -GATE).float(), base, ds)[0] > 0.5)
    gt = ((target != EVAL.empty_class) & keep).reshape(edims)
    val = keep.reshape(edims)
    tot["valid"] += int(val.sum())
    tot["lock_occ"] += int((lock_o & val).sum()); tot["lock_free"] += int((lock_f & val).sum())
    tot["open"] += int((~lock_o & ~lock_f & val).sum())
    tot["prior_on_lockfree"] += int((pri & lock_f & val).sum())
    tot["gt_on_lockfree"] += int((gt & lock_f & val).sum())
    silent = ~lock_o & ~lock_f
    for k, p in (("prior", pri), ("deployed", dep),
                 ("prior where map is silent", np.where(silent, pri, dep))):
        for j, x in enumerate(counts(p & val, gt, val)):
            agg[k][j] += x
    del m; torch.cuda.empty_cache(); n += 1
v = tot["valid"]
print(f"n={n} anchors, {v:,} evaluated cells")
print(f"  locked OCCUPIED by the map (|logodds|>=2, residual cannot act): "
      f"{100*tot['lock_occ']/v:5.2f}%")
print(f"  locked FREE by the map                                       : "
      f"{100*tot['lock_free']/v:5.2f}%")
print(f"  open to the residual                                         : "
      f"{100*tot['open']/v:5.2f}%")
lf = max(tot["lock_free"], 1)
print(f"\n  of cells the map locked FREE, the ground truth says occupied for "
      f"{100*tot['gt_on_lockfree']/lf:.1f}%")
print(f"  the blind prior would have predicted occupied on {100*tot['prior_on_lockfree']/lf:.1f}% "
      f"of them")
print(f"\n{'condition':28s} {'IoU':>7} {'prec':>7} {'recall':>7}")
for k, (tp, fp, fn) in agg.items():
    print(f"{k:28s} {100*tp/max(tp+fp+fn,1):7.2f} {100*tp/max(tp+fp,1):7.2f} "
          f"{100*tp/max(tp+fn,1):7.2f}")
