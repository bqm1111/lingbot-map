"""Is the 'prior' prediction literally one constant volume? If so, score it properly."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, "tools/gate8c1")
from _common import ART, REPO_ROOT
from gates.gate6 import grids as G6G, targets as G6T, vocab
from gates.gate8 import sources as S, vocab as V8
from gates.gate8.feed import CachedFeed
from gates.gate8.net import load_checkpoint
from gates.gate8a.regions import to_eval_grid
from tools.gate8c1.eval_target import MANIFEST, PAST5_OFFSETS, window_map
from tools.gate8c1.figure_3d_gallery import anchors
from tools.gate8c1.prior_ablation import blank
from tools.gate8c1.scale_ablation import moge_scale, counts

ds = sys.argv[1] if len(sys.argv) > 1 else "occ3d"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
fz = json.load(open(MANIFEST))["seeds"]["0"]; tau = float(fz["occupancy_threshold"])
dev = torch.device("cuda:2"); torch.cuda.set_device(dev)
C = len(vocab.load(ds)); MAP, EVAL = G6G.PREDICTION_GRID[ds], G6G.EVAL_GRID[ds]
edims = tuple(int(d) for d in EVAL.dims); into = V8.into_matrix(ds)
net = load_checkpoint(os.path.join(REPO_ROOT, fz["checkpoint"]), dev)

# PRIOR is filled from the first anchor's blanked query (the semantic width has to come
# from a real map -- the table carries the union vocabulary, not the dataset's own)
PRIOR = None
# confirm the network really is scene-independent when told nothing was observed
agg = {"constant prior": [0, 0, 0], "deployed": [0, 0, 0], "all valid occupied": [0, 0, 0]}
same = 0; n = 0
for seg, i in anchors(ds, N):
    f = seg.frames[i]
    target, keep = G6T.semantic_target(ds, f.gt_ref, REPO_ROOT)
    if target is None: continue
    s = moge_scale(ds, seg.name)
    feed = CachedFeed(seg, dev)
    m, _ = window_map(feed, seg, i, PAST5_OFFSETS, s, into, C, dev)
    P = feed.pose[i].copy(); P[:3, 3] *= s
    Tgw = P @ np.linalg.inv(np.asarray(f.T_cam_to_grid, np.float64))
    q = m.query(MAP, Tgw)
    q["age"] = torch.where(q["last_time"] >= 0, (i - q["last_time"]).float(),
                           torch.full_like(q["last_time"], -1).float())
    qb = blank(q)
    fin_b, _ = net.raw(qb, MAP)
    pb = (to_eval_grid(fin_b, qb["logodds"], ds)[0] >= tau).cpu().numpy().reshape(edims)
    if PRIOR is None:
        PRIOR = pb
        print(f"constant prior volume: {int(PRIOR.sum()):,} of {PRIOR.size:,} eval cells "
              f"({100*PRIOR.mean():.1f}%)")
        zc = (np.arange(edims[2]) + .5) * float(EVAL.voxel_size) + float(EVAL.origin[2])
        oz = PRIOR.sum(axis=(0, 1))
        for k in range(edims[2]):
            if oz[k]:
                print(f"    z={zc[k]:+5.1f} m  {int(oz[k]):7,}  "
                      f"({100*oz[k]/PRIOR[:, :, k].size:5.1f}% of that layer)")
    same += int(np.array_equal(pb, PRIOR))
    fin_d, _ = net.raw(q, MAP)
    pd = (to_eval_grid(fin_d, q["logodds"], ds)[0] >= tau).cpu().numpy().reshape(edims)
    gt = ((target != EVAL.empty_class) & keep).reshape(edims); val = keep.reshape(edims)
    for k, p in (("constant prior", PRIOR), ("deployed", pd),
                 ("all valid occupied", np.ones_like(val))):
        for j, x in enumerate(counts(p & val, gt, val)): agg[k][j] += x
    del m; torch.cuda.empty_cache(); n += 1
print(f"\nblank-input prediction identical to the constant prior on {same}/{n} anchors")
print(f"{'condition':22s} {'IoU':>7} {'prec':>7} {'recall':>7}")
for k, (tp, fp, fn) in agg.items():
    print(f"{k:22s} {100*tp/max(tp+fp+fn,1):7.2f} {100*tp/max(tp+fp,1):7.2f} "
          f"{100*tp/max(tp+fn,1):7.2f}")
