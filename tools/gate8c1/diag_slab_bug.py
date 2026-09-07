"""What is the network actually saying in the ceiling voxels?

If the slab is a confident hallucination, the final log-odds there will be clearly positive.
If it is 'no opinion' being read as occupied by a negative threshold, it will sit in a thin
band just above tau and just below zero.
"""
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

ds = sys.argv[1] if len(sys.argv) > 1 else "semantickitti"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
fz = json.load(open(MANIFEST))["seeds"]["0"]; tau = float(fz["occupancy_threshold"])
dev = torch.device("cuda:2"); torch.cuda.set_device(dev)
C = len(vocab.load(ds)); MAP, EVAL = G6G.PREDICTION_GRID[ds], G6G.EVAL_GRID[ds]
edims = tuple(int(d) for d in EVAL.dims); into = V8.into_matrix(ds)
net = load_checkpoint(os.path.join(REPO_ROOT, fz["checkpoint"]), dev)
seg = S.segments(ds, REPO_ROOT)[0]
z0, vs = float(EVAL.origin[2]), float(EVAL.voxel_size)
zc = (np.arange(edims[2]) + .5) * vs + z0
print(f"threshold tau = {tau:+.4f}   (a voxel is called occupied when final >= tau)")

acc = {"final": [], "z": [], "gt": [], "obs": []}
for seg_, i in anchors(ds, N):
    f = seg_.frames[i]
    target, keep = G6T.semantic_target(ds, f.gt_ref, REPO_ROOT)
    if target is None: continue
    feed = CachedFeed(seg_, dev)
    with np.load(S.scale_path(ds, seg_.name)) as z:
        sc = float(np.exp(np.median(z["log_s"][:5])))
    m, _ = window_map(feed, seg_, i, PAST5_OFFSETS, sc, into, C, dev)
    P = feed.pose[i].copy(); P[:3, 3] *= sc
    Tgw = P @ np.linalg.inv(np.asarray(f.T_cam_to_grid, np.float64))
    q = m.query(MAP, Tgw)
    q["age"] = torch.where(q["last_time"] >= 0, (i - q["last_time"]).float(),
                           torch.full_like(q["last_time"], -1).float())
    fin, _ = net.raw(q, MAP)
    s_fin = to_eval_grid(fin, q["logodds"], ds)[0].cpu().numpy().reshape(edims)
    obs = to_eval_grid(q["observed"].float(), q["logodds"], ds)[0].cpu().numpy().reshape(edims) > .5
    gt = ((target != EVAL.empty_class) & keep).reshape(edims)
    val = keep.reshape(edims)
    zz = np.broadcast_to(np.arange(edims[2]), edims)
    acc["final"].append(s_fin[val]); acc["z"].append(zz[val])
    acc["gt"].append(gt[val]); acc["obs"].append(obs[val])
    del m; torch.cuda.empty_cache()
F = np.concatenate(acc["final"]); Z = np.concatenate(acc["z"])
G = np.concatenate(acc["gt"]); O = np.concatenate(acc["obs"])

print(f"\n{'z (m)':>7} {'n':>10} {'observed%':>10} {'GT occ%':>9} "
      f"{'median final':>13} {'pred occ%':>10} {'| in (tau,0) %':>15}")
for k in range(edims[2]):
    s = Z == k
    if not s.any(): continue
    fk = F[s]
    band = ((fk >= tau) & (fk < 0)).mean() * 100
    print(f"{zc[k]:+7.1f} {s.sum():10,} {100*O[s].mean():10.2f} {100*G[s].mean():9.2f} "
          f"{np.median(fk):13.4f} {100*(fk >= tau).mean():10.2f} {band:15.2f}")

hi = zc[Z] >= 2.0
print(f"\n--- voxels above 2 m ({hi.sum():,}) ---")
fk = F[hi]
print(f"  predicted occupied            {100*(fk >= tau).mean():.2f}%")
print(f"  ... of those, final < 0       {100*((fk >= tau) & (fk < 0)).sum()/max((fk>=tau).sum(),1):.2f}%")
print(f"  would be occupied at tau = 0  {100*(fk >= 0).mean():.2f}%")
print(f"  median final log-odds         {np.median(fk):+.4f}")
print(f"  ground truth occupied         {100*G[hi].mean():.2f}%")
print(f"  observed by the mapper        {100*O[hi].mean():.2f}%")
