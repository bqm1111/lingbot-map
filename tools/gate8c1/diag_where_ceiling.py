"""Two questions about the residual ceiling FP.

1. Does an edge-on view make a sparse layer look solid? (integration along the ray)
2. Do the residual FP live in columns the sky rule could not reach -- those with no LiDAR
   return at all, which the rule skips because it has no column top to measure from?
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, "tools/gate8c1")
from _common import REPO_ROOT
from gates.gate6 import grids as G6G, vocab
from gates.gate8 import vocab as V8
from gates.gate8.net import load_checkpoint
from tools.gate8c1.eval_target import MANIFEST
from tools.gate8c1.figure_3d_gallery import anchors, predict

ds = "semantickitti"
fz = json.load(open(MANIFEST))["seeds"]["0"]; tau = float(fz["occupancy_threshold"])
dev = torch.device("cuda:1"); torch.cuda.set_device(dev)
C = len(vocab.load(ds)); MAP, EVAL = G6G.PREDICTION_GRID[ds], G6G.EVAL_GRID[ds]
edims = tuple(int(d) for d in EVAL.dims); into = V8.into_matrix(ds)
net = load_checkpoint(os.path.join(REPO_ROOT, fz["checkpoint"]), dev)
z0, vs = float(EVAL.origin[2]), float(EVAL.voxel_size)
zc = (np.arange(edims[2]) + .5) * vs + z0
hi = zc >= 2.0
A = dict(fp_hi=0, fp_all=0, fp_hi_nogt_col=0, fp_hi_gtcol=0, cols=0, cols_nogt=0,
         depth=[], solid=0)
for seg, i in anchors(ds, 12):
    r = predict(ds, seg, i, 0, dev, net, tau, into, C, MAP, EVAL, edims)
    if r is None: continue
    fp = r["pred"] & ~r["gt"] & r["valid"]
    gt_col = r["gt"].any(axis=2)                       # column has any real geometry
    A["fp_all"] += int(fp.sum()); A["fp_hi"] += int(fp[:, :, hi].sum())
    A["fp_hi_nogt_col"] += int(fp[:, :, hi][~gt_col].sum())
    A["fp_hi_gtcol"] += int(fp[:, :, hi][gt_col].sum())
    A["cols"] += gt_col.size; A["cols_nogt"] += int((~gt_col).sum())
    # how many FP voxels does a single edge-on line of sight cross?
    per_row = fp[:, :, hi].sum(axis=0)                 # integrate along x (the view axis)
    A["depth"].append(per_row[per_row > 0])
d = np.concatenate(A["depth"])
print(f"false positives: {A['fp_all']:,} total, {A['fp_hi']:,} above 2 m "
      f"({100*A['fp_hi']/A['fp_all']:.1f}%)")
print(f"\n--- 1. why a sparse ceiling looks solid from the side ---")
print(f"mean occupancy of a layer above 2 m is a few percent, but a side view integrates")
print(f"along 256 voxels of depth. Non-empty lines of sight cross a median of "
      f"{np.median(d):.0f} FP voxels (p90 {np.percentile(d, 90):.0f}, max {d.max():.0f}).")
print(f"Anything above ~2 reads as opaque, so the elevation panel exaggerates a thin "
      f"scattering into a sheet.")
print(f"\n--- 2. can the sky rule even reach them? ---")
print(f"columns with no ground-truth geometry at all: "
      f"{100*A['cols_nogt']/A['cols']:.1f}% of the grid")
print(f"  FP above 2 m in those columns : {A['fp_hi_nogt_col']:,} "
      f"({100*A['fp_hi_nogt_col']/max(A['fp_hi'],1):.1f}% of high FP)")
print(f"  FP above 2 m in columns that DO have geometry: {A['fp_hi_gtcol']:,} "
      f"({100*A['fp_hi_gtcol']/max(A['fp_hi'],1):.1f}%)")
