"""Per-layer false positives for the exact anchors the gallery figure draws."""
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
print("manifest:", MANIFEST)
print("checkpoint:", fz["checkpoint"], " tau", tau)
dev = torch.device("cuda:1"); torch.cuda.set_device(dev)
C = len(vocab.load(ds)); MAP, EVAL = G6G.PREDICTION_GRID[ds], G6G.EVAL_GRID[ds]
edims = tuple(int(d) for d in EVAL.dims); into = V8.into_matrix(ds)
net = load_checkpoint(os.path.join(REPO_ROOT, fz["checkpoint"]), dev)
z0, vs = float(EVAL.origin[2]), float(EVAL.voxel_size)
zc = (np.arange(edims[2]) + .5) * vs + z0
rows = []
for seg, i in anchors(ds, 6):
    r = predict(ds, seg, i, 0, dev, net, tau, into, C, MAP, EVAL, edims)
    if r is None: continue
    fp = r["pred"] & ~r["gt"] & r["valid"]
    rows.append((r["clip"], fp.sum(axis=(0, 1)), r["valid"].sum(axis=(0, 1))))
    print(f"{r['clip']}: total FP {int(fp.sum()):,}")
print(f"\n{'z (m)':>7}" + "".join(f"{c[:12]:>14}" for c, _, _ in rows))
print(f"{'':>7}" + "".join(f"{'FP  (% layer)':>14}" for _ in rows))
for k in range(edims[2] - 1, -1, -1):
    if zc[k] < 1.5: continue
    print(f"{zc[k]:+7.1f}" + "".join(
        f"{int(f[k]):8,}{100*f[k]/max(v[k],1):5.1f}%" for _, f, v in rows))
