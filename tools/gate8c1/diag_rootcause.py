"""The root cause, as one correlation.

Claim: the model over-predicts exactly where training never supervised it. If true, the
false-positive rate at a given height on SemanticKITTI should track the UNSUPERVISED
fraction at that height in the KITTI-360 training target -- two completely independent
datasets, linked only through what the loss was allowed to see.
"""
import glob, json, os, random, sys
import numpy as np, torch
sys.path.insert(0, "tools/gate8c1")
from _common import REPO_ROOT
from gates.gate6 import grids as G6G, vocab
from gates.gate8 import vocab as V8
from gates.gate8.net import load_checkpoint
from tools.gate8c1.eval_target import MANIFEST
from tools.gate8c1.figure_3d_gallery import anchors, predict

TGT = "/media/SSD1/MINH_DATASETS/lingbot_gate8c1_sky/targets"   # the shipped model's data
ds = "semantickitti"
fz = json.load(open(MANIFEST))["seeds"]["0"]; tau = float(fz["occupancy_threshold"])
dev = torch.device("cuda:1"); torch.cuda.set_device(dev)
C = len(vocab.load(ds)); MAP, EVAL = G6G.PREDICTION_GRID[ds], G6G.EVAL_GRID[ds]
edims = tuple(int(d) for d in EVAL.dims); into = V8.into_matrix(ds)
net = load_checkpoint(os.path.join(REPO_ROOT, fz["checkpoint"]), dev)
z0, vs = float(EVAL.origin[2]), float(EVAL.voxel_size)
zc = (np.arange(edims[2]) + .5) * vs + z0

# --- A. how much of each height did TRAINING supervise? (KITTI-360) ---
fs = glob.glob(f"{TGT}/*/*.npz"); random.seed(0); fs = random.sample(fs, 80)
sup = np.zeros(edims[2]); tot = np.zeros(edims[2])
for f in fs:
    with np.load(f) as z:
        d = tuple(int(x) for x in z["dims"]); n = int(np.prod(d))
        v = np.unpackbits(z["valid_packed"])[:n].astype(bool).reshape(d)
    sup += v.sum(axis=(0, 1)); tot += v[:, :, 0].size
unsup = 100 * (1 - sup / tot)

# --- B. how often does the model hallucinate at each height? (SemanticKITTI) ---
FP = np.zeros(edims[2]); NEG = np.zeros(edims[2])
for seg, i in anchors(ds, 12):
    r = predict(ds, seg, i, 0, dev, net, tau, into, C, MAP, EVAL, edims)
    if r is None: continue
    neg = ~r["gt"] & r["valid"]                       # voxels the benchmark says are empty
    FP += (r["pred"] & neg).sum(axis=(0, 1)); NEG += neg.sum(axis=(0, 1))
fpr = 100 * FP / np.maximum(NEG, 1)

print(f"{'z (m)':>7} {'UNSUPERVISED in training':>26} {'hallucination rate':>20}")
for k in range(edims[2] - 1, -1, -2):
    print(f"{zc[k]:+7.1f} {unsup[k]:24.1f} % {fpr[k]:18.1f} %")
r = np.corrcoef(unsup, fpr)[0, 1]
hi = zc >= 1.0
print(f"\nPearson r over all 32 heights            : {r:+.3f}")
print(f"Pearson r restricted to z >= 1.0 m       : "
      f"{np.corrcoef(unsup[hi], fpr[hi])[0, 1]:+.3f}")
print(f"\nunsupervised fraction, mean over z>=2 m  : {unsup[zc >= 2].mean():.1f} %")
print(f"hallucination rate,     mean over z>=2 m  : {fpr[zc >= 2].mean():.1f} %")
