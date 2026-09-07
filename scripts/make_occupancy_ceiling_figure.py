"""Publication figure: LingBot occupancy vs the GT-LiDAR ceiling, SemanticKITTI seq 08."""
import base64, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

D = json.load(open("outputs/prompted_lingbot/occupancy_geometry/semantickitti/vis3d/frame_data.json"))
TP, FP, FN = 0, 1, 2
C = {TP: (0.13, 0.55, 0.20), FP: (0.85, 0.33, 0.10), FN: (0.20, 0.40, 0.80)}
NAME = {TP: "true positive (hit)", FP: "false positive (wrongly filled)", FN: "false negative (missed)"}
FRAME = 100
rng = np.random.default_rng(0)

def load(frame, model):
    m = next(f for f in D["frames"] if f["frame"] == frame)["models"][model]
    xyz = np.frombuffer(base64.b64decode(m["xyz_b64"]), np.int16).reshape(-1, 3).astype(float)
    cls = np.frombuffer(base64.b64decode(m["cls_b64"]), np.uint8)
    return xyz, cls, m

# One sampling RATE for all three classes so the picture preserves true class
# proportions; per-class budgets would make a precise model look full of errors.
TARGET_POINTS = 46000
SIZE  = {TP: 1.7, FP: 1.7, FN: 0.9}
ALPHA = {TP: 0.95, FP: 0.9, FN: 0.15}

def panel(ax, frame, model):
    xyz, cls, m = load(frame, model)
    rate = min(1.0, TARGET_POINTS / max(len(cls), 1))
    counts = {}
    for c in (FN, FP, TP):                        # far-to-near class painting
        idx = np.flatnonzero(cls == c)
        counts[c] = m["counts"][str(c)]["total"]
        k = int(round(len(idx) * rate))
        if k < len(idx):
            idx = rng.choice(idx, k, replace=False)
        p = xyz[idx]
        ax.scatter(p[:, 0] * 0.2, p[:, 1] * 0.2, p[:, 2] * 0.2,
                   s=SIZE[c], c=[C[c]], alpha=ALPHA[c], linewidths=0, depthshade=False)
    ax.set_box_aspect((8, 8, 1.6))
    ax.view_init(elev=26, azim=-62)
    ax.set_xlim(0, 51.2); ax.set_ylim(-25.6, 25.6); ax.set_zlim(-2, 4.4)
    ax.set_xlabel("forward (m)", fontsize=8.5, labelpad=-3)
    ax.set_ylabel("lateral (m)", fontsize=8.5, labelpad=-3)
    ax.set_zticks([]); ax.tick_params(labelsize=7.5, pad=-2); ax.grid(False)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((1, 1, 1, 0)); a.pane.set_edgecolor((.82, .82, .82, 1))
    ax.text2D(0.0, 0.10,
              f"hit {counts[TP]:,}    wrong {counts[FP]:,}    missed {counts[FN]:,}",
              transform=ax.transAxes, fontsize=8.4, color="#333", family="monospace")
    return m

fig = plt.figure(figsize=(13.6, 10.2))

# ---- explicit bands: suptitle / legend / headers / 3D / charts / footer ----
fig.text(0.5, 0.972, "Frozen LingBot-MAP reaches 36% of its own sensor ceiling on SemanticKITTI seq 08",
         ha="center", fontsize=15.5)
fig.text(0.5, 0.944, "binary occupancy  ·  256×256×32 grid @ 0.2 m  ·  causal history 20  ·  160 anchor frames",
         ha="center", fontsize=10.5, color="#444")

handles = [Line2D([], [], marker="s", ls="", ms=9, color=C[c], label=NAME[c]) for c in (TP, FP, FN)]
fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.905), ncol=3,
           frameon=False, fontsize=10.5, columnspacing=3.0, handletextpad=0.55)

gs3d = fig.add_gridspec(1, 2, left=0.03, right=0.985, top=0.845, bottom=0.415, wspace=0.03)
axA = fig.add_subplot(gs3d[0], projection="3d")
mA = panel(axA, FRAME, "lingbot")
axB = fig.add_subplot(gs3d[1], projection="3d")
mB = panel(axB, FRAME, "lidar")

for x, name, sub, m in ((0.265, "LingBot-MAP", "frozen · predicted pose · anchored scale", mA),
                        (0.745, "GT-LiDAR ceiling", "ground-truth scans · GT poses · same history", mB)):
    fig.text(x, 0.868, name, ha="center", fontsize=12.5, weight="bold")
    fig.text(x, 0.848, sub, ha="center", fontsize=9, color="#555")
    fig.text(x, 0.824, f"IoU {m['iou']:.3f}      P {m['precision']:.3f}      R {m['recall']:.3f}",
             ha="center", fontsize=11, family="monospace")

gsb = fig.add_gridspec(1, 2, left=0.055, right=0.985, top=0.335, bottom=0.125, wspace=0.20)

# -- history sweep --------------------------------------------------------- #
ax1 = fig.add_subplot(gsb[0])
h = [1, 5, 20, 100]
lb = [0.0207, 0.0523, 0.0840, 0.0869]
ld = [0.1015, 0.1803, 0.2327, 0.2460]
ax1.plot(h, ld, "o-", color=C[TP], lw=2, ms=6, label="GT-LiDAR ceiling")
ax1.plot(h, lb, "s-", color=C[FP], lw=2, ms=6, label="LingBot-MAP")
ax1.set_xscale("log"); ax1.set_xticks(h); ax1.set_xticklabels(h)
ax1.set_xlabel("causal history (frames)", fontsize=9.5)
ax1.set_ylabel("binary occupancy IoU", fontsize=9.5)
ax1.set_ylim(0, 0.275)
ax1.set_title("Longer history lifts the ceiling, not the model", fontsize=11, pad=8)
ax1.legend(fontsize=9, frameon=False, loc="upper left")
ax1.tick_params(labelsize=8.5)
for s in ("top", "right"): ax1.spines[s].set_visible(False)
for x, a, b in zip(h, lb, ld):
    if x == 1:                       # too close to the axis corner to label
        continue
    ax1.annotate(f"{100*a/b:.0f}% of ceiling", (x, a), textcoords="offset points",
                 xytext=(0, 9), ha="center", fontsize=8, color="#666")

# -- per-frame ------------------------------------------------------------- #
ax2 = fig.add_subplot(gsb[1])
frames = [100, 1000, 2500]
fl = [0.1346, 0.0173, 0.0302]
fd = [0.2145, 0.3178, 0.4097]
x = np.arange(len(frames)); w = 0.36
ax2.bar(x - w/2, fl, w, color=C[FP], label="LingBot-MAP")
ax2.bar(x + w/2, fd, w, color=C[TP], label="GT-LiDAR ceiling")
for i, (a, b) in enumerate(zip(fl, fd)):
    ax2.text(i - w/2, a + 0.009, f"{a:.3f}", ha="center", fontsize=8.5)
    ax2.text(i + w/2, b + 0.009, f"{b:.3f}", ha="center", fontsize=8.5)
ax2.set_xticks(x); ax2.set_xticklabels([f"frame {f}" for f in frames], fontsize=9)
ax2.set_ylabel("binary occupancy IoU", fontsize=9.5)
ax2.set_ylim(0, 0.47)
ax2.set_title("Per-frame: the two move in opposite directions", fontsize=11, pad=8)
ax2.legend(fontsize=9, frameon=False, loc="upper left")
ax2.tick_params(labelsize=8.5)
for s in ("top", "right"): ax2.spines[s].set_visible(False)
ax2.annotate("frame 100 is LingBot's best of the three;\nthe ceiling is at its worst there",
             xy=(0.03, 0.60), xycoords="axes fraction", fontsize=8.2, color="#666")

fig.text(0.5, 0.046,
         f"3D panels: anchor frame {FRAME}. Both panels use one uniform sampling rate, so class proportions are preserved; statistics are computed on the full volume.",
         ha="center", fontsize=8.5, color="#555")
fig.text(0.5, 0.021,
         "Metric: IoU = tp/(tp+fp+fn) over valid voxels, identical target and support for both. LingBot-MAP is frozen throughout — 1,157,943,540 parameters, never trained for this task.",
         ha="center", fontsize=8.5, color="#555")

out = "outputs/prompted_lingbot/occupancy_geometry/semantickitti/vis3d/occupancy_ceiling_figure.png"
fig.savefig(out, dpi=200, facecolor="white")
print("wrote", out)
