"""Gate 17 qualitative examples, deterministically selected.

Selection rule, fixed before looking at any picture: anchors are ranked by their own (oracle A - B0) mIoU on P2 and
sampled at fixed quantiles, so the set spans the range including anchors where the oracle HURTS. Nothing is curated.

Panels per row: input image, GT semantics, B0, oracle instance, the oracle grouping itself, and which voxels the
oracle could reach at all (its coverage). Colours are the benchmark's own class colours; the instance panel uses an
arbitrary repeatable colour per component.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from gates.gate8c1 import render3d as R3  # noqa: E402
from tools.delayed_semantic import config as C  # noqa: E402
from tools.delayed_semantic.regions import build_regions  # noqa: E402
from tools.delayed_semantic.stream import DriveStream  # noqa: E402
from tools.gate8c1.figure_3d_compare import C_MISS, GHOST, SOLID  # noqa: E402
from tools.instance_oracle import config as G, oracle as OR  # noqa: E402
from tools.instance_oracle.analysis import load, miou  # noqa: E402
from tools.instance_oracle.evaluate import causal_state  # noqa: E402
from tools.occany_comparison import config as OC  # noqa: E402
from tools.semantic_completion.evidence import trident as TR  # noqa: E402
from tools.semantic_completion.frozen_geometry import load_frozen_geometry  # noqa: E402

QUANTILES = (0.05, 0.35, 0.65, 0.95)
PANELS = ("0 · input image", "1 · ground-truth semantics", "2 · B0 nearest evidence",
          "3 · oracle instance", "4 · oracle instance grouping", "5 · oracle coverage")
VIEW = dict(eye_offset=(-30.0, -24.0, 26.0), fov=46.0)
C_COVERED = (0.16, 0.42, 0.85)


def palette() -> np.ndarray:
    """KITTI-360 / SemanticKITTI class colours, index = sem label (0 = empty)."""
    spec = yaml.safe_load(open(OC.OCCANY_SK_LEARNING_MAP))
    inv, col = spec["learning_map_inv"], spec["color_map"]
    return np.asarray([[0, 0, 0]] + [np.asarray(col[inv[c]], float)[::-1] / 255.0 for c in range(1, 19)])


def instance_colours(n: int) -> np.ndarray:
    rng = np.random.default_rng(G.SEED)
    return rng.uniform(0.25, 0.95, size=(n + 1, 3))


def class_meshes(labels: np.ndarray, mask: np.ndarray, pal: np.ndarray, shrink: float = SOLID) -> List:
    out = []
    for c in np.unique(labels[mask]):
        idx = np.argwhere((mask & (labels == c)).reshape(OR.DIMS))
        if len(idx):
            out.append(R3.cube_mesh(idx, np.tile(pal[int(c)], (len(idx), 1)), C.VOXEL_SIZE_M, C.GRID_ORIGIN_M, shrink=shrink))
    return out


def flat_mesh(mask: np.ndarray, colour, shrink: float = SOLID):
    idx = np.argwhere(mask.reshape(OR.DIMS))
    return None if not len(idx) else R3.cube_mesh(idx, np.tile(colour, (len(idx), 1)), C.VOXEL_SIZE_M, C.GRID_ORIGIN_M, shrink=shrink)


def pick() -> List[Dict]:
    d = load()
    a = d["conf__C0_b0__P2_correct_completion"]; b = d["conf__A_oracle_instance__P2_correct_completion"]
    delta = np.array([miou(b[i]) - miou(a[i]) for i in range(len(a))])
    ok = np.isfinite(delta)
    ids = d["ids"][ok]; delta = delta[ok]
    order = np.argsort(delta, kind="stable")
    out = []
    for q in QUANTILES:
        r = int(round(q * (len(order) - 1))); i = int(order[r])
        out.append({"t": int(ids[i]), "delta_miou": float(delta[i]), "rank_pct": 100.0 * r / (len(order) - 1)})
    return out


def render(device: str = "cuda:0") -> Dict:
    paths = C.Paths(); dev = torch.device(device); torch.cuda.set_device(dev)
    text, _, _ = TR.load_text_embeddings(paths.g15.text_embeddings); text = text.to(dev)
    stream = DriveStream(G.DEV_DRIVE); net = load_frozen_geometry(C.GEOMETRY_SEED, dev, paths.g15)
    pal = palette()
    rows, picks = [], pick()
    for pk in picks:
        t = pk["t"]
        with np.load(paths.g15.eval_mask_path(t)) as z:
            sg = z["sem_gt"].astype(np.int64); vd = np.unpackbits(z["valid"])[:C.N_VOXELS].astype(bool)
        st = causal_state(stream, t, net, text, dev, paths)
        inst_np, _ = OR.instance_map(sg, vd)
        n_inst = int(inst_np.max())
        sem_gt = torch.from_numpy(sg).to(dev)
        E = st["E"]
        stats = OR.instance_stats(inst_np, E.cpu().numpy() & ((sg >= 1) & (sg <= 18) & vd), n_inst)
        inst = torch.from_numpy(inst_np.astype(np.int64)).to(dev)
        sem = OR.instance_semantics(inst, st["ev"].flat_index, st["ev"].feature, st["ev_label"], n_inst, text, dev)
        gen = torch.Generator(device=dev).manual_seed(G.SEED * 100003 + t)
        labs = OR.method_labels(st["b0"], inst, sem, sem_gt, gen)
        covered = labs.pop("_covered").cpu().numpy()
        R = torch.zeros(C.N_VOXELS, dtype=torch.bool, device=dev)
        reg = build_regions(sem_gt, E, st["Ec"], st["S"], st["F"], st["O"], R, R)
        P2 = reg["completed_never_observed"].cpu().numpy()
        gt_np = sg
        ic = instance_colours(n_inst)
        scenes = [class_meshes(gt_np, P2, pal),
                  class_meshes(labs["C0_b0"].cpu().numpy(), P2, pal),
                  class_meshes(labs["A_oracle_instance"].cpu().numpy(), P2, pal),
                  class_meshes(inst_np, P2, ic),
                  [flat_mesh(P2 & ~covered, C_MISS, GHOST), flat_mesh(P2 & covered, C_COVERED)]]
        ctr = ((np.argwhere(P2.reshape(OR.DIMS)) + 0.5) * C.VOXEL_SIZE_M + np.asarray(C.GRID_ORIGIN_M)).mean(0)
        imgs = [plt.imread(stream.frames[t].image_path)]
        imgs += [R3.render([m for m in ms if m is not None], size=(900, 660), look_at=ctr, **VIEW) for ms in scenes]
        rows.append({"images": imgs, "label": "t=%05d  delta mIoU %+.2f  (rank %.0f%%)  coverage %.0f%%"
                     % (t, pk["delta_miou"], pk["rank_pct"], 100 * covered[P2].mean() if P2.any() else 0)})
        print("  rendered t=%05d" % t, flush=True)

    n, cols = len(rows), len(PANELS)
    fig = plt.figure(figsize=(5.6 * cols, 4.6 * n + 2.2))
    grid = fig.add_gridspec(n + 1, cols, height_ratios=[4.6] * n + [1.6], top=0.94, bottom=0.02,
                            left=0.005, right=0.995, hspace=0.18, wspace=0.02)
    for r, row in enumerate(rows):
        for c, im in enumerate(row["images"]):
            ax = fig.add_subplot(grid[r, c]); ax.imshow(im); ax.set_axis_off(); ax.set_title(PANELS[c], fontsize=11)
            if c == 0:
                ax.text(0.0, -0.06, row["label"], transform=ax.transAxes, fontsize=9, va="top",
                        bbox=dict(fc="w", ec="0.7", alpha=.9))
    ax = fig.add_subplot(grid[n, :]); ax.set_axis_off()
    handles = [Patch(color=C_COVERED, label="oracle reached it (its instance has visible evidence)"),
               Patch(color=C_MISS, label="oracle could not reach it (instance never observed) -- falls back to B0")]
    handles += [Patch(color=pal[c], label=G.CLASS_NAMES[c - 1]) for c in range(1, 19)]
    ax.legend(handles=handles, loc="upper center", ncol=7, fontsize=10, frameon=False)
    fig.suptitle("Gate 17: oracle instance grounding, KITTI-360 drive 0006, completed-unobserved voxels", fontsize=16)
    os.makedirs(G.ARTIFACT_DIR, exist_ok=True)
    out = os.path.join(G.ARTIFACT_DIR, "qualitative.png")
    fig.savefig(out, dpi=95, bbox_inches="tight"); plt.close(fig)
    json.dump({"selection_rule": "anchors at fixed quantiles of per-anchor (oracle A - B0) mIoU on P2",
               "quantiles": list(QUANTILES), "picks": picks, "figure": out},
              open(os.path.join(G.ARTIFACT_DIR, "qualitative_index.json"), "w"), indent=1)
    print("wrote", out)
    return {"figure": out, "picks": picks}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0")
    print(json.dumps(render(ap.parse_args().device), indent=1))
