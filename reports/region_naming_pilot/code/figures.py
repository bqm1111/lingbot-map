"""Stage 6: a small set of readable figures. GT appears only in the evaluation panels.

Four cases are picked deterministically from the written predictions and scores (never by eye):
  partly_visible_object   -- queried region, modal GT an object class, seen in >= 2 frames, most grown voxels
  surface_behind_object   -- queried region, modal GT a surface class, most grown voxels the clip never observed
  region_without_seed     -- largest region with RGB support but no causal Trident evidence at all
  failure                 -- queried region costing the most correct voxels (B0 right, Q_R wrong)
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import torch

import evaluate as EV
import legacy as L
import manifest as MF
import paths as P
import predict as PD
import proposals as PR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.colors import ListedColormap                        # noqa: E402

NAMES = ["empty", "car", "bicycle", "motorcycle", "truck", "other vehicle", "person", "road", "parking",
         "sidewalk", "other ground", "building", "fence", "vegetation", "terrain", "pole", "traffic sign",
         "other structure", "other object"]
_C = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
CMAP = ListedColormap(np.vstack([[0.08, 0.08, 0.1, 1.0], _C[:18]]))


def bev(flat: torch.Tensor, occ: torch.Tensor) -> np.ndarray:
    """Top-down view: the label of the highest occupied voxel of each column, 0 where no column is occupied."""
    v = (flat * occ).reshape(L.GRID_DIMS).cpu().numpy()
    o = occ.reshape(L.GRID_DIMS).cpu().numpy()
    top = np.where(o.any(2), (o.shape[2] - 1) - o[:, :, ::-1].argmax(2), 0)
    ix, iy = np.meshgrid(np.arange(o.shape[0]), np.arange(o.shape[1]), indexing="ij")
    return np.where(o.any(2), v[ix, iy, top], 0)


def bev_panel(t: int, pr: Dict, sem_gt: torch.Tensor, out: str) -> str:
    O = pr["O"]
    mem = pr["member"]
    panels = [("frozen occupancy O", (O.long() * 7), O),
              ("region membership", (mem.clamp(min=0) % 18 + 1) * (mem >= 0), O & (mem >= 0)),
              ("B0 (deployed)", pr["labels"]["B0"], O),
              ("P_R pooled Trident + region", pr["labels"]["P_R"], O),
              ("Q_R Qwen + region", pr["labels"]["Q_R"], O),
              ("human labels (evaluation only)", sem_gt.clamp(0, 18), (sem_gt >= 1) & (sem_gt <= 18))]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.2), constrained_layout=True)
    for a, (title, v, m) in zip(ax.ravel(), panels):
        a.imshow(bev(v, m).T, origin="lower", cmap=CMAP, vmin=0, vmax=18, interpolation="nearest")
        a.set_title(title, fontsize=10); a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"anchor t={t}, drive {P.DRIVE} — bird's-eye view, label of the highest occupied voxel", fontsize=11)
    fig.savefig(P.under_experiment(out), dpi=120); plt.close(fig)
    return out


def pick_cases(ids: List[int], device) -> Dict:
    """Deterministic case selection over the evaluation anchors."""
    best = {k: None for k in ("partly_visible_object", "surface_behind_object", "region_without_seed", "failure")}
    for t in ids:
        pr = PD.load(t, device)
        prop = PR.load(t)
        sem_gt = L.load_sem_gt(t, device)
        reg = EV.regions_of(sem_gt, pr["E"], pr["O"], pr["S"], pr["F"])
        prim = reg[L.PRIMARY]
        obs = pr["S"] | pr["F"]
        rr = {r["region_id"]: r for r in prop["regions"]}
        rows = EV.region_recognition(t, pr, sem_gt, device)
        mem, member_ok = pr["member"], (pr["member"] >= 0) & ~pr["contested"]
        seeds = torch.from_numpy(prop["voxel_region"].astype(np.int64)).to(device) >= 0
        rb0, rq = pr["labels"]["B0"] == sem_gt, pr["labels"]["Q_R"] == sem_gt
        for row in rows:
            r = row["region_id"]
            grown = member_ok & (mem == r) & ~seeds
            cand = {"t": t, "region": r, "row": row, "n_grown": int(grown.sum()),
                    "n_grown_unobserved": int((grown & ~obs).sum()),
                    "cost": int((prim & (mem == r) & rb0 & ~rq).sum())}
            if row["modal_gt"] in EV.OBJECT_CLASSES and row["n_frames"] >= 2:
                if best["partly_visible_object"] is None or cand["n_grown"] > best["partly_visible_object"]["n_grown"]:
                    best["partly_visible_object"] = cand
            if row["modal_gt"] in EV.SURFACE_CLASSES:
                if best["surface_behind_object"] is None or cand["n_grown_unobserved"] > best["surface_behind_object"]["n_grown_unobserved"]:
                    best["surface_behind_object"] = cand
            if best["failure"] is None or cand["cost"] > best["failure"]["cost"]:
                best["failure"] = cand
        noseed = [r for r in prop["regions"] if r["n_evidence_voxels"] == 0]
        if noseed:
            top = max(noseed, key=lambda r: r["best_area"])
            cand = {"t": t, "region": top["region_id"], "row": None, "n_grown": 0,
                    "n_grown_unobserved": 0, "cost": 0, "area": top["best_area"]}
            if best["region_without_seed"] is None or cand["area"] > best["region_without_seed"]["area"]:
                best["region_without_seed"] = cand
    return best


def case_figure(cases: Dict, device, out: str) -> str:
    import recognize as RC
    rows = [(k, v) for k, v in cases.items() if v]
    fig, ax = plt.subplots(len(rows), 3, figsize=(16, 3.1 * len(rows)), constrained_layout=True)
    ax = np.atleast_2d(ax)
    for i, (name, c) in enumerate(rows):
        prop = PR.load(c["t"])
        reg = next(r for r in prop["regions"] if r["region_id"] == c["region"])
        imgs = [PR.native_image(k) for k in prop["stats"]["keys"]]
        views = RC.region_views(reg, prop["label_native"], imgs)
        for j in range(2):
            ax[i, j].imshow(np.asarray(views[j])); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        ax[i, 0].set_title(f"{name}  (t={c['t']}, region {c['region']})", fontsize=9, loc="left")
        ax[i, 1].set_title(f"frames {reg['frames']}, {reg['n_evidence_voxels']} evidence voxels", fontsize=9, loc="left")
        row = c["row"]
        if row:
            qrec = json.load(open(os.path.join(P.QWEN_CACHE, f"{c['t']:05d}.json")))["regions"].get(str(c["region"]), {})
            txt = (f"pooled Trident : {NAMES[row['p_label']]}\n"
                   f"Qwen           : {NAMES[row['q_label']]}   ({row['outcome']})\n"
                   f"human modal    : {NAMES[row['modal_gt']] if row['modal_gt'] else 'unknown'}"
                   f"   purity {row['purity']:.2f}\n" if row["modal_gt"] else
                   f"pooled Trident : {NAMES[row['p_label']]}\nQwen           : {NAMES[row['q_label']]}\nhuman modal    : unknown\n")
            txt += (f"grown voxels   : {c['n_grown']} ({c['n_grown_unobserved']} never observed)\n"
                    f"correct voxels lost vs B0 : {c['cost']}\n"
                    f'Qwen said      : "{str(qrec.get("parsed", {}).get("category"))}"')
        else:
            txt = (f"no causal Trident evidence at all\n{reg['best_area']} owned pixels, "
                   f"frames {reg['frames']}\nthis region can seed no name under either recognizer")
        ax[i, 2].axis("off")
        ax[i, 2].text(0, 0.95, txt, va="top", family="monospace", fontsize=8.5)
    fig.savefig(P.under_experiment(out), dpi=120); plt.close(fig)
    return out


def recognition_figure(analysis: Dict, out: str) -> str:
    """Region-level recognition: P and Q against the modal human class, plus the per-anchor paired deltas."""
    rows = [r for v in analysis["recognition"].values() for r in v if r["modal_gt"]]
    cm = {k: np.zeros((19, 19)) for k in ("p", "q")}
    for r in rows:
        cm["p"][r["modal_gt"], r["p_label"]] += 1
        cm["q"][r["modal_gt"], r["q_label"]] += 1
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)
    for a, k, title in ((ax[0], "p", "pooled Trident"), (ax[1], "q", "Qwen-VL")):
        n = cm[k][1:, 1:]
        a.imshow(n / np.maximum(n.sum(1, keepdims=True), 1), cmap="magma", vmin=0, vmax=1)
        acc = np.trace(cm[k]) / max(len(rows), 1)
        a.set_title(f"{title} — region accuracy {100*acc:.1f}% ({len(rows)} regions)", fontsize=10)
        a.set_xticks(range(18)); a.set_xticklabels(NAMES[1:], rotation=90, fontsize=6)
        a.set_yticks(range(18)); a.set_yticklabels(NAMES[1:], fontsize=6)
        a.set_xlabel("predicted"); a.set_ylabel("modal human class")
    pa = analysis["per_anchor"]
    ev = sorted(int(t) for t in pa if pa[t]["miou"]["B0"] is not None)
    ev = [t for t in ev if t in MF.select()["evaluation"]]
    x = np.arange(len(ev))
    for m, c in (("P_N", "tab:blue"), ("Q_N", "tab:orange"), ("P_R", "tab:green"), ("Q_R", "tab:red")):
        d = [100 * (pa[str(t)]["miou"][m] - pa[str(t)]["miou"]["B0"]) for t in ev]
        ax[2].plot(x, d, "o-", ms=3, lw=1, color=c, label=f"{m} − B0")
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_title("per-anchor mIoU change on the primary population", fontsize=10)
    ax[2].set_xlabel("evaluation anchor (one per temporal block)"); ax[2].set_ylabel("Δ mIoU (points)")
    ax[2].legend(fontsize=8)
    fig.savefig(P.under_experiment(out), dpi=120); plt.close(fig)
    return out


def main(device: str = "cuda:0") -> Dict:
    a = MF.select()
    analysis = json.load(open(os.path.join(P.ARTIFACTS, "analysis.json")))
    ids = [t for t in a["evaluation"] if os.path.exists(os.path.join(P.PREDICTIONS, f"{t:05d}.npz"))]
    made = {}
    cases = pick_cases(ids, device)
    with open(P.under_experiment(os.path.join(P.ARTIFACTS, "figure_cases.json")), "w") as f:
        json.dump(cases, f, indent=2, default=str)
    made["cases"] = case_figure(cases, device, os.path.join(P.FIGURES, "cases.png"))
    made["recognition"] = recognition_figure(analysis, os.path.join(P.FIGURES, "recognition.png"))
    t = ids[len(ids) // 2]
    pr = PD.load(t, device)
    made["bev"] = bev_panel(t, pr, L.load_sem_gt(t, device), os.path.join(P.FIGURES, f"bev_{t:05d}.png"))
    print(json.dumps(made, indent=1))
    return made


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
