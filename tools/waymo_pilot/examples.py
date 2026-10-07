"""Deterministic inspectable examples: what one anchor's supervision actually looks like.

Selection is by fixed quantiles of the anchor's own yield (``U / hidden``), so the set always
contains the worst case, the median and the best case rather than only the flattering ones.
Each figure shows, left to right: the five causal views, the causal semantic evidence in
bird's-eye view, the privileged view that first reveals the target region, and the privileged
semantic target projected on the same bird's-eye grid, with provenance printed underneath.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from tools.waymo_pilot import config as C, k360 as K, populations as POP

#: fixed quantiles of per-anchor yield; always includes the worst and the best
QUANTILES: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
#: an 18-class qualitative palette, fixed so two figures are comparable
PALETTE = np.array([
    (0.84, 0.15, 0.16), (0.12, 0.47, 0.71), (1.00, 0.50, 0.05), (0.58, 0.40, 0.74),
    (0.55, 0.34, 0.29), (0.89, 0.47, 0.76), (0.50, 0.50, 0.50), (0.74, 0.74, 0.13),
    (0.09, 0.75, 0.81), (0.68, 0.78, 0.91), (1.00, 0.73, 0.47), (0.60, 0.87, 0.54),
    (1.00, 0.60, 0.59), (0.77, 0.69, 0.84), (0.77, 0.61, 0.58), (0.97, 0.71, 0.82),
    (0.78, 0.78, 0.78), (0.86, 0.86, 0.55)], np.float32)


def pick(rows: Sequence[Dict], n: int = 6) -> List[Dict]:
    """Deterministic spread over yield, then over scenes, never only the best cases."""
    scored = sorted(rows, key=lambda r: (_yield(r), r["scene"], r["anchor"]))
    if not scored:
        return []
    idx = sorted({int(round(q * (len(scored) - 1))) for q in QUANTILES})
    out = [scored[i] for i in idx]
    # top up from unrepresented scenes so several segments appear
    seen = {r["scene"] for r in out}
    for r in scored:
        if len(out) >= n:
            break
        if r["scene"] not in seen:
            out.append(r)
            seen.add(r["scene"])
    for r in scored[::max(len(scored) // max(n, 1), 1)]:
        if len(out) >= n:
            break
        if r not in out:
            out.append(r)
    return out[:n]


def _yield(r: Dict) -> float:
    c = r["counts"]
    return (c["U"] / c["hidden"]) if c.get("hidden") else 0.0


def _bev(mask: torch.Tensor, label: Optional[torch.Tensor] = None) -> np.ndarray:
    """Top-down RGB of a flat voxel mask: highest occupied z wins, coloured by class."""
    X, Y, Z = C.GRID_DIMS
    m = mask.reshape(X, Y, Z)
    img = np.ones((Y, X, 3), np.float32)
    any_z = m.any(-1)
    if label is None:
        col = np.zeros((Y, X, 3), np.float32)
        col[...] = np.array([0.15, 0.35, 0.65], np.float32)
    else:
        lab = label.reshape(X, Y, Z)
        top = torch.where(m, torch.arange(Z, device=m.device).view(1, 1, Z), -1).argmax(-1)
        cls = torch.gather(lab, 2, top.unsqueeze(-1)).squeeze(-1).clamp(1, C.N_CLASSES)
        col = PALETTE[(cls.cpu().numpy() - 1).clip(0, C.N_CLASSES - 1)]
        col = np.transpose(col, (1, 0, 2))
    a = np.transpose(any_z.cpu().numpy())
    img[a] = col[a] if label is not None else col[a]
    return np.flipud(img)


def render(rec: Dict, device, text: torch.Tensor, out_dir: str) -> Dict:
    """One figure plus its provenance sidecar. Returns the paths and the facts shown."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import image as mpimg
    from tools.delayed_semantic.stream import DriveStream

    drive, t = rec["scene"], int(rec["anchor"])
    cs = K.causal_state(drive, t, device, text)
    centres = POP.grid_centres(device)
    from tools.semantic_completion.evidence import build as G15B
    obs = POP.causal_observability(centres, np.linalg.inv(G15B.camera_to_grid()),
                                  K.causal_views(cs["clip"], device), cs["S"], cs["F"],
                                  cs["evidence_mask"])
    O_priv, _v, _m = K.privileged_occupancy(drive, t, device)
    sem = K.privileged_semantics(drive, t, device)
    pop = POP.build(O_priv, sem["valid"], obs)
    readout = torch.zeros(C.N_VOXELS, dtype=torch.long, device=device)
    readout[sem["flat"]] = (1 + sem["prob"].argmax(1)).long()

    stream = DriveStream(drive)
    causal = [stream.frames[t + o] for o in C.CLIP_OFFSETS]
    # the privileged frame that first reveals the most targets: the median first_dt of U
    keep = pop.U[sem["flat"]]
    dt = sem["first_dt_s"][keep].float()
    med_dt = float(dt.median()) if dt.numel() else 0.0
    rev_idx = min(range(len(stream)),
                  key=lambda j: abs((stream.frames[j].time_s - stream.frames[t].time_s)
                                    - med_dt)) if dt.numel() else t
    reveal = stream.frames[rev_idx]

    fig = plt.figure(figsize=(16.0, 7.2), dpi=120)
    gs = fig.add_gridspec(3, 5, height_ratios=[1.0, 1.9, 0.5], hspace=0.22, wspace=0.08)
    for i, f in enumerate(causal):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(mpimg.imread(f.image_path))
        ax.set_title(f"causal t{C.CLIP_OFFSETS[i]:+d}  ({f.time_s - stream.frames[t].time_s:+.2f} s)",
                     fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 0:2])
    ax.imshow(_bev(cs["evidence_mask"], cs["evidence_label"]))
    ax.set_title(f"causal semantic evidence (BEV)  {rec['counts']['E_causal']:,} voxels", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(mpimg.imread(reveal.image_path))
    ax.set_title(f"privileged revealing view\n{reveal.time_s - stream.frames[t].time_s:+.2f} s "
                 f"(median reveal {med_dt:+.2f} s)", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 3])
    ax.imshow(_bev(pop.hidden))
    ax.set_title(f"hidden, before a target\n{rec['counts']['hidden']:,} voxels", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 4])
    ax.imshow(_bev(pop.U, readout))
    ax.set_title(f"privileged target U\n{rec['counts']['U']:,} voxels "
                 f"({100 * _yield(rec):.1f} % of hidden)", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

    prov = rec["provenance"]
    agr = rec["agreement"]
    rvn = rec["repeat_vs_new"]
    top = sorted(rec["readout_histogram"].items(), key=lambda kv: -kv[1])[:4]
    agr_pct = ("n/a" if agr["two_view_agreement"] is None
               else f"{100 * agr['two_view_agreement']:.0f} %")
    obs_med = prov["observation_count"].get("median")
    obs_p90 = prov["observation_count"].get("p90")
    dt_med = prov["first_observation_dt_s"].get("median")
    dt_p90 = prov["first_observation_dt_s"].get("p90")
    cnt = rec["counts"]
    txt = "\n".join([
        f"{drive[17:21]} anchor {t}    source {list(prov['sources'])[0]}    "
        f"geometry {prov['geometry_source']}    camera {prov['camera']}    "
        f"tier A {prov['tier']['A_occupied_at_t']:,} / B {prov['tier']['B_occupancy_uncertain']:,}",
        f"observations per target: median {obs_med}, p90 {obs_p90}    "
        f"first reveal: median {dt_med:+.2f} s, p90 {dt_p90:+.2f} s    "
        f"two-view consistency {agr_pct} (consistency, NOT correctness)",
        f"newly supported {rvn['n_newly_supported']:,} of {rvn['n_targets']:,}; "
        f"causal camera already had a return at {rvn['n_causal_camera_had_a_return']:,}    "
        f"readout: " + ", ".join(f"{k} {v:,}" for k, v in top),
        f"sub-cases of hidden: surface-observed/teacher-silent "
        f"{cnt['hidden_surface_observed_no_teacher']:,}    outside-frustum "
        f"{cnt['hidden_outside_every_frustum']:,}    occluded "
        f"{cnt['hidden_occluded_throughout']:,}    unresolved "
        f"{cnt['hidden_unresolved_visibility']:,} (carved free "
        f"{cnt['hidden_causal_free_disagreement']:,})",
    ])
    ax = fig.add_subplot(gs[2, :]); ax.axis("off")
    ax.text(0.0, 0.95, txt, fontsize=7.4, va="top", family="monospace")

    os.makedirs(out_dir, exist_ok=True)
    stem = f"{drive[17:21]}_{t:05d}"
    png = os.path.join(out_dir, stem + ".png")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    side = {"figure": os.path.relpath(png, C.REPO_ROOT), "scene": drive, "anchor": t,
            "yield_U_over_hidden": _yield(rec), "counts": rec["counts"],
            "provenance": prov, "agreement": agr, "repeat_vs_new": rvn,
            "causal_frames": [{"index": f.index, "dt_s": f.time_s - stream.frames[t].time_s,
                               "image": os.path.relpath(f.image_path, "/")} for f in causal],
            "revealing_frame": {"index": reveal.index,
                                "dt_s": reveal.time_s - stream.frames[t].time_s,
                                "image": os.path.relpath(reveal.image_path, "/")}}
    with open(os.path.join(out_dir, stem + ".json"), "w") as fh:
        json.dump(side, fh, indent=2, default=float)
    return side


__all__ = ["QUANTILES", "pick", "render"]
