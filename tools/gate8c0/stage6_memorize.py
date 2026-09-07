#!/usr/bin/env python
"""Gate 8C-0 Stage 6: can the unchanged completion model memorize a fixed KITTI-360 batch?

Two deterministic batches of four KITTI-360 *training-drive* samples, from different
locations, with **fixed crop origins** and no stochastic augmentation. Architecture,
inputs, target definition and loss are exactly Gate 8A cell B. If occupancy AP on the
editable region does not approach 1 here, the failure is not generalisation.

The reported diagnostics separate the possible causes the brief lists: gradient norms
catch an optimisation failure; the supervised-mask prevalence and editable fraction catch
a masking problem; the input-distance / target-disagreement pair catches information
collision and contradictory targets.

    python tools/gate8c0/stage6_memorize.py --device cuda:1
"""
from __future__ import annotations
import argparse, glob, os, sys, time
import numpy as np, torch, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, cfg, default_device                          # noqa: E402
from gates.scale_gate.config import write_json                                         # noqa: E402
from gates.gate8 import targets as TG                                                  # noqa: E402
from gates.gate8.net import CompletionUNet, apply_residual                             # noqa: E402
from gates.gate8a import scores as SC                                                  # noqa: E402
from gates.gate8a.regions import editable_native                                       # noqa: E402
from gates.gate8b.sources import G8B_ROOT                                              # noqa: E402
from tools.gate8a.train import step_loss                                         # noqa: E402


def fixed_batch(files, crop, dev, origins):
    out = {k: [] for k in ("input", "gt_occ", "gt_valid", "observed", "fut_p", "fut_valid",
                           "base_logodds")}
    for p, (x0, y0, z0) in zip(files, origins):
        with np.load(p) as z:
            d = TG.unpack_sample(z, dev)
        cx, cy, cz = crop
        sl = (slice(x0, x0 + cx), slice(y0, y0 + cy), slice(z0, z0 + cz))
        out["input"].append(d["input"][(slice(None),) + sl])
        for k in ("gt_occ", "gt_valid", "observed", "fut_valid", "base_logodds"):
            out[k].append(d[k][sl])
        out["fut_p"].append(d["fut_p"][sl].permute(3, 0, 1, 2))
    return {k: torch.stack(v) for k, v in out.items()}


@torch.no_grad()
def diagnose(net, b, cfg_d):
    occ_res, sem_log = net(b["input"])
    final = apply_residual(b["base_logodds"], occ_res[:, 0].float())
    edit = editable_native(b["base_logodds"], cfg_d["lock_logodds"])
    valid = b["gt_valid"]
    gt = b["gt_occ"] > 0
    out = {}
    for nm, mask in (("full", valid), ("edit", valid & edit)):
        lg = final[mask]; y = gt[mask]
        if lg.numel() == 0 or y.sum() == 0 or (~y).sum() == 0:
            out[nm] = {"ap": float("nan"), "prevalence": float("nan")}
            continue
        bi = SC.bin_index(lg)
        pos = torch.bincount(bi[y], minlength=SC.N_BINS).cpu().numpy()[None]
        tot = torch.bincount(bi, minlength=SC.N_BINS).cpu().numpy()[None]
        sw = SC.sweep(pos, tot - pos)
        prev = float(sw["n_pos"] / sw["n_tot"]); ap = SC.average_precision(sw)
        j = int(np.argmax(sw["iou"]))
        at0 = SC.at_threshold(sw, 0.0)
        out[nm] = {"ap": ap, "prevalence": prev, "ap_over_prevalence": ap / max(prev, 1e-9),
                   "auroc": SC.auroc(sw),
                   "best_iou": float(sw["iou"][j]), "best_threshold": float(sw["tau"][j]),
                   "iou_at_zero": at0["iou"], "precision_at_zero": at0["precision"],
                   "recall_at_zero": at0["recall"], "density_at_zero": at0["density"],
                   "n_voxels": int(sw["n_tot"]), "n_positive": int(sw["n_pos"])}
    out["editable_fraction_of_valid"] = float((valid & edit).sum() / max(valid.sum(), 1))
    out["valid_fraction"] = float(valid.float().mean())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()
    c = cfg()
    cfg_d = yaml.safe_load(open(os.path.join(REPO_ROOT, "configs/gate8a/cellB_uniform_focal.yaml")))
    steps = a.steps or c["memorize_steps"]
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    torch.manual_seed(c["seed"]); np.random.seed(c["seed"])
    crop = tuple(cfg_d["crop"])
    allf = sorted(glob.glob(f"{G8B_ROOT}/samples/k360_train/*.npz"))
    by_drive = {}
    for p in allf:
        by_drive.setdefault(os.path.basename(p).rsplit("_", 1)[0], []).append(p)
    d0 = sorted(by_drive)[0]; d1 = sorted(by_drive)[-1]
    # batch A: four consecutive-ish anchors early in one drive; batch B: a different drive
    # and a different part of it. Both fixed, no resampling, no augmentation.
    batches = {"A": by_drive[d0][10:40:8][:4], "B": by_drive[d1][-40:-8:8][:4]}
    dims = None
    with np.load(batches["A"][0]) as z:
        dims = tuple(int(x) for x in z["dims"])
    origins = [(int(0.25 * (dims[0] - crop[0])), int(0.5 * (dims[1] - crop[1])), 0),
               (int(0.50 * (dims[0] - crop[0])), int(0.5 * (dims[1] - crop[1])), 0),
               (int(0.25 * (dims[0] - crop[0])), int(0.3 * (dims[1] - crop[1])), 0),
               (int(0.50 * (dims[0] - crop[0])), int(0.7 * (dims[1] - crop[1])), 0)]
    out = {"config": "configs/gate8a/cellB_uniform_focal.yaml", "steps": steps, "crop": list(crop),
           "fixed_crop_origins": origins, "seed": c["seed"], "batches": {}, "runs": {}}
    for name, files in batches.items():
        out["batches"][name] = [os.path.basename(p) for p in files]
        b = fixed_batch(files, crop, dev, origins)
        # information-collision / contradictory-target diagnostics on the fixed batch
        X = b["input"].reshape(len(files), -1)
        Y = (b["gt_occ"] > 0).reshape(len(files), -1).float()
        V = b["gt_valid"].reshape(len(files), -1)
        dx = torch.cdist(X, X); dy = torch.cdist(Y, Y)
        pair = []
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                m = V[i] & V[j]
                pair.append({"i": i, "j": j, "input_l2": float(dx[i, j]),
                             "target_l2": float(dy[i, j]),
                             "target_disagreement_on_common_valid":
                                 float((Y[i][m] != Y[j][m]).float().mean()) if m.any() else None})
        out["batches"][name + "_pairs"] = pair
        net = CompletionUNet(width=cfg_d["width"]).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg_d["lr"],
                                weight_decay=cfg_d["weight_decay"])
        log, t0 = [], time.time()
        for it in range(1, steps + 1):
            net.train()
            loss, m = step_loss(net, b, cfg_d)
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), cfg_d["grad_clip"]))
            opt.step()
            if it % c["memorize_log_every"] == 0 or it == 1 or it == steps:
                net.eval(); dg = diagnose(net, b, cfg_d)
                row = {"step": it, "loss": loss.item(), "focal": m["focal"], "dice": m["dice"],
                       "sem_kl": m["sem_kl"], "grad_norm": gn,
                       "supervised_positive_prevalence": dg["full"]["prevalence"],
                       "editable_fraction_of_valid": dg["editable_fraction_of_valid"],
                       "valid_fraction": dg["valid_fraction"],
                       "full": dg["full"], "edit": dg["edit"]}
                log.append(row)
                e = dg["edit"]
                print(f"  [{name} {it:5d}/{steps}] loss {loss.item():.4f} focal {m['focal']:.4f} "
                      f"kl {m['sem_kl']:.4f} |grad| {gn:.3f} || edit AP {e['ap']:.4f} "
                      f"(x{e['ap_over_prevalence']:.2f}) bestIoU {e['best_iou']:.4f}"
                      f"@{e['best_threshold']:+.2f} IoU@0 {e['iou_at_zero']:.4f} "
                      f"P {e['precision_at_zero']:.4f} R {e['recall_at_zero']:.4f} "
                      f"dens {e['density_at_zero']:.4f}", flush=True)
        last = log[-1]
        out["runs"][name] = {"log": log, "seconds": time.time() - t0,
                             "peak_gpu_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30,
                             "final": last,
                             "memorized": bool(last["edit"]["ap"] > 0.9
                                               and last["edit"]["best_iou"] > 0.7),
                             "gradients_healthy": bool(all(
                                 np.isfinite(r["grad_norm"]) and r["grad_norm"] > 1e-6
                                 for r in log))}
    A, B = out["runs"]["A"], out["runs"]["B"]
    out["verdict"] = {
        "memorization_passes": bool(A["memorized"] and B["memorized"]),
        "gradients_healthy": bool(A["gradients_healthy"] and B["gradients_healthy"]),
        "final_edit_ap": {"A": A["final"]["edit"]["ap"], "B": B["final"]["edit"]["ap"]},
        "final_edit_best_iou": {"A": A["final"]["edit"]["best_iou"],
                                "B": B["final"]["edit"]["best_iou"]},
        "min_pairwise_input_l2": min(p["input_l2"] for n in ("A", "B")
                                     for p in out["batches"][n + "_pairs"]),
        "max_pairwise_target_disagreement": max(
            p["target_disagreement_on_common_valid"] or 0.0 for n in ("A", "B")
            for p in out["batches"][n + "_pairs"])}
    write_json(os.path.join(ART, "stage6_memorize.json"), out)
    print(f"\nstage 6 verdict: {out['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
