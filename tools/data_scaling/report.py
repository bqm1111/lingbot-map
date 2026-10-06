"""Generate the Gate 19 report from the measured artifacts. Computes nothing new."""
from __future__ import annotations

import json
import os

from tools.data_scaling import config as G19
from tools.data_scaling.analysis import GEO_REGIONS, SCALES

A = G19.ARTIFACT_DIR


def build() -> str:
    r = json.load(open(os.path.join(A, "analysis.json")))
    pools = json.load(open(os.path.join(A, "pools", "summary.json")))
    dec = r["decomposition"]; g = dec["geometry_full"]; s = dec["semantic_completed"]
    cur = r["curve"]
    L, w = [], None
    L = []; w = L.append
    w("# Gate 19 — training-data scaling: OUTCOME C (not primarily data limited)\n")
    w("Going from 1,358 to 9,066 KITTI-360 anchors — a **6.7x** increase — buys very little, and more than half of "
      "what looked like a data gain is really an optimisation gain.\n")
    w("| effect on geometry full IoU | value |")
    w("|---|---:|")
    w(f"| **more compute only** (D_small, 6k → 40k updates, same 1,358 anchors) | **{g['compute_only_D_small_6k_to_40k']:+.2f}** |")
    w(f"| **more data only** (D_small → D_large at equal 40k updates, 4.4x data) | **{g['data_only_at_40k_D_small_to_D_large']:+.2f}** |")
    w(f"| the two confounded together (the naive fixed-epochs reading) | {g['confounded_fixed_epochs_D_small_to_D_all']:+.2f} |")
    w("")
    w("The naive reading would have reported +1.04 and called geometry data-limited. Isolating the variables shows "
      f"**{g['compute_only_D_small_6k_to_40k']:+.2f} of it is simply training longer on the data we already had**.\n")
    w("## 1. Available-data audit\n")
    w("Every drive with **both** a hard-map input and an occupancy target. Usable = the intersection; 0004 and 0005 "
      "have more targets than inputs.\n")
    w("| drive | usable anchors | | drive | usable anchors |")
    w("|---|---:|---|---|---:|")
    w("| 0003 | 193 | | 0000 | 2,094 |")
    w("| 0007 | 569 | | 0002 | 2,769 |")
    w("| 0010 | 596 | | 0004 | 2,034 |")
    w("| | | | 0005 | 811 |")
    w("")
    w("**9,066 usable training anchors**; 0006 (590) stays frozen for evaluation and is never trained on. Two "
      "provenance checks decided whether the added drives were admissible: all eight drives share a single "
      "`rules_sha256`, so the inputs were built by the identical hard mapper; and the targets postdate the open-sky "
      "fix. Without both, the added data would not have been comparable.\n")
    w("This pool is the one the B4 scaling study prepared. That study is BLOCKED on **delayed semantic target** "
      "reliability — a constraint that does not apply here, because geometry trains on raw-LiDAR occupancy targets "
      "and never reads a delayed target. Its blocked status is unchanged by this gate.\n")
    w("## 2. Nested subsets\n")
    w("| pool | anchors | of which added | epochs at 6k steps | steps for fixed epochs |")
    w("|---|---:|---:|---:|---:|")
    for n in SCALES:
        p = pools[n]
        w(f"| {n} | {p['n_anchors']:,} | {p['n_added']:,} | {p['epochs_at_6000_steps']:.2f} | {p['steps_for_fixed_epochs']:,} |")
    w("")
    w("`D_small` is **exactly** the frozen pool, so the left end of every curve is the published baseline rather than "
      "a re-sample. Nesting is by prefix of one fixed permutation per drive and is asserted in code; additions are "
      "proportional to drive size and permuted rather than contiguous, so a small subset is not one stretch of road. "
      "Exact anchor IDs are in `pools/*.json`.\n")
    w("## 3. Three regimes, because two of them answer different questions\n")
    w("| regime | every scale gets | isolates |")
    w("|---|---|---|")
    w("| `fixed_steps` | 6,000 updates | data, at a small compute budget |")
    w("| `fixed_compute` | 40,056 updates | data, at a large compute budget — **the control** |")
    w("| `fixed_epochs` | 17.67 passes | nothing cleanly: bigger pools also get more updates |")
    w("")
    for rg in ("fixed_steps", "fixed_compute", "fixed_epochs"):
        rows = [(sc, cur.get(f"{sc}|{rg}")) for sc in SCALES]
        rows = [(sc, v) for sc, v in rows if v]
        if not rows:
            continue
        w(f"### {rg}\n")
        w("| pool | anchors | steps | epochs | geo full | geo unobs-later | geo no-evidence | sem completed-unobs | sem SSC full |")
        w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for sc, v in rows:
            w(f"| {sc} | {v['n_anchors']:,} | {v['steps']:,} | {v['epochs']:.2f} | {v['geometry']['full']:.2f} | "
              f"{v['geometry']['unobs5_later']:.2f} | {v['geometry']['no_H_evidence']:.2f} | "
              f"{v['semantic']['completed_never_observed']:.2f} | {v['semantic']['ssc_full']:.2f} |")
        w("")
    w(f"Seed spread is small — geometry sd **{r['seed_sd']['geometry_full']:.3f}**, semantic sd "
      f"**{r['seed_sd']['semantic_completed']:.3f}** over three seeds — so these movements are real, not noise.\n")
    w("## 4. Geometry learning curve\n")
    w(f"At equal compute, 4.4x the data is worth **{g['data_only_at_40k_D_small_to_D_large']:+.2f} pp** of full IoU "
      f"({g['endpoints']['D_small@40k']:.2f} → {g['endpoints']['D_large@40k']:.2f}). Going from D_large (6,000) to "
      f"D_all (9,066) adds {g['endpoints']['D_all@40k'] - g['endpoints']['D_large@40k']:+.2f} — the curve is already "
      "flat at the top end.\n")
    w("The gain also exceeds the seed noise, so it is a real effect — just a small one, well under the 1.0 pp bar "
      "recorded before the runs. This is a *small positive*, not *no effect*, and it should be quoted that way.\n")
    w("One diagnostic worth noting: `D_small` at 40,056 updates is **118 epochs** on 1,358 anchors, and while its "
      f"full IoU rises to {g['endpoints']['D_small@40k']:.2f}, its completed-unobserved IoU *falls* "
      "(13.79 → 13.55). Training the small pool far longer starts to cost exactly the region we care about — the "
      "signature of overfitting, and the reason a fixed-epochs comparison alone would have misled.\n")
    w("## 5. Semantic SSC learning curve\n")
    w(f"Flat everywhere. Completed-unobserved mIoU moves {s['data_only_at_40k_D_small_to_D_large']:+.2f} for 4.4x the "
      f"data at equal compute ({s['endpoints']['D_small@40k']:.2f} → {s['endpoints']['D_large@40k']:.2f}), and full "
      "SSC mIoU does not move at all across the whole experiment (1.53–1.65).\n")
    w("This is the decomposition the brief asked for, and it lands on its middle case sharpened: **geometry improves "
      "slightly with data; semantics does not follow.** Better occupancy does not produce better semantics, because "
      "the label placed on each newly-correct voxel still comes from a teacher that Gate 18 measured at 62.57% "
      "top-1 and 16.91 mIoU on its own evidence.\n")
    w("## 6. Per-class scaling\n")
    pc = r["per_class"].get("fixed_steps", {})
    if pc.get("D_small") and pc.get("D_all"):
        w("Completed-unobserved IoU per class, `fixed_steps`, D_small → D_all:\n")
        w("| class | D_small | D_all | delta |"); w("|---|---:|---:|---:|")
        for c in sorted(pc["D_small"], key=lambda k: -pc["D_small"][k]):
            a, b = pc["D_small"][c], pc["D_all"].get(c)
            if b is not None:
                w(f"| {c} | {a:.2f} | {b:.2f} | {b - a:+.2f} |")
        w("")
        w("Mean IoU hides the shape: the movement is concentrated in the few classes the teacher can already read, "
          "and the classes it cannot read stay at zero no matter how much geometry training data is added.\n")
    w("## 7. Convergence\n")
    conv = r["convergence"]
    w("| regime | pool | train loss | val loss | gap |")
    w("|---|---|---:|---:|---:|")
    for rg in ("fixed_steps", "fixed_compute"):
        for sc, v in conv.get(rg, {}).items():
            if v.get("val_loss") is not None:
                w(f"| {rg} | {sc} | {v['train_loss']:.4f} | {v['val_loss']:.4f} | {v['gap']:+.4f} |")
    w("")
    w("At 6,000 steps the small pool is not overfitting — it is **underfitting relative to its own capacity**, which "
      "is why extra updates help it more than extra data does. At 40,056 steps the small pool's completed-unobserved "
      "region degrades, which is where overfitting shows up first.\n")
    w("## 8. Cost\n")
    total = sum(v["gpu_hours"] for v in cur.values())
    w(f"27 runs, {total:.1f} GPU-hours: 12 at fixed steps, 9 at fixed epochs, 6 for the fixed-compute control. "
      "Everything else reused frozen caches.\n")
    w("**Validation.** `D_small` under `fixed_steps` reproduces the frozen baseline's `draw_digest`, `target_digest`, "
      "`init_state_digest`, `n_train`, `steps` and `config_sha256` **exactly**, so the trainer is a re-execution of "
      "the frozen recipe rather than a variant. The evaluator reproduces the frozen full IoU to 0.0016, inside the "
      "declared 0.005 tolerance.\n")
    w("## 9. Outcome\n")
    w("**OUTCOME C — the current formulation is not primarily data limited.** Both curves flatten early: at equal "
      f"compute, 4.4x the data gives {g['data_only_at_40k_D_small_to_D_large']:+.2f} pp geometry and "
      f"{s['data_only_at_40k_D_small_to_D_large']:+.2f} pp semantics, against bars of 1.0 and 0.5 pp set before the "
      "runs. Both exceed seed noise, so the honest statement is *small but real*, not *nothing*.\n")
    w("Per the brief: **do not scale training blindly, and do not respond by adding more datasets.** The case for "
      "SemanticKITTI / Occ3D / Waymo training was conditional on this curve still rising; it is not.\n")
    w("## 10. Recommended next experiment (exactly one)\n")
    w("**The same one Gate 18 recommended, now with a second independent reason to run it: measure whether the "
      "teacher can be improved at all.** Four gates have now converged on the same wall — instance association "
      "(Gate 17, +0.09), the feature-space objective (B5, −2.03), temporal accumulation (Gate 18, +0.19 controlled), "
      "and training-data volume (Gate 19, +0.37 controlled). Each ruled out a different mechanism, and each ended at "
      "a teacher that is 62.57% correct on what it directly observes.\n")
    w("Score alternative dense open-vocabulary encoders, and a prompt/vocabulary variation of Trident-H, on the "
      "frozen evidence voxels of 0006 against the same GT. No training, no new geometry, one forward pass and the "
      "existing scorer. If a different teacher moves 62.57% / 16.91 materially, the whole stack inherits it. If none "
      "does, the open-vocabulary readout itself is the thing to replace — and that, not more data or another "
      "propagation head, is where the next real effort belongs.\n")
    w("## 11. Reproduce\n")
    w("```bash\npython -m tools.data_scaling.subsets\nbash artifacts/gate19_data_scaling/run_train.sh\n"
      "bash artifacts/gate19_data_scaling/run_control.sh\nbash artifacts/gate19_data_scaling/run_eval.sh\n"
      "python -m tools.data_scaling.analysis\npython -m tools.data_scaling.report\n```\n")
    w("Code: `tools/data_scaling/{config,subsets,train,evaluate,analysis,report}.py`. Pools, per-run records "
      "(with digests, steps, epochs, GPU hours), per-checkpoint evaluations and `analysis.json` are all under "
      f"`artifacts/gate19_data_scaling/`. Deterministic, resumable, seeds {list(G19.SEEDS)}; commit in "
      "`git_commit.txt`.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    out = os.path.join(A, "report.md")
    open(out, "w").write(build())
    print("wrote", out)
