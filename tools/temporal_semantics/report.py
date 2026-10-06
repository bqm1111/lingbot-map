"""Generate the Gate 18 report from the measured artifacts. Computes nothing new."""
from __future__ import annotations

import json
import os

from tools.temporal_semantics import config as G18

A = G18.ARTIFACT_DIR


def build() -> str:
    tq = json.load(open(os.path.join(A, "teacher_quality.json")))
    sc = json.load(open(os.path.join(A, "scaling.json")))
    ov = json.load(open(os.path.join(A, "history_overlap.json")))
    lad = {int(k): v for k, v in sc["ladder"].items()}
    L, w = [], None
    L = []
    w = L.append
    k1, k5, k20 = lad[1], lad[5], lad[20]
    w("# Gate 18 — temporal semantic evidence scaling: OUTCOME C (saturates), with a structural reason\n")
    w("Accumulating more causal history **cannot** improve semantics of the current volume, and the reason is "
      "geometric rather than a property of the teacher: **the past is behind you.** Evidence beyond about four "
      "anchors back lies entirely outside the forward-looking grid, so there is nothing left to fuse.\n")
    w(f"Within the existing five-frame window history does help — T3 completed-unobserved mIoU rises "
      f"{k1['T3_miou']:.2f} → {k5['T3_miou']:.2f} from K=1 to K=5 — but essentially all of that is **new voxels being "
      f"observed**, not better fusion of the same voxels. Beyond K=5 everything is flat: "
      f"{k5['T3_miou']:.2f} → {k20['T3_miou']:.2f}.\n")
    w("## 1. Raw teacher quality (deployable, no oracle)\n")
    w(f"Frozen Trident-H on drive 0006, at voxels where it has evidence **and** ground truth has a label "
      f"({tq['n_scored']:,} voxels; only {100 * tq['label_coverage_of_evidence']:.1f}% of the {tq['n_evidence']:,} "
      f"evidence voxels carry a label, because the benchmark labels a smaller volume than the map covers).\n")
    w("| | top-1 accuracy | mIoU |")
    w("|---|---:|---:|")
    w(f"| all classes | **{tq['accuracy']:.2f}** | **{tq['miou']:.2f}** |")
    w(f"| thing | {tq['thing']['accuracy']:.2f} | {tq['thing']['miou']:.2f} |")
    w(f"| stuff | {tq['stuff']['accuracy']:.2f} | {tq['stuff']['miou']:.2f} |")
    w("")
    w(f"**This is the ceiling every propagation method inherits.** The teacher is right {tq['accuracy']:.0f}% of the "
      f"time on what it directly sees, and B0 reaches 8.80 mIoU on completed-unobserved against the teacher's own "
      f"{tq['miou']:.2f} — propagation loses roughly half, and the half that remains is bounded by a teacher that is "
      "already wrong a third of the time.\n")
    w("Per-class IoU is extremely uneven: vegetation "
      f"{tq['per_class_iou']['vegetation']:.1f}, building {tq['per_class_iou']['building']:.1f}, "
      f"car {tq['per_class_iou']['car']:.1f}, road {tq['per_class_iou']['road']:.1f}, but terrain "
      f"{tq['per_class_iou']['terrain']:.2f}, pole {tq['per_class_iou']['pole']:.2f}, parking "
      f"{tq['per_class_iou']['parking']:.2f}. Vegetation alone is "
      f"{100 * tq['per_class_gt_voxels']['vegetation'] / tq['n_scored']:.0f}% of the scored voxels.\n")
    w("## 2. The K ladder\n")
    w("Rule is the frozen one (normalised count-weighted mean of unit features). K≤5 is fully deployable; K>5 uses "
      "GT-pose registration and is an upper bound (see `plan.md`).\n")
    w("| K (frames) | seconds | evidence voxels | frames per voxel | T1 mIoU | T2 mIoU | **T3 completed-unobserved** |")
    w("|---:|---:|---:|---:|---:|---:|---:|")
    for k in G18.HISTORY_K:
        v = lad[k]
        w(f"| {k} | {v['seconds']:.2f} | {v['ev_voxels']:,.0f} | {v['frames_per_voxel']:.2f} | {v['T1_miou']:.2f} | "
          f"{v['T2_miou']:.2f} | **{v['T3_miou']:.2f}** |")
    w("")
    w("- **T1** population grows with K (all evidence voxels with a GT label).")
    w("- **T2** population is *fixed* at the K=1 evidence voxels, so only the fused feature changes.")
    w("- **T3** is `completed_never_observed`, fixed, propagated with the existing B0 rule.\n")
    w("**The T2 column is the one that matters, and it is flat.** Holding the voxel set fixed and only fusing more "
      f"observations of those same voxels moves mIoU {k1['T2_miou']:.2f} → {k20['T2_miou']:.2f} — "
      f"**{k20['T2_miou'] - k1['T2_miou']:+.2f} over the whole range**. Fusing more views of a voxel does almost "
      "nothing for its semantics.\n")
    w(f"All of the 1→5 gain is coverage: evidence voxels go {k1['ev_voxels']:,.0f} → {k5['ev_voxels']:,.0f} "
      f"({k5['ev_voxels'] / k1['ev_voxels']:.1f}x). More frames see *more places*, not the same place better.\n")
    w("### A correlation that does not survive control\n")
    w("Stratifying the teacher by how many distinct frames saw a voxel suggests a large effect:\n")
    w("| distinct frames | accuracy | mIoU | voxels |")
    w("|---|---:|---:|---:|")
    for nm, v in tq["by_frames"].items():
        w(f"| {nm} | {v['accuracy']:.2f} | {v['miou']:.2f} | {v['voxels']:,} |")
    w("")
    w("That looks like +6 mIoU from being seen more often. It is **confounded**: voxels seen by more frames are "
      "nearer and easier, not better estimated. The controlled version of the same question is T2 above — same "
      f"voxels, more observations — and it gives {k20['T2_miou'] - k1['T2_miou']:+.2f}. The stratified number should "
      "not be quoted as a temporal effect.\n")
    w("Binning by *ray* count instead of distinct frames raises mIoU (12.64 → 19.82) while accuracy **falls** "
      "(61.41 → 60.96), which is the same confound seen from another angle.\n")
    w("## 3. Why history saturates: the past is behind you\n")
    w("For each earlier anchor, how much of its evidence can reach the current grid at all (40 anchors):\n")
    w("| anchors back | frames | seconds | ego moved | lands in the grid | and on the hard map |")
    w("|---:|---:|---:|---:|---:|---:|")
    for o in sorted(ov, key=int):
        v = ov[o]
        w(f"| {o} | {v['frames']} | {v['seconds']:.2f} | {v['ego_m']:.1f} m | {100 * v['in_grid']:.1f}% | "
          f"{100 * v['on_S']:.1f}% |")
    w("")
    w(f"The ego vehicle moves **{ov['1']['ego_m']:.1f} m per anchor** while an anchor's own evidence reaches only "
      "**28.5 m forward** (the depth range of the reconstruction) into a grid that extends 51.2 m forward. After "
      "four anchors — 12 frames, 6.2 s, ~51 m — **0.0%** of the older evidence is still inside the current volume.\n")
    w("This is why K=10 and K=20 add only ~2% more evidence voxels than K=5. It is not that old observations are "
      "uninformative; it is that they are observations of space the vehicle has already driven past. Forward-facing "
      "driving means the volume you must complete is the volume you have *least* history for.\n")
    w("## 4. Distance and class breakdowns\n")
    w("T3 mIoU by distance to the nearest evidence:\n")
    w("| distance | " + " | ".join(f"K={k}" for k in G18.HISTORY_K) + " |")
    w("|---|" + "---:|" * len(G18.HISTORY_K))
    for nm, row in sc["distance"].items():
        w(f"| {nm} | " + " | ".join(f"{row[str(k)]:.2f}" if str(k) in row else f"{row[k]:.2f}" for k in G18.HISTORY_K) + " |")
    w("")
    w("History helps only where evidence is already close (0–1 m: 11.04 → 12.42) and does nothing past 4 m "
      "(>8 m: 3.09 → 3.11). Better-fused nearby evidence does not reach distant completed voxels.\n")
    w("thing vs stuff on T3: thing " + " → ".join(f"{sc['thing_stuff_T3']['thing'][str(k)]:.2f}" if str(k) in sc['thing_stuff_T3']['thing'] else f"{sc['thing_stuff_T3']['thing'][k]:.2f}" for k in G18.HISTORY_K) +
      "; stuff " + " → ".join(f"{sc['thing_stuff_T3']['stuff'][str(k)]:.2f}" if str(k) in sc['thing_stuff_T3']['stuff'] else f"{sc['thing_stuff_T3']['stuff'][k]:.2f}" for k in G18.HISTORY_K) +
      ". Both saturate by K=3–5.\n")
    w("Per-class T3 (K=1 → K=5 → K=20) is in `scaling.json`. The 1→5 gains concentrate in the frequent classes "
      "(building 28.00 → 35.05, road 12.20 → 17.52, vegetation 49.29 → 52.77); the classes the teacher cannot read "
      "at all stay at zero (pole 0.03 → 0.04, traffic-sign 0.03 → 0.04, terrain 0.38 → 0.26).\n")
    w("## 5. Aggregation rules (K=5)\n")
    w("| rule | T1 | T2 | T3 |")
    w("|---|---:|---:|---:|")
    for r, v in sc["rules"].items():
        w(f"| {r} | {v['T1']:.2f} | {v['T2']:.2f} | {v['T3']:.2f} |")
    w("")
    w("`count_weighted` is not listed because it is **identical** to `mean_feature` on this evidence: every accepted "
      "ray contributes one unit feature, so summing rays already is the count weighting. The rule choice is worth "
      "at most ~0.4 mIoU (`latest` is the worst, as expected), which is small beside everything else here.\n")
    w("## 6. Interpretation\n")
    w("Three things are now measured rather than assumed:\n")
    w(f"1. **The teacher is the ceiling.** {tq['accuracy']:.0f}% top-1, {tq['miou']:.2f} mIoU on its own evidence. "
      "Nothing downstream can exceed it, and B0 already loses half of it in propagation.")
    w("2. **Fusing more views of the same voxel is worth almost nothing** "
      f"({k20['T2_miou'] - k1['T2_miou']:+.2f} mIoU from 1 to 20 frames on a fixed population). The apparent "
      "temporal effect in the stratified view is a confound.")
    w("3. **Longer history cannot reach the volume being completed**, because forward motion carries past evidence "
      "out of the forward grid within ~6 seconds.\n")
    w("Together these say the streaming nature of the map is not, by itself, a source of better semantics for "
      "forward completion. A persistent memory would retain space the vehicle has passed — useful for revisiting or "
      "for a surround volume, not for the benchmark's forward cone.\n")
    w("## 7. Outcome\n")
    w("**OUTCOME C — temporal evidence saturates.** Additional history barely improves teacher or map semantics, and "
      "the completed-unobserved region is flat beyond K=5. Per the brief: the teacher/representation itself is the "
      "thing that needs replacing or adapting, not the amount of temporal evidence.\n")
    w("No persistent-memory formulation is proposed, because that is warranted only under Outcome A.\n")
    w("## 8. Recommended next experiment (exactly one)\n")
    w("**Test whether the teacher can be improved at all on this data, before anything else is built.** Gate 17 ruled "
      "out association, Gate B5 ruled out the feature-space objective, and Gate 18 now rules out temporal "
      f"accumulation — all three land on the same {tq['accuracy']:.0f}% teacher. The smallest decisive next step is a "
      "**teacher comparison on the frozen evidence voxels of 0006**: score one or two alternative open-vocabulary "
      "dense encoders, and a prompt/vocabulary variation of Trident-H itself, against the same GT on the same "
      "voxels. It needs no training and no new geometry — only a forward pass and the existing scorer. If a "
      "different teacher moves the 62.57% / 16.91 numbers materially, the whole completion stack inherits it; if "
      "none does, the bottleneck is the open-vocabulary readout itself and that is what must change.\n")
    w("Gate 19 (training-data scaling) is a separate hypothesis and is untouched by this result — it should be run "
      "on its own terms, as the brief specifies, now that Gate 18 is frozen.\n")
    w("## 9. Reproduce\n")
    w("```bash\npython -m tools.temporal_semantics.teacher_quality --device cuda:0 --shard 0 --shards 4\n"
      "python -m tools.temporal_semantics.scaling --device cuda:0 --shard 0 --shards 4\npython -m tools.temporal_semantics.report\n```\n")
    w("Code: `tools/temporal_semantics/{config,accumulate,teacher_quality,scaling,report}.py`. "
      "**Validation:** K=5 with the frozen rule reproduces the frozen evidence cache exactly — identical voxel index "
      "sets, cosine 1.000000, identical counts. No training. ~0.3 GPU-h. Commit in `git_commit.txt`.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    out = os.path.join(A, "report.md")
    open(out, "w").write(build())
    print("wrote", out)
