"""Generate the Gate 17 report from the analysis artifacts. Computes nothing new."""
from __future__ import annotations

import json
import os

from tools.instance_oracle import config as G

A = G.ARTIFACT_DIR
P1, P2, P3 = "P1_ssc_deployable", "P2_correct_completion", "P3_oracle_geometry"


def build() -> str:
    r = json.load(open(os.path.join(A, "analysis.json")))
    acc = json.load(open(os.path.join(A, "accuracy_P2.json")))
    nn = json.load(open(os.path.join(A, "nearest_same_instance.json")))
    q = json.load(open(os.path.join(A, "qualitative_index.json")))
    m = r["miou"]; ciA = r["paired_ci_P2"]["A_oracle_instance"]; ciS = r["paired_ci_P2"]["C1_shuffled_instance"]
    L = []
    w = L.append
    w("# Gate 17 — instance grounding: OUTCOME C (insufficient headroom)\n")
    w("**Perfect instance association does not fix semantic propagation.** With ground-truth grouping between visible "
      "evidence and completed unobserved voxels, and semantics still taken from the frozen teacher, mIoU on correctly "
      f"completed voxels moves from **{m['C0_b0'][P2]:.2f} to {m['A_oracle_instance'][P2]:.2f}** "
      f"({r['gains']['A_minus_B0_P2']:+.2f}, paired 95% CI [{ciA[1]:+.2f}, {ciA[2]:+.2f}] — includes zero). On the "
      f"deployable SSC read it moves {r['gains']['A_minus_B0_P1']:+.2f}. "
      "**Do not build the instance transformer on this evidence.**\n")
    w(f"The oracle is wired correctly — destroying the grouping costs {-r['gains']['shuffled_minus_B0_P2']:.2f} mIoU "
      f"(CI [{ciS[1]:+.2f}, {ciS[2]:+.2f}]), so the association carries real signal. It simply does not carry signal "
      "that B0 was missing.\n")
    w("## A. What was reused\n")
    w("The causal 5-frame state, hard map, frozen CompletionUNet occupancy (seed 0), Trident-H evidence, B0's "
      "nearest-evidence rule, `regions.build_regions`, `semantic_metrics.summarize`, the Gate 15 eval masks and the "
      "predict/eval file firewalls — all unchanged. Only the oracle grouping is new. See `plan.md`.\n")
    w("## B. What is oracle and what stays deployable\n")
    w("**Ground truth supplies only *which voxels belong together*.** For every method except the `C2` ceiling, the "
      "class still comes from the frozen teacher read through the frozen text embeddings.\n")
    for k, v in r["oracle_use"].items():
        w(f"- `{k}` — {v}")
    w("")
    w("**KITTI-360 instance annotations are not on this machine.** The tree holds only `data_2d_raw` and "
      "`data_3d_raw`; the SSCBench voxel labels are `float32` semantic-only with no instance channel; the sole "
      "instance annotations anywhere belong to nuScenes, which is firewalled out of source work. So instances are "
      "**6-connected components of same-class GT-occupied voxels** — the strongest legitimate substitute, and an "
      "*upper bound*: it knows GT class boundaries exactly, and adjacent same-class objects merge. A low ceiling "
      "under an optimistic oracle is decisive; a high one would have needed re-checking against real annotations "
      "before building anything.\n")
    w("## C. Baseline reproduction (gate passed)\n")
    w("B0 reproduces the frozen pilot record exactly on all eight regions that do not depend on the revealed set: "
      "observed_current 15.21, observed_past_only 17.38, never_observed 8.39, completed_never_observed 8.80, "
      "completed_geo_observed 9.72, completed_geo_unobserved 7.73, ssc_full 1.55, ssc_never_observed 1.39.\n")
    w("Getting there found two bugs in my own Gate B5 analysis, both fixed: a reimplemented mIoU that included the "
      "empty class (inflating SSC regions ~4x), and a shard glob that merged a leftover smoke-test file (4 duplicated "
      "anchors). B5's conclusion is unchanged.\n")
    w("## D. Oracle results\n")
    w("590 anchors, drive 0006. mIoU by the repository's frozen definition (classes 1..18, empty excluded).\n")
    w("| method | P1 deployable SSC | P2 correct completion | P3 oracle geometry |")
    w("|---|---:|---:|---:|")
    for k in G.METHODS:
        w(f"| {k} | {m[k][P1]:.2f} | {m[k][P2]:.2f} | {m[k][P3]:.2f} |")
    w("")
    w(f"Populations: P1 {r['population_voxels'][P1]:,} voxels, P2 {r['population_voxels'][P2]:,}, "
      f"P3 {r['population_voxels'][P3]:,}. Instance coverage of P2 is **{100 * r['coverage_P2']:.1f}%**.\n")
    w("`C2_gt_class` reads 100.00 on P2 and P3 **by construction** — those populations are GT-occupied, so a GT label "
      f"is perfect. It is not an informative ceiling there. The informative one is P1 = {m['C2_gt_class'][P1]:.2f}: "
      "even perfect semantics on the current predicted occupancy reaches only that, because occupancy errors cap the "
      "deployable number.\n")
    w(f"Accuracy tells the same story as mIoU on P2: B0 {acc['C0_b0']['accuracy']:.2f}, oracle instance "
      f"{acc['A_oracle_instance']['accuracy']:.2f}, shuffled {acc['C1_shuffled_instance']['accuracy']:.2f}. "
      "The oracle buys about **1.3 points of accuracy and 0.09 of mIoU**.\n")
    w("### Per-class on P2 (B0 → oracle instance)\n")
    w("| class | B0 | oracle | delta |"); w("|---|---:|---:|---:|")
    pc = r["per_class_P2"]
    for c in ("car", "truck", "person", "bicycle", "building", "road", "sidewalk", "vegetation", "terrain", "fence", "pole"):
        a, b = pc["C0_b0"].get(c), pc["A_oracle_instance"].get(c)
        if a is not None and b is not None:
            w(f"| {c} | {a:.2f} | {b:.2f} | {b - a:+.2f} |")
    w("")
    w("## E. Why it does not help — geometry vs semantics vs association\n")
    w("Three measurements, in the order that matters:\n")
    frac_inst = 100 * r["instances_with_evidence_per_anchor"] / r["instances_per_anchor"]
    w(f"**1. The association is mostly unavailable.** Only **{frac_inst:.1f}% of instances per anchor have any "
      f"visible evidence** ({r['instances_with_evidence_per_anchor']:.0f} of {r['instances_per_anchor']:.0f}), and "
      f"only **{100 * r['coverage_P2']:.1f}% of P2 voxels** belong to an instance the model saw at all. For the other "
      "two thirds the oracle has nothing to offer and falls back to B0. You cannot ground to an object you never saw.\n")
    w("**2. Where it IS available, it changes the answer but not the quality.** I expected B0's nearest evidence to "
      "already come from the same instance, which would have made the oracle a no-op. It does not: among P2 voxels "
      f"whose instance has evidence, B0's nearest evidence is in the same instance only "
      f"**{100 * nn['nearest_same_instance_frac']:.1f}%** of the time "
      f"(**{100 * nn['all_nearest_same_instance_frac']:.1f}%** across all of P2). So the oracle really does relabel "
      "most covered voxels from a different, instance-correct source — and mIoU still barely moves. The label it "
      "hands over is no better than the one proximity was handing over.\n")
    w("**3. The teacher is the binding constraint.** Perfect grouping inherits the teacher's read of the instance's "
      f"visible part, and that read is right about **{acc['A_oracle_instance']['accuracy']:.0f}%** of the time. That "
      "ceiling, not the association, is what caps the result — the same teacher-capped conclusion the frozen "
      "zero-shot evaluation reached.\n")
    cg = r["paired_ci_P1"]["C2_gt_class"]
    w(f"**Geometry's share, separately:** perfect semantics on the current occupancy gives P1 {m['C2_gt_class'][P1]:.2f} "
      f"against B0's {m['C0_b0'][P1]:.2f} (paired {cg[0]:+.2f}, CI [{cg[1]:+.2f}, {cg[2]:+.2f}]). So the deployable "
      "gap is bounded by geometry and dominated by semantic quality; instance association is not where either "
      "bottleneck lives.\n")
    w("## F. Distance from visible evidence (P2 mIoU)\n")
    w("The hypothesis was that instance grounding would pull ahead exactly where Euclidean propagation degrades. "
      "It does not.\n")
    w("| distance | " + " | ".join(k.split("_")[0] for k in G.METHODS) + " |")
    w("|---|" + "---:|" * len(G.METHODS))
    for b in G.DISTANCE_NAMES:
        w(f"| {b} | " + " | ".join(f"{r['bins']['distance'][b][k]:.2f}" for k in G.METHODS) + " |")
    w("")
    w("B0 and the oracle track each other within ~0.2 mIoU at every distance; both decay from ~12 to ~3 as distance "
      "grows. The advantage does not appear where proximity fails — it is not there.\n")
    w("## G. Instance visibility — the one real positive\n")
    w("| visible fraction | " + " | ".join(k.split("_")[0] for k in G.METHODS) + " |")
    w("|---|" + "---:|" * len(G.METHODS))
    for b in G.VISIBILITY_NAMES:
        w(f"| {b} | " + " | ".join(f"{r['bins']['visibility'][b][k]:.2f}" for k in G.METHODS) + " |")
    w("")
    d020 = r["bins"]["visibility"]["0-20%"]["A_oracle_instance"] - r["bins"]["visibility"]["0-20%"]["C0_b0"]
    w("**When an object is actually well observed, instance grounding does work**: +2.4 at 20-40% visible, +3.4 at "
      f"40-60%, +3.1 at 80-100%. But the 0-20% bin dominates the population and shows {d020:+.2f}. The mechanism is "
      "real; the data is not in the regime where it pays.\n")
    w("### thing vs stuff\n")
    w("| split | voxels | B0 | oracle | delta |"); w("|---|---:|---:|---:|---:|")
    for s in ("thing", "stuff"):
        a, b = r["thing_stuff"][s]["C0_b0"], r["thing_stuff"][s]["A_oracle_instance"]
        w(f"| {s} | {r['thing_stuff_voxels'][s]:,} | {a:.2f} | {b:.2f} | {b - a:+.2f} |")
    w("")
    w("On **things** — the classes where an instance is genuinely an object — the oracle is **worse** than B0 "
      f"({r['thing_stuff']['thing']['A_oracle_instance']:.2f} vs {r['thing_stuff']['thing']['C0_b0']:.2f}). Things are "
      f"{r['thing_stuff_voxels']['thing']:,} of {r['thing_stuff_voxels']['stuff']:,} P2 voxels. The small positive on "
      "stuff comes from large connected regions (vegetation +2.12, building +0.27), which is region smoothing, not "
      "object grounding.\n")
    w("## H. Qualitative examples\n")
    w(f"`qualitative.png`. Selection rule fixed in advance: anchors at quantiles {q['quantiles']} of per-anchor "
      "(oracle − B0) mIoU on P2, so the set spans anchors where the oracle hurts as well as helps: " +
      ", ".join(f"t={p['t']:05d} {p['delta_miou']:+.2f}" for p in q["picks"]) + ". The coverage panel is the one to "
      "look at — grey is every completed voxel the oracle could not reach.\n")
    w("## I. Decision\n")
    w("**OUTCOME C — instance grounding has insufficient headroom.** The brief's own reading applies: this is a valid "
      "and useful negative result. Do not build the persistent-instance transformer. Per the brief no next "
      "architecture is proposed, because a design document is warranted only under Outcome A or B.\n")
    w("Scope: source domain only, KITTI-360 drive 0006, connected-component instances. No target dataset was opened.\n")
    w("## J. Next experiment (one, smallest)\n")
    w("**Measure the teacher's ceiling on visible evidence directly, and nothing else.** Every completion result so "
      "far — B1, B4, B5 and now the instance oracle — lands on the same wall: whatever you propagate, you propagate "
      "a teacher label that is right about half the time. Gate 17 shows association is not the missing piece; B5 "
      "showed the feature-space objective is not either. The cheapest decisive next measurement: on observed evidence "
      "voxels that have GT labels, what is frozen Trident-H's per-class accuracy, and how much of the completed-region "
      "error is inherited from it rather than created by propagation? That is a scoring exercise on caches that "
      "already exist, needs no training, and decides whether any propagation mechanism can pay before another one is "
      "built.\n")
    w("## K. Reproduce\n")
    w("```bash\nbash artifacts/gate17_instance_oracle/run_eval.sh\npython -m tools.instance_oracle.analysis\n"
      "python -m tools.instance_oracle.visualize --device cuda:0\n```\n")
    w("Code: `tools/instance_oracle/{config,oracle,evaluate,analysis,visualize,report}.py`. Commit in "
      f"`git_commit.txt`. Deterministic (seed {G.SEED}), resumable per shard, no frozen output modified. "
      "Cost: ~0.1 GPU-h.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    out = os.path.join(A, "report.md")
    open(out, "w").write(build())
    print("wrote", out)
