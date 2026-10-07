"""Aggregate per-anchor records into the §5 comparison, with every denominator named.

Three rules the brief insists on, implemented rather than remembered:

* **counts are summed, never averaged as per-anchor ratios.** A fraction is always the ratio
  of two summed counts, so one dense anchor cannot be weighted like one sparse anchor;
* **every reported fraction carries its denominator** in the key name, so a changed
  denominator cannot silently inflate a coverage gain. ``w_temporal`` and ``w_extended`` are
  additionally compared on a *common* geometry-supported denominator -- the union's ``hidden``
  set -- with extra geometry coverage reported as its own line;
* **agreement is labelled consistency, not correctness,** wherever it appears.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from tools.waymo_pilot import config as C, populations as POP


def _sum(rows: Sequence[Dict], *path: str) -> int:
    tot = 0
    for r in rows:
        v = r
        for k in path:
            v = v.get(k, {}) if isinstance(v, dict) else {}
        tot += int(v) if isinstance(v, (int, float)) else 0
    return tot


def _sum_dict(rows: Sequence[Dict], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        for k, v in (r.get(key) or {}).items():
            if isinstance(v, (int, float)):
                out[k] = out.get(k, 0) + int(v)
    return out


def frac(num: int, den: int) -> Optional[float]:
    return (num / den) if den else None


def aggregate(rows: Sequence[Dict], variant: str) -> Dict:
    """One variant's pooled statistics. ``rows`` are ``measure_anchor`` records."""
    if not rows:
        return {"variant": variant, "n_anchors": 0}
    counts = _sum_dict(rows, "counts")
    scenes = sorted({r["scene"] for r in rows})
    hidden, U = counts.get("hidden", 0), counts.get("U", 0)
    # The §4 sub-cases need the causal clip. Anchors without one still contribute coverage,
    # so they are pooled for hidden/U and EXCLUDED from the sub-case sums -- which therefore
    # carry their own, smaller denominator and say so.
    sub_rows = [r for r in rows if r.get("subcases_available", True)]
    sub_counts = _sum_dict(sub_rows, "counts")
    sub_hidden = sub_counts.get("hidden", 0)
    agr_two = _sum(rows, "agreement", "n_with_two_views")
    agr_same = _sum(rows, "agreement", "n_two_views_agree")
    rvn = _sum_dict(rows, "repeat_vs_new")
    out = {
        "variant": variant,
        "n_scenes": len(scenes), "scenes": scenes, "n_anchors": len(rows),
        "n_usable_pairs": len([r for r in rows if r["counts"]["U"] > 0]),
        # ---- geometry side, before any semantic target is required
        "n_privileged_occupied": counts.get("O_priv", 0),
        "n_causal_evidence": counts.get("E_causal", 0),
        "n_hidden_before_target": hidden,
        "hidden_per_anchor": hidden / len(rows),
        # ---- semantic side
        "n_with_privileged_semantics": U,
        "fraction_of_hidden_with_target": frac(U, hidden),
        "U_per_anchor": U / len(rows),
        # ---- §4 decomposition, both before and after requiring a target
        "n_anchors_with_subcases": len(sub_rows),
        "subcase_denominator_hidden": sub_hidden,
        "subcases_hidden": {k: sub_counts.get(f"hidden_{k}", 0) for k in POP.SUBCASES},
        "subcases_U": {k: sub_counts.get(f"U_{k}", 0) for k in POP.SUBCASES},
        "subcase_fractions_of_hidden": {k: frac(sub_counts.get(f"hidden_{k}", 0), sub_hidden)
                                        for k in POP.SUBCASES},
        "diagnostics_hidden": {k: sub_counts.get(f"hidden_{k}", 0) for k in
                               ("causal_free_disagreement", "unresolved_not_carved")},
        "diagnostics_U": {k: sub_counts.get(f"U_{k}", 0) for k in
                          ("causal_free_disagreement", "unresolved_not_carved")},
        # ---- provenance
        "sources": _sum_dict([r["provenance"] for r in rows], "sources"),
        "overlaps": _sum_dict([r["provenance"] for r in rows], "overlaps"),
        "exclusive_to_source": _sum_dict([r["provenance"] for r in rows],
                                         "exclusive_to_source"),
        "tier": _sum_dict([r["provenance"] for r in rows], "tier"),
        # ---- exclusions
        "exclusions": _sum_dict(rows, "exclusions"),
        # ---- distributions
        "distance_U": _sum_dict(rows, "distance"),
        "distance_hidden": _sum_dict(rows, "distance_hidden"),
        "range_U": _sum_dict(rows, "range"),
        "range_hidden": _sum_dict(rows, "range_hidden"),
        "readout_histogram_U": _sum_dict(rows, "readout_histogram"),
        "b0_histogram_U": _sum_dict(rows, "b0_histogram"),
        # ---- agreement, and what it is not
        "agreement": {"n_with_two_views": agr_two, "n_agree": agr_same,
                      "two_view_agreement": frac(agr_same, agr_two),
                      "denominator": "target voxels in U with >= 2 privileged views",
                      "note": "teacher consistency between views, NOT semantic correctness"},
        "repeat_vs_new": rvn,
        "fraction_newly_supported": frac(rvn.get("n_newly_supported", 0),
                                         rvn.get("n_targets", 0)),
        "fraction_single_observation": frac(rvn.get("n_single_observation", 0),
                                           rvn.get("n_targets", 0)),
    }
    out["class_diversity"] = class_diversity(out["readout_histogram_U"])
    out["timing"] = _clip_span(rows)
    return out


def class_diversity(hist: Dict[str, int]) -> Dict:
    """Is the target set dominated by one class? A road-only pile is not useful diversity."""
    tot = sum(hist.values())
    if not tot:
        return {"n_total": 0}
    ordered = sorted(hist.items(), key=lambda kv: -kv[1])
    p = np.asarray([v / tot for v in hist.values() if v > 0], np.float64)
    thing = sum(v for k, v in hist.items()
                if C.CLASS_NAMES.index(k) + 1 in C.THING_LABELS)
    return {"n_total": tot, "top_class": ordered[0][0],
            "top_class_fraction": ordered[0][1] / tot,
            "top3_fraction": sum(v for _k, v in ordered[:3]) / tot,
            "n_classes_present": int((np.asarray(list(hist.values())) > 0).sum()),
            "n_classes_above_1pct": int(sum(1 for v in hist.values() if v / tot > 0.01)),
            "entropy_bits": float(-(p * np.log2(p)).sum()),
            "thing_fraction": thing / tot}


def _clip_span(rows: Sequence[Dict]) -> Dict:
    """The measured temporal protocol, so a difference is recorded rather than inherited."""
    got = [r.get("clip_span_s") or {} for r in rows]
    def med(k):
        v = [g[k] for g in got if isinstance(g.get(k), (int, float))]
        return float(np.median(v)) if v else None
    return {"causal_span_s": med("causal_span_s"), "stream_dt_s": med("stream_dt_s"),
            "n_causal_frames": med("n_causal_frames"),
            "n_future_frames": med("n_future_frames"),
            "future_last_dt_s": med("future_last_dt_s")}


def common_denominator_comparison(a: Dict, b: Dict, name_a: str, name_b: str) -> Dict:
    """Compare two variants of the SAME anchors on one geometry-supported denominator.

    The common denominator is the larger variant's ``hidden`` set, which is the geometry
    both variants had available before semantics were required. Extra geometry coverage --
    the case where the variants do not even agree on ``hidden`` -- is reported on its own
    line instead of being folded into the semantic gain.
    """
    den = max(a.get("n_hidden_before_target", 0), b.get("n_hidden_before_target", 0))
    geom_delta = b.get("n_hidden_before_target", 0) - a.get("n_hidden_before_target", 0)
    return {
        "variants": [name_a, name_b],
        "common_denominator_hidden": den,
        "extra_geometry_coverage_voxels": geom_delta,
        f"{name_a}_targets": a.get("n_with_privileged_semantics", 0),
        f"{name_b}_targets": b.get("n_with_privileged_semantics", 0),
        f"{name_a}_coverage_of_common": frac(a.get("n_with_privileged_semantics", 0), den),
        f"{name_b}_coverage_of_common": frac(b.get("n_with_privileged_semantics", 0), den),
        "coverage_gain_pp": (
            None if not den else
            100.0 * (b.get("n_with_privileged_semantics", 0)
                     - a.get("n_with_privileged_semantics", 0)) / den),
        "note": ("a positive coverage gain on a common denominator is a supervision-recipe "
                 "effect; extra_geometry_coverage_voxels is a separate, geometric effect"),
    }


def table(aggs: Sequence[Dict]) -> str:
    """The report's headline comparison, markdown."""
    cols = [a for a in aggs if a.get("n_anchors")]
    if not cols:
        return "_no variant produced a measurement_\n"
    rows = [
        ("scenes", lambda a: f"{a['n_scenes']}"),
        ("anchors", lambda a: f"{a['n_anchors']}"),
        ("usable pairs", lambda a: f"{a['n_usable_pairs']}"),
        ("privileged occupied voxels", lambda a: f"{a['n_privileged_occupied']:,}"),
        ("causal semantic evidence voxels", lambda a: f"{a['n_causal_evidence']:,}"),
        ("**hidden (O_priv & ~E_causal)**", lambda a: f"**{a['n_hidden_before_target']:,}**"),
        ("hidden per anchor", lambda a: f"{a['hidden_per_anchor']:,.0f}"),
        ("**U (hidden & target valid)**", lambda a: f"**{a['n_with_privileged_semantics']:,}**"),
        ("U / hidden", lambda a: _pct(a["fraction_of_hidden_with_target"])),
        ("U per anchor", lambda a: f"{a['U_per_anchor']:,.0f}"),
        ("sub-cases below: anchors / hidden voxels",
         lambda a: f"{a['n_anchors_with_subcases']} / {a['subcase_denominator_hidden']:,}"),
        ("— surface observed, teacher silent", lambda a: _sub(a, "surface_observed_no_teacher")),
        ("— outside every causal frustum", lambda a: _sub(a, "outside_every_frustum")),
        ("— occluded throughout", lambda a: _sub(a, "occluded_throughout")),
        ("— unresolved visibility", lambda a: _sub(a, "unresolved_visibility")),
        ("   of which causal map carved it free",
         lambda a: ("—" if not a["n_anchors_with_subcases"] else
                    f"{a['diagnostics_hidden']['causal_free_disagreement']:,} "
                    f"({_pct(frac(a['diagnostics_hidden']['causal_free_disagreement'], a['subcase_denominator_hidden']), 0)})")),
        ("newly supported / all targets", lambda a: _pct(a["fraction_newly_supported"])),
        ("single-observation targets", lambda a: _pct(a["fraction_single_observation"])),
        ("two-view agreement (consistency)", lambda a: _pct(a["agreement"]["two_view_agreement"])),
        ("top class share of U", lambda a: _pct(a["class_diversity"].get("top_class_fraction"))
                                            + f" ({a['class_diversity'].get('top_class')})"),
        ("classes > 1 % of U", lambda a: f"{a['class_diversity'].get('n_classes_above_1pct')}/18"),
        ("thing share of U", lambda a: _pct(a["class_diversity"].get("thing_fraction"))),
        ("causal clip span (s)", lambda a: _f(a["timing"].get("causal_span_s"))),
        ("stream dt (s)", lambda a: _f(a["timing"].get("stream_dt_s"))),
    ]
    head = "| quantity | " + " | ".join(a["variant"] for a in cols) + " |"
    sep = "|---|" + "".join("---:|" for _ in cols)
    body = [f"| {label} | " + " | ".join(fn(a) for a in cols) + " |" for label, fn in rows]
    return "\n".join([head, sep, *body]) + "\n"


def _sub(a: Dict, key: str) -> str:
    """A sub-case count with the share of its OWN denominator.

    The sub-case denominator is smaller than ``hidden`` whenever some anchors lack a causal
    clip, so the raw counts of two variants are not comparable. The percentage is.
    """
    if not a.get("n_anchors_with_subcases"):
        return "— (no causal clip)"
    n = a["subcases_hidden"][key]
    return f"{n:,} ({_pct(frac(n, a['subcase_denominator_hidden']), 0)})"


def _pct(x: Optional[float], nd: int = 1) -> str:
    return "—" if x is None else f"{100.0 * x:.{nd}f} %"


def _f(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.3f}"


__all__ = ["aggregate", "class_diversity", "common_denominator_comparison", "table", "frac"]
