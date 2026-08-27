"""Render the Phase-1 metrics JSON into the markdown tables used by the verdict report."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

from research.lingbot_semantic_memory.config import REPO_ROOT

ORDER = ["lingbot_encoder", "lingbot_gct_b04", "lingbot_gct_mid", "lingbot_gct_b17",
         "lingbot_gct_final"]


def _f(x, n=4):
    return "—" if x is None else f"{x:.{n}f}"


def raw_table(d: Dict) -> str:
    raw = d["raw_token_diagnostic"]
    lines = ["| representation | cross-view cos | negative control | margin | recall@1 | variance | diversity |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ORDER + ["external_dino_teacher"]:
        if name not in raw:
            continue
        cv = raw[name]["cross_view"]["all"]
        lines.append(f"| `{name}` | {_f(cv['pos'])} | {_f(cv['neg'])} | **{_f(cv['margin'])}** | "
                     f"{_f(cv['recall@1'])} | {_f(cv['variance'])} | {_f(raw[name]['diversity'])} |")
    return "\n".join(lines)


def main_table(d: Dict, budget: float = 1.0, geom: str = "predicted") -> str:
    res = d["results"]
    base = res[f"lingbot_encoder@{budget:g}"]
    b_pos = base["cross_view"][geom]["all"]["pos"]
    b_cos = base["fidelity"]["cos_mean"]
    lines = ["| representation | params | teacher cos | cos (centred) | diversity | cross-view cos | vs encoder | margin | recall@1 | depth-boundary margin |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ORDER:
        k = f"{name}@{budget:g}"
        if k not in res:
            continue
        r = res[k]
        cv = r["cross_view"][geom]["all"]
        f = r["fidelity"]
        ratio = cv["pos"] / b_pos if b_pos else float("nan")
        lines.append(
            f"| `{name}` | {r['params']:,} | {_f(f['cos_mean'])} | {_f(f['cos_centered'])} | "
            f"{_f(f.get('diversity'))} | {_f(cv['pos'])} | {ratio:+.1%} | {_f(cv['margin'])} | "
            f"{_f(cv['recall@1'])} | {_f(f.get('depth_boundary_margin'))} |")
    tr = d.get("teacher_reference", {})
    lines.append(f"| `external_dino_teacher` (reference) | 0 | 1.0000 | 1.0000 | "
                 f"{_f(tr.get('diversity'))} | "
                 f"{_f(d['raw_token_diagnostic']['external_dino_teacher']['cross_view']['all']['pos'])} | — | "
                 f"{_f(d['raw_token_diagnostic']['external_dino_teacher']['cross_view']['all']['margin'])} | "
                 f"{_f(d['raw_token_diagnostic']['external_dino_teacher']['cross_view']['all']['recall@1'])} | "
                 f"{_f(tr.get('depth_boundary_margin'))} |")
    return "\n".join(lines)


def geometry_table(d: Dict, budget: float = 1.0) -> str:
    res = d["results"]
    counts = d["correspondence_counts"]
    frac = d["correspondence_valid_fraction"]
    lines = ["| representation | predicted cos | oracle cos | Δ | predicted recall@1 | oracle recall@1 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in ORDER:
        k = f"{name}@{budget:g}"
        if k not in res:
            continue
        p = res[k]["cross_view"]["predicted"]["all"]
        o = res[k]["cross_view"]["oracle"]["all"]
        lines.append(f"| `{name}` | {_f(p['pos'])} | {_f(o['pos'])} | {o['pos']-p['pos']:+.4f} | "
                     f"{_f(p['recall@1'])} | {_f(o['recall@1'])} |")
    lines.append("")
    lines.append("| correspondence set | matches | valid fraction |")
    lines.append("|---|---:|---:|")
    for k in ("predicted", "predicted_conf", "oracle", "oracle_conf"):
        lines.append(f"| `{k}` | {counts[k]:,} | {frac[k]:.3f} |")
    return "\n".join(lines)


def budget_table(d: Dict) -> str:
    res = d["results"]
    budgets = sorted({r["budget"] for r in res.values()})
    lines = ["| representation | " + " | ".join(f"teacher cos @{b:.0%}" for b in budgets)
             + " | retention 10%/100% |", "|---|" + "---:|" * (len(budgets) + 1)]
    for name in ORDER:
        cells, ref, lo = [], None, None
        for b in budgets:
            k = f"{name}@{b:g}"
            v = res[k]["fidelity"]["cos_mean"] if k in res else None
            cells.append(_f(v))
            if b == max(budgets):
                ref = v
            if b == min(budgets):
                lo = v
        ret = f"{lo/ref:.1%}" if ref and lo else "—"
        lines.append(f"| `{name}` | " + " | ".join(cells) + f" | {ret} |")
    return "\n".join(lines)


def band_table(d: Dict, budget: float = 1.0, geom: str = "predicted") -> str:
    res = d["results"]
    gaps = sorted({int(k[3:]) for k in res[f"lingbot_encoder@{budget:g}"]["cross_view"][geom]
                   if k.startswith("gap")})
    lines = ["| representation | short gaps | long gaps | " + " | ".join(f"gap {g}" for g in gaps) + " |",
             "|---|" + "---:|" * (2 + len(gaps))]
    for name in ORDER:
        k = f"{name}@{budget:g}"
        if k not in res:
            continue
        cv = res[k]["cross_view"][geom]
        cells = [_f(cv["short"]["pos"]), _f(cv["long"]["pos"])]
        cells += [_f(cv[f"gap{g}"]["pos"]) for g in gaps]
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def conf_table(d: Dict, budget: float = 1.0) -> str:
    res = d["results"]
    lines = ["| representation | no conf filter | conf filter (top 75%) | Δ |", "|---|---:|---:|---:|"]
    for name in ORDER:
        k = f"{name}@{budget:g}"
        if k not in res:
            continue
        a = res[k]["cross_view"]["predicted"]["all"]["pos"]
        b = res[k]["cross_view"]["predicted_conf"]["all"]["pos"]
        lines.append(f"| `{name}` | {_f(a)} | {_f(b)} | {b-a:+.4f} |")
    return "\n".join(lines)


def cross_dataset_table(runs: Dict[str, Dict], budget: float = 1.0) -> str:
    """Side-by-side gate arithmetic for two datasets under the identical protocol."""
    lines = ["| representation | " + " | ".join(
        f"{n}: consistency | {n}: fidelity" for n in runs) + " |",
        "|---|" + "---:|" * (2 * len(runs))]
    for name in ORDER[1:]:
        cells = []
        for d in runs.values():
            res = d["results"]
            base = res[f"lingbot_encoder@{budget:g}"]
            r = res[f"{name}@{budget:g}"]
            c = r["cross_view"]["predicted"]["all"]["pos"] / base["cross_view"]["predicted"]["all"]["pos"]
            fdl = r["fidelity"]["cos_mean"] / base["fidelity"]["cos_mean"]
            cells += [f"{c:.3f}", f"{fdl:.3f}"]
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("| quantity | " + " | ".join(runs) + " |")
    lines.append("|---|" + "---:|" * len(runs))
    rows = [
        ("patch grid", lambda d: "x".join(str(x) for x in d["grid_hw"])),
        ("held-out frame pairs", lambda d: str(d["n_pairs"])),
        ("valid corr. (predicted)", lambda d: f"{d['correspondence_valid_fraction']['predicted']:.3f}"),
        ("valid corr. (oracle)", lambda d: f"{d['correspondence_valid_fraction']['oracle']:.3f}"),
        ("encoder cross-view cos", lambda d: _f(d["results"][f"lingbot_encoder@{budget:g}"]["cross_view"]["predicted"]["all"]["pos"])),
        ("encoder teacher cos", lambda d: _f(d["results"][f"lingbot_encoder@{budget:g}"]["fidelity"]["cos_mean"])),
        ("encoder recall@1 (pred)", lambda d: _f(d["results"][f"lingbot_encoder@{budget:g}"]["cross_view"]["predicted"]["all"]["recall@1"])),
        ("encoder recall@1 (oracle)", lambda d: _f(d["results"][f"lingbot_encoder@{budget:g}"]["cross_view"]["oracle"]["all"]["recall@1"])),
        ("best GCT consistency gain", lambda d: _best_gain(d, budget, "predicted")),
        ("best GCT gain (oracle)", lambda d: _best_gain(d, budget, "oracle")),
        ("raw-token gain, gct_final", lambda d: f"{d['raw_token_diagnostic']['lingbot_gct_final']['cross_view']['all']['pos'] / d['raw_token_diagnostic']['lingbot_encoder']['cross_view']['all']['pos'] - 1:+.1%}"),
        ("raw-token margin, gct_final", lambda d: f"{d['raw_token_diagnostic']['lingbot_gct_final']['cross_view']['all']['margin'] / d['raw_token_diagnostic']['lingbot_encoder']['cross_view']['all']['margin'] - 1:+.1%}"),
        ("verdict", lambda d: "`" + d["verdict"]["verdict"] + "`"),
    ]
    for label, fn in rows:
        lines.append(f"| {label} | " + " | ".join(fn(d) for d in runs.values()) + " |")
    return "\n".join(lines)


def _best_gain(d: Dict, budget: float, geom: str) -> str:
    res = d["results"]
    base = res[f"lingbot_encoder@{budget:g}"]["cross_view"][geom]["all"]["pos"]
    best = max(res[f"{n}@{budget:g}"]["cross_view"][geom]["all"]["pos"] for n in ORDER[1:])
    return f"{best / base - 1:+.1%}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="research/lingbot_semantic_memory/outputs/phase1/metrics.json")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--compare", nargs="*", default=None,
                    metavar="NAME=PATH", help="additional runs for a cross-dataset table")
    a = ap.parse_args()
    d = json.load(open(os.path.join(REPO_ROOT, a.metrics)))
    out = []
    out.append("### Raw-token diagnostic (no probe)\n\n" + raw_table(d))
    out.append("\n### Main comparison (100 % teacher, predicted geometry)\n\n" + main_table(d))
    out.append("\n### Predicted vs oracle geometry\n\n" + geometry_table(d))
    out.append("\n### Teacher budget\n\n" + budget_table(d))
    out.append("\n### Temporal band\n\n" + band_table(d))
    out.append("\n### Depth-confidence filtering\n\n" + conf_table(d))
    if a.compare:
        runs = {"KITTI (outdoor)": d}
        for spec in a.compare:
            n, _, path = spec.partition("=")
            runs[n] = json.load(open(os.path.join(REPO_ROOT, path)))
        out.append("\n### Cross-dataset comparison\n\n" + cross_dataset_table(runs))
    text = "\n".join(out)
    print(text)
    if a.output_dir:
        p = os.path.join(REPO_ROOT, a.output_dir, "tables.md")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(text)


if __name__ == "__main__":
    main()
