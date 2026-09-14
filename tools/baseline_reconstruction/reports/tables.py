"""Small helpers shared by every report writer: markdown tables and CSV files."""
from __future__ import annotations

import csv
import os
from typing import Dict, Iterable, List, Sequence


def markdown_table(header: Sequence[str], rows: Iterable[Sequence]) -> str:
    lines = ["| " + " | ".join(str(h) for h in header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines) + "\n"


def write_csv(path: str, rows: List[Dict], fieldnames: Sequence[str] | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        open(path, "w").close(); return
    fieldnames = list(fieldnames or rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def pct(x: float, nd: int = 2) -> str:
    return f"{100 * x:.{nd}f}"


__all__ = ["markdown_table", "write_csv", "pct"]
