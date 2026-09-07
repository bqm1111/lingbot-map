#!/usr/bin/env python
"""Verify a DDAD root has what the scale gate needs. Downloads nothing."""
import argparse, os, sys

REQUIRED = ["rgb", "point_cloud", "calibration"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    if not os.path.isdir(a.root):
        print(f"MISSING root: {a.root}"); return 1
    base = os.path.join(a.root, "ddad_train_val")
    base = base if os.path.isdir(base) else a.root
    scenes = [d for d in sorted(os.listdir(base)) if os.path.isdir(os.path.join(base, d))]
    if not scenes:
        print(f"no scene directories under {base}"); return 1
    problems, ok = [], 0
    for s in scenes:
        p = os.path.join(base, s)
        miss = [r for r in REQUIRED if not os.path.isdir(os.path.join(p, r))]
        if miss:
            problems.append(f"{s}: missing {miss}")
        else:
            ok += 1
    print(f"DDAD at {base}: {len(scenes)} scenes, {ok} complete")
    for p in problems[:20]:
        print("  -", p)
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
