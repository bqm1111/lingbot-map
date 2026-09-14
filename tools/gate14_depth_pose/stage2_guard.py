"""Gate 14 Stage 2 guard. Every Stage 2 entry point must call require_promotion() before doing anything.

Stage 2 (completion training with NEW-L input) is allowed only if Stage 1's frozen promotion rule passed on 0006:
artifacts/gate14_depth_pose/promotion_result.json exists, has promote == true, and was produced under the exact
promotion_rule.json on disk (sha256 match).
"""
from __future__ import annotations
import hashlib, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g14                                                                                   # noqa: E402

RESULT = os.path.join(g14.ART, "promotion_result.json")


class Stage2Locked(RuntimeError):
    pass


def require_promotion():
    if not os.path.exists(RESULT):
        raise Stage2Locked("Stage 2 locked: no promotion_result.json (Stage 1 not analysed)")
    res = json.load(open(RESULT)); rule_sha = hashlib.sha256(open(g14.PROMOTION_RULE, "rb").read()).hexdigest()
    if res.get("rule_sha256") != rule_sha:
        raise Stage2Locked("Stage 2 locked: promotion_result.json was not produced under the frozen promotion_rule.json")
    if res.get("promote") is not True:
        raise Stage2Locked(f"Stage 2 locked: NEW-L did not pass the frozen promotion rule ({ {k: res.get(k) for k in ('C1_iou_gain', 'C2_free_space', 'C3_not_density', 'C4_range_consistency')} })")
    return res


if __name__ == "__main__":
    try:
        print("Stage 2 unlocked:", require_promotion()["values"])
    except Stage2Locked as e:
        print(e); raise SystemExit(3)
