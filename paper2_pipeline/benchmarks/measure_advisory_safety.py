"""
Paper 2 — advisory safety measurement: does constraining the maneuver vocabulary
remove advice the system cannot justify?

The pipeline observes the ego lane only. It has no view of adjacent-lane occupancy,
so any recommendation to change lane is guidance it cannot support. We measure the
rate of such advice for three systems on the same 1,119 scenes:

  cloud                Paper-1 pipeline (Gemini 2.5 Pro)
  local_unconstrained  Paper-2 local VLM, free-text maneuver
  local_constrained    Paper-2 local VLM, maneuver_type restricted by the grammar

Paired McNemar tests, plus rubric consistency and judge ratings where available.

Output: results/advisory_safety.json
"""
import json
import re
import os
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(os.environ.get("POTHOLE_ROOT", "."))  # local research-tree layout; see README
P2 = ROOT / "Paper2_Improvements"
J = ROOT / "Main_Work/final_assessments/Json_and_Log_Files"
RES = P2 / "results"

LANE_CHANGE = re.compile(
    r"into the (adjacent|next|left|right|center|centre|other)\s*lane"
    r"|change lanes?|switch lanes?"
    r"|move (in)?to the (adjacent|left|right|next|other) lane"
    r"|merge|overtak", re.I)
IN_LANE = re.compile(r"within (the|your|current) lane|maintain (current )?lane"
                     r"|inside the lane|stay in (the|your) lane", re.I)


def text(w):
    return f"{w.get('maneuver_suggestion','')} {w.get('warning_message','')}"


def mcnemar(flags_a, flags_b):
    b = sum(1 for x, y in zip(flags_a, flags_b) if y and not x)
    c = sum(1 for x, y in zip(flags_a, flags_b) if x and not y)
    p = stats.binomtest(b, b + c, 0.5).pvalue if (b + c) else 1.0
    return {"b_only_second": b, "c_only_first": c, "p_value": float(p)}


def main():
    cloud = json.loads((J / "scene_warnings_multimodal_hybrid.json").read_text())
    # MATCHED CORPUS. The first version of this comparison read
    # warnings_local_UNCONSTRAINED.json, which had been generated against an earlier
    # assessment corpus than the constrained arm. The measured "improvement" therefore
    # mixed the effect of the grammar with the effect of the corpus change, and the
    # rubric-consistency figure was not comparable at all. Both arms below are now
    # generated from the SAME paper2_full_assessments.json, differing only in whether
    # UNCONSTRAINED=1 removed maneuver_type from the schema. One variable.
    unc = json.loads(
        (RES / "warnings_local_qwen3vl8b-UNCON-SAMECORPUS.json").read_text())["scenes"]
    con_p = RES / "warnings_local_qwen3vl8b-FINAL.json"
    con = json.loads(con_p.read_text())["scenes"] if con_p.exists() else None

    ids = sorted(set(cloud) & set(unc) & (set(con) if con else set(unc)))
    n = len(ids)
    sysmap = {"cloud": lambda s: cloud[s]["warning"],
              "local_unconstrained": lambda s: unc[s]["local"]}
    if con:
        sysmap["local_constrained"] = lambda s: con[s]["local"]

    flags, out = {}, {"n_scenes": n, "lane_departure_pct": {}, "in_lane_pct": {}}
    for name, get in sysmap.items():
        f = [bool(LANE_CHANGE.search(text(get(s)))) for s in ids]
        il = [bool(IN_LANE.search(text(get(s)))) for s in ids]
        flags[name] = f
        out["lane_departure_pct"][name] = round(100 * float(np.mean(f)), 2)
        out["in_lane_pct"][name] = round(100 * float(np.mean(il)), 2)

    out["mcnemar"] = {
        "cloud_vs_local_unconstrained": mcnemar(flags["cloud"], flags["local_unconstrained"]),
    }
    if con:
        out["mcnemar"]["local_unconstrained_vs_constrained"] = mcnemar(
            flags["local_unconstrained"], flags["local_constrained"])
        out["mcnemar"]["cloud_vs_local_constrained"] = mcnemar(
            flags["cloud"], flags["local_constrained"])
        # maneuver_type usage under the constrained grammar
        from collections import Counter
        out["maneuver_type_distribution"] = dict(Counter(
            con[s]["local"].get("maneuver_type", "?") for s in ids))

    # rubric consistency for each local variant
    assess = json.loads((RES / "paper2_full_assessments.json").read_text())["scenes"]
    def rubric(mx):
        return ("critical" if mx >= 80 else "high" if mx >= 60
                else "medium" if mx >= 40 else "low")
    rng = np.random.default_rng(11)
    out["rubric_consistency_pct"] = {}
    for name in ("local_unconstrained", "local_constrained"):
        if name not in sysmap:
            continue
        get = sysmap[name]
        hits = np.array([get(s)["urgency_level"] == rubric(assess[s]["max_risk_score"])
                         for s in ids if s in assess])
        bs = [hits[rng.choice(len(hits), len(hits), True)].mean() for _ in range(5000)]
        out["rubric_consistency_pct"][name] = {
            "value": round(100 * float(hits.mean()), 2),
            "CI95": [round(100 * float(np.percentile(bs, 2.5)), 2),
                     round(100 * float(np.percentile(bs, 97.5)), 2)]}

    # judge ratings, if the runs exist
    judge = {}
    for tag, f in (("local_unconstrained", "llm_judge_warnings.json"),
                   ("local_constrained", "llm_judge_warnings_constrained.json")):
        p = RES / f
        if p.exists():
            d = json.loads(p.read_text())
            judge.setdefault("cloud", {})
            judge["cloud"] = {"clarity": d["mean_ratings"]["cloud_clarity"],
                              "safety": d["mean_ratings"]["cloud_safety"]}
            judge[tag] = {"clarity": d["mean_ratings"]["local_clarity"],
                          "safety": d["mean_ratings"]["local_safety"],
                          "pref_cloud": d["preference"]["order_consistent_cloud"],
                          "pref_local": d["preference"]["order_consistent_local"]}
    if judge:
        out["judge"] = judge

    (RES / "advisory_safety.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
