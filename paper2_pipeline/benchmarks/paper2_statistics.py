"""
Paper 2 — statistical hardening, applying every methodological lesson from the
Paper-1 review before submission rather than after it.

  S1  Teacher vs student depth: paired Stuart-Maxwell marginal-homogeneity test
      (NOT a test of independence), weighted kappa, bootstrap CIs, confusion.
  S2  Stage-2 fidelity: fp16+compile vs fp32 masks - paired comparison of the
      derived risk categories (McNemar-style exact statement).
  S3  Warning urgency: Paper-1 pipeline vs Paper-2 local pipeline on the same
      1,119 scenes - paired Stuart-Maxwell, with the caveat that the inputs differ.
  S4  Rubric consistency with a bootstrap CI rather than a bare percentage.

Output: Paper2_Improvements/results/paper2_statistics.json
"""
import json
from collections import Counter
import os
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix

ROOT = Path(os.environ.get("POTHOLE_ROOT", "."))  # local research-tree layout; see README
P2 = ROOT / "Paper2_Improvements"
J = ROOT / "Main_Work/final_assessments/Json_and_Log_Files"
DEPTH = ["low", "medium", "high"]
URG = ["low", "medium", "high", "critical"]
rng = np.random.default_rng(17)
out = {}


def stuart_maxwell(a, b, labels):
    k = len(labels)
    N = np.zeros((k, k))
    idx = {l: i for i, l in enumerate(labels)}
    for x, y in zip(a, b):
        N[idx[x], idx[y]] += 1
    d = np.array([N[i, :].sum() - N[:, i].sum() for i in range(k)])[:-1]
    S = np.zeros((k - 1, k - 1))
    for i in range(k - 1):
        for j in range(k - 1):
            S[i, j] = (N[i, :].sum() + N[:, i].sum() - 2 * N[i, i]) if i == j \
                else -(N[i, j] + N[j, i])
    chi2 = float(d @ np.linalg.solve(S, d))
    return {"chi2": round(chi2, 2), "df": k - 1,
            "p_value": float(stats.chi2.sf(chi2, k - 1)),
            "exact_agreement_pct": round(100 * float(np.mean(np.array(a) == np.array(b))), 2)}


def boot_kappa(y, p, labels, weights=None, n=5000):
    yi = np.array([labels.index(v) for v in y])
    pi = np.array([labels.index(v) for v in p])
    vals = []
    for _ in range(n):
        s = rng.choice(len(yi), len(yi), replace=True)
        if len(set(yi[s])) > 1:
            vals.append(cohen_kappa_score(yi[s], pi[s], weights=weights))
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


# ---------------------------------------------------------------- S1
p2a = json.loads((P2 / "results/paper2_full_assessments.json").read_text())["scenes"]
t, s = [], []
for sc in p2a.values():
    for p in sc["potholes"]:
        if p["depth"]["paper1_gemini"]:
            t.append(p["depth"]["paper1_gemini"])
            s.append(p["depth"]["paper2_distilled"])
out["S1_teacher_vs_student_depth"] = {
    "n": len(t),
    "teacher_distribution": dict(Counter(t)),
    "student_distribution": dict(Counter(s)),
    "kappa": round(float(cohen_kappa_score(t, s, labels=DEPTH)), 4),
    "kappa_CI95": boot_kappa(t, s, DEPTH),
    "quadratic_weighted_kappa": round(float(cohen_kappa_score(
        [DEPTH.index(x) for x in t], [DEPTH.index(x) for x in s], weights="quadratic")), 4),
    "weighted_kappa_CI95": boot_kappa(t, s, DEPTH, weights="quadratic"),
    "stuart_maxwell_paired": stuart_maxwell(t, s, DEPTH),
    "confusion_rows_teacher": confusion_matrix(t, s, labels=DEPTH).tolist(),
    "note": "Paired data (same potholes). Marginal homogeneity is rejected only if the two "
            "engines assign different overall class proportions; agreement is reported "
            "separately by kappa.",
}

# ---------------------------------------------------------------- S2
b = json.loads((P2 / "results/bench_stage2_vith-fp16-comp.json").read_text())
out["S2_stage2_fidelity"] = {
    "iou_vs_fp32": b["fidelity_iou_vs_vith_fp32"],
    "area_relative_error_mean": b["area_rel_err_mean"],
    "risk_category_flip_rate": b["risk_category_flip_rate"],
    "n_potholes": b["total_boxes"],
    "statement": "Zero risk-category flips over the evaluated potholes; the paired "
                 "comparison is therefore degenerate (no discordant pairs) and no test "
                 "statistic is meaningful. We report the exact count rather than a p-value.",
}

# ---------------------------------------------------------------- S3
w1 = json.loads((J / "scene_warnings_multimodal_hybrid.json").read_text())
w2 = json.loads((P2 / "results/warnings_local_qwen3vl8b-FULL.json").read_text())["scenes"]
common = sorted(set(w1) & set(w2))
a = [w1[k]["warning"]["urgency_level"] for k in common]
bb = [w2[k]["local"]["urgency_level"] for k in common]
out["S3_warning_urgency_paper1_vs_paper2"] = {
    "n_scenes": len(common),
    "paper1_distribution": dict(Counter(a)),
    "paper2_distribution": dict(Counter(bb)),
    "stuart_maxwell_paired": stuart_maxwell(a, bb, URG),
    "kappa": round(float(cohen_kappa_score(a, bb, labels=URG)), 4),
    "caveat": "The two systems consume different depth inputs (cloud VLM vs distilled "
              "ensemble), so this measures end-to-end divergence of the full pipelines, "
              "not warning-generator quality in isolation.",
}

# ---------------------------------------------------------------- S4
def rubric(mx):
    return ("critical" if mx >= 80 else "high" if mx >= 60 else "medium" if mx >= 40 else "low")
hits = []
for k in common:
    sc = p2a.get(k)
    if sc:
        hits.append(w2[k]["local"]["urgency_level"] == rubric(sc["max_risk_score"]))
hits = np.array(hits)
bs = [hits[rng.choice(len(hits), len(hits), True)].mean() for _ in range(5000)]
out["S4_rubric_consistency"] = {
    "n_scenes": int(len(hits)),
    "consistency_pct": round(100 * float(hits.mean()), 2),
    "bootstrap_CI95_pct": [round(100 * float(np.percentile(bs, 2.5)), 2),
                           round(100 * float(np.percentile(bs, 97.5)), 2)],
    "definition": "Fraction of scenes whose generated urgency equals the urgency implied by "
                  "the deterministic risk-score bands. Measures internal calibration of the "
                  "generator against its own inputs; it is NOT an accuracy against "
                  "independent hazard labels, which do not exist for this dataset.",
}

(P2 / "results/paper2_statistics.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1)[:2600])
print("\nSaved: results/paper2_statistics.json")
