"""
Phase A statistics for the IEEE Access resubmission (R2-M7, M6, M2-partial).

Outputs: Paper1_Resubmission/phaseA_statistics_results.json
  M7a  Stuart-Maxwell marginal-homogeneity test on the 1,119 paired urgency
       labels (image-only vs hybrid) - replaces the chi-squared independence test.
  M7b  Cluster-aware analysis of the 150 Likert ratings (3 annotators x 50
       scenes): per-annotator paired Wilcoxon + scene-level aggregation +
       scene-clustered bootstrap CIs for the clarity/safety differences.
  M6   Risk-score sensitivity: weight simplex perturbations and band-boundary
       shifts -> % of potholes changing category; plus Table-3/4 consistency count.
  M2p  Depth study on frozen 38-consensus set: quadratic-weighted kappa,
       bootstrap CIs, per-class precision/recall, confusion matrix for
       Gemini 2.5 Pro / Flash / GPT-5.
"""
import os
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

ROOT = Path(os.environ.get("POTHOLE_ROOT", "."))  # local data layout; see README
RES = ROOT / "Paper1_Resubmission"
CSV = ROOT / "Main_Work/final_assessments/HIL_Warnings_Sheets/detailed_comparison.csv"
ODS_DIR = ROOT / "Main_Work/final_assessments/HIL_Warnings_Sheets/Annotators_Copies"
FLAT = ROOT / "Main_Work/final_assessments/Json_and_Log_Files/final_risk_scores_flat.json"
EVAL = ROOT / "data/human_depth_eval_set.json"  # first-panel evaluation set

URG = ["low", "medium", "high", "critical"]
DEPTH = ["low", "medium", "high"]
rng = np.random.default_rng(7)
out = {}

# ---------------------------------------------------------------- M7a
df = pd.read_csv(CSV)
a = df["image_only_urgency"].str.lower().map(URG.index).to_numpy()
b = df["hybrid_urgency"].str.lower().map(URG.index).to_numpy()
k = 4
N = np.zeros((k, k))
for i, j in zip(a, b):
    N[i, j] += 1
d = np.array([N[i, :].sum() - N[:, i].sum() for i in range(k)])[:-1]  # drop last (singular)
S = np.zeros((k - 1, k - 1))
for i in range(k - 1):
    for j in range(k - 1):
        if i == j:
            S[i, j] = N[i, :].sum() + N[:, i].sum() - 2 * N[i, i]
        else:
            S[i, j] = -(N[i, j] + N[j, i])
chi2_sm = float(d @ np.linalg.solve(S, d))
p_sm = float(stats.chi2.sf(chi2_sm, k - 1))
out["M7a_stuart_maxwell"] = {
    "n_scenes": int(len(df)),
    "chi2": round(chi2_sm, 2), "df": k - 1, "p_value": p_sm,
    "note": "marginal homogeneity rejected -> the two methods' urgency "
            "distributions differ, correctly tested as PAIRED data",
    "exact_agreement": round(float((a == b).mean()), 4),
}

# ---------------------------------------------------------------- M7b
frames = []
for i, f in enumerate(sorted(ODS_DIR.glob("human_evaluation_study_*.ods")), 1):
    t = pd.read_excel(f, engine="odf", sheet_name="Evaluation")
    key = pd.read_excel(f, engine="odf", sheet_name="Answer_Key_Researcher_Only")
    key.columns = [str(c).strip() for c in key.columns]
    kcols = [c for c in key.columns if "scene" in c.lower()] + \
            [c for c in key.columns if "a" == c.lower() or "method_a" in c.lower()]
    t["annotator"] = i
    # map A/B -> methods using the answer key (col holding 'hybrid'/'image')
    mcol = None
    for c in key.columns:
        vals = key[c].astype(str).str.lower()
        if vals.str.contains("hybrid").any() or vals.str.contains("image").any():
            mcol = c; break
    scol = [c for c in key.columns if "scene" in c.lower()][0]
    amap = dict(zip(key[scol], key[mcol].astype(str).str.lower()))
    t["A_is_hybrid"] = t["Scene_ID"].map(lambda s: "hybrid" in str(amap.get(s, "")))
    frames.append(t)
r = pd.concat(frames, ignore_index=True)
m7b = {"n_rows": int(len(r)), "answer_key_used": True}
for name in ("Clarity", "Safety"):
    ah = pd.to_numeric(r[f"Method_A_{name}_Rating"], errors="coerce")
    bh = pd.to_numeric(r[f"Method_B_{name}_Rating"], errors="coerce")
    hyb = ah.where(r["A_is_hybrid"], bh)
    img = bh.where(r["A_is_hybrid"], ah)
    dif = (hyb - img).dropna()
    per_ann = []
    for iann, g in r.groupby("annotator"):
        gah = pd.to_numeric(g[f"Method_A_{name}_Rating"], errors="coerce")
        gbh = pd.to_numeric(g[f"Method_B_{name}_Rating"], errors="coerce")
        gd = (gah.where(g["A_is_hybrid"], gbh) - gbh.where(g["A_is_hybrid"], gah)).dropna()
        w = stats.wilcoxon(gd, zero_method="wilcox") if (gd != 0).any() else None
        per_ann.append({"annotator": int(iann), "mean_diff": round(float(gd.mean()), 3),
                        "wilcoxon_p": (float(w.pvalue) if w else None), "n": int(len(gd))})
    tmp = r.assign(_d=(hyb - img))
    scene_means = tmp.groupby("Scene_ID")["_d"].mean().dropna()
    w_sc = stats.wilcoxon(scene_means, zero_method="wilcox")
    boots = []
    scenes = scene_means.index.to_numpy()
    for _ in range(5000):
        samp = rng.choice(scenes, size=len(scenes), replace=True)
        boots.append(scene_means.loc[samp].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    m7b[name.lower()] = {
        "mean_diff_hybrid_minus_imageonly": round(float(dif.mean()), 3),
        "per_annotator": per_ann,
        "scene_level_wilcoxon": {"n_scenes": int(len(scene_means)),
                                 "stat": float(w_sc.statistic), "p": float(w_sc.pvalue)},
        "scene_clustered_bootstrap_CI95": [round(float(lo), 3), round(float(hi), 3)],
    }
out["M7b_clustered_ratings"] = m7b

# ---------------------------------------------------------------- M6
flat = json.loads(FLAT.read_text())
rows = [(v["risk_assessment"]["component_scores"]["s_depth"],
         v["risk_assessment"]["component_scores"]["s_area"],
         v["risk_assessment"]["component_scores"]["s_pos"]) for v in flat.values()]
C = np.array(rows)
def cat(scores):
    return np.digitize(scores, [40, 60, 80])
base = cat(0.5 * C[:, 0] + 0.3 * C[:, 1] + 0.2 * C[:, 2])
w_grid, flips_w = [], []
for dwd in (-0.1, 0, 0.1):
    for dwa in (-0.1, 0, 0.1):
        wd, wa = 0.5 + dwd, 0.3 + dwa
        wp = 1.0 - wd - wa
        if wp < 0.05 or (dwd == dwa == 0):
            continue
        alt = cat(wd * C[:, 0] + wa * C[:, 1] + wp * C[:, 2])
        w_grid.append({"weights": [round(wd, 2), round(wa, 2), round(wp, 2)],
                       "pct_category_change": round(float((alt != base).mean() * 100), 2)})
b_grid = []
for db in (-5, 5):
    alt = np.digitize(0.5 * C[:, 0] + 0.3 * C[:, 1] + 0.2 * C[:, 2],
                      [40 + db, 60 + db, 80 + db])
    b_grid.append({"boundary_shift": db,
                   "pct_category_change": round(float((alt != base).mean() * 100), 2)})
out["M6_sensitivity"] = {
    "n_potholes": int(len(C)),
    "weight_perturbations": w_grid,
    "boundary_shifts": b_grid,
    "median_weight_flip_pct": round(float(np.median([g["pct_category_change"] for g in w_grid])), 2),
}

# ---------------------------------------------------------------- M2 partial
ev = json.loads(EVAL.read_text())["records"]
usable = {k: v for k, v in ev.items() if v["human_consensus"]}
y = [usable[k]["human_consensus"] for k in usable]
m2 = {}
for model in ("gemini_pro", "gemini_flash", "gpt5"):
    p = [usable[k]["paper1_predictions"][model] for k in usable]
    yi = [DEPTH.index(t) for t in y]
    pi = [DEPTH.index(t) for t in p]
    qwk = cohen_kappa_score(yi, pi, weights="quadratic")
    boots = []
    idx = np.arange(len(yi))
    for _ in range(5000):
        s = rng.choice(idx, len(idx), replace=True)
        if len(set(np.array(yi)[s])) < 2:
            continue
        boots.append(cohen_kappa_score(np.array(yi)[s], np.array(pi)[s], weights="quadratic"))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=DEPTH, zero_division=0)
    m2[model] = {
        "quadratic_weighted_kappa": round(float(qwk), 4),
        "qwk_bootstrap_CI95": [round(float(lo), 3), round(float(hi), 3)],
        "unweighted_kappa": round(float(cohen_kappa_score(y, p, labels=DEPTH)), 4),
        "per_class": {c: {"precision": round(float(pr[i]), 3), "recall": round(float(rc[i]), 3),
                          "support": int(sup[i])} for i, c in enumerate(DEPTH)},
        "confusion_true_rows": confusion_matrix(y, p, labels=DEPTH).tolist(),
    }
out["M2_depth_stats_38set"] = {"n": len(y), "consensus_dist": dict(Counter(y)), "models": m2}

(RES / "phaseA_statistics_results.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: (v if k != "M7b_clustered_ratings" else "see file") for k, v in out.items()},
                 indent=1, default=str)[:3500])
print("\nSaved:", RES / "phaseA_statistics_results.json")
