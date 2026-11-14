"""
Paper 2 — final analysis of the round-2 human warning study (the reported system).

Round 2 rates warnings_local_qwen3vl8b-FINAL.json (constrained Qwen3-VL-8B) against
the published cloud warnings (Gemini 2.5 Pro) on a fresh 50-scene sample, blind,
per-scene randomised A/B order, three independent raters.

The unit of analysis is the SCENE (n = 50). The three ratings of one scene are not
independent observations of the system, so tests on 150 (scene, rater) pairs
overstate significance; we average raters within a scene first and test on scene
means. The (scene, rater)-level numbers are kept only for the descriptive table.

  python benchmarks/analyze_human_round2.py

Output: results/human_eval_round2.json
"""
import itertools
import json
from collections import Counter
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(os.environ.get("POTHOLE_ROOT", "."))  # local research-tree layout; see README
P2 = ROOT / "Paper2_Improvements"
KIT = P2 / "human_eval_kit_v2"
KEY = json.loads((KIT / "assignment_key_DO_NOT_SHARE.json").read_text())["key"]
RATERS = {  # anonymised rater id (rating file suffix) -> description used in the paper
    "R1": "researcher (United States)",
    "R2": "researcher (Germany)",
    "R3": "non-researcher driver",
}
WARN = json.loads((P2 / "results/warnings_local_qwen3vl8b-FINAL.json").read_text())["scenes"]
ROUND1 = ROOT / "Annotations_Results/v1_paper2_by_system.json"
JUDGE_150 = P2 / "results/llm_judge_warnings_constrained.json"
JUDGE_SAME = P2 / "results/llm_judge_warnings_human50_FINAL.json"   # optional
rng = np.random.default_rng(2026)


def load():
    """Long table: one row per (scene, rater, system)."""
    rows = []
    for r in RATERS:
        df = pd.read_excel(KIT / f"Results/Warning_Evaluation_{r}.xlsx", sheet_name="Ratings")
        for _, x in df.iterrows():
            sid = x["Scene_ID"]
            m = KEY[sid]
            p = str(x["Preference (A/B/Tie)"]).strip().upper()
            pref = "tie" if p == "TIE" else ("local" if m[p] == "local" else "cloud")
            for side in "AB":
                rows.append({
                    "scene": sid, "rater": r,
                    "system": "local" if m[side] == "local" else "cloud",
                    "clarity": int(x[f"Clarity_{side} (1-5)"]),
                    "safety": int(x[f"Safety_{side} (1-5)"]),
                    "pref": pref,
                    "urgency": WARN[sid]["local"]["urgency_level"],
                })
    d = pd.DataFrame(rows)
    assert d.scene.nunique() == 50 and len(d) == 300, "incomplete ratings"
    return d


def boot_ci(x, n=10000):
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def paired(d, metric):
    """Scene-level paired comparison, local minus cloud."""
    s = d.pivot_table(index="scene", columns="system", values=metric, aggfunc="mean")
    diff = (s["local"] - s["cloud"]).to_numpy()
    w = stats.wilcoxon(s["local"], s["cloud"])
    t = stats.ttest_rel(s["local"], s["cloud"])
    return {
        "local_mean": round(float(s["local"].mean()), 3),
        "cloud_mean": round(float(s["cloud"].mean()), 3),
        "diff_mean": round(float(diff.mean()), 3),
        "bootstrap_CI95_over_scenes": boot_ci(diff),
        "wilcoxon_p": float(w.pvalue),
        "paired_t_p": float(t.pvalue),
        "cohens_dz": round(float(diff.mean() / diff.std(ddof=1)), 3),
        "scenes_local_higher": int((diff > 0).sum()),
        "scenes_cloud_higher": int((diff < 0).sum()),
        "scenes_equal": int((diff == 0).sum()),
        "n_scenes": int(len(diff)),
    }


def kripp_alpha(units, metric="interval"):
    """Krippendorff's alpha; units = list of per-item rating lists (no missing)."""
    vals = sorted({v for u in units for v in u})
    idx = {v: i for i, v in enumerate(vals)}
    o = np.zeros((len(vals), len(vals)))
    for u in units:
        m = len(u)
        for a, b in itertools.permutations(range(m), 2):
            o[idx[u[a]], idx[u[b]]] += 1.0 / (m - 1)
    nc = o.sum(1)
    n = nc.sum()

    def delta(c, k):
        if metric == "interval":
            return (vals[c] - vals[k]) ** 2
        lo, hi = min(c, k), max(c, k)          # ordinal
        return (nc[lo:hi + 1].sum() - (nc[lo] + nc[hi]) / 2) ** 2

    D = np.array([[delta(c, k) for k in range(len(vals))] for c in range(len(vals))])
    do = (o * D).sum() / n
    de = (np.outer(nc, nc) * D).sum() / (n * (n - 1))
    return float(1 - do / de)


def icc2(mat):
    """ICC(2,1) and ICC(2,k), two-way random, absolute agreement. mat: items x raters."""
    n, k = mat.shape
    gm = mat.mean()
    msr = k * ((mat.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((mat.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((mat - mat.mean(1, keepdims=True) - mat.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    single = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    avg = (msr - mse) / (msr + (msc - mse) / n)
    return round(float(single), 3), round(float(avg), 3)


def fleiss(table):
    t = np.asarray(table, float)
    n = t.sum(1)[0]
    p = t.sum(0) / t.sum()
    P = ((t ** 2).sum(1) - n) / (n * (n - 1))
    return float((P.mean() - (p ** 2).sum()) / (1 - (p ** 2).sum()))


def agreement(d):
    out = {}
    for metric in ("clarity", "safety"):
        m = d.pivot_table(index=["scene", "system"], columns="rater", values=metric)
        mat = m[list(RATERS)].to_numpy(float)
        prs = [stats.pearsonr(mat[:, i], mat[:, j])[0]
               for i, j in itertools.combinations(range(mat.shape[1]), 2)]
        s1, sk = icc2(mat)
        out[metric] = {
            "mean_pairwise_pearson": round(float(np.mean(prs)), 3),
            "icc2_1": s1, "icc2_k": sk,
            "krippendorff_alpha_ordinal": round(kripp_alpha(mat.tolist(), "ordinal"), 3),
            "n_items": int(mat.shape[0]),
        }
    # within-rater direction agreement: does each pair of raters agree on which
    # system is safer for a scene (sign of local - cloud)?
    sg = d.pivot_table(index=["scene", "rater"], columns="system", values="safety")
    sg = np.sign(sg["local"] - sg["cloud"]).unstack()
    out["safety_sign_agreement_pairwise"] = {
        f"{a}-{b}": round(float((sg[a] == sg[b]).mean()), 3)
        for a, b in itertools.combinations(RATERS, 2)}
    # preference agreement in system space (local / cloud / tie)
    pr = d[d.system == "local"].pivot_table(index="scene", columns="rater",
                                            values="pref", aggfunc="first")
    cats = ["local", "cloud", "tie"]
    out["preference_fleiss_kappa"] = round(
        fleiss([[list(r).count(c) for c in cats] for r in pr.to_numpy()]), 3)
    return out


def main():
    d = load()
    res = {"STUDY": "HUMAN evaluation round 2: 3 independent raters, 50 scenes, blind, "
                    "per-scene randomised A/B; constrained local system (FINAL)",
           "raters": RATERS,
           "unit_of_analysis": "scene (ratings averaged over raters within scene)"}

    # descriptive, rating level
    res["descriptive"] = {
        s: {m: {"mean": round(float(g[m].mean()), 3), "sd": round(float(g[m].std(ddof=1)), 3),
                "n": int(len(g))} for m in ("clarity", "safety")}
        for s, g in d.groupby("system")}

    res["paired_scene_level"] = {m: paired(d, m) for m in ("clarity", "safety")}

    # preference
    p = d[d.system == "local"]
    c = Counter(p.pref)
    dec = c["local"] + c["cloud"]
    maj = Counter()
    for _, g in p.groupby("scene"):
        cc = Counter(g.pref).most_common()
        maj[cc[0][0] if cc[0][1] >= 2 else "split"] += 1
    res["preference"] = {
        "ratings": {"local": c["local"], "cloud": c["cloud"], "tie": c["tie"]},
        "binomial_p_local_vs_cloud_excl_ties": float(stats.binomtest(c["local"], dec).pvalue),
        "scene_majority": dict(maj),
        "binomial_p_scene_majority": float(stats.binomtest(maj["local"], maj["local"] + maj["cloud"]).pvalue),
    }

    # per rater
    res["per_rater"] = {}
    for r, g in d.groupby("rater"):
        e = {}
        for m in ("clarity", "safety"):
            s = g.pivot_table(index="scene", columns="system", values=m)
            w = stats.wilcoxon(s["local"], s["cloud"]) if (s["local"] != s["cloud"]).any() else None
            e[m] = {"local": round(float(s["local"].mean()), 2), "cloud": round(float(s["cloud"].mean()), 2),
                    "diff": round(float((s["local"] - s["cloud"]).mean()), 2),
                    "wilcoxon_p": float(w.pvalue) if w else None}
        pc = Counter(g[g.system == "local"].pref)
        e["preference"] = {"local": pc["local"], "cloud": pc["cloud"], "tie": pc["tie"]}
        res["per_rater"][r] = e

    # by local urgency stratum (sampling strata)
    res["by_urgency"] = {}
    for u, g in d.groupby("urgency"):
        s = g.pivot_table(index="scene", columns="system", values="safety", aggfunc="mean")
        res["by_urgency"][u] = {"n_scenes": int(len(s)),
                                "safety_local": round(float(s["local"].mean()), 2),
                                "safety_cloud": round(float(s["cloud"].mean()), 2)}

    res["agreement"] = agreement(d)

    # what the rated warnings actually say (simple lexical cues, 50 scenes each)
    import re
    cues = {"asks_to_slow": r"reduce|slow|brake|decelerat",
            "gives_target_speed": r"\d+\s*mph",
            "says_maintain_speed": r"maintain (your )?(current )?speed",
            "asks_to_steer": r"steer|swerve|shift|straddle"}
    res["content_cues"] = {}
    for sysn, k in (("local", "local"), ("cloud", "paper1_gemini")):
        txt = [(WARN[s][k]["warning_message"] + " " + WARN[s][k]["maneuver_suggestion"]).lower()
               for s in KEY]
        res["content_cues"][sysn] = {c: round(float(np.mean([bool(re.search(p, t)) for t in txt])), 2)
                                     for c, p in cues.items()}
        res["content_cues"][sysn]["mean_words"] = round(float(np.mean([len(t.split()) for t in txt])), 1)
    res["content_cues"]["urgency_agreement_local_vs_cloud"] = round(float(np.mean(
        [WARN[s]["local"]["urgency_level"] == WARN[s]["paper1_gemini"].get("urgency_level")
         for s in KEY])), 2)

    # round 1 (unconstrained generator, pre-final corpus) for the comparison table
    r1 = json.loads(ROUND1.read_text())
    res["round1_reference"] = {
        "system": "unconstrained local generator, pre-final corpus (49 scenes)",
        "per_system": r1["per_system"], "preference": r1["preference"],
        "differences_local_minus_cloud": r1["differences_local_minus_cloud"]}

    # automated judge on the reported system (150 random scenes)
    j = json.loads(JUDGE_150.read_text())
    res["judge_150_constrained"] = {
        "judge": j["config"]["judge"], "n_scenes": j["config"]["n_scenes"],
        "local_source": j["config"]["local_source"],
        "mean_ratings": j["mean_ratings"], "preference": j["preference"],
        "overlap_with_human_scenes": len({r["scene"] for r in j["rows"]} & set(KEY))}
    if JUDGE_SAME.exists():   # same-item comparison, written by llm_judge_warnings.py
        js = json.loads(JUDGE_SAME.read_text())
        if js["config"]["n_scenes"] == 50:
            jt = {r["scene"]: r for r in js["rows"]}
            item = {}
            for m in ("clarity", "safety"):
                h, a = [], []
                for (sid, sysn), g in d.groupby(["scene", "system"]):
                    h.append(g[m].mean())
                    a.append(np.mean(jt[sid][f"{sysn}_{m}"]))
                item[m] = {"spearman": round(float(stats.spearmanr(h, a)[0]), 3),
                           "judge_minus_human_mean": round(float(np.mean(a) - np.mean(h)), 3)}
            res["judge_same_50"] = {"mean_ratings": js["mean_ratings"],
                                    "preference": js["preference"], "item_level": item}

    out = P2 / "results/human_eval_round2.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print("Saved:", out)


if __name__ == "__main__":
    main()
