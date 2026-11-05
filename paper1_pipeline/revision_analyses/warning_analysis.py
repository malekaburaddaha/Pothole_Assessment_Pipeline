"""Warning-generation statistics (paper Tables 9, 10, 11, 12 and 13).

    python paper1_pipeline/revision_analyses/warning_analysis.py

Runs from the released files only:
  data/enpad/warnings_hybrid.json, data/enpad/warnings_image_only.json   all 1,119 scenes
  data/human_studies/warning_study_ratings.csv                            50 scenes x 3 raters

The two methods' urgency labels are paired on the same scenes, so their
distributions are compared with the Stuart-Maxwell test of marginal homogeneity.
The 150 ratings are clustered by scene, so rating differences are averaged over
raters within each scene (n = 50) and tested with the Wilcoxon signed-rank test,
with scene-resampling bootstrap intervals. Preferences are analysed per scene.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score as ck

ROOT = Path(__file__).resolve().parents[2]
URG = ["low", "medium", "high", "critical"]
rng = np.random.default_rng(7)


def stuart_maxwell(a, b, k):
    N = np.zeros((k, k))
    for i, j in zip(a, b):
        N[i, j] += 1
    d = np.array([N[i, :].sum() - N[:, i].sum() for i in range(k)])[:-1]
    S = np.array([[N[i, :].sum() + N[:, i].sum() - 2 * N[i, i] if i == j else -(N[i, j] + N[j, i])
                   for j in range(k - 1)] for i in range(k - 1)])
    chi2 = float(d @ np.linalg.solve(S, d))
    return chi2, float(stats.chi2.sf(chi2, k - 1)), N


def fleiss(table):
    n = table.sum(axis=1)[0]
    p = table.sum(axis=0) / table.sum()
    P = ((table ** 2).sum(axis=1) - n) / (n * (n - 1))
    return float((P.mean() - (p ** 2).sum()) / (1 - (p ** 2).sum()))


def main():
    out = {}
    h = json.loads((ROOT / "data/enpad/warnings_hybrid.json").read_text())
    im = json.loads((ROOT / "data/enpad/warnings_image_only.json").read_text())
    ids = sorted(set(h) & set(im))
    a = [URG.index(im[s]["warning"]["urgency_level"].lower()) for s in ids]
    b = [URG.index(h[s]["warning"]["urgency_level"].lower()) for s in ids]
    chi2, p, N = stuart_maxwell(a, b, 4)
    out["urgency"] = {"n_scenes": len(ids),
                      "image_only": dict(Counter(URG[i] for i in a)), "hybrid": dict(Counter(URG[i] for i in b)),
                      "stuart_maxwell_chi2": round(chi2, 2), "df": 3, "p": p,
                      "high_to_medium": int(N[2, 1]), "high_to_critical": int(N[2, 3]),
                      "cohen_kappa": round(ck(a, b), 4),
                      "exact_agreement_pct": round(100 * float(np.mean(np.array(a) == np.array(b))), 1),
                      "adjacent_agreement_pct": round(100 * float(np.mean(np.abs(np.array(a) - np.array(b)) <= 1)), 1)}

    r = pd.read_csv(ROOT / "data/human_studies/warning_study_ratings.csv")
    A_h = r.method_A == "hybrid"
    for crit in ("clarity", "safety"):
        r[f"{crit}_hybrid"] = np.where(A_h, r[f"{crit}_A"], r[f"{crit}_B"])
        r[f"{crit}_image_only"] = np.where(A_h, r[f"{crit}_B"], r[f"{crit}_A"])
        r[f"{crit}_diff"] = r[f"{crit}_hybrid"] - r[f"{crit}_image_only"]
        sc = r.groupby("scene_id")[f"{crit}_diff"].mean()
        scenes = sc.index.to_numpy()
        boots = [sc.loc[rng.choice(scenes, size=len(scenes), replace=True)].mean() for _ in range(5000)]
        out[crit] = {"hybrid_mean_sd": [round(r[f"{crit}_hybrid"].mean(), 2), round(r[f"{crit}_hybrid"].std(), 2)],
                     "image_only_mean_sd": [round(r[f"{crit}_image_only"].mean(), 2), round(r[f"{crit}_image_only"].std(), 2)],
                     "mean_difference": round(r[f"{crit}_diff"].mean(), 3),
                     "scene_bootstrap_ci95": [round(np.percentile(boots, 2.5), 2), round(np.percentile(boots, 97.5), 2)],
                     "scene_level_wilcoxon_p": float(stats.wilcoxon(sc, zero_method="wilcox").pvalue),
                     "per_rater": {k: {"mean_difference": round(g[f"{crit}_diff"].mean(), 2),
                                       "wilcoxon_p": float(stats.wilcoxon(g[f"{crit}_diff"], zero_method="wilcox").pvalue)}
                                   for k, g in r.groupby("rater")}}
    # inter-rater reliability per method x criterion (Fleiss kappa on the 1-5 ratings)
    irr = {}
    for meth in ("image_only", "hybrid"):
        for crit in ("clarity", "safety"):
            w = r.pivot(index="scene_id", columns="rater", values=f"{crit}_{meth}")
            table = np.stack([np.bincount(w.loc[s].astype(int), minlength=7)[1:] for s in w.index])
            irr[f"{meth}_{crit}"] = round(fleiss(table), 3)
    out["fleiss_kappa"] = irr
    pref = np.where(r.preference == "TIE", 0, np.where((r.preference == "A") == A_h, 1, -1))
    r["vote"] = pref
    votes = Counter(pref)
    per_scene = r.groupby("scene_id")["vote"].sum()
    win, lose = int((per_scene > 0).sum()), int((per_scene < 0).sum())
    g = {k: v.vote.values for k, v in r.groupby("scene_id")}
    keys = list(g)
    ratios = []
    for _ in range(10000):
        bvec = np.concatenate([g[k] for k in rng.choice(keys, len(keys))])
        ratios.append((bvec == 1).sum() / max(1, (bvec == -1).sum()))
    out["preference"] = {"hybrid": int(votes[1]), "image_only": int(votes[-1]), "tie": int(votes[0]),
                         "ratio": round(votes[1] / votes[-1], 2),
                         "ratio_scene_bootstrap_ci95": [round(np.percentile(ratios, 2.5), 1), round(np.percentile(ratios, 97.5), 1)],
                         "scenes_hybrid_more": win, "scenes_image_only_more": lose, "scenes_tied": int((per_scene == 0).sum()),
                         "scene_binomial_p": float(stats.binomtest(win, win + lose, 0.5).pvalue),
                         "per_rater": {k: [int((v.vote == 1).sum()), int((v.vote == -1).sum()), int((v.vote == 0).sum())] for k, v in r.groupby("rater")}}
    dst = Path(__file__).parent / "results/warning_analysis.json"
    dst.parent.mkdir(exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
