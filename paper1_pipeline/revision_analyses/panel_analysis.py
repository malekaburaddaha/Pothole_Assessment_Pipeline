"""Severity agreement against the two human panels (paper Tables 6, 7, 8 and 16).

    python paper1_pipeline/revision_analyses/panel_analysis.py

Runs from the released files only:
  data/human_studies/depth_panel1.csv   first panel (model predictions were shown to annotators)
  data/human_studies/depth_panel2.csv   second, prediction-free panel
  data/enpad/severity_gemini_2_5_pro.json   released Stage-4 labels

The `shown_*` columns of panel 1 are the validation-run predictions of the three
candidate models (the run used for model selection). The released labels come from
the separate full-dataset run; the two disagree on 8 of the 38 consensus items.
Consensus is the majority label; three-way splits are excluded, never tie-broken.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score as ck

ROOT = Path(__file__).resolve().parents[2]
D = ["low", "medium", "high"]
rng = np.random.default_rng(17)


def idx(v):
    return [D.index(x) for x in v]


def block(y, p, boot=5000):
    y, p = np.array(idx(y)), np.array(idx(p))
    k, q = [], []
    for _ in range(boot):
        s = rng.choice(len(y), len(y))
        if len(set(y[s])) > 1 and len(set(p[s])) > 1:
            k.append(ck(y[s], p[s])); q.append(ck(y[s], p[s], weights="quadratic"))
    cm = [[int(((y == r) & (p == c)).sum()) for c in range(3)] for r in range(3)]
    pr = {D[c]: {"precision": round(cm[c][c] / max(1, sum(cm[r][c] for r in range(3))), 3),
                 "recall": round(cm[c][c] / max(1, sum(cm[c])), 3)} for c in range(3)}
    return {"n": len(y), "kappa": round(ck(y, p), 4), "kappa_ci95": [round(np.percentile(k, 2.5), 3), round(np.percentile(k, 97.5), 3)],
            "weighted_kappa": round(ck(y, p, weights="quadratic"), 4),
            "weighted_kappa_ci95": [round(np.percentile(q, 2.5), 3), round(np.percentile(q, 97.5), 3)],
            "agreement_pct": round(100 * float((y == p).mean()), 1), "confusion_rows_consensus": cm, "per_class": pr}


def consensus(df, raters):
    out = {}
    for pid, row in df.iterrows():
        c = Counter(row[r] for r in raters).most_common(1)[0]
        out[pid] = c[0] if c[1] >= 2 else None
    return out


def pairwise(df, raters, ids):
    pairs = [(a, b) for i, a in enumerate(raters) for b in raters[i + 1:]]
    vals = {f"{a}_vs_{b}": round(ck(idx(df.loc[ids, a]), idx(df.loc[ids, b])), 4) for a, b in pairs}
    agree = {f"{a}_vs_{b}": round(100 * float((df.loc[ids, a] == df.loc[ids, b]).mean()), 1) for a, b in pairs}
    return {"pairwise_kappa": vals, "mean_pairwise_kappa": round(float(np.mean(list(vals.values()))), 4), "pairwise_agreement_pct": agree}


def main():
    hs = ROOT / "data/human_studies"
    p1 = pd.read_csv(hs / "depth_panel1.csv").set_index("pothole_id")
    p2 = pd.read_csv(hs / "depth_panel2.csv").set_index("pothole_id")
    rel = {k: v["depth_category"] for k, v in json.loads((ROOT / "data/enpad/severity_gemini_2_5_pro.json").read_text()).items()}
    R = ["R1", "R2", "R3"]
    out = {}

    c1 = consensus(p1, R)
    cnt = Counter(max(Counter(p1.loc[i, R]).values()) for i in p1.index)
    ids1 = [i for i in p1.index if c1[i]]
    out["panel1"] = {"n_items": len(p1), "unanimous": cnt[3], "majority": cnt[2], "three_way_split": cnt[1],
                     "n_consensus": len(ids1), **pairwise(p1, R, list(p1.index))}
    out["panel1_models"] = {name: block([c1[i] for i in ids1], [p1.loc[i, col] for i in ids1])
                            for name, col in (("gemini_2_5_pro_validation_run", "shown_pro_prediction"),
                                              ("gemini_2_5_flash_validation_run", "shown_flash_prediction"),
                                              ("gpt5_validation_run", "shown_gpt5_prediction"))}
    out["panel1_models"]["gemini_2_5_pro_released_labels"] = block([c1[i] for i in ids1], [rel[i] for i in ids1])
    out["validation_vs_released_disagreements"] = sum(p1.loc[i, "shown_pro_prediction"] != rel[i] for i in ids1)

    c2 = consensus(p2, R)
    first = [i for i in p2.index if p2.loc[i, "in_first_panel"]]
    new = [i for i in p2.index if not p2.loc[i, "in_first_panel"]]
    out["panel2"] = {"n_items": len(p2), "three_way_split": sum(v is None for v in c2.values()),
                     "all_items": pairwise(p2, R, list(p2.index)), "first_panel_items": pairwise(p2, R, first),
                     "new_items": pairwise(p2, R, new)}
    for name, ids in (("same_42_items", first), ("new_items", new), ("all_items", list(p2.index))):
        keep = [i for i in ids if c2[i]]
        out[f"released_labels_vs_panel2_{name}"] = block([c2[i] for i in keep], [rel[i] for i in keep])
    both = [i for i in ids1 if c2.get(i)]
    out["panel1_vs_panel2_consensus_kappa"] = round(ck(idx([c1[i] for i in both]), idx([c2[i] for i in both])), 4)

    dst = Path(__file__).parent / "results/panel_analysis.json"
    dst.parent.mkdir(exist_ok=True)
    dst.write_text(json.dumps(out, indent=1, default=int))
    print(json.dumps({k: (v if not isinstance(v, dict) else {x: v[x] for x in v if x in ("n", "kappa", "weighted_kappa", "mean_pairwise_kappa", "n_consensus")}) for k, v in out.items()}, indent=1, default=int))


if __name__ == "__main__":
    main()
