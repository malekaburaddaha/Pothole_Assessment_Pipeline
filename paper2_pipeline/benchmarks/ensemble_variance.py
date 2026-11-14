"""
Paper 2 — how variable is the distilled ensemble, really?

The originally reported ensemble kappa (0.404) came from one particular set of six
trained models. Two runs with the same seed later produced models differing by
kappa 0.13 vs 0.33 on the 38-item human set, revealing that training was
nondeterministic and that the reported bootstrap CI (over test items) omitted the
larger source of variance: training itself.

Training is now deterministic. This script quantifies both variance components:
  - across INDEPENDENT ensembles (different seed triples), and
  - across items (bootstrap), for the pooled prediction.

Output: results/ensemble_variance.json
"""
import json
import sys
from collections import Counter
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import cohen_kappa_score

ROOT = Path(os.environ.get("POTHOLE_ROOT", "."))  # local research-tree layout; see README
P2 = ROOT / "Paper2_Improvements"
CKPT = P2 / "stage4_5_local_vlm/ckpt"
D = ["low", "medium", "high"]
DEV = "cuda:0"
ENSEMBLES = {"A": [42, 43, 44], "B": [45, 46, 47], "C": [48, 49, 50]}

sys.path.insert(0, str(P2 / "stage4_5_local_vlm"))
from train_depth_distilled import TwoTower, geo_vector  # noqa: E402
import torchvision.transforms as T  # noqa: E402

TF = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def load(variant, seed):
    p = CKPT / f"distill_convnext_tiny_{variant}_s{seed}.pt"
    if not p.exists():
        return None
    m = TwoTower("convnext_tiny", use_geo=(variant == "twotower"))
    m.load_state_dict(torch.load(p, map_location="cpu"))
    return m.to(DEV).eval()


def main():
    ev = json.loads((P2 / "results/human_depth_eval_set.json").read_text())["records"]
    gem = json.loads((ROOT / "Main_Work/final_assessments/Json_and_Log_Files/"
                      "gemini-2_5-pro_full_dataset.json").read_text())
    usable = {k: v for k, v in ev.items() if v["human_consensus"]}
    ids = sorted(usable)
    y = [usable[k]["human_consensus"] for k in ids]

    # pre-load inputs once
    inputs = []
    for pid in ids:
        rec = usable[pid]
        cp = ROOT / "Main_Work/test_data/image_patches" / rec["crop_image"].split("image_patches/")[1]
        img = TF(Image.open(cp).convert("RGB"))[None].to(DEV)
        g = gem[pid]
        geo = torch.tensor([geo_vector(g.get("geometric_features", {}),
                                       g.get("positional_features", {}))],
                           dtype=torch.float32).to(DEV)
        inputs.append((img, geo))

    def predict(models):
        out = []
        with torch.no_grad():
            for img, geo in inputs:
                v = np.zeros(3, int)
                for m in models:
                    v[int(m(img, geo).argmax(1))] += 1
                out.append(D[int(v.argmax())])
        return out

    res = {"per_model": {}, "per_ensemble": {}}
    # individual models
    singles = []
    for variant in ("image", "twotower"):
        for seeds in ENSEMBLES.values():
            for s in seeds:
                m = load(variant, s)
                if m is None:
                    continue
                p = predict([m])
                k = float(cohen_kappa_score(y, p, labels=D))
                res["per_model"][f"{variant}_s{s}"] = round(k, 4)
                singles.append(k)
                del m
                torch.cuda.empty_cache()
    # independent ensembles of six (image+twotower over a seed triple)
    ens_k, ens_pred = [], {}
    for name, seeds in ENSEMBLES.items():
        models = [m for v in ("image", "twotower") for s in seeds
                  if (m := load(v, s)) is not None]
        if len(models) < 6:
            continue
        p = predict(models)
        ens_pred[name] = p
        k = float(cohen_kappa_score(y, p, labels=D))
        qwk = float(cohen_kappa_score([D.index(a) for a in y],
                                      [D.index(a) for a in p], weights="quadratic"))
        res["per_ensemble"][name] = {
            "seeds": seeds, "kappa": round(k, 4), "weighted_kappa": round(qwk, 4),
            "agreement_pct": round(100 * float(np.mean(np.array(y) == np.array(p))), 2),
            "distribution": dict(Counter(p))}
        ens_k.append(k)
        for m in models:
            del m
        torch.cuda.empty_cache()

    rng = np.random.default_rng(3)
    summary = {
        "n_items": len(ids),
        "single_model_kappa": {"mean": round(float(np.mean(singles)), 4),
                               "sd": round(float(np.std(singles, ddof=1)), 4),
                               "min": round(float(np.min(singles)), 4),
                               "max": round(float(np.max(singles)), 4),
                               "n_models": len(singles)},
        "ensemble_kappa_across_independent_ensembles": {
            "mean": round(float(np.mean(ens_k)), 4),
            "sd": round(float(np.std(ens_k, ddof=1)), 4) if len(ens_k) > 1 else None,
            "values": [round(k, 4) for k in ens_k], "n_ensembles": len(ens_k)},
    }
    # item-level bootstrap on the pooled 18-model vote, for comparison
    if ens_pred:
        allm = [m for v in ("image", "twotower") for seeds in ENSEMBLES.values()
                for s in seeds if (m := load(v, s)) is not None]
        pooled = predict(allm)
        yi = np.array([D.index(a) for a in y]); pi = np.array([D.index(a) for a in pooled])
        bs = []
        for _ in range(5000):
            s = rng.choice(len(yi), len(yi), replace=True)
            if len(set(yi[s])) > 1:
                bs.append(cohen_kappa_score(yi[s], pi[s]))
        summary["pooled_18_model_vote"] = {
            "kappa": round(float(cohen_kappa_score(y, pooled, labels=D)), 4),
            "item_bootstrap_CI95": [round(float(np.percentile(bs, 2.5)), 4),
                                    round(float(np.percentile(bs, 97.5)), 4)],
            "distribution": dict(Counter(pooled))}
    res["summary"] = summary
    res["note"] = ("Training is now deterministic (seeded RNGs, cudnn.deterministic, "
                   "deterministic algorithms, seeded workers), so these checkpoints "
                   "are reproducible. The spread across independent ensembles is the "
                   "quantity the paper reports; an item-level bootstrap alone "
                   "understates uncertainty because it omits training variance.")
    (P2 / "results/ensemble_variance.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(summary, indent=1))
    print(json.dumps(res["per_ensemble"], indent=1))


if __name__ == "__main__":
    main()
