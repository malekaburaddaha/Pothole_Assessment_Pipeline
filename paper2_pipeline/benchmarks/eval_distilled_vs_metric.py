"""
Paper 2 — validate the distilled depth ensemble against PHYSICALLY MEASURED depth.

Paper 1's Stage-4 engine (Gemini 2.5 Pro) was validated this way and agreed only
weakly with sensor depth (kappa 0.098, recovering 7% of >5 cm defects). The obvious
question for Paper 2 is whether the distilled student inherits that bias, since it
was trained on the teacher's labels. This runs the identical protocol on the
6-model ConvNeXt ensemble.

Uses the metric ground truth produced for Paper 1 (m1_metric_depth_gt.json) so the
two systems are scored on exactly the same potholes.

Output: Paper2_Improvements/results/distilled_vs_metric.json
"""
import json
import random
import sys
from collections import Counter
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(os.environ.get("POTHOLE_ROOT", "."))  # local research-tree layout; see README
P2 = ROOT / "Paper2_Improvements"
GT = ROOT / "Paper1_Resubmission/m1_metric_depth_gt.json"
CROPS = ROOT / "Paper1_Resubmission/m1_crops"
DS = ROOT / "PothRGBD_Dataset/PUBLIC_POTHOLE_DATASET"
CKPT = P2 / "stage4_5_local_vlm/ckpt"
DEPTH = ["low", "medium", "high"]
MAX_MM = 200
DEVICE = "cuda:0"

sys.path.insert(0, str(P2 / "stage4_5_local_vlm"))
from train_depth_distilled import TwoTower, geo_vector  # noqa: E402
import torchvision.transforms as T  # noqa: E402

TF = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
ENSEMBLE = [("image", 42), ("image", 43), ("image", 44),
            ("twotower", 42), ("twotower", 43), ("twotower", 44)]

# SCALE=1 evaluates the scale-augmented ensemble instead. The deployed ensemble
# collapses to 'high' out of domain because PothRGBD potholes are captured
# close-range and fill the crop (median 68,560 px) whereas training crops are
# distant dashcam defects (median 3,606 px) - and large, high-contrast, detail-rich
# crops are exactly what the student learned to call 'high'. Scale jitter during
# training is the obvious mitigation, so we test it rather than assert it. If the
# collapse persists, the honest conclusion is that the failure is not fixable by
# augmentation alone and the engine must be re-distilled per capture geometry.
SUFFIX = "_scaleaug" if __import__("os").getenv("SCALE", "0") == "1" else ""


def load_models():
    ms = []
    for variant, seed in ENSEMBLE:
        p = CKPT / f"distill_convnext_tiny_{variant}_s{seed}{SUFFIX}.pt"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing; train it before evaluating")
        m = TwoTower("convnext_tiny", use_geo=(variant == "twotower"))
        m.load_state_dict(torch.load(p, map_location="cpu"))
        m.to(DEVICE).eval()
        ms.append(m)
    return ms


def crop_for(rec):
    """Same crop the Paper-1 metric study used, if present; else regenerate."""
    name = f"{Path(rec['image']).stem}_p{rec['poly_id']}.jpg"
    p = CROPS / name
    if p.exists():
        return p
    img = cv2.imread(str(DS / "images" / rec["image"]))
    if img is None:
        return None
    H, W = img.shape[:2]
    x1, y1, x2, y2 = rec["bbox"]
    pad = int(0.25 * max(x2 - x1, y2 - y1))
    c = img[max(0, y1-pad):min(H, y2+pad), max(0, x1-pad):min(W, x2+pad)]
    if c.size == 0:
        return None
    CROPS.mkdir(exist_ok=True)
    cv2.imwrite(str(p), c)
    return p


def main():
    gt = [r for r in json.loads(GT.read_text())["potholes"]
          if 0 <= r["depth_mm_p90"] <= MAX_MM]
    by = {c: [r for r in gt if r["metric_category"] == c] for c in DEPTH}
    random.seed(13)                      # same seed/sample as the Paper-1 study
    sample = []
    for c in DEPTH:
        sample += random.sample(by[c], min(55, len(by[c])))
    print(f"sample: {len(sample)}  {dict(Counter(r['metric_category'] for r in sample))}")

    models = load_models()
    preds, kept = [], []
    for rec in sample:
        p = crop_for(rec)
        if p is None:
            continue
        img = TF(Image.open(p).convert("RGB"))[None].to(DEVICE)
        w = rec["bbox"][2] - rec["bbox"][0]
        h = rec["bbox"][3] - rec["bbox"][1]
        per = 2 * (w + h)
        metrics = {"area": rec["area_px"],
                   "circularity": 4 * np.pi * rec["area_px"] / (per ** 2) if per else 0,
                   "solidity": min(rec["area_px"] / max(w * h, 1), 1.0),
                   "aspect_ratio": w / h if h else 0}
        geo = torch.tensor([geo_vector(metrics, {"Distance_Category": "Near",
                                                 "Lateral_Position": "in_path_right"})],
                           dtype=torch.float32).to(DEVICE)
        votes = np.zeros(3, int)
        with torch.no_grad():
            for m in models:
                votes[int(m(img, geo).argmax(1))] += 1
        preds.append(DEPTH[int(votes.argmax())])
        kept.append(rec)

    from sklearn.metrics import (cohen_kappa_score, confusion_matrix,
                                 precision_recall_fscore_support)
    from scipy import stats
    y = [r["metric_category"] for r in kept]
    yi = [DEPTH.index(v) for v in y]
    pi = [DEPTH.index(v) for v in preds]
    mm = np.array([r["depth_mm_p90"] for r in kept])
    rng = np.random.default_rng(4)
    boots = []
    for _ in range(5000):
        s = rng.choice(len(yi), len(yi), replace=True)
        if len(set(np.array(yi)[s])) > 1:
            boots.append(cohen_kappa_score(np.array(yi)[s], np.array(pi)[s],
                                           weights="quadratic"))
    pr, rc, f1, sup = precision_recall_fscore_support(y, preds, labels=DEPTH,
                                                      zero_division=0)
    rho = stats.spearmanr(pi, mm)
    means = {c: round(float(mm[[i for i, v in enumerate(preds) if v == c]].mean()), 1)
             for c in DEPTH if any(v == c for v in preds)}

    out = {
        "config": {"engine": "distilled ConvNeXt-Tiny 6-model ensemble"
                             + (" WITH SCALE AUGMENTATION" if SUFFIX else ""),
                   "checkpoint_suffix": SUFFIX or "(none)",
                   "dataset": "PothRGBD (independent RGB-D)", "n": len(kept),
                   "protocol": "identical sample/seed to the Paper-1 Gemini validation"},
        "metrics": {
            "kappa": round(float(cohen_kappa_score(y, preds, labels=DEPTH)), 4),
            "quadratic_weighted_kappa": round(float(
                cohen_kappa_score(yi, pi, weights="quadratic")), 4),
            "qwk_bootstrap_CI95": [round(float(np.percentile(boots, 2.5)), 3),
                                   round(float(np.percentile(boots, 97.5)), 3)],
            "exact_agreement_pct": round(100 * float(np.mean(np.array(y) == np.array(preds))), 2),
            "within_one_category_pct": round(100 * float(np.mean(
                [abs(a - b) <= 1 for a, b in zip(yi, pi)])), 2),
            "spearman_pred_vs_mm": [round(float(rho.correlation), 4), float(rho.pvalue)],
            "mean_measured_mm_by_prediction": means,
            "per_class": {c: {"precision": round(float(pr[i]), 3),
                              "recall": round(float(rc[i]), 3), "support": int(sup[i])}
                          for i, c in enumerate(DEPTH)},
            "confusion_rows_metric_truth": confusion_matrix(y, preds, labels=DEPTH).tolist(),
            "prediction_distribution": dict(Counter(preds)),
        },
        "gemini_reference_same_sample": {
            "kappa": 0.0981, "quadratic_weighted_kappa": 0.1781,
            "exact_agreement_pct": 42.38, "high_recall": 0.073,
            "note": "from Paper1_Resubmission/m1_vlm_vs_metric.json",
        },
    }
    p = P2 / f"results/distilled_vs_metric{SUFFIX}.json"
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["metrics"], indent=1))
    print("Saved:", p)


if __name__ == "__main__":
    main()
