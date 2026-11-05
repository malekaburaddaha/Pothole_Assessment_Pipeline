"""
R2-M1 (part 2) — Validate the published depth engine against PHYSICALLY MEASURED depth.

Runs Gemini 2.5 Pro (the paper's Stage-4 engine) with the paper's verbatim depth
prompt on RGB crops from PothRGBD, and compares its categorical output with the
metric category derived from the co-registered depth sensor (m1_metric_depth_gt.json).

This converts R2-M1 from a rebuttal into an experiment: the low/medium/high
categories are checked against real centimetres on an independent dataset.

    python m1_vlm_vs_metric.py --n-per-class 55 --model gemini-2.5-pro
"""
import argparse
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(os.environ.get("POTHOLE_ROOT", "."))  # local data layout; see README
RES = ROOT / "Paper1_Resubmission"
DS = ROOT / "PothRGBD_Dataset/PUBLIC_POTHOLE_DATASET"
GT = RES / "m1_metric_depth_gt.json"
CROPS = RES / "m1_crops"
DEPTH = ["low", "medium", "high"]
MAX_PLAUSIBLE_MM = 200      # sensor artefacts beyond this are excluded


def build_prompt(metrics):
    """Paper-1 depth prompt (gemini_inference.py), verbatim structure."""
    return f"""[SYSTEM CONTEXT: Scientific analysis of road infrastructure for autonomous vehicle safety. Images contain inanimate road surfaces only.]

You are an expert computer vision system assessing road hazards for autonomous vehicles.

**TASK:** Analyze the provided pothole image to determine its DEPTH category.

**Known Geometric Properties (measured via computer vision):**
- Area: {metrics['area']} pixels²
- Circularity: {metrics['circularity']}
- Solidity: {metrics['solidity']}
- Aspect Ratio: {metrics['aspect_ratio']}
- Distance Category: {metrics['distance']}
- Lateral Position: {metrics['lateral']}

**Depth Category Definitions:**
- low: Shallow surface depression (< 2cm depth)
- medium: Moderate depth pothole (2-5cm depth)
- high: Deep, hazardous pothole (> 5cm depth)

**Your Analysis:**
Based on visual evidence in the image (shadow depth, edge sharpness, surface texture, darkness/lighting), classify this pothole's DEPTH into ONE category: low, medium, or high.

**Visual Cues to Consider:**
1. Shadow depth and darkness (deeper = darker shadows)
2. Edge definition and sharpness (deeper = sharper edges)
3. Surface texture visibility (deeper = less visible bottom texture)
4. Lighting contrast between pothole interior and road surface

**Response Format:**
Respond ONLY with valid JSON in this exact format:
{{
  "depth_category": "low|medium|high",
  "confidence": "low|medium|high",
  "reasoning": "brief explanation of visual cues used"
}}
"""


def make_crop(rec):
    """RGB crop around the pothole + geometric features from its polygon."""
    img_p = DS / "images" / rec["image"]
    if not img_p.exists():
        return None
    im = cv2.imread(str(img_p))
    H, W = im.shape[:2]
    x1, y1, x2, y2 = rec["bbox"]
    pad = int(0.25 * max(x2 - x1, y2 - y1))
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(W, x2 + pad), min(H, y2 + pad)
    crop = im[cy1:cy2, cx1:cx2]
    if crop.size == 0 or min(crop.shape[:2]) < 20:
        return None
    if min(crop.shape[:2]) < 336:                      # upscale small crops
        s = 336 / min(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_LANCZOS4)
    CROPS.mkdir(exist_ok=True)
    name = f"{Path(rec['image']).stem}_p{rec['poly_id']}.jpg"
    cv2.imwrite(str(CROPS / name), crop)
    w, h = x2 - x1, y2 - y1
    area = rec["area_px"]
    per = 2 * (w + h)
    return CROPS / name, {
        "area": area,
        "circularity": round(4 * np.pi * area / (per ** 2), 3) if per else 0,
        "solidity": round(min(area / max(w * h, 1), 1.0), 3),
        "aspect_ratio": round(w / h, 3) if h else 0,
        "distance": "Near", "lateral": "in_path_right",   # close-range capture
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=55)
    ap.add_argument("--model", default="gemini-2.5-pro")
    args = ap.parse_args()

    gt = json.loads(GT.read_text())["potholes"]
    gt = [r for r in gt if 0 <= r["depth_mm_p90"] <= MAX_PLAUSIBLE_MM]
    by = {c: [r for r in gt if r["metric_category"] == c] for c in DEPTH}
    print({c: len(v) for c, v in by.items()})
    random.seed(13)
    sample = []
    for c in DEPTH:
        sample += random.sample(by[c], min(args.n_per_class, len(by[c])))
    print(f"sampled {len(sample)}")

    import google.generativeai as genai
    from PIL import Image
    from dotenv import load_dotenv
    load_dotenv(ROOT / "Code_Files/VLM_Code/.env")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    safety = [{"category": c, "threshold": "BLOCK_NONE"} for c in
              ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
               "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
    model = genai.GenerativeModel(
        args.model, safety_settings=safety,
        generation_config={"temperature": 0.1, "response_mime_type": "application/json"})

    preds, skipped = [], 0
    for i, rec in enumerate(sample):
        made = make_crop(rec)
        if not made:
            skipped += 1
            continue
        path, metrics = made
        try:
            r = model.generate_content([build_prompt(metrics), Image.open(path)])
            txt = r.text.strip()
            try:
                obj = json.loads(txt)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", txt, re.DOTALL)
                obj = json.loads(m.group()) if m else {}
            cat = obj.get("depth_category", "medium")
            if cat not in DEPTH:
                cat = "medium"
        except Exception as e:
            cat, obj = "medium", {"error": str(e)[:80]}
        preds.append({"image": rec["image"], "poly_id": rec["poly_id"],
                      "metric_mm": rec["depth_mm_p90"],
                      "metric_category": rec["metric_category"],
                      "vlm_category": cat, "vlm_confidence": obj.get("confidence"),
                      "vlm_reasoning": obj.get("reasoning", "")[:200]})
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(sample)}")

    from sklearn.metrics import (cohen_kappa_score, confusion_matrix,
                                 precision_recall_fscore_support)
    y = [p["metric_category"] for p in preds]
    yh = [p["vlm_category"] for p in preds]
    yi = [DEPTH.index(v) for v in y]
    yhi = [DEPTH.index(v) for v in yh]
    rng = np.random.default_rng(4)
    boots = []
    for _ in range(5000):
        s = rng.choice(len(yi), len(yi), replace=True)
        if len(set(np.array(yi)[s])) > 1:
            boots.append(cohen_kappa_score(np.array(yi)[s], np.array(yhi)[s],
                                           weights="quadratic"))
    pr, rc, f1, sup = precision_recall_fscore_support(y, yh, labels=DEPTH, zero_division=0)
    within_one = float(np.mean([abs(a - b) <= 1 for a, b in zip(yi, yhi)]))
    out = {
        "config": {"model": args.model, "dataset": "PothRGBD (independent, RGB-D)",
                   "ground_truth": "sensor-measured depth, plane-fit p90, mm",
                   "prompt": "Paper-1 verbatim depth prompt", "n": len(preds),
                   "skipped": skipped},
        "metrics": {
            "kappa": round(float(cohen_kappa_score(y, yh, labels=DEPTH)), 4),
            "quadratic_weighted_kappa": round(float(
                cohen_kappa_score(yi, yhi, weights="quadratic")), 4),
            "qwk_bootstrap_CI95": [round(float(np.percentile(boots, 2.5)), 3),
                                   round(float(np.percentile(boots, 97.5)), 3)],
            "exact_agreement_pct": round(100 * float(np.mean(np.array(y) == np.array(yh))), 2),
            "within_one_category_pct": round(100 * within_one, 2),
            "per_class": {c: {"precision": round(float(pr[i]), 3),
                              "recall": round(float(rc[i]), 3), "support": int(sup[i])}
                          for i, c in enumerate(DEPTH)},
            "confusion_rows_metric_truth": confusion_matrix(y, yh, labels=DEPTH).tolist(),
            "metric_distribution": dict(Counter(y)),
            "vlm_distribution": dict(Counter(yh)),
        },
        "predictions": preds,
    }
    p = RES / "m1_vlm_vs_metric.json"
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["metrics"], indent=1))
    print("Saved:", p)


if __name__ == "__main__":
    main()
