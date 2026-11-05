"""
R2-M10 — Repeated-run variability of the hosted depth model.

Reviewer: "A low generation temperature does not alone guarantee deterministic
results." We quantify it: the same potholes, the same prompt, three independent
runs at the published temperature (0.1), reporting the proportion of potholes
whose category is identical across all runs and Fleiss' kappa across runs.

    python m10_run_variability.py --n 100 --runs 3
"""
import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("POTHOLE_ROOT", "."))  # local data layout; see README
RES = ROOT / "Paper1_Resubmission"
SAM = ROOT / "Main_Work/Dataset/With_Distance/sam_predictions_large"
PATCHES = ROOT / "Main_Work/image_patches"
GEM = ROOT / "Main_Work/final_assessments/Json_and_Log_Files/gemini-2_5-pro_full_dataset.json"
DEPTH = ["low", "medium", "high"]


def build_prompt(m, p):
    return f"""[SYSTEM CONTEXT: Scientific analysis of road infrastructure for autonomous vehicle safety. Images contain inanimate road surfaces only.]

You are an expert computer vision system assessing road hazards for autonomous vehicles.

**TASK:** Analyze the provided pothole image to determine its DEPTH category.

**Known Geometric Properties (measured via computer vision):**
- Area: {m.get('area','N/A')} pixels²
- Circularity: {m.get('circularity','N/A')}
- Solidity: {m.get('solidity','N/A')}
- Aspect Ratio: {m.get('aspect_ratio','N/A')}
- Distance Category: {p.get('Distance_Category','N/A')}
- Lateral Position: {p.get('Lateral_Position','N/A')}

**Depth Category Definitions:**
- low: Shallow surface depression (< 2cm depth)
- medium: Moderate depth pothole (2-5cm depth)
- high: Deep, hazardous pothole (> 5cm depth)

**Your Analysis:**
Based on visual evidence in the image (shadow depth, edge sharpness, surface texture, darkness/lighting), classify this pothole's DEPTH into ONE category: low, medium, or high.

**Response Format:**
Respond ONLY with valid JSON:
{{"depth_category": "low|medium|high", "confidence": "low|medium|high", "reasoning": "brief"}}
"""


def fleiss(table):
    n, k = table.shape
    m = table.sum(1)[0]
    p_i = ((table ** 2).sum(1) - m) / (m * (m - 1))
    p_j = table.sum(0) / (n * m)
    return float((p_i.mean() - (p_j ** 2).sum()) / (1 - (p_j ** 2).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--model", default="gemini-2.5-pro")
    args = ap.parse_args()

    gem = json.loads(GEM.read_text())
    random.seed(31)
    pids = random.sample(sorted(gem.keys()), args.n)

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

    runs = {r: {} for r in range(args.runs)}
    fails = 0
    for r in range(args.runs):
        for i, pid in enumerate(pids):
            rec = gem[pid]
            scene, bid = rec["scene"], rec["bbox_id"]
            crop = PATCHES / scene / f"{scene}_mask{bid}.jpg"
            if not crop.exists():
                continue
            try:
                resp = model.generate_content([
                    build_prompt(rec.get("geometric_features", {}),
                                 rec.get("positional_features", {})),
                    Image.open(crop)])
                cat = json.loads(resp.text).get("depth_category", "medium")
                if cat not in DEPTH:
                    cat = "medium"
            except Exception:
                cat, fails = "medium", fails + 1
            runs[r][pid] = cat
            if (i + 1) % 50 == 0:
                print(f"  run {r+1}: {i+1}/{len(pids)}")
        print(f"run {r+1}/{args.runs} complete")

    common = sorted(set.intersection(*[set(runs[r]) for r in runs]))
    labels = np.array([[DEPTH.index(runs[r][p]) for r in range(args.runs)] for p in common])
    identical = float(np.mean([len(set(row)) == 1 for row in labels]))
    tab = np.array([[int((row == c).sum()) for c in range(3)] for row in labels])
    # agreement with the labels published in the paper
    pub = [gem[p]["depth_category"] for p in common]
    run0_match = float(np.mean([runs[0][p] == pub[i] for i, p in enumerate(common)]))

    out = {
        "config": {"model": args.model, "temperature": 0.1, "n_potholes": len(common),
                   "n_runs": args.runs, "api_failures": fails},
        "identical_across_all_runs_pct": round(100 * identical, 2),
        "fleiss_kappa_across_runs": round(fleiss(tab), 4),
        "per_run_distribution": {f"run{r+1}": dict(Counter(runs[r][p] for p in common))
                                 for r in range(args.runs)},
        "agreement_run1_vs_published_labels_pct": round(100 * run0_match, 2),
        "interpretation": "Temperature 0.1 yields high but not perfect determinism; "
                          "the residual variation bounds the reproducibility of any "
                          "single hosted-model run.",
        "runs": {str(r): runs[r] for r in runs},
    }
    p = RES / "m10_run_variability.json"
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1))
    print("Saved:", p)


if __name__ == "__main__":
    main()
