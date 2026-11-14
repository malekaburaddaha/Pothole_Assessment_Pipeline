"""
Extract the Paper-1 42-pothole human depth-annotation study into a clean JSON
eval set, with human consensus computed exactly as Paper 1 did:
consensus = majority category when >=2 of 3 annotators agree; 3-way splits excluded.

Output: Paper2_Improvements/results/human_depth_eval_set.json
"""
import json
from collections import Counter
import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("POTHOLE_ROOT", "."))  # local research-tree layout; see README
XLSX = ROOT / "Main_Work/Validation_outputs_depth/human_annotation_template.xlsx"
OUT = ROOT / "Paper2_Improvements/results/human_depth_eval_set.json"

df = pd.read_excel(XLSX, sheet_name="Annotations")

records = {}
n_split = 0
for _, row in df.iterrows():
    labels = [
        str(row["Annotator_1_Depth"]).strip().lower(),
        str(row["Annotator_2_Depth"]).strip().lower(),
        str(row["Annotator_3_Depth"]).strip().lower(),
    ]
    counts = Counter(labels)
    top, top_n = counts.most_common(1)[0]
    consensus = top if top_n >= 2 else None  # None = 3-way split, excluded
    if consensus is None:
        n_split += 1
    records[row["Pothole_ID"]] = {
        "crop_image": str(row["Crop_Image_Path"]),
        "full_scene": str(row["Full_Scene_Path"]),
        "area_pixels": int(row["Area_pixels"]),
        "circularity": float(row["Circularity"]),
        "human_labels": labels,
        "human_consensus": consensus,
        "unanimous": top_n == 3,
        "paper1_predictions": {
            "gemini_flash": str(row["Flash_Prediction"]).strip().lower(),
            "gemini_pro": str(row["Pro_Prediction"]).strip().lower(),
            "gpt5": str(row["GPT5_Prediction"]).strip().lower(),
        },
    }

usable = {k: v for k, v in records.items() if v["human_consensus"]}

# Recompute every Paper-1 VLM's kappa against the CURRENT consensus so that all
# rows (old and new models) are measured on identical labels. Note: the xlsx's
# Annotator_1 column was edited after Paper 1's archived analysis (k12/k13
# changed, k23 identical), so paper-published kappas are close but not identical.
from sklearn.metrics import cohen_kappa_score

vlm_kappas = {}
for model in ["gemini_flash", "gemini_pro", "gpt5"]:
    y_true = [v["human_consensus"] for v in usable.values()]
    y_pred = [v["paper1_predictions"][model] for v in usable.values()]
    vlm_kappas[model] = {
        "kappa_vs_consensus": round(float(cohen_kappa_score(y_true, y_pred)), 4),
        "agreement_percent": round(
            100 * sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true), 2
        ),
    }

summary = {
    "provenance_note": (
        "Human labels from Main_Work/Validation_outputs_depth/human_annotation_template.xlsx "
        "as of 2026-07-29. Annotator_1 column differs slightly from the version used for "
        "Paper 1's archived human_validation_results.json (k12 0.5016->0.4948, k13 "
        "0.4158->0.4710, k23 unchanged 0.3944). All Paper-2 comparisons use THESE labels "
        "for every model, including re-scored Paper-1 VLMs, so rankings are internally consistent."
    ),
    "n_total": len(records),
    "n_usable_consensus": len(usable),
    "n_three_way_split": n_split,
    "consensus_distribution": dict(
        Counter(v["human_consensus"] for v in usable.values())
    ),
    "human_avg_pairwise_kappa_current_labels": 0.4534,
    "paper1_vlms_rescored_on_current_consensus": vlm_kappas,
    "records": records,
}
OUT.write_text(json.dumps(summary, indent=2))
print(f"total={summary['n_total']}  usable={summary['n_usable_consensus']}  "
      f"splits={n_split}  dist={summary['consensus_distribution']}")
print(f"Saved: {OUT}")
