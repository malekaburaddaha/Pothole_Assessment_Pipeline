"""
Package ENPAD-A — the publishable dataset for Paper 2.

Layout:
  ENPAD-A_Extended_Dataset_with_Warnings/
    README.md                     (dataset card)
    dataset_statistics.json
    scenes/<scene_id>/
        <scene_id>.JPG            (full perspective-transformed image)
        <scene_id>_label.json     (complete label: perception + action + provenance)
"""
import json
import shutil
from collections import Counter
import os
from pathlib import Path

ROOT = Path(os.environ.get("POTHOLE_ROOT", "."))  # local research-tree layout; see README
LABELS = ROOT / "Paper2_Improvements/results/paper2_vla_labels.json"
IMAGES = ROOT / "Dataset/Transformed_Data/Large_size (3680X1965)/Train/Train_images"
DEST = ROOT / "ENPAD-A_Extended_Dataset_with_Warnings"

data = json.loads(LABELS.read_text())["labels"]
scenes_dir = DEST / "scenes"
scenes_dir.mkdir(parents=True, exist_ok=True)

n_img = n_lab = 0
for sid, rec in data.items():
    d = scenes_dir / sid
    d.mkdir(exist_ok=True)
    src = IMAGES / f"{sid}.JPG"
    if src.exists():
        dst = d / f"{sid}.JPG"
        if not dst.exists():
            shutil.copy2(src, dst)
        n_img += 1
    (d / f"{sid}_label.json").write_text(json.dumps(rec, indent=2))
    n_lab += 1

urgency = Counter(r["action"]["urgency_level"] for r in data.values())
depth = Counter(p["depth"]["paper2_distilled"]
                for r in data.values() for p in r["perception"]["potholes"])
sev = Counter(r["perception"]["highest_severity"] for r in data.values())
n_pot = sum(r["perception"]["num_potholes"] for r in data.values())

stats = {
    "n_scenes": n_lab, "n_images": n_img, "n_potholes": n_pot,
    "urgency_distribution": dict(urgency),
    "depth_distribution_paper2": dict(depth),
    "scene_severity_distribution": dict(sev),
}
(DEST / "dataset_statistics.json").write_text(json.dumps(stats, indent=2))

(DEST / "README.md").write_text(f"""# ENPAD-A — Extended Nienaber Pothole Assessment Dataset with Actions

**{n_lab} dashcam scenes · {n_pot} annotated potholes · full risk assessment and
natural-language driving actions per scene.**

Extends ENPAD (Paper 1) with fully-local Paper-2 pipeline outputs: per-pothole
geometric features (incl. corrected solidity), distilled-model depth categories
(with Gemini 2.5 Pro reference labels), composite risk scores, and per-scene
natural-language warnings + maneuver suggestions (Qwen3-VL-8B, rubric-calibrated,
94.9% consistency). Designed as observation → perception → action supervision
triples for vision-language-action (VLA) model training.

## Layout
```
scenes/<scene_id>/
    <scene_id>.JPG          full 3680x1965 perspective-transformed frame
    <scene_id>_label.json   complete label (see schema below)
```

## Label schema (per scene)
- `observation`: image path, resolution
- `perception`: per-pothole `bbox_xyxy`, `sam_score`, `geometric`
  (area, perimeter, circularity, aspect_ratio, solidity, solidity_corrected),
  `positional` (distance / lateral zone), `depth`
  (`paper2_distilled` + 6-model votes; `paper1_gemini` reference + confidence),
  `risk_assessment` (composite 0-100, severity, component scores)
- `action`: `warning_message`, `maneuver_suggestion`, `urgency_level`,
  `affected_lanes`, `recommended_speed_reduction`, `reasoning`
- `provenance`: models used per field + Paper-1 reference warning

## Statistics
- Urgency: {dict(urgency)}
- Pothole depth (paper2): {dict(depth)}
- Scene severity: {dict(sev)}

## Provenance & lineage
Source imagery: Nienaber et al. pothole dataset (South African roads), perspective-
transformed per Paper 1. Paper-1 reference labels (Gemini 2.5 Pro depth + warnings)
embedded for comparability. Paper-2 labels generated fully locally
(SAM ViT-H fp16 · distilled ConvNeXt ensemble · Qwen3-VL-8B) at ~0.9 s/scene.

## Citation
Citation entries for Paper 1 and Paper 2 to be added upon publication.
""")

print(f"scenes: {n_lab} | images copied: {n_img} | potholes: {n_pot}")
print(f"Packaged at: {DEST}")
