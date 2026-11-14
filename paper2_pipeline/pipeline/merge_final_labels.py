"""
Paper 2 — Final merge: Phase A assessments + Phase B warnings
   → one VLA-ready detailed label per scene.

Label design (for the next project's VLA finetuning):
  observation : scene image path (perspective-transformed frame)
  perception  : per-pothole bbox, mask stats, geometric + positional features,
                depth (paper2_distilled + paper1_gemini reference), risk scores
  action      : warning_message, maneuver_suggestion, urgency_level,
                affected_lanes, recommended_speed_reduction, reasoning
This is the image -> (perceive, decide, act) supervision triple.

Output: results/paper2_vla_labels.json + summary stats
"""
import json
import os
from pathlib import Path
from collections import Counter

P2 = Path(os.environ.get("POTHOLE_ROOT", "."), "Paper2_Improvements")
ASSESS = P2 / "results/paper2_full_assessments.json"
WARN = P2 / "results/warnings_local_qwen3vl8b-FULL.json"
OUT = P2 / "results/paper2_vla_labels.json"

assess = json.loads(ASSESS.read_text())
warn = json.loads(WARN.read_text())["scenes"]

labels, missing_warn = {}, 0
for sid, sc in assess["scenes"].items():
    w = warn.get(sid, {}).get("local")
    if w is None:
        missing_warn += 1
        continue
    labels[sid] = {
        "scene_id": sid,
        "observation": {"image": sc["image"], "resolution": "3680x1965 perspective-transformed"},
        "perception": {
            "num_potholes": sc["num_potholes"],
            "potholes": sc["potholes"],
            "max_risk_score": sc["max_risk_score"],
            "highest_severity": sc["highest_severity"],
        },
        "action": {
            "warning_message": w["warning_message"],
            "maneuver_suggestion": w["maneuver_suggestion"],
            "urgency_level": w["urgency_level"],
            "affected_lanes": w.get("affected_lanes", []),
            "recommended_speed_reduction": w.get("recommended_speed_reduction", ""),
            "reasoning": w.get("reasoning", ""),
        },
        "provenance": {
            "segmentation": "SAM ViT-H fp16+compile (Paper-2)",
            "depth": "distilled ConvNeXt 6-ensemble (teacher: Gemini 2.5 Pro)",
            "warning_model": "Qwen3-VL-8B-Instruct (local)",
            "paper1_reference_warning": warn.get(sid, {}).get("paper1_gemini"),
        },
    }

urgency = Counter(v["action"]["urgency_level"] for v in labels.values())
n_pot = sum(v["perception"]["num_potholes"] for v in labels.values())
summary = {
    "n_scenes": len(labels),
    "n_potholes": n_pot,
    "missing_warnings": missing_warn,
    "urgency_distribution": dict(urgency),
    "labels": labels,
}
OUT.write_text(json.dumps(summary, indent=1))
print(f"scenes: {len(labels)}  potholes: {n_pot}  missing warnings: {missing_warn}")
print("urgency distribution:", dict(urgency))
print(f"Saved: {OUT}")
