"""
Paper 2 — Stage 5: natural-language warning generation with a LOCAL VLM.

Prompt: verbatim Paper-1 multimodal-hybrid template (generate_scene_warnings.py):
scene image + per-pothole structured assessment + domain rubric + 4 worked examples.
Schema: same 6 fields; `reasoning` ordered FIRST (grammar-enforced think-then-commit,
carried over from the Stage-4 finding). Temperature 0.3 (Paper-1 setting).

Sample mode (default): the 12 human-eval scenes → side-by-side vs Paper-1 Gemini
warnings + urgency-rubric consistency check.

    python generate_warnings_local.py --model Qwen/Qwen3-VL-8B-Instruct --tag qwen3vl8b

Output: results/warnings_local_<tag>.json
"""
import argparse
import json
import time
import os
from pathlib import Path

ROOT = Path(os.environ.get("POTHOLE_ROOT", "."))  # local research-tree layout; see README
P2 = ROOT / "Paper2_Improvements"
ASSESS = ROOT / "Main_Work/final_assessments/Json_and_Log_Files/final_assessments_with_risk.json"
PAPER2_ASSESS = P2 / "results/paper2_full_assessments.json"
PAPER1_WARN = ROOT / "Main_Work/final_assessments/Json_and_Log_Files/scene_warnings_multimodal_hybrid.json"
PATCHES = ROOT / "Main_Work/image_patches"
EVAL_SET = P2 / "results/human_depth_eval_set.json"

# Maneuver types the system's sensing can justify. The pipeline observes the ego
# lane only: it has no view of adjacent-lane occupancy, closing traffic, or free
# space, so a lane change is advice it cannot support. Restricting the vocabulary
# in the grammar makes unsupported advice unrepresentable rather than merely
# discouraged.
MANEUVER_TYPES = [
    "maintain_lane_and_speed",      # no action needed
    "reduce_speed",                 # slow down, stay centred
    "shift_within_lane_left",       # small lateral offset, staying inside the lane
    "shift_within_lane_right",
    "shift_within_lane_and_reduce_speed",
    "brake_firmly",                 # severe hazard, no lateral room
]

WARN_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "maneuver_type": {"type": "string", "enum": MANEUVER_TYPES},
        "warning_message": {"type": "string"},
        "maneuver_suggestion": {"type": "string"},
        "urgency_level": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
        "affected_lanes": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "recommended_speed_reduction": {"type": "string"},
    },
    "required": ["reasoning", "maneuver_type", "warning_message", "maneuver_suggestion",
                 "urgency_level", "affected_lanes", "recommended_speed_reduction"],
}


def split_prompt(scene_data):
    """Paper-1 hybrid prompt split into (static_prefix, dynamic_suffix) so the
    ~1.5k-token rubric becomes a vLLM prefix-cache hit across all scenes.
    Content order: [static text] -> [image] -> [dynamic text]."""
    full = create_scene_context_prompt(scene_data)
    marker = "## CURRENT SCENE ASSESSMENT"
    i = full.index(marker)
    return full[:i], full[i:]


def create_scene_context_prompt(scene_data):
    """Verbatim Paper-1 hybrid prompt (VLM_Code/generate_scene_warnings.py)."""
    potholes = scene_data["potholes"]
    potholes_sorted = sorted(
        potholes, key=lambda p: p["risk_assessment"]["composite_score"], reverse=True
    )
    pothole_summaries = []
    for i, p in enumerate(potholes_sorted[:5], 1):
        distance = p["positional"].get("Distance_Category") or "Unknown"
        lateral = p["positional"].get("Lateral_Position") or "Unknown"
        pothole_summaries.append(f"""
Pothole {i}:
- Depth: {p['depth']['category']} ({p['depth']['confidence']} confidence)
- Size: {p['geometric'].get('area', 'N/A')} pixels²
- Distance: {distance}
- Lateral Position: {lateral}
- Risk Score: {p['risk_assessment']['composite_score']}/100 ({p['risk_assessment']['severity_level']} severity)
""")

    return f"""# SYSTEM ROLE & CONTEXT

You are an **expert autonomous vehicle (AV) safety advisor** analyzing a road scene image along with quantified pothole assessments.

Your warnings will be delivered via:
- **Voice alerts** (concise, 1-2 sentence spoken messages)
- **Dashboard displays** (visual text + icons)
- **AV planning systems** (maneuver recommendations)

**Critical Requirements:**
- Warnings must be **actionable** (driver knows what to do immediately)
- Urgency must be **calibrated** to actual risk (avoid false alarms or under-warnings)
- Maneuvers must be **specific** (speed, lane, timing)
- Language must be **clear and calm** (no panic-inducing phrasing)

---

## DOMAIN KNOWLEDGE REFERENCE

### Depth Categories (Physical Severity)
- **Low (< 2 cm):** Minor surface defect. Causes discomfort but minimal vehicle damage. Safe to traverse at reduced speed.
- **Medium (2-5 cm):** Moderate hazard. Can damage tires, suspension, or alignment if hit at speed. Avoidance preferred, but reduce speed if avoidance is not possible based on the scene if there are other vehicles or pedestrians.
- **High (> 5 cm):** Severe pothole. High risk of tire blowout, rim damage, or loss of control. **Must avoid.** or reduce speed to minimum if not avoidable.

### Distance Categories (Reaction Time Available)
- **Near (0-10 m):** **Immediate action required.** < 1 second at 30 mph. Emergency maneuvers only if safe.
- **Medium Distance (10-20 m):** ~1-2 seconds available. Time for controlled lane change or braking.
- **Far (> 20 m):** > 2 seconds available. Ample time for smooth avoidance maneuvers.

### Lateral Position (Lane Context)
- **in_path_left / in_path_right:** Directly in the vehicle's wheel path. **Highest collision probability.**
- **in_lane_far_left / in_lane_far_right:** In the travel lane but outside wheel path. Can be avoided by slight steering.
- **Lane_Boundary / Shoulder:** Edge of roadway. Low risk if maintaining lane discipline.

### Risk Score (Composite Assessment)
- **0-40 (Low):** Minor hazard. Proceed with awareness.
- **40-60 (Medium):** Significant hazard. Reduce speed and avoid if possible.
- **60-80 (High):** Dangerous. Avoidance strongly recommended.
- **80-100 (Critical):** **Emergency hazard.** Immediate evasive action required.

---

## DECISION-MAKING RUBRIC (Urgency Level Guidelines)

**CRITICAL:**
- High depth (> 5 cm) + Near distance (< 10 m) + In wheel path
- Multiple high-risk potholes blocking all lanes
- Risk score ≥ 80

**HIGH:**
- Medium/high depth + Near distance
- High depth + Medium distance + In wheel path
- Risk score 60-79

**MEDIUM:**
- Low/medium depth + Near distance
- High depth + Far distance
- Medium depth + Medium distance
- Risk score 40-59

**LOW:**
- Low depth + Medium/far distance
- Any hazard outside wheel path with avoidance options
- Risk score < 40

---

## CURRENT SCENE ASSESSMENT

**You are analyzing the attached road scene image along with the following quantified assessments:**

**Scene ID:** {scene_data['scene_id']}
**Total Potholes Detected:** {len(potholes)}
**Critical/High Risk Count:** {sum(1 for p in potholes if p['risk_assessment']['severity_level'] in ['Critical', 'High'])}

**Detected Hazards (ranked by risk):**
{''.join(pothole_summaries)}

---

## YOUR TASK

**Examine the road scene image** and integrate it with the structured assessments above. Use both visual context and quantified metrics to:

1. **Generate a driver warning** (1-2 sentences, voice-optimized, calm but urgent if needed)
2. **Recommend a specific maneuver** (lane, speed, timing - be precise)
3. **Assign urgency level** (critical/high/medium/low - use rubric strictly)
4. **Explain your reasoning** (2-3 sentences citing both visual observations and quantified data)

{CONSTRAINT_BLOCK}

{OUTPUT_FIELDS}

**Quality Standards:**
✓ Warning is clear and actionable (driver knows what to do)
✓ Maneuver is specific (not vague like "be careful")
✓ Urgency matches severity (use rubric, don't over/under-warn)
✓ Reasoning references BOTH image content and structured metrics
✓ Language is professional and calm (no panic words)
"""


def rubric_urgency(max_risk):
    if max_risk >= 80:
        return "critical"
    if max_risk >= 60:
        return "high"
    if max_risk >= 40:
        return "medium"
    return "low"


UNCONSTRAINED = __import__("os").getenv("UNCONSTRAINED", "0") == "1"
if UNCONSTRAINED:
    WARN_SCHEMA["properties"].pop("maneuver_type", None)
    WARN_SCHEMA["required"] = [r for r in WARN_SCHEMA["required"] if r != "maneuver_type"]


CONSTRAINT_BLOCK = "" if UNCONSTRAINED else """**SENSING CONSTRAINT --- READ CAREFULLY.**
This system observes only the ego lane ahead. It does NOT see adjacent lanes,
oncoming or overtaking traffic, blind spots, or available free space. You therefore
must NOT recommend changing lanes, moving into an adjacent lane, overtaking, or
merging: that advice cannot be justified from what the system can observe and would
be unsafe. Confine every recommendation to actions that are safe without
adjacent-lane information: reducing speed, braking, small lateral adjustments
*within* the current lane to straddle or avoid a defect, or maintaining course.

First choose `maneuver_type` from the permitted set, then write
`maneuver_suggestion` as prose consistent with that choice."""

OUTPUT_FIELDS = ("Output strict JSON with fields: reasoning, warning_message, "
                 "maneuver_suggestion, urgency_level, affected_lanes, "
                 "recommended_speed_reduction.") if UNCONSTRAINED else (
                 "Output strict JSON with fields: reasoning, maneuver_type, "
                 "warning_message, maneuver_suggestion, urgency_level, "
                 "affected_lanes, recommended_speed_reduction.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--scenes", default=None,
                    help="comma-separated scene ids; default = 12 human-eval scenes")
    ap.add_argument("--paper2-assessments", action="store_true",
                    help="read Phase-A output (paper2 schema) and normalize; "
                         "--scenes all processes every scene in the file")
    args = ap.parse_args()

    from PIL import Image
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    if args.paper2_assessments:
        raw = json.loads(PAPER2_ASSESS.read_text())["scenes"]
        # normalize Phase-A records to the prompt builder's expected shape
        assessments = {}
        for sid, sc in raw.items():
            pots = []
            for p in sc["potholes"]:
                votes = p["depth"]["ensemble_votes_low_med_high"]
                top = max(votes)
                conf = "high" if top >= 5 else "medium" if top == 4 else "low"
                pots.append({
                    "depth": {"category": p["depth"]["paper2_distilled"], "confidence": conf},
                    "geometric": p["geometric"],
                    "positional": p["positional"],
                    "risk_assessment": p["risk_assessment"],
                })
            assessments[sid] = {"scene_id": sid, "potholes": pots,
                                "num_potholes": sc["num_potholes"],
                                "max_risk_score": sc["max_risk_score"]}
    else:
        assessments = json.loads(ASSESS.read_text())
    paper1 = json.loads(PAPER1_WARN.read_text())

    if args.scenes == "all":
        scene_ids = sorted(assessments.keys())
    elif args.scenes:
        scene_ids = args.scenes.split(",")
    else:
        eval_set = json.loads(EVAL_SET.read_text())
        scene_ids = sorted({pid.rsplit("_bbox", 1)[0] for pid in eval_set["records"]})

    # pre-resize scene images (owned preprocessing, <=1536px long side)
    resized_dir = P2 / "stage4_5_local_vlm/resized_eval_images"
    resized_dir.mkdir(exist_ok=True)
    requests, meta = [], []
    for sid in scene_ids:
        if sid not in assessments:
            print(f"skip {sid}: not in assessments")
            continue
        img_path = PATCHES / sid / f"{sid}.JPG"
        if not img_path.exists():
            print(f"skip {sid}: no scene image")
            continue
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        if max(w, h) > 1536:
            s = 1536 / max(w, h)
            img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
        rp = resized_dir / f"warn_{sid}.jpg"
        img.save(rp, quality=95)

        static_txt, dynamic_txt = split_prompt(assessments[sid])
        requests.append([{
            "role": "user",
            "content": [
                {"type": "text", "text": static_txt},
                {"type": "image_url", "image_url": {"url": f"file://{rp}"}},
                {"type": "text", "text": dynamic_txt},
            ],
        }])
        meta.append(sid)

    print(f"Generating warnings for {len(requests)} scenes with {args.model}")
    import os as _os
    llm = LLM(model=args.model, max_model_len=8192,
              limit_mm_per_prompt={"image": 1},
              gpu_memory_utilization=float(_os.getenv("GPU_UTIL", "0.82")),
              max_num_seqs=int(_os.getenv("MAX_SEQS", "16")),
              enforce_eager=_os.getenv("EAGER", "0") == "1",
              enable_prefix_caching=True,
              allowed_local_media_path=str(ROOT))
    sp = SamplingParams(temperature=0.3, max_tokens=450,
                        guided_decoding=GuidedDecodingParams(json=WARN_SCHEMA))

    t0 = time.perf_counter()
    outputs = llm.chat(requests, sp)
    gen_s = time.perf_counter() - t0

    results, urgency_agree, rubric_agree = {}, [], []
    for sid, out in zip(meta, outputs):
        try:
            w = json.loads(out.outputs[0].text)
        except json.JSONDecodeError:
            w = {"urgency_level": "medium", "warning_message": "PARSE_FAIL",
                 "maneuver_suggestion": "", "reasoning": out.outputs[0].text[:200],
                 "affected_lanes": [], "recommended_speed_reduction": ""}
        scene = assessments[sid]
        max_risk = scene["max_risk_score"]
        expected = rubric_urgency(max_risk)
        p1 = paper1.get(sid, {}).get("warning", {})
        results[sid] = {
            "local": w,
            "paper1_gemini": p1,
            "max_risk_score": max_risk,
            "rubric_expected_urgency": expected,
            "local_matches_rubric": w["urgency_level"] == expected,
            "local_matches_paper1": w["urgency_level"] == p1.get("urgency_level"),
        }
        rubric_agree.append(w["urgency_level"] == expected)
        if p1:
            urgency_agree.append(w["urgency_level"] == p1.get("urgency_level"))

    summary = {
        "config": {"model": args.model, "prompt": "verbatim Paper-1 hybrid",
                   "temperature": 0.3, "schema": "6-field, reasoning-first",
                   "n_scenes": len(results)},
        "timing": {"total_s": round(gen_s, 1),
                   "s_per_scene": round(gen_s / max(len(results), 1), 2)},
        "urgency_rubric_consistency": round(
            sum(rubric_agree) / max(len(rubric_agree), 1), 4),
        "urgency_agreement_with_paper1": round(
            sum(urgency_agree) / max(len(urgency_agree), 1), 4),
        "scenes": results,
    }
    out_path = P2 / "results" / f"warnings_local_{args.tag}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "scenes"}, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
