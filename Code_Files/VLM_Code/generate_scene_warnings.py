"""
Generate natural language warnings using VLM with BOTH scene images and structured labels
This enables fair comparison with image-only baseline in ablation study
"""
import json
from pathlib import Path
from tqdm import tqdm
import logging
import google.generativeai as genai
import os
from PIL import Image
from dotenv import load_dotenv

# Paths
INPUT_FILE = Path("/data/Malek/Pothole_Assessment_Pipeline/Masks_Batches_Dataset/final_assessments/final_assessments_with_risk.json")
SCENE_IMAGES_DIR = Path("/data/Malek/Pothole_Assessment_Pipeline/Masks_Batches_Dataset/image_patches")
OUTPUT_DIR = Path("/data/Malek/Pothole_Assessment_Pipeline/Masks_Batches_Dataset/final_assessments")
LOG_FILE = OUTPUT_DIR / "scene_warnings.log"

logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Configure API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def create_scene_context_prompt(scene_data):
    """
    Create structured prompt with domain knowledge, examples, and rubric
    """
    potholes = scene_data['potholes']
    
    # Sort by risk score (highest first)
    potholes_sorted = sorted(potholes, key=lambda p: p['risk_assessment']['composite_score'], reverse=True)
    
    pothole_summaries = []
    for i, p in enumerate(potholes_sorted[:5], 1):  # Top 5 highest risk
        # Handle field name variations
        distance = p['positional'].get('Distance_Category') or p['positional'].get('distance_category', 'Unknown')
        lateral = p['positional'].get('Lateral_Position') or p['positional'].get('lateral_position', 'Unknown')
        
        summary = f"""
Pothole {i}:
- Depth: {p['depth']['category']} ({p['depth']['confidence']} confidence)
- Size: {p['geometric'].get('area', 'N/A')} pixels²
- Distance: {distance}
- Lateral Position: {lateral}
- Risk Score: {p['risk_assessment']['composite_score']}/100 ({p['risk_assessment']['severity_level']} severity)
"""

        pothole_summaries.append(summary)
    
    prompt = f"""# SYSTEM ROLE & CONTEXT

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

## EXAMPLES (Learn from these scenarios)

### Example 1: Critical Urgency
**Scenario:** High depth, Near distance, in_path_right, Risk Score: 85/100

**Output:**
{{
  "warning_message": "Critical pothole ahead in right wheel path. Take immediate action.",
  "maneuver_suggestion": "If safe, move left immediately. Otherwise, reduce speed to 10 mph.",
  "urgency_level": "critical",
  "reasoning": "A 6+ cm deep pothole is directly in the vehicle's path with less than 10 meters distance. High risk of tire damage or loss of control. Emergency avoidance is necessary if adjacent lane is clear; otherwise, severe speed reduction required.",
  "affected_lanes": ["right"],
  "recommended_speed_reduction": "Reduce to 10 mph if avoidance impossible"
}}

### Example 2: High Urgency
**Scenario:** Medium depth, Near distance, in_path_left + Low depth in right lane

**Output:**
{{
  "warning_message": "Significant pothole ahead in left wheel path. Move to right lane.",
  "maneuver_suggestion": "Shift right within lane to avoid left-side pothole. Reduce speed to 20 mph.",
  "urgency_level": "high",
  "reasoning": "A 3-4 cm deep pothole is in the left wheel path with limited reaction time. Right lane has minor hazards but provides safer passage. Lane shift recommended over hard braking.",
  "affected_lanes": ["center", "left"],
  "recommended_speed_reduction": "20 mph"
}}

### Example 3: Medium Urgency
**Scenario:** Low depth, Medium distance, multiple minor potholes

**Output:**
{{
  "warning_message": "Multiple minor potholes ahead. Proceed with caution.",
  "maneuver_suggestion": "Reduce speed to 25 mph and steer slightly left to minimize impact.",
  "urgency_level": "medium",
  "reasoning": "Two shallow potholes (< 2 cm) are present with 10-20 meters available for reaction. Moderate speed reduction allows safe traversal with minimal steering adjustment. Not severe enough for emergency maneuvers.",
  "affected_lanes": ["center", "right"],
  "recommended_speed_reduction": "25 mph"
}}

### Example 4: Low Urgency
**Scenario:** Low depth, Far distance, outside wheel path

**Output:**
{{
  "warning_message": "Minor road defect ahead on right side of lane.",
  "maneuver_suggestion": "Maintain current lane position. No speed reduction necessary.",
  "urgency_level": "low",
  "reasoning": "A shallow surface defect (< 2 cm) is located outside the wheel path with over 20 meters distance. Vehicle can safely pass without adjustment. Notification provided for driver awareness only.",
  "affected_lanes": ["right"],
  "recommended_speed_reduction": "Maintain speed"
}}

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

**Output Format (strict JSON):**
{{
  "warning_message": "string (concise, actionable, 1-2 sentences)",
  "maneuver_suggestion": "string (specific guidance: which lane, what speed, when to act)",
  "urgency_level": "critical|high|medium|low",
  "reasoning": "string (explain using BOTH visual observations from image AND quantified metrics)",
  "affected_lanes": ["center"|"left"|"right"],
  "recommended_speed_reduction": "string (e.g., '15 mph', 'maintain speed', 'reduce to 10 mph')"
}}

**Quality Standards:**
✓ Warning is clear and actionable (driver knows what to do)
✓ Maneuver is specific (not vague like "be careful")
✓ Urgency matches severity (use rubric, don't over/under-warn)
✓ Reasoning references BOTH image content and structured metrics
✓ Language is professional and calm (no panic words)
"""
    return prompt

def generate_scene_warning_multimodal(scene_id, scene_data, image_path, model_name="gemini-2.5-pro"):
    """
    Call VLM with both scene image and structured labels
    Uses multimodal fusion for enhanced reasoning
    """
    prompt = create_scene_context_prompt(scene_data)
    
    try:
        # Load scene image
        img = Image.open(image_path)
        
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "object",
                    "properties": {
                        "warning_message": {"type": "string"},
                        "maneuver_suggestion": {"type": "string"},
                        "urgency_level": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                        "reasoning": {"type": "string"},
                        "affected_lanes": {"type": "array", "items": {"type": "string"}},
                        "recommended_speed_reduction": {"type": "string"}
                    },
                    "required": ["warning_message", "maneuver_suggestion", "urgency_level", "reasoning"]
                }
            }
        )
        
        # Multimodal input: [text prompt, image]
        response = model.generate_content([prompt, img])
        result = json.loads(response.text)
        
        logger.info(f"✓ Generated multimodal warning for {scene_id}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to generate warning for {scene_id}: {e}")
        return {
            "warning_message": "Multiple hazards detected ahead",
            "maneuver_suggestion": "Proceed with caution",
            "urgency_level": "medium",
            "reasoning": "Error in VLM processing",
            "affected_lanes": [],
            "recommended_speed_reduction": "unknown"
        }

def process_all_scenes(sample_size=None):
    """
    Generate warnings for all scenes (or sample for testing)
    Uses multimodal approach: scene image + structured labels
    """
    with open(INPUT_FILE) as f:
        assessments = json.load(f)
    
    # Sample if requested (for testing)
    if sample_size:
        import random
        scene_ids = random.sample(list(assessments.keys()), min(sample_size, len(assessments)))
        assessments = {k: assessments[k] for k in scene_ids}
    
    logger.info(f"Processing {len(assessments)} scenes for multimodal warning generation")
    logger.info(f"Method: Scene Image + Structured Labels (Hybrid Approach)")
    
    scene_warnings = {}
    skipped = 0
    
    for scene_id, scene_data in tqdm(assessments.items(), desc="Generating warnings"):
        # Find scene image
        image_path = SCENE_IMAGES_DIR / scene_id / f"{scene_id}.JPG"
        
        if not image_path.exists():
            logger.warning(f"Image not found for {scene_id}, skipping")
            skipped += 1
            continue
        
        warning = generate_scene_warning_multimodal(scene_id, scene_data, image_path)
        
        scene_warnings[scene_id] = {
            'scene_id': scene_id,
            'num_potholes': scene_data['num_potholes'],
            'method': 'multimodal_hybrid',  # For ablation tracking
            'warning': warning,
            'potholes': scene_data['potholes'],  # Keep full assessment data
            'image_path': str(image_path)
        }
    
    # Save results
    output_file = OUTPUT_DIR / 'scene_warnings_multimodal_hybrid.json'
    with open(output_file, 'w') as f:
        json.dump(scene_warnings, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"MULTIMODAL WARNING GENERATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Method: Scene Image + Structured Labels")
    logger.info(f"Total scenes processed: {len(scene_warnings)}")
    logger.info(f"Skipped (no image): {skipped}")
    logger.info(f"Results saved: {output_file}")
    
    # Statistics
    urgency_dist = {}
    for sw in scene_warnings.values():
        urgency = sw['warning']['urgency_level']
        urgency_dist[urgency] = urgency_dist.get(urgency, 0) + 1
    
    logger.info(f"\nUrgency Distribution:")
    for urgency, count in sorted(urgency_dist.items()):
        logger.info(f"  {urgency}: {count}")
    
    return scene_warnings

if __name__ == "__main__":
    # Test on sample first
    # logger.info("Running on sample (10 scenes) for testing...")
    # process_all_scenes(sample_size=2)
    
    # Uncomment to run on full dataset
    logger.info("Running on full dataset...")
    process_all_scenes()
