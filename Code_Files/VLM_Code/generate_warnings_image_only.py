"""
ABLATION BASELINE: Image-only warning generation (no structured labels)
For fair comparison with multimodal hybrid approach
"""
import json
from pathlib import Path
from tqdm import tqdm
import logging
import google.generativeai as genai
import os
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Paths
INPUT_FILE = Path("/data/Malek/Pothole_Assessment_Pipeline/Masks_Batches_Dataset/final_assessments/final_assessments_with_risk.json")
SCENE_IMAGES_DIR = Path("/data/Malek/Pothole_Assessment_Pipeline/Masks_Batches_Dataset/image_patches")
OUTPUT_DIR = Path("/data/Malek/Pothole_Assessment_Pipeline/Masks_Batches_Dataset/final_assessments")
LOG_FILE = OUTPUT_DIR / "warnings_image_only.log"

logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])
logger = logging.getLogger(__name__)

def generate_warning_image_only(image_path, scene_id):
    """Generate warning from raw image ONLY (no labels)"""
    
    prompt = """You are an autonomous vehicle safety advisor analyzing a road scene.

**TASK:**
Examine this road image and identify any potholes or road hazards visible.
Based on what you see, provide:

1. **Driver Warning**: A concise alert message (1-2 sentences)
2. **Maneuver Suggestion**: Specific navigation recommendation (speed, lane, timing)
3. **Urgency Level**: low/medium/high/critical
4. **Explanation**: Brief reasoning (2-3 sentences)

Respond ONLY with valid JSON:
{
  "warning_message": "string",
  "maneuver_suggestion": "string",
  "urgency_level": "low|medium|high|critical",
  "reasoning": "string",
  "num_potholes_detected": number,
  "affected_lanes": ["center"|"left"|"right"],
  "recommended_speed_reduction": "string"
}
"""
    
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-pro",
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "object",
                    "properties": {
                        "warning_message": {"type": "string"},
                        "maneuver_suggestion": {"type": "string"},
                        "urgency_level": {"type": "string"},
                        "reasoning": {"type": "string"},
                        "num_potholes_detected": {"type": "integer"},
                        "affected_lanes": {"type": "array", "items": {"type": "string"}},
                        "recommended_speed_reduction": {"type": "string"}
                    }
                }
            }
        )
        
        img = Image.open(image_path)
        response = model.generate_content([prompt, img])
        result = json.loads(response.text)
        
        logger.info(f"✓ Generated image-only warning for {scene_id}")
        return result
        
    except Exception as e:
        logger.error(f"Failed for {scene_id}: {e}")
        return {
            "warning_message": "Unable to analyze scene",
            "maneuver_suggestion": "Proceed with caution",
            "urgency_level": "unknown",
            "reasoning": "Processing error",
            "num_potholes_detected": 0,
            "affected_lanes": [],
            "recommended_speed_reduction": "unknown"
        }

def process_image_only_ablation(sample_size=None):
    """Run image-only baseline on same scenes as hybrid"""
    
    with open(INPUT_FILE) as f:
        assessments = json.load(f)
    
    if sample_size:
        import random
        scene_ids = random.sample(list(assessments.keys()), min(sample_size, len(assessments)))
        assessments = {k: assessments[k] for k in scene_ids}
    
    logger.info(f"Processing {len(assessments)} scenes for IMAGE-ONLY baseline")
    
    results = {}
    skipped = 0
    
    for scene_id, scene_data in tqdm(assessments.items(), desc="Image-only warnings"):
        image_path = SCENE_IMAGES_DIR / scene_id / f"{scene_id}.JPG"
        
        if not image_path.exists():
            logger.warning(f"Image not found for {scene_id}")
            skipped += 1
            continue
        
        warning = generate_warning_image_only(image_path, scene_id)
        
        results[scene_id] = {
            'scene_id': scene_id,
            'num_potholes_ground_truth': scene_data['num_potholes'],
            'method': 'image_only_baseline',
            'warning': warning,
            'image_path': str(image_path)
        }
    
    output_file = OUTPUT_DIR / 'warnings_image_only_ablation.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"IMAGE-ONLY BASELINE COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total scenes processed: {len(results)}")
    logger.info(f"Skipped: {skipped}")
    logger.info(f"Results saved: {output_file}")
    
    return results

if __name__ == "__main__":
    # Test first
    # logger.info("Testing image-only baseline (2 scenes)...")
    # process_image_only_ablation(sample_size=2)
    
    # Uncomment for full run
    logger.info("Full image-only baseline (full dataset)...")
    process_image_only_ablation()
