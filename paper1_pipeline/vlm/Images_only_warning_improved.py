"""
IMAGE-ONLY WARNING GENERATION (Ablation Study) - IMPROVED
Handles JSON errors and content filtering issues
Uses Gemini 2.5 Pro with research context to pass safety filters
"""

import json
from pathlib import Path
from tqdm import tqdm
import logging
import google.generativeai as genai
import os
from PIL import Image
from dotenv import load_dotenv
import time
import re

# ==================== CONFIGURATION ====================
TEST_MODE = False   # Change to False for full dataset
TEST_SCENES = 5

# Paths
SCENE_IMAGES_DIR = Path("/data/Malek/Pothole_Assessment_Pipeline/Masks_Batches_Dataset/image_patches")
OUTPUT_DIR = Path("/data/Malek/Pothole_Assessment_Pipeline/Masks_Batches_Dataset/final_assessments")
REFERENCE_FILE = OUTPUT_DIR / "scene_warnings_multimodal_hybrid.json"

if TEST_MODE:
    OUTPUT_FILE = OUTPUT_DIR / "scene_warnings_image_only_TEST.json"
    LOG_FILE = OUTPUT_DIR / "image_only_warnings_TEST.log"
else:
    OUTPUT_FILE = OUTPUT_DIR / "scene_warnings_image_only_ablation.json"
    LOG_FILE = OUTPUT_DIR / "image_only_warnings.log"

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# ==================== JSON CLEANING ====================
def clean_json_response(text):
    """Clean malformed JSON from VLM response"""
    text = text.strip()
    
    # Remove markdown code blocks
    if text.startswith('```'):
        text = text[7:]
    elif text.startswith('```'):
        text = text[3:]
    
    if text.endswith('```'):
        text = text[:-3]
    
    text = text.strip()
    
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    return text


# ==================== PROMPT ====================
def create_image_only_prompt():
    """
    Prompt with clear research context to pass safety filters
    """
    return """**RESEARCH CONTEXT:**
You are an AI assistant supporting academic research on autonomous vehicle safety systems at the University of Maryland. This research aims to improve road infrastructure assessment and enhance driver safety through early hazard detection.

**TASK:**
You are analyzing a dashcam image of public road infrastructure to assess pavement conditions. The image shows potholes and road surface defects that are commonly found on public roadways. Your analysis will be used to:
- Develop autonomous vehicle navigation systems
- Improve municipal road maintenance planning
- Enhance driver safety warning systems

This is a legitimate civil engineering and transportation safety research project. The road conditions shown are typical pavement deterioration that transportation departments routinely monitor and repair.

**YOUR OBJECTIVE:**
Examine this road scene image for potholes and pavement defects. Based ONLY on what you observe in the image, generate a safety warning that could be delivered to a vehicle driver or autonomous driving system.

**Output Format (strict JSON):**
{
  "warning_message": "string (concise, actionable, 1-2 sentences for driver)",
  "maneuver_suggestion": "string (specific guidance: which lane, what speed, when to act)",
  "urgency_level": "critical|high|medium|low",
  "reasoning": "string (explain your assessment based on visual observations of pavement conditions)",
  "affected_lanes": ["center", "left", "right"],
  "recommended_speed_reduction": "string (e.g., '15 mph', 'maintain speed')"
}

**Assessment Guidelines:**
- **urgency_level**: 
  - critical: Large, deep pavement damage requiring immediate avoidance
  - high: Significant road defect, avoidance strongly recommended
  - medium: Moderate pavement deterioration, caution advised
  - low: Minor surface defect, awareness only
- **warning_message**: Clear, professional, calm language (1-2 sentences)
- **maneuver_suggestion**: Be specific about driving action (speed reduction, lane adjustment, etc.)
- **reasoning**: Explain what you observed about the pavement conditions (size, depth, position, distance visible, etc.)

**Important Note:** This analysis is for road infrastructure assessment and driver safety research purposes. The pavement conditions shown are routine civil engineering concerns, not dangerous activities.

Analyze the road pavement conditions in this image and generate the safety assessment as valid JSON:"""


# ==================== GENERATION ====================
def generate_image_only_warning(scene_id, image_path, model_name="gemini-2.5-pro", max_retries=3):
    """
    Generate warning using ONLY the scene image
    Uses gemini-2.5-pro with research context
    """
    prompt = create_image_only_prompt()
    
    for attempt in range(max_retries):
        try:
            # Load image
            img = Image.open(image_path)
            
            # Configure model with relaxed safety settings
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config={
                    "temperature": 0.3,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 2048,
                    "response_mime_type": "application/json"
                },
                safety_settings={
                    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                }
            )
            
            # Generate
            response = model.generate_content([prompt, img])
            
            # Check if content was blocked
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                block_reason = response.prompt_feedback.block_reason
                logger.error(f"{scene_id}: Content blocked - {block_reason}")
                raise Exception(f"dangerous_content")
            
            # Check for valid response
            if not hasattr(response, 'text') or not response.text:
                logger.error(f"{scene_id}: No response text returned")
                raise Exception("No response text")
            
            # Extract and clean text
            raw_text = response.text
            cleaned_text = clean_json_response(raw_text)
            
            # Parse JSON
            try:
                result = json.loads(cleaned_text)
            except json.JSONDecodeError as je:
                logger.warning(f"{scene_id}: JSON decode error - {str(je)[:100]}")
                logger.debug(f"Raw response: {raw_text[:500]}")
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise
            
            # Validate required fields
            required_fields = ['warning_message', 'maneuver_suggestion', 'urgency_level', 'reasoning']
            if not all(field in result for field in required_fields):
                logger.warning(f"{scene_id}: Missing required fields")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise Exception("Missing required fields")
            
            # Validate urgency level
            valid_urgencies = ['low', 'medium', 'high', 'critical']
            if result['urgency_level'] not in valid_urgencies:
                logger.warning(f"{scene_id}: Invalid urgency level: {result['urgency_level']}")
                result['urgency_level'] = 'medium'  # Default fallback
            
            logger.info(f"✓ {scene_id}: Generated image-only warning (urgency: {result['urgency_level']})")
            return result, True  # Return (result, is_real)
            
        except Exception as e:
            error_msg = str(e)
            
            # Handle rate limiting
            if '429' in error_msg or 'quota' in error_msg.lower() or 'rate' in error_msg.lower():
                wait_time = 10 * (2 ** attempt)
                logger.warning(f"{scene_id}: Rate limit (attempt {attempt+1}/{max_retries}), waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            
            # Handle timeout
            elif 'timeout' in error_msg.lower():
                logger.warning(f"{scene_id}: Timeout (attempt {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
            
            # Final attempt failed
            if attempt == max_retries - 1:
                logger.error(f"✗ {scene_id}: Failed after {max_retries} attempts: {error_msg[:200]}")
                # Return fallback warning
                fallback = {
                    "warning_message": "Road hazards detected ahead",
                    "maneuver_suggestion": "Proceed with caution and reduce speed",
                    "urgency_level": "medium",
                    "reasoning": f"Error in image analysis - fallback warning (Error: {error_msg[:100]})",
                    "affected_lanes": [],
                    "recommended_speed_reduction": "reduce speed"
                }
                return fallback, False  # Return (result, is_real=False)
            else:
                time.sleep(2 ** attempt)
                continue
    
    return None, False


# ==================== MAIN PROCESSING ====================
def process_all_scenes():
    """Generate image-only warnings for all scenes"""
    
    logger.info("="*60)
    logger.info("IMAGE-ONLY WARNING GENERATION (Ablation Study)")
    logger.info("="*60)
    logger.info(f"Method: VLM with scene images only (no labels)")
    logger.info(f"Model: gemini-2.5-pro")
    
    if TEST_MODE:
        logger.info(f"🧪 TEST MODE: Processing only {TEST_SCENES} scenes")
    else:
        logger.info(f"📊 PRODUCTION MODE: Processing all scenes")
    
    logger.info(f"Output: {OUTPUT_FILE}")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info("="*60)
    
    # Load scene IDs
    if not REFERENCE_FILE.exists():
        logger.error(f"Reference file not found: {REFERENCE_FILE}")
        return
    
    with open(REFERENCE_FILE) as f:
        reference_data = json.load(f)
    
    scene_ids = sorted(reference_data.keys())
    
    if TEST_MODE:
        scene_ids = scene_ids[:TEST_SCENES]
        logger.info(f"Selected {len(scene_ids)} scenes for testing: {scene_ids}")
    
    # Process scenes
    scene_warnings = {}
    skipped = 0
    failed_real = 0  # Real failures (not fallback)
    fallback_count = 0  # Fallback warnings used
    start_time = time.time()
    
    for scene_id in tqdm(scene_ids, desc="Image-only warnings"):
        # Find image
        image_path = SCENE_IMAGES_DIR / scene_id / f"{scene_id}.JPG"
        
        if not image_path.exists():
            logger.warning(f"Image not found for {scene_id}, skipping")
            skipped += 1
            continue
        
        # Generate warning
        warning, is_real = generate_image_only_warning(scene_id, image_path)
        
        if warning:
            scene_warnings[scene_id] = {
                'scene_id': scene_id,
                'method': 'image_only_ablation',
                'is_real_warning': is_real,  # Track if fallback was used
                'warning': warning,
                'image_path': str(image_path)
            }
            
            if not is_real:
                fallback_count += 1
        else:
            failed_real += 1
        
        # Rate limiting
        time.sleep(1.5)  # Slightly longer delay
    
    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(scene_warnings, f, indent=2)
    
    # Summary
    elapsed_time = time.time() - start_time
    real_success = len(scene_warnings) - fallback_count
    
    logger.info("\n" + "="*60)
    logger.info("IMAGE-ONLY WARNING GENERATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Total scenes: {len(scene_ids)}")
    logger.info(f"Successfully processed: {len(scene_warnings)} total")
    logger.info(f"  - Real VLM warnings: {real_success}")
    logger.info(f"  - Fallback warnings: {fallback_count}")
    logger.info(f"Skipped (no image): {skipped}")
    logger.info(f"Complete failures: {failed_real}")
    logger.info(f"Real success rate: {real_success/len(scene_ids)*100:.1f}%")
    logger.info(f"Total runtime: {elapsed_time/60:.1f} minutes")
    logger.info(f"Avg time per scene: {elapsed_time/len(scene_ids):.1f} seconds")
    logger.info(f"Results saved: {OUTPUT_FILE}")
    
    # Urgency distribution (real warnings only)
    urgency_dist = {}
    for sw in scene_warnings.values():
        if sw['is_real_warning']:  # Only count real warnings
            urgency = sw['warning']['urgency_level']
            urgency_dist[urgency] = urgency_dist.get(urgency, 0) + 1
    
    logger.info("\nUrgency Distribution (real warnings only):")
    for urgency in ['low', 'medium', 'high', 'critical']:
        count = urgency_dist.get(urgency, 0)
        pct = (count / real_success * 100) if real_success > 0 else 0
        logger.info(f"  {urgency}: {count} ({pct:.1f}%)")
    
    return scene_warnings


if __name__ == "__main__":
    try:
        process_all_scenes()
    except KeyboardInterrupt:
        logger.warning("\n\nProcess interrupted by user.")
    except Exception as e:
        logger.error(f"\n\nFatal error: {e}", exc_info=True)
