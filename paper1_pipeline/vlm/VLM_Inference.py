"""
VLM Depth Inference with Adaptive Multi-Image Strategy
Handles blurriness by sending full scene + crop for small/distant potholes
"""

import os
import json
import base64
import time
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI, APIError, RateLimitError, APIConnectionError

from config import *

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Cost tracking
cost_tracker = {
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "total_cost_usd": 0.0,
    "requests_count": 0,
    "start_time": datetime.now().isoformat()
}


def encode_image_base64(image_path: str) -> str:
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_crop_dimensions(bbox: List[int]) -> Tuple[int, int]:
    """Calculate width and height from bbox [x1, y1, x2, y2]"""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return width, height


def should_use_dual_image(bbox: List[int], distance_category: str) -> bool:
    """
    Determine if we should send full scene + crop (vs. crop only)
    Based on crop size and distance category
    """
    width, height = get_crop_dimensions(bbox)
    min_dimension = min(width, height)
    
    # Use dual-image if crop is small OR pothole is far
    if min_dimension < MIN_CROP_SIZE_PX:
        logger.debug(f"Using dual-image: crop too small ({min_dimension}px)")
        return True
    
    if distance_category in DUAL_IMAGE_DISTANCE_CATEGORIES:
        logger.debug(f"Using dual-image: distance category = {distance_category}")
        return True
    
    return False


def construct_prompt(metrics: Dict, positional_features: Dict, use_dual_image: bool = False) -> str:
    """
    Construct VLM prompt with metrics and task description
    """
    base_prompt = f"""You are an expert computer vision system assessing road hazards for autonomous vehicles.

**TASK:** Analyze the provided pothole image to determine its DEPTH category.

**Known Geometric Properties (measured via computer vision):**
- Area: {metrics.get('area', 'N/A')} pixels²
- Circularity: {metrics.get('circularity', 'N/A'):.3f} (0=irregular, 1=perfect circle)
- Solidity: {metrics.get('solidity', 'N/A'):.3f} (ratio of pothole area to convex hull)
- Aspect Ratio: {metrics.get('aspect_ratio', 'N/A'):.3f}
- Distance Category: {positional_features.get('Distance_Category', 'N/A')}
- Lateral Position: {positional_features.get('Lateral_Position', 'N/A')}

**Depth Category Definitions:**
- low: {DEPTH_DEFINITIONS['low']}
- medium: {DEPTH_DEFINITIONS['medium']}
- high: {DEPTH_DEFINITIONS['high']}

**Your Analysis:**
Based on visual evidence in the image (shadow depth, edge sharpness, surface texture, darkness/lighting), classify this pothole's DEPTH into ONE category: low, medium, or high.

**Visual Cues to Consider:**
1. Shadow depth and darkness (deeper = darker shadows)
2. Edge definition and sharpness (deeper = sharper edges)
3. Surface texture visibility (deeper = less visible bottom texture)
4. Lighting contrast between pothole interior and road surface
"""

    if use_dual_image:
        base_prompt += """
**Note:** Two images are provided:
1. Full road scene (for spatial context and lighting reference)
2. Cropped pothole region (target for depth assessment)

Use the full scene to understand overall lighting and road texture, then focus on the cropped region to assess depth characteristics.
"""
    
    return base_prompt


def call_vlm_api(
    crop_image_path: str,
    full_image_path: Optional[str],
    metrics: Dict,
    positional_features: Dict,
    use_dual_image: bool
) -> Dict:
    """
    Call GPT-4o Vision API with adaptive multi-image strategy
    
    Returns:
        Dict with keys: depth_category, confidence, reasoning, tokens_used
    """
    try:
        # Encode images
        crop_b64 = encode_image_base64(crop_image_path)
        
        # Construct prompt
        prompt = construct_prompt(metrics, positional_features, use_dual_image)
        
        # Build message content
        content = [{"type": "text", "text": prompt}]
        
        # Add full scene if using dual-image strategy
        if use_dual_image and full_image_path and os.path.exists(full_image_path):
            scene_b64 = encode_image_base64(full_image_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{scene_b64}",
                    "detail": "high"
                }
            })
        
        # Add cropped patch (always included)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{crop_b64}",
                "detail": "high"
            }
        })
        
        # API call with structured output
        response = client.chat.completions.create(
            model=VLM_MODEL,
            messages=[{"role": "user", "content": content}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "depth_assessment_schema",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "depth_category": {
                                "type": "string",
                                "enum": DEPTH_CATEGORIES,
                                "description": "Predicted depth category"
                            },
                            "confidence": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                                "description": "Confidence in prediction"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Brief explanation of visual cues used"
                            }
                        },
                        "required": ["depth_category", "confidence", "reasoning"],
                        "additionalProperties": False
                    }
                }
            },
            temperature=0.1,  # Low temperature for consistency
            max_tokens=500,
            timeout=REQUEST_TIMEOUT
        )
        
        # Parse response
        result = json.loads(response.choices[0].message.content)
        
        # Track token usage and cost
        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        
        # GPT-4o pricing: $5/1M input, $20/1M output
        cost = (input_tokens * 5.0 / 1_000_000) + (output_tokens * 20.0 / 1_000_000)
        
        result['tokens_used'] = {
            'input': input_tokens,
            'output': output_tokens,
            'cost_usd': cost
        }
        
        # Update global cost tracker
        cost_tracker['total_input_tokens'] += input_tokens
        cost_tracker['total_output_tokens'] += output_tokens
        cost_tracker['total_cost_usd'] += cost
        cost_tracker['requests_count'] += 1
        
        return result
        
    except (APIError, RateLimitError, APIConnectionError) as e:
        logger.error(f"API Error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in VLM call: {str(e)}")
        raise


def infer_depth_with_retry(
    crop_path: str,
    full_image_path: Optional[str],
    metrics: Dict,
    positional_features: Dict,
    bbox: List[int],
    distance_category: str
) -> Dict:
    """
    Infer depth with retry logic for robustness
    """
    use_dual_image = should_use_dual_image(bbox, distance_category)
    
    for attempt in range(MAX_RETRIES):
        try:
            result = call_vlm_api(
                crop_path,
                full_image_path,
                metrics,
                positional_features,
                use_dual_image
            )
            result['used_dual_image'] = use_dual_image
            result['attempt'] = attempt + 1
            return result
            
        except RateLimitError as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY_SECONDS * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 2}/{MAX_RETRIES}")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed after {MAX_RETRIES} attempts: {str(e)}")
                raise
                
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}, retrying...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logger.error(f"Failed after {MAX_RETRIES} attempts: {str(e)}")
                raise


def process_all_potholes() -> Dict[str, Dict]:
    """
    Main processing function: iterate through all summaries and infer depth for each pothole
    """
    # Find all summary files
    summary_pattern = str(SUMMARIES_DIR / "*" / "summary.json")
    summary_files = glob.glob(summary_pattern)
    
    logger.info(f"Found {len(summary_files)} summary files to process")
    
    # Load existing results if resuming
    checkpoint_path = OUTPUT_DIR / "vlm_depth_predictions.json"
    if checkpoint_path.exists():
        logger.info("Loading existing checkpoint...")
        with open(checkpoint_path, 'r') as f:
            all_results = json.load(f)
        logger.info(f"Resuming from {len(all_results)} completed potholes")
    else:
        all_results = {}
    
    # Process each summary file
    total_potholes = 0
    processed_potholes = 0
    failed_potholes = []
    
    for summary_path in tqdm(summary_files, desc="Processing scenes"):
        try:
            with open(summary_path, 'r') as f:
                summary_data = json.load(f)
            
            image_name = summary_data['image_name']
            scene_dir = Path(summary_path).parent.name
            
            # Full scene image path
            full_image_path = TRANSFORMED_IMAGES_DIR / image_name
            
            # Process each pothole in this scene
            for result in summary_data.get('results', []):
                total_potholes += 1
                
                # Create unique pothole ID
                bbox_id = result.get('bbox_id', result.get('id', 0))
                pothole_id = f"{image_name}_bbox{bbox_id}"
                
                # Skip if already processed
                if pothole_id in all_results:
                    logger.debug(f"Skipping already processed: {pothole_id}")
                    continue
                
                # Paths
                crop_filename = f"mask{bbox_id}.jpg"
                crop_path = VLM_PATCHES_DIR / scene_dir / crop_filename
                
                if not crop_path.exists():
                    logger.warning(f"Crop not found: {crop_path}")
                    failed_potholes.append({
                        'id': pothole_id,
                        'reason': 'crop_not_found'
                    })
                    continue
                
                # Extract metrics
                metrics = result.get('mask_metrics', {})
                positional_features = result.get('Positional_feature', {})
                bbox = result.get('gt_bbox', [0, 0, 0, 0])
                distance_category = positional_features.get('Distance_Category', 'Unknown')
                
                # Infer depth
                try:
                    depth_result = infer_depth_with_retry(
                        str(crop_path),
                        str(full_image_path) if full_image_path.exists() else None,
                        metrics,
                        positional_features,
                        bbox,
                        distance_category
                    )
                    
                    # Store result
                    all_results[pothole_id] = {
                        'scene': image_name,
                        'bbox_id': bbox_id,
                        'depth_category': depth_result['depth_category'],
                        'confidence': depth_result['confidence'],
                        'reasoning': depth_result['reasoning'],
                        'used_dual_image': depth_result['used_dual_image'],
                        'tokens': depth_result['tokens_used'],
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    processed_potholes += 1
                    
                    # Checkpoint save
                    if processed_potholes % SAVE_CHECKPOINT_EVERY == 0:
                        with open(checkpoint_path, 'w') as f:
                            json.dump(all_results, f, indent=2)
                        logger.info(f"Checkpoint saved: {processed_potholes} potholes completed")
                        
                        # Save cost tracking
                        with open(COST_LOG_FILE, 'w') as f:
                            json.dump(cost_tracker, f, indent=2)
                    
                except Exception as e:
                    logger.error(f"Failed to process {pothole_id}: {str(e)}")
                    failed_potholes.append({
                        'id': pothole_id,
                        'reason': str(e)
                    })
                    
        except Exception as e:
            logger.error(f"Failed to process summary {summary_path}: {str(e)}")
    
    # Final save
    with open(checkpoint_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save failures log
    if failed_potholes:
        with open(OUTPUT_DIR / "failed_potholes.json", 'w') as f:
            json.dump(failed_potholes, f, indent=2)
    
    # Final cost report
    cost_tracker['end_time'] = datetime.now().isoformat()
    cost_tracker['total_potholes_processed'] = processed_potholes
    cost_tracker['failed_count'] = len(failed_potholes)
    with open(COST_LOG_FILE, 'w') as f:
        json.dump(cost_tracker, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"PROCESSING COMPLETE")
    logger.info(f"Total potholes found: {total_potholes}")
    logger.info(f"Successfully processed: {processed_potholes}")
    logger.info(f"Failed: {len(failed_potholes)}")
    logger.info(f"Total cost: ${cost_tracker['total_cost_usd']:.2f}")
    logger.info(f"Total tokens: {cost_tracker['total_input_tokens'] + cost_tracker['total_output_tokens']:,}")
    logger.info(f"Results saved to: {checkpoint_path}")
    logger.info(f"{'='*60}\n")
    
    return all_results


def main():
    """Main entry point"""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set!")
    
    logger.info("Starting VLM depth inference pipeline...")
    logger.info(f"Model: {VLM_MODEL}")
    logger.info(f"Adaptive strategy: dual-image for crops <{MIN_CROP_SIZE_PX}px or {DUAL_IMAGE_DISTANCE_CATEGORIES}")
    
    results = process_all_potholes()
    
    logger.info("Pipeline completed successfully!")
    return results


if __name__ == "__main__":
    main()
