"""
Test VLM Inference - Gemini API
Tests depth classification on 12 sample scenes
"""
import json
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime
# Import Gemini functions
from gemini_inference import call_gemini_api, should_use_dual_image
# For GPT 
from gpt5_inference import call_gpt5_api 

# Import config
from test_config import (
    SAM_PREDICTIONS_DIR,
    IMAGE_BATCHES_DIR,
    OUTPUT_DIR,
    TEST_MODELS,
    LOG_FILE
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def test_model(model_name: str):
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing Model: {model_name}")
    logger.info(f"{'='*60}\n")

    scene_dirs = sorted([p for p in SAM_PREDICTIONS_DIR.iterdir() if p.is_dir()])
    logger.info(f"Found {len(scene_dirs)} scene folders for testing.")

    results = {}
    total_potholes = 0
    processed_potholes = 0
    failed_potholes = []
    dual_image_count = 0
    single_image_count = 0

    for scene_dir in tqdm(scene_dirs, desc=f"Processing scenes with {model_name}"):
        scene_id = scene_dir.name
        summary_path = scene_dir / "summary.json"
        if not summary_path.exists():
            logger.warning(f"No summary.json in {scene_dir}")
            continue
        try:
            with open(summary_path) as f:
                summary_data = json.load(f)
            full_image_path = IMAGE_BATCHES_DIR / scene_id / f"{scene_id}.JPG"
            
            for result in summary_data.get('results', []):
                bbox_id = result.get('bbox_id', result.get('id', 0))
                pothole_id = f"{scene_id}_bbox{bbox_id}"
                crop_path = IMAGE_BATCHES_DIR / scene_id / f"{scene_id}_mask{bbox_id}.jpg"
               
                if not crop_path.exists():
                    logger.warning(f"Crop image not found: {crop_path}")
                    failed_potholes.append({'id': pothole_id, 'reason': 'crop_not_found'})
                    continue
                
                # Gather required info for inference
                metrics = result.get('mask_metrics', {})
                positional_features = result.get('Positional_feature', {})
                bbox = result.get('gt_bbox', [0, 0, 0, 0])
                distance_category = positional_features.get('Distance_Category', 'Unknown')
                use_dual = should_use_dual_image(bbox, distance_category)
                
                if use_dual:
                    dual_image_count += 1
                else:
                    single_image_count += 1
                
                try:
                    # Route to correct API based on model name
                    if 'gpt' in model_name.lower():
                        depth_result = call_gpt5_api(
                            str(crop_path),
                            str(full_image_path) if full_image_path.exists() else None,
                            metrics,
                            positional_features,
                            use_dual
                        )
                    else:
                        # ✓ CHANGED: Now uses Gemini API
                        depth_result = call_gemini_api(
                            str(crop_path),
                            str(full_image_path) if full_image_path.exists() else None,
                            metrics,
                            positional_features,
                            use_dual,
                            model_name=model_name
                        )
                    
                    results[pothole_id] = {
                        'model': model_name,
                        'scene': scene_id,
                        'bbox_id': bbox_id,
                        'depth_category': depth_result['depth_category'],
                        'confidence': depth_result['confidence'],
                        'reasoning': depth_result['reasoning'],
                        'used_dual_image': depth_result['used_dual_image'],
                        'timestamp': datetime.now().isoformat()
                    }
                    processed_potholes += 1
                    
                except Exception as e:
                    logger.error(f"Failed to process {pothole_id}: {e}")
                    failed_potholes.append({'id': pothole_id, 'reason': str(e)})
                
                total_potholes += 1
                
        except Exception as e:
            logger.error(f"Failed to process scene {scene_id}: {e}")

    # Save results
    model_name_safe = model_name.replace(':', '_').replace('.', '_')
    output_file = OUTPUT_DIR / f"{model_name_safe}_predictions.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n{'='*60}\nTEST COMPLETE: {model_name}")
    logger.info(f"Processed: {processed_potholes}/{total_potholes} potholes")
    logger.info(f"Failed: {len(failed_potholes)}")
    logger.info(f"Dual-image: {dual_image_count}, single-image: {single_image_count}")
    logger.info(f"Results: {output_file}\n{'='*60}\n")
    
    return results


def main():
    logger.info("="*60)
    logger.info("STARTING VLM TEST PIPELINE")
    logger.info("="*60)
    
    for model in TEST_MODELS:
        test_model(model)
    
    logger.info("\nTest pipeline complete.")


if __name__ == "__main__":
    main()
