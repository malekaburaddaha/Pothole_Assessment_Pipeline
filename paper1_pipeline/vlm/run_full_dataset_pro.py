"""
Run chosen VLM (Gemini Pro) on full dataset (ALL potholes, including 'Far')
"""
import json
from pathlib import Path
from tqdm import tqdm
import logging
from gemini_inference import call_gemini_api, should_use_dual_image

# Configuration
FULL_SAM_DIR = Path("/data/Malek/Pothole_Assessment_Pipeline/Masks_Batches_Dataset/sam_predictions_large")
FULL_IMAGE_DIR = Path("/data/Malek/Pothole_Assessment_Pipeline/Masks_Batches_Dataset/image_patches")
OUTPUT_DIR = Path("/data/Malek/Pothole_Assessment_Pipeline/Masks_Batches_Dataset/final_assessments")
OUTPUT_DIR.mkdir(exist_ok=True)

# Updated to Gemini Pro as requested
CHOSEN_MODEL = "gemini-2.5-pro" 
LOG_FILE = OUTPUT_DIR / f"full_dataset_{CHOSEN_MODEL.replace('.', '_').replace(':', '_')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_full_dataset():
    """
    Process all scenes in full dataset
    """
    scene_dirs = sorted([d for d in FULL_SAM_DIR.iterdir() if d.is_dir()])
    
    logger.info(f"Found {len(scene_dirs)} scenes to process")
    logger.info(f"Using model: {CHOSEN_MODEL}")
    
    all_results = {}
    total_processed = 0
    total_failed = 0
    
    for scene_dir in tqdm(scene_dirs, desc="Processing scenes"):
        scene_id = scene_dir.name
        summary_path = scene_dir / "summary.json"
        
        if not summary_path.exists():
            logger.warning(f"No summary for {scene_id}, skipping")
            continue
        
        with open(summary_path) as f:
            summary_data = json.load(f)
        
        full_image_path = FULL_IMAGE_DIR / scene_id / f"{scene_id}.JPG"
        
        for result in summary_data.get('results', []):
            bbox_id = result.get('bbox_id', result.get('id', 0))
            pothole_id = f"{scene_id}_bbox{bbox_id}"
            
            # Get features
            metrics = result.get('mask_metrics', {})
            positional = result.get('Positional_feature', {})
            bbox = result.get('gt_bbox', [0, 0, 0, 0])
            distance_cat = positional.get('Distance_Category', 'Unknown')
            
            # NOTE: "Far" potholes are now INCLUDED as requested
            
            crop_path = FULL_IMAGE_DIR / scene_id / f"{scene_id}_mask{bbox_id}.jpg"
            
            if not crop_path.exists():
                logger.warning(f"Missing crop: {crop_path}")
                total_failed += 1
                continue
            
            # Determine dual-image strategy
            use_dual = should_use_dual_image(bbox, distance_cat)
            
            try:
                depth_result = call_gemini_api(
                    str(crop_path),
                    str(full_image_path) if full_image_path.exists() else None,
                    metrics,
                    positional,
                    use_dual,
                    model_name=CHOSEN_MODEL
                )
                
                all_results[pothole_id] = {
                    'scene': scene_id,
                    'bbox_id': bbox_id,
                    'depth_category': depth_result['depth_category'],
                    'depth_confidence': depth_result['confidence'],
                    'depth_reasoning': depth_result['reasoning'],
                    'geometric_features': metrics,
                    'positional_features': positional,
                    'used_dual_image': use_dual
                }
                
                total_processed += 1
                
                # Save checkpoint every 100 samples
                if total_processed % 100 == 0:
                    checkpoint_file = OUTPUT_DIR / f"checkpoint_{CHOSEN_MODEL.replace('.', '_')}.json"
                    with open(checkpoint_file, 'w') as f:
                        json.dump(all_results, f, indent=2)
                    logger.info(f"Checkpoint: {total_processed} processed")
                
            except Exception as e:
                logger.error(f"Failed {pothole_id}: {e}")
                total_failed += 1
    
    # Save final results
    final_output = OUTPUT_DIR / f"{CHOSEN_MODEL.replace('.', '_').replace(':', '_')}_full_dataset.json"
    with open(final_output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"FULL DATASET PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total processed: {total_processed}")
    logger.info(f"Total failed: {total_failed}")
    logger.info(f"Results saved: {final_output}")
    
    return all_results

if __name__ == "__main__":
    run_full_dataset()
