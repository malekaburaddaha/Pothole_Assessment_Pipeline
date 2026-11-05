"""
Merge VLM depth predictions back into original summary JSONs
Creates enriched final assessment files
"""

import json
import glob
import shutil
from pathlib import Path
from typing import Dict
import logging
from tqdm import tqdm

from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_vlm_predictions(predictions_path: Path) -> Dict[str, Dict]:
    """Load VLM predictions from JSON file"""
    logger.info(f"Loading VLM predictions from {predictions_path}")
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)
    logger.info(f"Loaded {len(predictions)} depth predictions")
    return predictions


def fuse_data():
    """
    Merge VLM depth predictions into original summary files
    Save enriched assessments to final_assessments directory
    """
    # Load VLM predictions
    vlm_predictions_path = OUTPUT_DIR / "vlm_depth_predictions.json"
    if not vlm_predictions_path.exists():
        raise FileNotFoundError(f"VLM predictions not found at {vlm_predictions_path}")
    
    vlm_predictions = load_vlm_predictions(vlm_predictions_path)
    
    # Find all summary files
    summary_pattern = str(SUMMARIES_DIR / "*" / "summary.json")
    summary_files = glob.glob(summary_pattern)
    
    logger.info(f"Found {len(summary_files)} summary files to enrich")
    
    # Statistics
    stats = {
        'scenes_processed': 0,
        'potholes_enriched': 0,
        'potholes_missing_depth': 0
    }
    
    # Process each summary
    for summary_path in tqdm(summary_files, desc="Fusing data"):
        try:
            with open(summary_path, 'r') as f:
                summary_data = json.load(f)
            
            image_name = summary_data['image_name']
            
            # Enrich each pothole
            for result in summary_data.get('results', []):
                bbox_id = result.get('bbox_id', result.get('id', 0))
                pothole_id = f"{image_name}_bbox{bbox_id}"
                
                # Look up VLM prediction
                if pothole_id in vlm_predictions:
                    vlm_data = vlm_predictions[pothole_id]
                    
                    # Add VLM depth information
                    result['vlm_depth_inference'] = {
                        'depth_category': vlm_data['depth_category'],
                        'confidence': vlm_data['confidence'],
                        'reasoning': vlm_data['reasoning'],
                        'used_dual_image': vlm_data['used_dual_image'],
                        'inference_timestamp': vlm_data['timestamp']
                    }
                    
                    stats['potholes_enriched'] += 1
                else:
                    logger.warning(f"No VLM prediction found for {pothole_id}")
                    result['vlm_depth_inference'] = {
                        'depth_category': None,
                        'confidence': None,
                        'reasoning': "No VLM inference available",
                        'used_dual_image': None,
                        'inference_timestamp': None
                    }
                    stats['potholes_missing_depth'] += 1
            
            # Save enriched summary
            output_filename = Path(image_name).stem + "_assessment.json"
            output_path = FINAL_ASSESSMENTS_DIR / output_filename
            
            with open(output_path, 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            stats['scenes_processed'] += 1
            
        except Exception as e:
            logger.error(f"Failed to process {summary_path}: {str(e)}")
    
    # Save fusion statistics
    stats_path = FINAL_ASSESSMENTS_DIR / "fusion_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"DATA FUSION COMPLETE")
    logger.info(f"Scenes processed: {stats['scenes_processed']}")
    logger.info(f"Potholes enriched: {stats['potholes_enriched']}")
    logger.info(f"Potholes missing depth: {stats['potholes_missing_depth']}")
    logger.info(f"Enriched files saved to: {FINAL_ASSESSMENTS_DIR}")
    logger.info(f"{'='*60}\n")


def main():
    """Main entry point"""
    logger.info("Starting data fusion...")
    fuse_data()
    logger.info("Fusion completed successfully!")


if __name__ == "__main__":
    main()
