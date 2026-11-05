"""
Master script to run complete pipeline end-to-end
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import vlm_inference
import data_fusion
import calculate_risk_scores

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_full_pipeline():
    """Execute complete pipeline"""
    
    logger.info("="*60)
    logger.info("STARTING POTHOLE ASSESSMENT PIPELINE")
    logger.info("="*60)
    
    try:
        # Step 1: VLM Inference
        logger.info("\n[STEP 1/3] Running VLM depth inference...")
        vlm_inference.main()
        
        # Step 2: Data Fusion
        logger.info("\n[STEP 2/3] Fusing VLM predictions with original data...")
        data_fusion.main()
        
        # Step 3: Risk Score Calculation
        logger.info("\n[STEP 3/3] Calculating unified risk scores...")
        calculate_risk_scores.main()
        
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"\nPIPELINE FAILED: {str(e)}")
        raise


if __name__ == "__main__":
    run_full_pipeline()
