"""
Calculate Unified Pothole Risk Scores
Combines geometric metrics + VLM depth into single 1-10 severity score
"""

import json
import glob
from pathlib import Path
from typing import Dict
import logging
from tqdm import tqdm

from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============== RISK SCORE CONFIGURATION ==============

# Weights (must sum to 1.0)
WEIGHTS = {
    'depth': 0.40,      # Depth is primary damage predictor
    'area': 0.25,       # Size affects avoidability
    'distance': 0.20,   # Proximity affects urgency
    'solidity': 0.15    # Edge irregularity affects impact sharpness
}

# Scoring functions
DEPTH_SCORES = {
    'low': 2.0,
    'medium': 5.5,
    'high': 9.0
}

DISTANCE_SCORES = {
    'Near': 9.0,
    'Medium Distance': 5.0,
    'Far': 2.0
}

# Area normalization (adjust based on your data distribution)
AREA_MAX = 10000  # pixels² - areas above this get max score

# Solidity: lower solidity = more irregular = higher score


def calculate_area_score(area: float) -> float:
    """Convert area to 0-10 score (larger = higher score)"""
    return min(10.0, (area / AREA_MAX) * 10.0)


def calculate_solidity_score(solidity: float) -> float:
    """Convert solidity to 0-10 score (lower solidity = higher score)"""
    return (1.0 - solidity) * 10.0


def calculate_risk_score(pothole_data: Dict) -> float:
    """
    Calculate unified risk score from pothole metrics
    
    Returns: float in range [0, 10]
    """
    try:
        # Extract data
        depth_category = pothole_data.get('vlm_depth_inference', {}).get('depth_category')
        area = pothole_data.get('mask_metrics', {}).get('area', 0)
        solidity = pothole_data.get('mask_metrics', {}).get('solidity', 1.0)
        distance_category = pothole_data.get('Positional_feature', {}).get('Distance_Category')
        
        # Calculate component scores
        depth_score = DEPTH_SCORES.get(depth_category, 5.0)  # Default to medium if missing
        area_score = calculate_area_score(area)
        solidity_score = calculate_solidity_score(solidity)
        distance_score = DISTANCE_SCORES.get(distance_category, 5.0)
        
        # Weighted sum
        risk_score = (
            WEIGHTS['depth'] * depth_score +
            WEIGHTS['area'] * area_score +
            WEIGHTS['distance'] * distance_score +
            WEIGHTS['solidity'] * solidity_score
        )
        
        # Clamp to [0, 10]
        risk_score = max(0.0, min(10.0, risk_score))
        
        return round(risk_score, 2)
        
    except Exception as e:
        logger.warning(f"Error calculating risk score: {str(e)}")
        return 5.0  # Default to medium risk


def add_risk_scores():
    """
    Add risk scores to all final assessment files
    """
    # Find all assessment files
    assessment_files = glob.glob(str(FINAL_ASSESSMENTS_DIR / "*_assessment.json"))
    
    logger.info(f"Adding risk scores to {len(assessment_files)} assessment files")
    
    stats = {
        'files_processed': 0,
        'potholes_scored': 0,
        'risk_distribution': {
            'low (0-3)': 0,
            'medium (3-7)': 0,
            'high (7-10)': 0
        }
    }
    
    for assessment_path in tqdm(assessment_files, desc="Calculating risk scores"):
        try:
            with open(assessment_path, 'r') as f:
                assessment_data = json.load(f)
            
            # Add risk score to each pothole
            for result in assessment_data.get('results', []):
                risk_score = calculate_risk_score(result)
                result['risk_score'] = risk_score
                
                # Update statistics
                stats['potholes_scored'] += 1
                if risk_score < 3.0:
                    stats['risk_distribution']['low (0-3)'] += 1
                elif risk_score < 7.0:
                    stats['risk_distribution']['medium (3-7)'] += 1
                else:
                    stats['risk_distribution']['high (7-10)'] += 1
            
            # Save updated assessment
            with open(assessment_path, 'w') as f:
                json.dump(assessment_data, f, indent=2)
            
            stats['files_processed'] += 1
            
        except Exception as e:
            logger.error(f"Failed to process {assessment_path}: {str(e)}")
    
    # Save statistics
    stats_path = FINAL_ASSESSMENTS_DIR / "risk_score_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"RISK SCORE CALCULATION COMPLETE")
    logger.info(f"Files processed: {stats['files_processed']}")
    logger.info(f"Potholes scored: {stats['potholes_scored']}")
    logger.info(f"Distribution:")
    for category, count in stats['risk_distribution'].items():
        pct = (count / stats['potholes_scored'] * 100) if stats['potholes_scored'] > 0 else 0
        logger.info(f"  {category}: {count} ({pct:.1f}%)")
    logger.info(f"{'='*60}\n")


def main():
    """Main entry point"""
    logger.info("Calculating risk scores...")
    add_risk_scores()
    logger.info("Risk score calculation completed!")


if __name__ == "__main__":
    main()
