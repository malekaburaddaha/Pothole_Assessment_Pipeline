"""
Test configuration for pilot run (10-20 images)
"""
import os
from pathlib import Path

# Absolute path to the test_data directory 
BASE_TEST_DATA = Path('data/Malek/Pothole_Assessment_Pipeline/Masks_Batches_Dataset/test_data')

# Path to image batches (Full scenes + crops) 
IMAGE_BATCHES_DIR = BASE_TEST_DATA/ 'images_patches'   # e.g., images_patches/G0024826/G0024826.JPG
# Path to the summary Json files (per scene)
SAM_PREDICTIONS_DIR = BASE_TEST_DATA/ 'sam_prediction_large'  # e.g., sam_predictions_large/G0024826/summary.json

# Output paths
OUTPUT_DIR = BASE_TEST_DATA.parent / 'test_output'  # Appending this to the output path, e.g., /data/Malek/Pothole_Assessment_Pipeline/Masks_Batches_Dataset/test_outputs
FINAL-ASSESSMENTS_DIR = OUTPUT_DIR/ 'final_assessment' # Appending the final assessment path to the output directory as well. 
# Creating the directories and making sure they exist
OUTPUT_DIR.makdir(parents=True, exist_ok=True)
FINAL_ASSESSMENT_DIR.mkdir(parents=True, exist_ok=True) 

# Model/API variables remain unchanged
TEST_MODELS = ["gpt-5", "gemini-2.5-flash"]
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Thresholds and categories unchanged
MIN_CROP_SIZE_PX = 100
DUAL_IMAGE_DISTANCE_CATEGORIES = ["Far"]
DEPTH_CATEGORIES = ["low", "medium", "high"]
DEPTH_DEFINITIONS = {
    "low": "Shallow surface depression (< 2cm depth)",
    "medium": "Moderate depth pothole (2-5cm depth)",
    "high": "Deep, hazardous pothole (> 5cm depth)"
}

# API settings for testing
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 3
REQUEST_TIMEOUT = 30
SAVE_CHECKPOINT_EVERY = 5

# Logging
LOG_FILE = OUTPUT_DIR / "test_inference.log"
ERROR_LOG_FILE = OUTPUT_DIR / "test_errors.log"
COST_LOG_FILE = OUTPUT_DIR / "test_cost_tracking.json"


