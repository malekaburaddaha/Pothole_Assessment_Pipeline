"""
Configuration for Pothole Assessment VLM Pipeline
"""
import os
from pathlib import Path

# ============== PATHS ==============
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# Input directories
TRANSFORMED_IMAGES_DIR = DATA_ROOT / "Transformed_Data" / "Large_size (3680X1965)" / "Train" / "Train_images"
SUMMARIES_DIR = DATA_ROOT / "With_Distance" / "sam_predictions_large"
VLM_PATCHES_DIR = DATA_ROOT / "vlm_input_patches"

# Output directories
OUTPUT_DIR = DATA_ROOT / "vlm_outputs"
FINAL_ASSESSMENTS_DIR = DATA_ROOT / "final_assessments"

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_ASSESSMENTS_DIR.mkdir(parents=True, exist_ok=True)

# ============== VLM SETTINGS ==============
VLM_MODEL = "gpt-4o"  # or "gpt-4o-mini" for faster/cheaper testing
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Adaptive strategy thresholds
MIN_CROP_SIZE_PX = 100  # If crop smaller than this, use dual-image
DUAL_IMAGE_DISTANCE_CATEGORIES = ["Far"]  # Always use dual-image for these

# ============== DEPTH CATEGORIES ==============
DEPTH_CATEGORIES = ["low", "medium", "high"]
DEPTH_DEFINITIONS = {
    "low": "Shallow surface depression (< 2cm depth)",
    "medium": "Moderate depth pothole (2-5cm depth)", 
    "high": "Deep, hazardous pothole (> 5cm depth)"
}

# ============== API SETTINGS ==============
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
REQUEST_TIMEOUT = 60
SAVE_CHECKPOINT_EVERY = 100  # Save progress every N potholes

# ============== LOGGING ==============
LOG_FILE = OUTPUT_DIR / "inference.log"
ERROR_LOG_FILE = OUTPUT_DIR / "errors.log"
COST_LOG_FILE = OUTPUT_DIR / "cost_tracking.json"
