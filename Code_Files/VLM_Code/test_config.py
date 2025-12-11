"""
Test configuration for pilot run (10-20 images)
Loads API keys from .env file
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Verify keys are loaded
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in .env file!")

# Paths
BASE_TEST_DATA = Path('/data/Malek/Pothole_Assessment_Pipeline/Masks_Batches_Dataset/test_data')
IMAGE_BATCHES_DIR = BASE_TEST_DATA / 'image_patches'
SAM_PREDICTIONS_DIR = BASE_TEST_DATA / 'sam_predictions_large'
OUTPUT_DIR = BASE_TEST_DATA.parent / 'test_outputs'
FINAL_ASSESSMENTS_DIR = OUTPUT_DIR / 'final_assessments'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_ASSESSMENTS_DIR.mkdir(parents=True, exist_ok=True)

# VLM settings - Load from environment
TEST_MODELS = ["gemini-2.5-flash", "gemini-2.5-pro", "gpt-5"] # Testing with 3 models.  
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  

# Thresholds
MIN_CROP_SIZE_PX = 100
DUAL_IMAGE_DISTANCE_CATEGORIES = ["Far"]

# Depth categories
DEPTH_CATEGORIES = ["low", "medium", "high"]
DEPTH_DEFINITIONS = {
    "low": "Shallow surface depression (< 2cm depth)",
    "medium": "Moderate depth pothole (2-5cm depth)",
    "high": "Deep, hazardous pothole (> 5cm depth)"
}

# API settings
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 3
REQUEST_TIMEOUT = 30
SAVE_CHECKPOINT_EVERY = 5

# Logging
LOG_FILE = OUTPUT_DIR / "test_inference.log"
ERROR_LOG_FILE = OUTPUT_DIR / "test_errors.log"
COST_LOG_FILE = OUTPUT_DIR / "test_cost_tracking.json"

# Print confirmation (optional, for debugging)
print(f"✓ Configuration loaded")
print(f"✓ Google API Key: {'Set' if GOOGLE_API_KEY else 'Missing'}")
print(f"✓ OpenAI API Key: {'Set' if OPENAI_API_KEY else 'Not Set (optional)'}")
