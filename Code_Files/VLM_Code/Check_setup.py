"""
Verify API keys and configuration
"""
import os
from dotenv import load_dotenv

load_dotenv()

print("="*60)
print("Configuration Check")
print("="*60)

# Check API keys
google_key = os.getenv("GOOGLE_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

print(f"\n✓ Google API Key: {'Set (' + google_key[:10] + '...)' if google_key else '✗ MISSING'}")
print(f"✓ OpenAI API Key: {'Set (' + openai_key[:10] + '...)' if openai_key else '✗ Not Set (optional)'}")

# Check file structure
from pathlib import Path
base_path = Path('/data/Malek/Pothole_Assessment_Pipeline/Masks_Batches_Dataset/test_data')
print(f"\n✓ Test data directory: {'Found' if base_path.exists() else '✗ NOT FOUND'}")
print(f"✓ SAM predictions: {'Found' if (base_path / 'sam_predictions_large').exists() else '✗ NOT FOUND'}")
print(f"✓ Image patches: {'Found' if (base_path / 'image_patches').exists() else '✗ NOT FOUND'}")

# Check imports
try:
    import google.generativeai as genai
    print(f"\n✓ google-generativeai: Installed")
except ImportError:
    print(f"\n✗ google-generativeai: NOT INSTALLED - run 'pip install google-generativeai'")

print("\n" + "="*60)
if google_key:
    print("✓ Ready to run! Execute: python test_vlm_inference.py")
else:
    print("✗ Fix issues above before running")
print("="*60)
