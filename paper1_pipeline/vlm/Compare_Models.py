import json
from pathlib import Path

# official path 
home_path = Path('/data/Malek/Pothole_Assessment_Pipeline/Masks_Batches_Dataset/test_outputs')
flash = json.load(open(home_path / "gemini-2_5-flash_predictions.json"))
pro = json.load(open(home_path / "gemini-2_5-pro_predictions.json"))

agreement = sum(1 for k in flash if flash[k]['depth_category'] == pro[k]['depth_category'])
print(f"Agreement: {agreement}/{len(flash)} ({agreement/len(flash)*100:.1f}%)")

# Where they disagree:
for k in flash:
    if flash[k]['depth_category'] != pro[k]['depth_category']:
        print(f"{k}: Flash={flash[k]['depth_category']}, Pro={pro[k]['depth_category']}")
