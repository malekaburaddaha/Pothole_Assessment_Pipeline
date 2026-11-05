"""
Generate Excel template for human annotation with full context
"""
import pandas as pd
import json
import random
from pathlib import Path
from test_config import OUTPUT_DIR, SAM_PREDICTIONS_DIR, IMAGE_BATCHES_DIR

def create_annotation_template(n_samples=50):
    """
    Create Excel template with sample images for human annotation
    Includes geometric features and VLM predictions for context
    """
    
    # Load VLM predictions
    flash_preds = json.load(open(OUTPUT_DIR / "gemini-2_5-flash_predictions.json"))
    pro_preds = json.load(open(OUTPUT_DIR / "gemini-2_5-pro_predictions.json"))
    gpt5_preds = json.load(open(OUTPUT_DIR / "gpt-5_predictions.json"))
    
    # Randomly sample n_samples
    sample_ids = random.sample(list(flash_preds.keys()), min(n_samples, len(flash_preds)))
    
    rows = []
    
    for pid in sample_ids:
        scene_id, bbox_str = pid.split('_bbox')
        bbox_id = int(bbox_str)
        
        # Get paths
        crop_path = IMAGE_BATCHES_DIR / scene_id / f"{scene_id}_mask{bbox_id}.jpg"
        full_path = IMAGE_BATCHES_DIR / scene_id / f"{scene_id}.JPG"
        
        # Load summary data for geometric features
        summary_path = SAM_PREDICTIONS_DIR / scene_id / "summary.json"
        with open(summary_path) as f:
            summary = json.load(f)
        
        # Find matching result
        result = next((r for r in summary['results'] if r.get('bbox_id', r.get('id', 0)) == bbox_id), None)
        
        if result:
            metrics = result.get('mask_metrics', {})
            positional = result.get('Positional_feature', {})
            
            rows.append({
                'Pothole_ID': pid,
                'Crop_Image_Path': str(crop_path.relative_to(crop_path.parents[3])),
                'Full_Scene_Path': str(full_path.relative_to(full_path.parents[3])),
                'Area_pixels': metrics.get('area', 'N/A'),
                'Circularity': f"{metrics.get('circularity', 0):.3f}" if metrics.get('circularity') else 'N/A',
                'Distance_Category': positional.get('Distance_Category', 'N/A'),
                'Lateral_Position': positional.get('Lateral_Position', 'N/A'),
                'Flash_Prediction': flash_preds[pid]['depth_category'],
                'Flash_Confidence': flash_preds[pid]['confidence'],
                'Pro_Prediction': pro_preds[pid]['depth_category'],
                'GPT5_Prediction': gpt5_preds[pid]['depth_category'],
                'Annotator_1_Depth': '',  # To be filled
                'Annotator_2_Depth': '',  # To be filled
                'Annotator_3_Depth': '',  # To be filled
                'Notes': ''
            })
    
    df = pd.DataFrame(rows)
    
    # Save to Excel
    excel_path = OUTPUT_DIR / 'human_annotation_template.xlsx'
    
    # Create Excel with instructions sheet
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Instructions sheet
        instructions = pd.DataFrame({
            'Instructions': [
                '=== HUMAN ANNOTATION INSTRUCTIONS ===',
                '',
                '1. Open each pothole image (see Crop_Image_Path and Full_Scene_Path)',
                '2. Consider ALL available information:',
                '   - Visual appearance (shadows, texture)',
                '   - Size (Area)',
                '   - Shape (Circularity)',
                '   - Context (Distance, Lateral Position)',
                '   - VLM predictions (Flash, Pro, GPT-5)',
                '',
                '3. Classify depth category in your column:',
                '   - low: Shallow surface depression (< 2cm depth)',
                '   - medium: Moderate depth pothole (2-5cm depth)',
                '   - high: Deep, hazardous pothole (> 5cm depth)',
                '',
                '4. Rules:',
                '   - Annotate independently (do not discuss with others)',
                '   - Use lowercase only (low/medium/high)',
                '   - Add notes if uncertain',
                '',
                '5. Save file when complete',
                '',
                'Time estimate: 45-60 minutes'
            ]
        })
        instructions.to_excel(writer, sheet_name='Instructions', index=False)
        
        # Data sheet
        df.to_excel(writer, sheet_name='Annotations', index=False)
    
    print(f"✓ Created annotation template: {excel_path}")
    print(f"✓ Contains {len(df)} samples")
    print(f"\nNext steps:")
    print(f"1. Share file with 2 colleagues")
    print(f"2. Each person fills their column independently")
    print(f"3. Collect completed files by Friday")
    print(f"\nTemplate includes:")
    print(f"  - Crop and full scene paths")
    print(f"  - Geometric features (area, circularity)")
    print(f"  - Positional context (distance, lateral position)")
    print(f"  - All 3 VLM predictions for reference")
    
    return excel_path

if __name__ == "__main__":
    create_annotation_template(n_samples=50)