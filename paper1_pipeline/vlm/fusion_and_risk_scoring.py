"""
Step 6: Fuse VLM Depth with Geometric Features and Calculate Composite Risk Score
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Configuration
VLM_RESULTS_FILE = Path("/data/Malek/Pothole_Assessment_Pipeline/Masks_Batches_Dataset/final_assessments/gemini-2_5-pro_full_dataset.json")
OUTPUT_DIR = Path("/data/Malek/Pothole_Assessment_Pipeline/Masks_Batches_Dataset/final_assessments")

# Risk Weights (Paper Methodology)
WEIGHTS = {
    'depth': 0.50,      # VLM Depth is most critical
    'area': 0.30,       # Size matters
    'position': 0.20    # Lane position / trajectory
}

DEPTH_SCORES = {'high': 1.0, 'medium': 0.6, 'low': 0.2}

def calculate_risk_score(depth_cat, area_pixels, lateral_pos):
    """
    Calculate normalized risk score (0-100)
    """
    # 1. Depth Score (0-100)
    s_depth = DEPTH_SCORES.get(depth_cat, 0.2) * 100
    
    # 2. Area Score (0-100) - Logarithmic scaling
    # Assuming area ranges from ~100 to ~50000 pixels
    # Log10(100)=2, Log10(50000)=4.7
    try:
        if area_pixels < 10: area_pixels = 10
        log_area = np.log10(area_pixels)
        # Normalize: map 2.0->0, 4.5->100
        s_area = np.clip((log_area - 2.0) / 2.5 * 100, 0, 100)
    except:
        s_area = 0
        
    # 3. Position Score (0-100)
    # 'in_path' is highest risk, 'in_lane' is medium, others are low
    if lateral_pos in ['in_path_left', 'in_path_right']:
        s_pos = 100  # Directly in the wheel path
    elif lateral_pos in ['in_lane_far_left', 'in_lane_far_right']:
        s_pos = 70   # In the lane but not directly in path
    elif lateral_pos in ['Lane_Boundary', 'Shoulder']:
        s_pos = 50   # On the edge of the drivable area
    else:
        s_pos = 30   # Unknown or other
        
    # Composite Weighted Sum
    composite_score = (
        s_depth * WEIGHTS['depth'] +
        s_area * WEIGHTS['area'] +
        s_pos * WEIGHTS['position']
    )
    
    return round(composite_score, 1), {
        's_depth': s_depth, 
        's_area': s_area, 
        's_pos': s_pos
    }

def categorize_severity(score):
    if score >= 80: return "Critical"
    if score >= 60: return "High"
    if score >= 40: return "Medium"
    return "Low"

def run_fusion():
    print(f"Loading VLM results from: {VLM_RESULTS_FILE}")
    with open(VLM_RESULTS_FILE) as f:
        vlm_data = json.load(f)
        
    fused_results = {}
    
    print("Calculating risk scores...")
    
    # Organize by Scene ID for LLM step later
    scenes = {}
    
    for pid, data in vlm_data.items():
        scene_id = data['scene']
        
        # Extract features
        depth = data.get('depth_category', 'low')
        area = data['geometric_features'].get('area', 0)
        lat_pos = data['positional_features'].get('Lateral_Position', 'Unknown')
        
        # Calculate Risk
        score, components = calculate_risk_score(depth, area, lat_pos)
        severity = categorize_severity(score)
        
        # Create consolidated record
        record = {
            'pothole_id': pid,
            'scene_id': scene_id,
            'bbox_id': data['bbox_id'],
            'risk_assessment': {
                'composite_score': score,
                'severity_level': severity,
                'component_scores': components
            },
            'depth': {
                'category': depth,
                'confidence': data.get('depth_confidence'),
                'reasoning': data.get('depth_reasoning')
            },
            'geometric': data['geometric_features'],
            'positional': data['positional_features']
        }
        
        fused_results[pid] = record
        
        # Group by scene
        if scene_id not in scenes:
            scenes[scene_id] = {'scene_id': scene_id, 'potholes': []}
        scenes[scene_id]['potholes'].append(record)
        
    # Add scene-level summary
    for sid in scenes:
        scenes[sid]['num_potholes'] = len(scenes[sid]['potholes'])
        scenes[sid]['max_risk_score'] = max(p['risk_assessment']['composite_score'] for p in scenes[sid]['potholes'])
        scenes[sid]['highest_severity'] = categorize_severity(scenes[sid]['max_risk_score'])

    # Save outputs
    output_flat = OUTPUT_DIR / "final_risk_scores_flat.json"
    output_scenes = OUTPUT_DIR / "final_assessments_with_risk.json"
    
    with open(output_flat, 'w') as f:
        json.dump(fused_results, f, indent=2)
        
    with open(output_scenes, 'w') as f:
        json.dump(scenes, f, indent=2)

    # Generate distribution stats
    scores = [d['risk_assessment']['composite_score'] for d in fused_results.values()]
    severities = [d['risk_assessment']['severity_level'] for d in fused_results.values()]
    
    print(f"\n{'='*60}")
    print(f"FUSION COMPLETE")
    print(f"{'='*60}")
    print(f"Total Potholes Processed: {len(fused_results)}")
    print(f"Total Scenes: {len(scenes)}")
    print(f"\nRisk Severity Distribution:")
    print(pd.Series(severities).value_counts())
    print(f"\nAverage Risk Score: {np.mean(scores):.1f}")
    print(f"Outputs saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_fusion()
