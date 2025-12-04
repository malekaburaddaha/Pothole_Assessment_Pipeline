"""
Validate VLM predictions against RGB-D ground truth depth maps
PothRGBD Dataset with .npy depth files and YOLO format labels
"""
import numpy as np
import cv2
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from test_config import OUTPUT_DIR

# Dataset paths
RGBD_DATASET_ROOT = Path("/data/Malek/Pothole_Assessment_Pipeline/PothRGBD_Dataset/PUBLIC_POTHOLE_DATASET") 
RGB_DIR = RGBD_DATASET_ROOT / "images"
DEPTH_DIR = RGBD_DATASET_ROOT / "depths"
LABELS_DIR = RGBD_DATASET_ROOT / "labels"

# Depth thresholds (in centimeters)
DEPTH_THRESHOLDS = {
    'low': (0, 2.0),      # < 2cm
    'medium': (2.0, 5.0),  # 2-5cm
    'high': (5.0, 100.0)   # > 5cm
}

def load_yolo_bbox(label_path, img_width, img_height):
    """
    Load bounding box from YOLO format label file
    
    Args:
        label_path: Path to .txt label file
        img_width, img_height: Image dimensions
    
    Returns:
        [x1, y1, x2, y2] in pixel coordinates or None
    """
    if not label_path.exists():
        return None
    
    try:
        with open(label_path) as f:
            # YOLO format: class_id center_x center_y width height (normalized 0-1)
            line = f.readline().strip()
            if not line:
                return None
            
            parts = line.split()
            if len(parts) < 5:
                return None
            
            # Extract normalized coordinates
            _, cx, cy, w, h = map(float, parts[:5])
            
            # Convert to pixel coordinates
            x1 = int((cx - w/2) * img_width)
            y1 = int((cy - h/2) * img_height)
            x2 = int((cx + w/2) * img_width)
            y2 = int((cy + h/2) * img_height)
            
            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_width, x2), min(img_height, y2)
            
            return [x1, y1, x2, y2]
    
    except Exception as e:
        print(f"Error loading bbox from {label_path}: {e}")
        return None

def categorize_depth(depth_cm):
    """Convert continuous depth to category"""
    if depth_cm is None:
        return None
    
    for category, (min_d, max_d) in DEPTH_THRESHOLDS.items():
        if min_d <= depth_cm < max_d:
            return category
    
    return 'high'  # Default to high if > 5cm

def run_all_models_rgbd(models=["gemini-2.5-flash", "gemini-2.5-pro", "gpt-5"], n_samples=50):
    """
    Run all models on RGB-D validation and compare
    """
    from gemini_inference import call_gemini_api
    from gpt5_inference import call_gpt5_api
    
    # Get sample of RGB images
    rgb_images = sorted(list(RGB_DIR.glob("*.jpg")) + list(RGB_DIR.glob("*.png")))[:n_samples]
    
    print(f"\nFound {len(rgb_images)} RGB images to process")
    
    all_results = {}
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"Processing model: {model}")
        print(f"{'='*60}")
        
        results = []
        skipped = 0
        
        for rgb_path in tqdm(rgb_images, desc=f"Running {model} on RGB-D"):
            # FIXED: Extract timestamp prefix correctly
            # From: "20250227_135438_color_png.rf.984e9768eb679052fd79e702413a94c5.jpg"
            # To:   "20250227_135438"
            filename_parts = rgb_path.stem.split('_')
            
            if len(filename_parts) < 2:
                skipped += 1
                continue
            
            timestamp = f"{filename_parts[0]}_{filename_parts[1]}"
            
            # Construct matching filenames
            depth_path = DEPTH_DIR / f"{timestamp}_depth.npy"
            label_path = LABELS_DIR / f"{rgb_path.stem}.txt" 

            
            if not depth_path.exists():
                skipped += 1
                continue
            
            try:
                # Load image
                img = cv2.imread(str(rgb_path))
                if img is None:
                    skipped += 1
                    continue
                
                h, w = img.shape[:2]
                
                # Try to load ground truth bbox, otherwise use full image
                bbox = load_yolo_bbox(label_path, w, h)
                if bbox is None:
                    bbox = [0, 0, w, h]  # Full image fallback
                
                # Load depth from .npy file
                depth_array = np.load(depth_path)
                
                # Extract depth within bbox
                x1, y1, x2, y2 = bbox
                depth_roi = depth_array[y1:y2, x1:x2]
                
                # Filter valid depths
                valid_depths = depth_roi[(depth_roi > 0) & (~np.isnan(depth_roi))]
                
                if len(valid_depths) == 0:
                    skipped += 1
                    continue
                
                # Calculate ground truth depth (in cm)
                gt_depth_cm = np.median(valid_depths) * 100  # meters to cm
                gt_category = categorize_depth(gt_depth_cm)
                
                if gt_category is None:
                    skipped += 1
                    continue
                
                # Run VLM inference with dummy metrics (RGB-D dataset has no geometric features)
                dummy_metrics = {
                    'area': 0,
                    'circularity': 0.0,
                    'solidity': 0.0
                }
                dummy_positional = {
                    'Distance_Category': 'Unknown'
                }

                if 'gpt' in model.lower():
                    depth_result = call_gpt5_api(
                        str(rgb_path), None, dummy_metrics, dummy_positional, False
                    )
                else:
                    depth_result = call_gemini_api(
                        str(rgb_path), None, dummy_metrics, dummy_positional, False, model_name=model
                    )

                pred_category = depth_result['depth_category']
                
                results.append({
                    'image': rgb_path.name,
                    'gt_depth_cm': float(gt_depth_cm),
                    'gt_category': gt_category,
                    'pred_category': pred_category,
                    'model': model,
                    'confidence': depth_result.get('confidence', 'unknown'),
                    'reasoning': depth_result.get('reasoning', ''),
                    'used_bbox': bbox != [0, 0, w, h]  # Track if GT bbox was used
                })
                
            except Exception as e:
                print(f"\nError processing {rgb_path.name}: {e}")
                skipped += 1
        
        # Calculate metrics for this model
        if len(results) > 0:
            gt_labels = [r['gt_category'] for r in results]
            pred_labels = [r['pred_category'] for r in results]
            accuracy = accuracy_score(gt_labels, pred_labels)
            
            print(f"\n{model}: {accuracy*100:.1f}% accuracy on {len(results)} samples")
            print(f"Skipped: {skipped} images (no depth match or invalid data)")
            
            # Classification report
            print(f"\nPer-category performance:")
            report = classification_report(
                gt_labels, pred_labels,
                labels=['low', 'medium', 'high'],
                target_names=['low', 'medium', 'high'],
                digits=3,
                zero_division=0
            )
            print(report)
            
            # Save individual model results
            output_file = OUTPUT_DIR / f'rgbd_validation_{model.replace(":", "_").replace(".", "_")}.json'
            with open(output_file, 'w') as f:
                json.dump({
                    'model': model,
                    'accuracy': accuracy,
                    'num_samples': len(results),
                    'num_skipped': skipped,
                    'results': results
                }, f, indent=2)
            
            all_results[model] = {
                'accuracy': accuracy,
                'num_samples': len(results),
                'results': results
            }
        else:
            print(f"\n{model}: NO VALID SAMPLES PROCESSED!")
            print(f"Skipped: {skipped} images")
    
    # Summary comparison
    if all_results:
        print(f"\n{'='*60}")
        print("RGB-D VALIDATION SUMMARY")
        print(f"{'='*60}")
        print("\nModel Performance Comparison:")
        for model, metrics in all_results.items():
            print(f"  {model}: {metrics['accuracy']*100:.1f}% (n={metrics['num_samples']})")
        
        # Save summary
        with open(OUTPUT_DIR / 'rgbd_validation_summary.json', 'w') as f:
            json.dump({k: {'accuracy': v['accuracy'], 'num_samples': v['num_samples']} 
                       for k, v in all_results.items()}, f, indent=2)
    else:
        print("\n⚠️ NO RESULTS FROM ANY MODEL - Check file matching!")
    
    return all_results

if __name__ == "__main__":
    print("Starting RGB-D Ground Truth Validation...")
    print(f"RGB Dir: {RGB_DIR}")
    print(f"Depth Dir: {DEPTH_DIR}")
    print(f"Labels Dir: {LABELS_DIR}")
    
    run_all_models_rgbd(
        models=["gemini-2.5-flash", "gemini-2.5-pro", "gpt-5"],
        n_samples=50
    )
    
    print("\n" + "="*60)
    print("RGB-D validation complete!")
    print("="*60)
