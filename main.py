# %%
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import json
from datetime import datetime
import sys

sys.path.append("..")

sam_checkpoint = "sam_vit_h.pth"
model_type = "vit_h"

device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize SAM model
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# Use SamPredictor instead of SamAutomaticMaskGenerator
predictor = SamPredictor(sam)


def calculate_mask_metrics(mask):
    """Calculate various metrics for a mask"""
    area = np.sum(mask)

    # Find contours to calculate perimeter and other shape metrics
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest_contour, True)

        # Calculate circularity (4π*area/perimeter²)
        circularity = (
            4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        )

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 0

        # Calculate solidity (area/convex_hull_area)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

    else:
        perimeter = 0
        circularity = 0
        aspect_ratio = 0
        solidity = 0

    return {
        "area": int(area),
        "perimeter": float(perimeter),
        "circularity": float(circularity),
        "aspect_ratio": float(aspect_ratio),
        "solidity": float(solidity),
    }


def save_mask_data(mask, mask_info, save_dir, mask_id):
    """Save mask and its associated data"""
    # Save the binary mask
    mask_path = os.path.join(save_dir, f"mask_{mask_id}.png")
    cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

    # Save the colored mask for visualization
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colored_mask[mask] = [0, 255, 0]  # Green color for pothole
    colored_mask_path = os.path.join(save_dir, f"mask_{mask_id}_colored.png")
    cv2.imwrite(colored_mask_path, colored_mask)

    # Save metadata as JSON
    metadata_path = os.path.join(save_dir, f"mask_{mask_id}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(mask_info, f, indent=2)

    return mask_path, colored_mask_path, metadata_path


def visualize_results(image, gt_bboxes, results, save_path):
    """Create a comprehensive visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # Original image with ground truth boxes
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image with Ground Truth Boxes")
    for i, bbox in enumerate(gt_bboxes):
        x1, y1, x2, y2 = bbox
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=3,
            edgecolor="red",
            facecolor="none",
        )
        axes[0, 0].add_patch(rect)
        axes[0, 0].text(
            x1, y1 - 10, f"GT_{i}", color="red", fontsize=12, fontweight="bold"
        )
        print(
            f"Drawing GT box {i}: ({x1}, {y1}) to ({x2}, {y2}), size: {x2-x1}x{y2-y1}"
        )
    axes[0, 0].axis("off")

    # Image with all generated masks overlaid
    axes[0, 1].imshow(image)
    axes[0, 1].set_title(f"Generated Masks ({len(results)} masks)")

    # Create a combined overlay for all masks
    overlay = np.zeros_like(image)
    colors = [
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
    ]

    for i, result in enumerate(results):
        mask = result["best_mask"]
        color = colors[i % len(colors)]
        overlay[mask] = color

        # Also draw the bounding box for reference
        bbox = result["gt_bbox"]
        x1, y1, x2, y2 = bbox
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor=np.array(color) / 255.0,
            facecolor="none",
        )
        axes[0, 1].add_patch(rect)
        axes[0, 1].text(
            x1, y1 - 5, f"M_{i}", color=np.array(color) / 255.0, fontsize=10
        )

        print(f"Mask {i} area: {np.sum(mask)} pixels")

    axes[0, 1].imshow(overlay, alpha=0.6)
    axes[0, 1].axis("off")

    # Individual masks
    if results:
        num_masks = len(results)
        for i, result in enumerate(results[:4]):  # Show up to 4 masks
            row = 1
            col = i % 2
            if i < 4:
                axes[row, col].imshow(result["best_mask"], cmap="gray")
                axes[row, col].set_title(
                    f"Mask {i} (Score: {result['score']:.3f})"
                )
                axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# Main processing loop
IMAGES_DIR_PATH = (
    "Dataset/Transformed_Data/Large_size (3680X1965)/Train/Train_images/"
)
IMAGES = os.listdir(IMAGES_DIR_PATH)

# Create output directory structure
output_base_dir = "data/With_Distance/sam_predictions_large/"
os.makedirs(output_base_dir, exist_ok=True)

# Summary statistics
total_processed = 0
total_successful = 0
processing_stats = []

# For testing - set to a small number, remove for production
MAX_IMAGES_TO_PROCESS = None

for image_name in IMAGES:
    if not image_name.endswith(".JPG"):
        continue

    print(f"\n{'='*50}")
    print(f"Processing: {image_name}")
    print(f"{'='*50}")

    # Load image
    image_file_path = os.path.join(IMAGES_DIR_PATH, image_name)

    if not os.path.exists(image_file_path):
        print(f"Image not found: {image_file_path}")
        continue

    image = cv2.imread(image_file_path)
    if image is None:
        print(f"Failed to read image: {image_file_path}")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Set image for predictor
    predictor.set_image(image_rgb)

    # Load ground truth labels
    IMAGE_FILE_NAME = os.path.splitext(image_name)[0]
    LABELS_FILE_PATH = os.path.join(
        "Dataset",
        "Transformed_Data",
        "Large_size (3680X1965)",
        "Train",
        "Train_bboxes",
        f"{IMAGE_FILE_NAME}.txt",
    )
    #     LABELS_FILE_PATH = os.path.join(      # Testing Path
    #     "Dataset",
    #     "Test",
    #     "bbox",
    #     f"{IMAGE_FILE_NAME}.txt",
    # )

    if not os.path.exists(LABELS_FILE_PATH):
        print(f"Labels file not found: {LABELS_FILE_PATH}")
        continue

    with open(LABELS_FILE_PATH, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print("No labels found in file")
        continue

    # Parse ground truth bounding boxes (format: x, y, w, h)
    gt_bboxes = []
    for line in lines:
        try:
            x, y, w, h = map(float, line.split())
            # Convert from x,y,w,h to x1,y1,x2,y2 format for SAM
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            gt_bboxes.append([x1, y1, x2, y2])
        except ValueError:
            print(f"Error parsing line: {line}")
            continue

    if not gt_bboxes:
        print("No valid bounding boxes found")
        continue

    print(f"Found {len(gt_bboxes)} ground truth bounding boxes")
    print(f"Image shape: {image_rgb.shape}")

    # Process each ground truth bounding box with SamPredictor
    results = []

    for i, bbox in enumerate(gt_bboxes):
        print(f"\nProcessing bounding box {i+1}/{len(gt_bboxes)}: {bbox}")

        # Validate bounding box is within image bounds
        img_h, img_w = image_rgb.shape[:2]
        x1, y1, x2, y2 = bbox

        # Clamp coordinates to image bounds
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))

        # Skip if box is too small or invalid
        if x2 <= x1 or y2 <= y1:
            print(f"Invalid bounding box: {bbox}")
            continue

        bbox = [x1, y1, x2, y2]  # Update with clamped coordinates

        # Calculating the vertical distance for large transformed images
        IMAGE_HEIGHT = 1965
        IMAGE_WIDTH = 3680

        # FAR_PERCENTAGE = 43.6
        # MEDIUM_PERCENTAGE = 38.229572
        # NEAR_PERCENTAGE = 100 - FAR_PERCENTAGE - MEDIUM_PERCENTAGE

        Distance_Category = None
        if 0 <= bbox[3] <= 449:
            Distance_Category = "Far"
        elif 450 <= bbox[3] <= 514:
            Distance_Category = "Medium Distance"
        elif 515 <= bbox[3] <= 1965:
            Distance_Category = "Near"

        # Calculating the lateral distance for large transformed images
        Lateral_Position = None
        bbox_center_x = (bbox[0] + bbox[2]) // 2
        if 1000 <= bbox_center_x <= 1839:
            Lateral_Position = "in_path_left"
        elif 1840 <= bbox_center_x <= 2700:
            Lateral_Position = "in_path_right"
        elif 0 <= bbox_center_x <= 999:
            Lateral_Position = "in_lane_far_left"
        else:
            Lateral_Position = "in_lane_far_right"

        # Save the positional feature in a dictioanry
        positional_feature = {
            "Distance_Category": Distance_Category,
            "Lateral_Position": Lateral_Position,
        }

        try:
            # Use the bounding box as a prompt for SAM
            input_box = np.array(bbox)

            # Generate masks for this specific bounding box
            masks, scores, logits = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],  # Add batch dimension
                multimask_output=True,  # Get multiple mask candidates
            )

            print(f"Generated {len(masks)} mask candidates")

            # Select the best mask (highest score)
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]
            best_score = scores[best_idx]

            print(f"Best mask score: {best_score:.3f}")

            # Calculate mask metrics
            mask_metrics = calculate_mask_metrics(best_mask)

            # Store result
            result = {
                "gt_bbox": bbox,
                "best_mask": best_mask,
                "score": float(best_score),
                "all_scores": scores.tolist(),
                "mask_metrics": mask_metrics,
                "bbox_id": i,
                "Positional_feature": positional_feature,  # Added
            }

            results.append(result)

        except Exception as e:
            print(f"Error processing bounding box {i}: {str(e)}")
            continue

    # Save results if we got any
    if results:
        total_successful += 1

        # Create image-specific directory
        image_output_dir = os.path.join(output_base_dir, IMAGE_FILE_NAME)
        os.makedirs(image_output_dir, exist_ok=True)

        # Save individual masks and metadata
        for i, result in enumerate(results):
            mask_info = {
                "image_name": image_name,
                "image_shape": image_rgb.shape,
                "gt_bbox": result["gt_bbox"],
                "sam_score": result["score"],
                "all_sam_scores": result["all_scores"],
                "mask_metrics": result["mask_metrics"],
                "Positional_feature": positional_feature,  # Added
                "processing_timestamp": datetime.now().isoformat(),
            }

            mask_path, colored_path, metadata_path = save_mask_data(
                result["best_mask"], mask_info, image_output_dir, i
            )

            print(f"Saved mask {i}: {mask_path}")

        # Create visualization
        viz_path = os.path.join(image_output_dir, "visualization.png")
        visualize_results(image_rgb, gt_bboxes, results, viz_path)

        # Save summary for this image
        summary = {
            "image_name": image_name,
            "total_gt_boxes": len(gt_bboxes),
            "successful_predictions": len(results),
            "average_score": np.mean([r["score"] for r in results]),
            "results": [
                {
                    "bbox_id": r["bbox_id"],
                    "gt_bbox": r["gt_bbox"],
                    "sam_score": r["score"],
                    "mask_metrics": r["mask_metrics"],
                    "Positional_feature": r["Positional_feature"],  # Added
                }
                for r in results
            ],
        }

        summary_path = os.path.join(image_output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        processing_stats.append(summary)

        print(
            f"\nSuccessfully processed {len(results)} masks for {image_name}"
        )
        print(f"Average SAM score: {summary['average_score']:.3f}")

    total_processed += 1

    # For testing - limit number of images
    if (
        MAX_IMAGES_TO_PROCESS is not None
        and total_processed >= MAX_IMAGES_TO_PROCESS
    ):
        break

# Save overall statistics
overall_stats = {
    "total_images_processed": total_processed,
    "successful_images": total_successful,
    "success_rate": (
        total_successful / total_processed if total_processed > 0 else 0
    ),
    "processing_timestamp": datetime.now().isoformat(),
    "per_image_stats": processing_stats,
}

stats_path = os.path.join(output_base_dir, "overall_statistics.json")
with open(stats_path, "w") as f:
    json.dump(overall_stats, f, indent=2)

print(f"\n{'='*60}")
print("PROCESSING COMPLETE")
print(f"{'='*60}")
print(f"Total images processed: {total_processed}")
print(f"Successful images: {total_successful}")
print(f"Success rate: {overall_stats['success_rate']:.2%}")
print(f"Results saved to: {output_base_dir}")

# %%
