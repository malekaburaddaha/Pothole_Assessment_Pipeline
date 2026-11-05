#%%
import cv2
import matplotlib.pyplot as plt
import os

# Test with one image to see the bounding boxes
image_path = "Dataset/Transformed_Data/Large_size (3680X1965)/Train/Train_images/G0010117.JPG"
bbox_path = "Dataset/Transformed_Data/Large_size (3680X1965)/Train/Train_bboxes/G0010117.txt"

# Load image
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(f"Image shape: {image_rgb.shape}")

# Load bounding boxes
with open(bbox_path, 'r') as f:
    lines = [line.strip() for line in f if line.strip()]

print(f"Bounding box lines: {lines}")

# Parse and visualize
plt.figure(figsize=(15, 10))
plt.imshow(image_rgb)
plt.title("Test: Ground Truth Bounding Boxes")

for i, line in enumerate(lines):
    x, y, w, h = map(float, line.split())
    print(f"Box {i}: x={x}, y={y}, w={w}, h={h}")
    
    # Draw rectangle (x, y, w, h format)
    rect = plt.Rectangle((x, y), w, h, linewidth=3, edgecolor='red', facecolor='none')
    plt.gca().add_patch(rect)
    plt.text(x, y-10, f'Box_{i}', color='red', fontsize=12, fontweight='bold')

plt.axis('off')
plt.tight_layout()
plt.savefig("bbox_test.png", dpi=150, bbox_inches='tight')
plt.show()

print("Test visualization saved as bbox_test.png") 
# %%
