import cv2
import json
import os
import shutil
from tqdm import tqdm


IMAGES_DIR_PATH = "/data/Malek/Pothole_Motion_Project/Dataset/Transformed_Data/Large_size (3680X1965)/Train/Train_images"
JSON_PATH = "/data/Malek/Pothole_Motion_Project/data/With_Distance/sam_predictions_large/overall_statistics.json"
EXTRACT_ROOT_PATH = "/data/Malek/Pothole_Motion_Project/data/image_batches"
with open(JSON_PATH) as f:
    data = json.load(f)

for i in tqdm(range(len(data["per_image_stats"]))):
    image_data = data["per_image_stats"][i]
    IMAGE_NAME = image_data["image_name"]
    POTHOLES_COUNT = image_data["successful_predictions"]
    RESULTS = image_data["results"]

    IMG_OUTPUT_DIR = os.path.join(EXTRACT_ROOT_PATH, IMAGE_NAME.split(".")[0])
    os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)
    for pothole in range(POTHOLES_COUNT):
        pothole_coords_xyxy = RESULTS[pothole]["gt_bbox"]
        x1, y1, x2, y2 = pothole_coords_xyxy
        img_path = os.path.join(IMAGES_DIR_PATH, IMAGE_NAME)
        img = cv2.imread(img_path)
        pothole_mask = img[y1:y2, x1:x2]
        cv2.imwrite(
            os.path.join(
                IMG_OUTPUT_DIR, f"{IMAGE_NAME.split('.')[0]}_mask{pothole}.jpg"
            ),
            pothole_mask,
        )
        shutil.copy(img_path, IMG_OUTPUT_DIR)
