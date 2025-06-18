import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
from copy import deepcopy


def get_iou(box_a, box_b, eps=0.01):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    width = x2 - x1
    height = y2 - y1
    if width <= 0.0 or height <= 0.0:
        return 0.0

    def _get_box_area(box):
        area = (box[2] - box[0]) * (box[3] - box[1])
        return area

    box_a_area = _get_box_area(box_a)
    box_b_area = _get_box_area(box_b)

    intersection_area = width * height

    combined_area = box_a_area + box_b_area - intersection_area

    iou = intersection_area / (combined_area + eps)
    return iou


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


IMAGE_FILE_PATH = (
    # "data/Transformed_Data/Large_size (3680X1965)/Train/Train_images/G0041416.JPG" # correct
    # "data/Transformed_Data/Large_size (3680X1965)/Train/Train_images/G0067078.JPG" # 3 missed 1 correct
    # "data/Transformed_Data/Large_size (3680X1965)/Train/Train_images/G0067001.JPG" # correct
    "data/Transformed_Data/Large_size (3680X1965)/Train/Train_images/G0011491.JPG"
)
assert os.path.exists(IMAGE_FILE_PATH)
image = cv2.imread(IMAGE_FILE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

IMAGE_FILE_NAME = os.path.split(IMAGE_FILE_PATH)[-1].split(".")[0]
LABELS_FILE_PATH = os.path.join(
    "data",
    "Transformed_Data",
    "Large_size (3680X1965)",
    "Train",
    "Train_bboxes",
    f"{IMAGE_FILE_NAME}.txt",
)
print(LABELS_FILE_PATH)
with open(LABELS_FILE_PATH) as f:
    lines = [line.rstrip() for line in f]

print(lines)

print(image.shape)
image_to_ann = image.copy()

gt_bboxes = []
for line in lines:
    x1, y1, w, h = line.split(" ")
    x2 = eval(x1) + eval(w)
    y2 = eval(y1) + eval(h)
    x1, y1, x2, y2 = int(eval(x1)), int(eval(y1)), int(x2), int(y2)
    cv2.rectangle(
        image_to_ann,
        (x1, y1),
        (x2, y2),
        color=(255, 0, 0),
        thickness=3,
    )
    gt_bboxes.append([x1, y1, x2, y2])
# plt.figure(figsize=(20, 20))
# plt.imshow(image_to_ann)
# plt.axis("off")
# plt.show()

potholes_images = []

import sys

sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "sam_vit_h.pth"
model_type = "vit_h"

device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

masks = mask_generator.generate(image)

seg_bboxes = [mask["bbox"] for mask in masks]
seg_bboxes = [
    [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]] for bbox in seg_bboxes
]

print(len(masks))
print(masks[0].keys())

# plt.figure(figsize=(20, 20))
# plt.imshow(image)
# show_anns(masks)
# plt.axis("off")
# plt.show()


def check_which_seg_bbox_match_gt_bbox(seg_bbox, gt_bbox):
    ious = []
    iou = get_iou(seg_bbox, gt_bbox)
    ious.append(iou)

    if iou > 0.5:
        print(max(ious))
        return True
    # print(max(ious))
    return False


masks_to_be_extracted = len(gt_bboxes)
potholes_masks = []
should_save = False
for gt_bbox in gt_bboxes:
    for seg_box in seg_bboxes:
        if not check_which_seg_bbox_match_gt_bbox(seg_box, gt_bbox):
            continue
        print("Found a match!")
        masks_to_be_extracted -= 1
        pothole_mask = masks[seg_bboxes.index(seg_box)]["segmentation"]
        pothole_mask = pothole_mask.astype(np.uint8) * 255
        pothole_mask = cv2.cvtColor(pothole_mask, cv2.COLOR_GRAY2RGB)
        potholes_masks.append(pothole_mask)
        # plt.figure(figsize=(20, 20))
        # plt.imshow(pothole_mask)
        # plt.axis("off")
        # plt.show()
        if masks_to_be_extracted == 0:
            should_save = True
            break
print(should_save)

if should_save:
    # dir_path = f"data/potholes_masks/{IMAGE_FILE_NAME}"
    dir_path = os.path.join("data", "potholes_masks", IMAGE_FILE_NAME)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    for i, mask in enumerate(potholes_masks):
        cv2.imwrite(os.path.join(dir_path, f"mask_{i}.png"), mask)


img = image.copy()
for i, seg_box in enumerate(seg_bboxes):
    cv2.rectangle(
        img,
        (int(seg_box[0]), int(seg_box[1])),
        (int(seg_box[2]), int(seg_box[3])),
        color=(0, 255, 0),
        thickness=3,
    )
# plt.figure(figsize=(20, 20))
# plt.imshow(img)
# plt.axis("off")
