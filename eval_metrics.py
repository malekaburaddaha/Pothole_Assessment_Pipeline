 
#%%
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% 
#install pycocotools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
#%% The input directory 
input_dir = "./"


#%% 
"""Detection labels json file"""
# Read the contents of "new pred.json" as a string
with open(os.path.join(input_dir, 'dt_labels.json'), 'r') as file:
    json_str = file.read()

# Load the JSON data from the string using json.loads
data = json.loads(json_str)

# Get the list of annotations from the "annotations" key in the dictionary
anns = data["annotations"]

# Check the type and content of anns (print the first 5 annotations)
print(type(anns))
print(anns[:5])  # Print the first 5 annotations for debugging purposes

#%%
"""Ground truth labels json file"""
# Now you can proceed with creating the COCO instance
# resfile = 'new pred.json' 

coco_gt = COCO(os.path.join(input_dir, 'gt_labels.json'))
coco_dt = coco_gt.loadRes(anns)


# %%
imgIds = coco_gt.getImgIds()
imgIds

# %%
cocoEval = COCOeval(coco_gt,coco_dt,'bbox')
# Modify the threshold range
# iou_threshold = [0.25, 0.3]
# cocoEval.params.iouThrs = iou_threshold

cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
# %%


# A dictionary to store the results 
results_dict = {
    "AP@[0.50:0.95]": cocoEval.stats[0],
    "AP@0.50": cocoEval.stats[1],
    "AP@0.75": cocoEval.stats[2],
    "AP@[0.50:0.95] Small": cocoEval.stats[3],
    "AP@[0.50:0.95] Medium": cocoEval.stats[4],
    "AP@[0.50:0.95] Large": cocoEval.stats[5],
    "AR@[0.50:0.95]": cocoEval.stats[6],
    "AR@0.50": cocoEval.stats[7],
    "AR@0.75": cocoEval.stats[8],
    "AR@[0.50:0.95] Small": cocoEval.stats[9],
    "AR@[0.50:0.95] Medium": cocoEval.stats[10],
    "AR@[0.50:0.95] Large": cocoEval.stats[11],    
}
  
print(results_dict)    
#%% Create a DataFrame from the dictionary
df = pd.DataFrame(results_dict, index = ['Values'])

# Save the DataFrame to an Excel sheet
df.to_excel("coco_eval_results_Bottom_cropped_0.2_800.xlsx", index=True)


























#%%
"""Another modified code for the coco eval"""
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import json

# Load ground truth annotations (new labels.json)
input_dir = "./"  
coco_gt = COCO(os.path.join(input_dir, 'new labels.json'))

# Load predicted annotations (new pred.json)
# with open(os.path.join(input_dir, 'fake pred test.json'), 'r') as file:
#     json_str = file.read()
# anns_pred = json.loads(json_str)

# anns = anns_pred['annotations']
anns = 'fake pred test.json'
#%% Load predictions into COCO format
coco_dt = coco_gt.loadRes(anns)

# Get image IDs
img_ids = coco_gt.getImgIds()
img_ids

#%% Initialize COCO evaluation object
cocoEval = COCOeval(coco_gt, coco_dt, 'bbox')
iou_threshold = [0.30, 0.95]
cocoEval.params.iouThrs = iou_threshold

# Set image IDs for evaluation
cocoEval.params.imgIds = img_ids

# Run evaluation
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

#%% Get evaluation metrics
ap_values = cocoEval.stats[:12]  # AP for different IoU thresholds
ar_values = cocoEval.stats[12:25]  # AR for different IoU thresholds
# mAP = cocoEval.stats[23]  # Mean Average Precision

# Print evaluation results
print("Average Precision (AP) for IoU thresholds:")
print(ap_values)
print("\nAverage Recall (AR) for IoU thresholds:")
print(ar_values)
# print("\nMean Average Precision (mAP):", mAP)

# %%







"""Evaluation for 3 classes"""


#%%
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% 
#install pycocotools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
#%% The input directory 
input_dir = "./"


#%% 
"""Detection labels json file"""
# Read the contents of "new pred.json" as a string
with open(os.path.join(input_dir, 'dt_labels.json'), 'r') as file:
    json_str = file.read()

# Load the JSON data from the string using json.loads
data = json.loads(json_str)

# Get the list of annotations from the "annotations" key in the dictionary
anns = data["annotations"]

# Check the type and content of anns (print the first 5 annotations)
print(type(anns))
print(anns[:5])  # Print the first 5 annotations for debugging purposes

#%%
"""Ground truth labels json file"""
# Now you can proceed with creating the COCO instance
# resfile = 'new pred.json' 

coco_gt = COCO(os.path.join(input_dir, 'gt_labels.json'))
coco_dt = coco_gt.loadRes(anns)


# %%
imgIds = coco_gt.getImgIds()
imgIds

# %%
cocoEval = COCOeval(coco_gt,coco_dt,'bbox')

#%% Get the list of category IDs from the ground truth annotations
catIds = coco_gt.getCatIds()
catIds
#%% Create a dictionary to store the results for each class
results_per_class = {}

#%% Iterate over each category
for catId in catIds:
    # Set the category IDs to evaluate
    cocoEval.params.catIds = [catId]
    
    
    # Evaluate
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    # Store the results for the current category
    results_per_class[catId] = {
        "AP@[0.50:0.95]": cocoEval.stats[0],
        "AP@0.50": cocoEval.stats[1],
        "AP@0.75": cocoEval.stats[2],
        "AP@[0.50:0.95] Small": cocoEval.stats[3],
        "AP@[0.50:0.95] Medium": cocoEval.stats[4],
        "AP@[0.50:0.95] Large": cocoEval.stats[5],
        "AR@[0.50:0.95]": cocoEval.stats[6],
        "AR@0.50": cocoEval.stats[7],
        "AR@0.75": cocoEval.stats[8],
        "AR@[0.50:0.95] Small": cocoEval.stats[9],
        "AR@[0.50:0.95] Medium": cocoEval.stats[10],
        "AR@[0.50:0.95] Large": cocoEval.stats[11]
    }
    
#%%


cocoEval.params.imgIds = imgIds
# Evaluate
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

cocoEval.stats
print(cocoEval.stats)

# %%Print the results for each class
for catId, catResults in results_per_class.items():
    print(f"Results for category {catId}:")
    print(catResults)    

# %%Convert the results_per_class dictionary to a DataFrame
df_results = pd.DataFrame(results_per_class).T

# Save the DataFrame to an Excel sheet
df_results.to_excel("Image_as_is_eval_small_3classes_800.xlsx")

# %%




























































#%%
"""Testing"""


#%%
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

#%% The input directory 
input_dir = "./"

#%% Detection labels json file
# Read the contents of dt_labels.json
with open(os.path.join(input_dir, 'dt_labels.json'), 'r') as file:
    dt_data = json.load(file)

# Get the list of annotations from the "annotations" key in the dictionary
dt_anns = dt_data["annotations"]

# Check the type and content of dt_anns (print the first 5 annotations)
print(type(dt_anns))
print(dt_anns[:5])  # Print the first 5 annotations for debugging purposes

#%% Ground truth labels json file
coco_gt = COCO(os.path.join(input_dir, 'gt_labels.json'))

# Verify if the detected annotations are correctly matched with ground truth
image_ids_gt = set(img['id'] for img in coco_gt.dataset['images'])
image_ids_dt = set(ann['image_id'] for ann in dt_anns)

print(f"Number of images in gt_labels: {len(image_ids_gt)}")
print(f"Number of images in dt_labels: {len(image_ids_dt)}")
print(f"Common image IDs: {len(image_ids_gt.intersection(image_ids_dt))}")

# Check if any image IDs are missing in dt_labels
missing_image_ids = image_ids_gt - image_ids_dt
print(f"Missing image IDs in dt_labels: {missing_image_ids}")

# Load detection results into COCO format
coco_dt = coco_gt.loadRes(dt_anns)

# %% Get image IDs from ground truth
imgIds = coco_gt.getImgIds()

# %% Initialize COCOeval object
cocoEval = COCOeval(coco_gt, coco_dt, 'bbox')

# %% Evaluate
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

# Print the evaluation stats
print("Evaluation stats:")
print(cocoEval.stats)

# %% Results per category (optional, commented out)
# results_per_class = {}
# for catId in catIds:
#     cocoEval.params.catIds = [catId]
#     cocoEval.evaluate()
#     cocoEval.accumulate()
#     cocoEval.summarize()
#     results_per_class[catId] = {
#         "AP@[0.50:0.95]": cocoEval.stats[0],
#         "AP@0.50": cocoEval.stats[1],
#         "AP@0.75": cocoEval.stats[2],
#         "AP@[0.50:0.95] Small": cocoEval.stats[3],
#         "AP@[0.50:0.95] Medium": cocoEval.stats[4],
#         "AP@[0.50:0.95] Large": cocoEval.stats[5],
#         "AR@[0.50:0.95]": cocoEval.stats[6],
#         "AR@0.50": cocoEval.stats[7],
#         "AR@0.75": cocoEval.stats[8],
#         "AR@[0.50:0.95] Small": cocoEval.stats[9],
#         "AR@[0.50:0.95] Medium": cocoEval.stats[10],
#         "AR@[0.50:0.95] Large": cocoEval.stats[11]
#     }

# %% Print the results for each class (optional, commented out)
# for catId, catResults in results_per_class.items():
#     print(f"Results for category {catId}:")
#     print(catResults)

# %% Convert the results_per_class dictionary to a DataFrame (optional, commented out)
# df_results = pd.DataFrame(results_per_class).T
# Save the DataFrame to an Excel sheet (optional, commented out)
# df_results.to_excel("coco_eval_results_per_class.xlsx")
