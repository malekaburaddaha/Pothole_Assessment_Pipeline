
#%%
import os
import cv2

#%%
bbox_folder = "abs_gt"  # Folder containing text files with bounding boxes
images_folder = "Detected_images"  # Folder containing images
output_folder = "gt_vs_dt"

#%% Iterate over the text files in the bbox folder
for filename in os.listdir(bbox_folder):
    if filename.endswith(".txt"):  # Check for text files
        image_path = os.path.join(images_folder, filename.replace(".txt", ".JPG"))  # Assume .jpg images
        bbox_path = os.path.join(bbox_folder, filename)

        if os.path.exists(image_path):  # Check if the image exists
            image = cv2.imread(image_path)

            with open(bbox_path, 'r') as file:
                for line in file:
                    values = line.split()
                    x = float(values[0])
                    y = float(values[1])
                    w = float(values[2])
                    h = float(values[3])

                    # Convert center coordinates to top-left and bottom-right corner coordinates to create a rectangle
                    x1 = int(x - w / 2)
                    y1 = int(y - h / 2)
                    x2 = int(x + w / 2)
                    y2 = int(y + h / 2)
                    
                    # Extract the bottom right, and we already have the top left coords
                    # x1 = int(x)
                    # y1 = int(y)
                    # x2 = int(x + w)
                    # y2 = int(y + h)

                    # Draw the bounding box
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Save the image with bounding boxes to the output folder
            output_path = os.path.join(output_folder, filename.replace(".txt", ".JPG"))
            cv2.imwrite(output_path, image)

# %%




















"""Plot for transformed images"""


import os
import cv2

#%%
bbox_folder = "abs_gt"  # Folder containing text files with bounding boxes
images_folder = "Detected_images"  # Folder containing images
output_folder = "gt_vs_dt"

#%% Iterate over the text files in the bbox folder
for filename in os.listdir(bbox_folder):
    if filename.endswith(".txt"):  # Check for text files
        image_path = os.path.join(images_folder, filename.replace(".txt", ".JPG"))  # Assume .jpg images
        bbox_path = os.path.join(bbox_folder, filename)

        if os.path.exists(image_path):  # Check if the image exists
            image = cv2.imread(image_path)

            with open(bbox_path, 'r') as file:
                for line in file:
                    values = line.split()
                    x = float(values[0]) # Start from 1 if you are using bboxes with classes values included
                    y = float(values[1])
                    w = float(values[2])
                    h = float(values[3])

                    # Convert from center coordinates to top-left and bottom-right corner coordinates to create a rectangle
                    x1 = int(x - w / 2)
                    y1 = int(y - h / 2)
                    x2 = int(x + w / 2)
                    y2 = int(y + h / 2)
                    
                    # # Extract the bottom right, if we already have the top left coords
                    # x1 = int(x)
                    # y1 = int(y)
                    # x2 = int(x + w)
                    # y2 = int(y + h)

                    # Draw the bounding box
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            

            # Save the image with bounding boxes to the output folder
            output_path = os.path.join(output_folder, filename.replace(".txt", ".JPG"))
            cv2.imwrite(output_path, image)

# %%


"""This code is to resize any image to a 800x800 image"""

from PIL import Image

# Open an image file
with Image.open('no_anotation_fig2.jpg') as img:
    # Resize image to 800x800
    resized_img = img.resize((800, 800))

    # Save the resized image
    resized_img.save('Transformed_resized.png')

# %%
