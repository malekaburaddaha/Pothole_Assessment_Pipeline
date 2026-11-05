#%%
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import cv2

#%%
# Load an image
# image = cv2.imread('F:/Malek/code/Three_classes/Transf_dataset/Train data/positive/G0011474.JPG') 
# # Convert to RGB and read the image
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# # plot the image
# plt.imshow(image) 
# print(image.shape)
#%%
"""Modify the classes values based on the y coordiante into three classes (Near:0, Medium:1, Far:2)"""


def modify_class_values(file_path):
    # First open the file and read each line
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # Write the new class values
    with open(file_path, 'w') as file:
        #Iterate over each line in the file
        for line in lines:
            values = line.split()
            
            # This code is to add 0 as a class value at the 0 index for each line in the text file
            values.insert(0, 0)
            
            # Get the y coordinate value
            y_value = float(values[2])
            # Create a new class variable
            new_class = 0 # This is Near

            # This is for images that are only bottom cropped or image as is
            # if 1196 <= y_value <= 1350:
            #     new_class = 2 # Far
            # elif 1351 <= y_value <= 1500:
            #     new_class = 1 # Medium
            # elif 1501 <= y_value <= 1963:
            #     new_class = 0 # Near
            
            # This is for images and bboxes that are double cropped
            # if 4.0 <= y_value <= 235.0:
            #     new_class = 2 # Far
            # elif 236.0 <= y_value <= 370.0:
            #     new_class = 1 # Medium
            # elif 371.0 <= y_value <= 771.0:
            #     new_class = 0 # Near

            # Modify the class label value
            values[0] = new_class
            # This is for the detectd labels
            file.write(f"{values[0]} {values[1]} {values[2]} {values[3]} {values[4]} {values[5]}\n")
            
            # This is for the ground truth labels
            # file.write(f"{values[0]} {values[1]} {values[2]} {values[3]} {values[4]}\n")

#%%
"""Processes all text files in a folder"""            
def process_all_files(folder_path):
    # Iterate over all the files in the path
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'): # Ensure that only text files are proceessed
            file_path = os.path.join(folder_path, filename)
            modify_class_values(file_path) # Modify class values
            
#%%
bboxes_path = 'abs_dt'

process_all_files(bboxes_path)
# %%

"""count the number of each class before cleaning:
    Near: some number #4425
    Medium: some number #2126
    Far: some number. #784
    """




"""Plot all bounding boxes on the images showing all the new class labels"""

# Dictionary to map integer class labels to word labels
class_labels = {0: 'Near', 1: 'Medium', 2: 'Far'}

# First define a function to plot a bounding boxes with class labels on an image
def plot_bounding_boxes(image_path, bounding_box_path):
    image = cv2.imread(image_path)  # Reading the image
    font = cv2.FONT_HERSHEY_SIMPLEX  # Setting the font used for plotting the text

    class_counts = {'Near': 0, 'Medium': 0, 'Far': 0}  # Create a dictionary to count the classes

    with open(bounding_box_path, 'r') as file:  # Open the bbox and read its lines
        lines = file.readlines()

        for line in lines:
            values = line.split()
            class_label = int(values[0])  # Read class label as an integer
            class_label_word = class_labels[class_label]  # Map to word label
            class_counts[class_label_word] += 1  # Increment the count for the word label
            # Extracting coordinates
            x = int(float(values[1]))
            y = int(float(values[2]))
            w = int(float(values[3]))
            h = int(float(values[4]))
            # Count the extracted class value
            class_label = str(values[0])  # Ensure class label is a string
            # class_counts[class_label] += 1
            # print("class counts so far: ", class_counts)
            # print('\n')


            # prepare for plotting by choosing the top left and bottom right for bboxes
            # Top left
            x1 = x
            y1 = y
            # Bottom right
            x2 = x + w 
            y2 = y + h

            # Drawing the bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green bbox

            # plotting the text of bboxes
            text = f"{'Near Pothole.' if class_label == '0' else 'Medium Pothole.' if class_label == '1' else 'Far Pothole.'}"
            cv2.putText(image, text, (x1, y1 - 5), font, 0.55, (0, 0, 255), 1)

    return image, class_counts





#%%
"""Plot all images and bounding boxes in their respective folders"""

def process_all_images(images_folder, bounding_boxes_folder, output_folder):
    # Create the output folder if it does not exist
    # os.makedirs(output_folder, exist_ok= True)

    # Get the names of the bbox files to match the images filename
    bbox_filenames = [f for f in os.listdir(bounding_boxes_folder) if f.endswith('.txt')]
    # print("bboxes names we have form the second function: ",bbox_filenames)
    # print('\n')
    total_class_counts = {'Near': 0, 'Medium': 0, 'Far': 0}  # Initialize for accumulation

    # iterate over the images with JPG extension
    for filename in os.listdir(images_folder):
        # Only JPEG images with matching bounding boxes
        if filename.endswith('.JPG') and filename.replace('.JPG', '.txt') in bbox_filenames: 
            # print("The filename in the second: ", filename)
            # print('\n')

            # Join the image and images folder path
            image_path = os.path.join(images_folder, filename)
            # Do the same for bboxes with the same image name.
            bounding_boxes_path = os.path.join(bounding_boxes_folder, filename.replace('.JPG', '.txt'))

            # Apply the plot function on the image and bbox
            plotted_image, class_counts = plot_bounding_boxes(image_path, bounding_boxes_path)

            # Where to save the new images
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, plotted_image)

        # Update total class counts
        total_class_counts['Near'] += class_counts['Near']
        total_class_counts['Medium'] += class_counts['Medium']
        total_class_counts['Far'] += class_counts['Far']

    
    # Print final class counts
    print("Near:", total_class_counts['Near'])
    print("Medium:", total_class_counts['Medium'])
    print("Far:", total_class_counts['Far'])
        

#%%
# Plot the images with their classes and print their counts
images_folder = 'F:/Malek/code/Three_classes/Double_cropped/Double_Cropped_data/test'
bounding_boxes_folder = 'F:/Malek/code/Three_classes/Double_cropped/Double_Cropped_data/Test_Classes_labels'      
output_folder = 'F:/Malek/code/Three_classes/Double_cropped/Double_Cropped_data/test_plots'

process_all_images(images_folder, bounding_boxes_folder, output_folder)
# %%







###############################################################################################
###############################################################################################











"""This code is for transfomed images"""

#%%
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import cv2

#%%
# # Load an image
# image = cv2.imread('G0011876.JPG') 
# # Convert to RGB and read the image
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# # plot the image
# plt.imshow(image) 
# print(image.shape)
#%%
"""Modify the classes values based on the y coordiante into three classes (Near:0, Medium:1, Far:2)"""

def modify_class_values(file_path):
    # Initialize counters for each class
    near_count = 0
    medium_count = 0
    far_count = 0

    # First open the file and read each line
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # Write the new class values
    with open(file_path, 'w') as file:
        #Iterate over each line in the file
        for line in lines:
            values = line.split()
            
            # This code is to add 0 as a class value at the 0 index for each line in the text file
            values.insert(0, 0)
            
            # Get the y coordinate value
            y_value = float(values[2])
            # Create a new class variable
            new_class = 0 # This is Near
            
            # # For auto transformed bboxes
            if 0 <= y_value <= 285:
                new_class = 2 # Far
            elif 286 <= y_value <= 430:
                new_class = 1 # Medium
            elif 431 <= y_value <= 800:
                new_class = 0 # Near
            
            # For the new bigger auto transformed bboxes (gt), 0-670, 671-1099, 1100-1965 official values
            # if 0 <= y_value <= 670:
            #     new_class = 2 # Far
            #     far_count += 1
            # elif 671 <= y_value <= 1099:
            #     new_class = 1 # Medium
            #     medium_count += 1
            # elif 1100 <= y_value <= 1965:
            #     new_class = 0 # Near
            #     near_count += 1
            
            # This is for images and bboxes that are transformed
            # if 0 <= y_value <= 350:
            #     new_class = 2 # Far
            # elif 351 <= y_value <= 550:
            #     new_class = 1 # Medium
            # elif 551 <= y_value <= 800:
            #     new_class = 0 # Near

            # Value[5] is for the confidence value for the detected bboxes
            # Modify the class label value
            values[0] = new_class
            # for the predicted labels
            file.write(f"{values[0]} {values[1]} {values[2]} {values[3]} {values[4]} {values[5]}\n")
            
            # For the ground truth folder
            # file.write(f"{values[0]} {values[1]} {values[2]} {values[3]} {values[4]}\n")

        return near_count, medium_count, far_count


#%%
"""Processes all text files in a folder"""            
def process_all_files(folder_path):
    total_near_count = 0
    total_medium_count = 0
    total_far_count = 0

    # Iterate over all the files in the path
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'): # Ensure that only text files are processed
            file_path = os.path.join(folder_path, filename)
            near_count, medium_count, far_count = modify_class_values(file_path)
            total_near_count += near_count
            total_medium_count += medium_count
            total_far_count += far_count

    # Print the total counts of each class
    print(f"Total Near count: {total_near_count}")
    print(f"Total Medium count: {total_medium_count}")
    print(f"Total Far count: {total_far_count}")

# Apply the function
bboxes_path = 'abs_dt'
process_all_files(bboxes_path)
# %%
 





"""Same as the above code for transformed datasets, and it counts the number of classes"""

# import os

# def modify_class_values(file_path):
#     # Initialize counters for each class
#     near_count = 0
#     medium_count = 0
#     far_count = 0

#     # First open the file and read each line
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
    
#     # Write the new class values
#     with open(file_path, 'w') as file:
#         # Iterate over each line in the file
#         for line in lines:
#             values = line.split()
            
#             # This code is to add 0 as a class value at the 0 index for each line in the text file
#             values.insert(0, 0)
            
#             # Get the y coordinate value
#             y_value = float(values[2])
#             # Create a new class variable
#             new_class = 0 # This is Near
            
#             # For the new bigger auto transformed bboxes (gt)
#             if 0 <= y_value <= 650:
#                 new_class = 2 # Far
#                 far_count += 1
#             elif 651 <= y_value <= 1151:
#                 new_class = 1 # Medium
#                 medium_count += 1
#             elif 1152 <= y_value <= 1965:
#                 new_class = 0 # Near
#                 near_count += 1
            
#             # Modify the class label value
#             values[0] = new_class
            
#             # For the ground truth folder
#             file.write(f"{values[0]} {values[1]} {values[2]} {values[3]} {values[4]}\n")
    
#     return near_count, medium_count, far_count

# """Processes all text files in a folder"""            
# def process_all_files(folder_path):
#     total_near_count = 0
#     total_medium_count = 0
#     total_far_count = 0

#     # Iterate over all the files in the path
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.txt'): # Ensure that only text files are processed
#             file_path = os.path.join(folder_path, filename)
#             near_count, medium_count, far_count = modify_class_values(file_path)
#             total_near_count += near_count
#             total_medium_count += medium_count
#             total_far_count += far_count

#     # Print the total counts of each class
#     print(f"Total Near count: {total_near_count}")
#     print(f"Total Medium count: {total_medium_count}")
#     print(f"Total Far count: {total_far_count}")

# # Apply the function
# bboxes_path = 'abs_gt'
# process_all_files(bboxes_path)




















"""count the number of each class before cleaning:
    Near: some number #4425
    Medium: some number #2126
    Far: some number. #784
    """




"""Plot all bounding boxes on the images showing all the new class labels"""

# Dictionary to map integer class labels to word labels
class_labels = {0: 'Near', 1: 'Medium', 2: 'Far'}

# First define a function to plot a bounding boxes with class labels on an image
def plot_bounding_boxes(image_path, bounding_box_path):
    image = cv2.imread(image_path)  # Reading the image
    font = cv2.FONT_HERSHEY_SIMPLEX  # Setting the font used for plotting the text

    class_counts = {'Near': 0, 'Medium': 0, 'Far': 0}  # Create a dictionary to count the classes

    with open(bounding_box_path, 'r') as file:  # Open the bbox and read its lines
        lines = file.readlines()

        for line in lines:
            values = line.split()
            class_label = int(values[0])  # Read class label as an integer
            class_label_word = class_labels[class_label]  # Map to word label
            class_counts[class_label_word] += 1  # Increment the count for the word label
            # Extracting coordinates
            x = int(float(values[1]))
            y = int(float(values[2]))
            w = int(float(values[3]))
            h = int(float(values[4]))
            # Count the extracted class value
            class_label = str(values[0])  # Ensure class label is a string
            # class_counts[class_label] += 1
            # print("class counts so far: ", class_counts)
            # print('\n')


            # prepare for plotting by choosing the top left and bottom right for bboxes
            # Top left
            x1 = x
            y1 = y
            # Bottom right
            x2 = x + w 
            y2 = y + h

            # Drawing the bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green bbox

            # plotting the text of bboxes
            text = f"{'Near Pothole.' if class_label == '0' else 'Medium Pothole.' if class_label == '1' else 'Far Pothole.'}"
            cv2.putText(image, text, (x1, y1 - 5), font, 0.55, (0, 0, 255), 1)

    return image, class_counts

#%%
"""Plot all images and bounding boxes in their respective folders"""

def process_all_images(images_folder, bounding_boxes_folder, output_folder):
    # Create the output folder if it does not exist
    # os.makedirs(output_folder, exist_ok= True)

    # Get the names of the bbox files to match the images filename
    bbox_filenames = [f for f in os.listdir(bounding_boxes_folder) if f.endswith('.txt')]
    # print("bboxes names we have form the second function: ",bbox_filenames)
    # print('\n')
    total_class_counts = {'Near': 0, 'Medium': 0, 'Far': 0}  # Initialize for accumulation

    # iterate over the images with JPG extension
    for filename in os.listdir(images_folder):
        # Only JPEG images with matching bounding boxes
        if filename.endswith('.JPG') and filename.replace('.JPG', '.txt') in bbox_filenames: 
            # print("The filename in the second: ", filename)
            # print('\n')

            # Join the image and images folder path
            image_path = os.path.join(images_folder, filename)
            # Do the same for bboxes with the same image name.
            bounding_boxes_path = os.path.join(bounding_boxes_folder, filename.replace('.JPG', '.txt'))

            # Apply the plot function on the image and bbox
            plotted_image, class_counts = plot_bounding_boxes(image_path, bounding_boxes_path)

            # Where to save the new images
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, plotted_image)

        # Update total class counts
        total_class_counts['Near'] += class_counts['Near']
        total_class_counts['Medium'] += class_counts['Medium']
        total_class_counts['Far'] += class_counts['Far']

    
    # Print final class counts
    print("Near:", total_class_counts['Near'])
    print("Medium:", total_class_counts['Medium'])
    print("Far:", total_class_counts['Far'])
        

#%%
# Plot the images with their classes and print their counts
images_folder = 'test'
bounding_boxes_folder = bboxes_path    
output_folder = 'classes_plot_2'

process_all_images(images_folder, bounding_boxes_folder, output_folder)
