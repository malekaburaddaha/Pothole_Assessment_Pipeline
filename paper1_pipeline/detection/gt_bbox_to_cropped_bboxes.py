#%% Libraries
import numpy
import os

#%%


""" Crop bboxes by subtracting 1192 from the y_value of the center coordinates or the top left coords
"""

def process_file(input_path, output_path):
    with open(input_path, 'r') as infile:
        lines = infile.readlines()

    updated_lines = []
    for line in lines:
        # Split the line into x, y, w, h
        x, y, w, h = map(float, line.split())

        # Subtract 1192 from the y-coordinate
        updated_y = y - 1192.0

        # Build the updated line and append to the list
        updated_line = f"{x} {updated_y} {w} {h}\n"
        updated_lines.append(updated_line)

    with open(output_path, 'w') as outfile:
        outfile.writelines(updated_lines)


def process_all_files(gt_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(gt_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(gt_folder, filename)
            output_path = os.path.join(output_folder, filename)
            process_file(input_path, output_path)


#%%

gt_folder = 'F:/Malek/code/evaluation/New_evaluation/Small_cropped/centered_gt'
output_folder = 'F:/Malek/code/evaluation/New_evaluation/Small_cropped/Cropped_GT'

process_all_files(gt_folder, output_folder)


# %%






"""Same as the above function, but this is meant for training labels with classes as a first element of each line
"""

""" Crop bboxes by subtracting 1192 from the y_value of the center coordinates or the top left coords
"""

#%%

import numpy
import os


def process_file(input_path, output_path):
    with open(input_path, 'r') as infile:
        lines = infile.readlines()

    updated_lines = []
    for line in lines:
        # Split the line into x, y, w, h
        c, x, y, w, h = map(float, line.split())

        # Subtract 1192 from the y-coordinate
        updated_y = y - 1192.0

        # Build the updated line and append to the list
        updated_line = f"{0} {x} {updated_y} {w} {h}\n"
        updated_lines.append(updated_line)

    with open(output_path, 'w') as outfile:
        outfile.writelines(updated_lines)


def process_all_files(gt_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(gt_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(gt_folder, filename)
            output_path = os.path.join(output_folder, filename)
            process_file(input_path, output_path)


#%%

gt_folder = 'F:/Malek/code/Perspective_transform/Bottom_cropped/training_labels'
output_folder = 'F:/Malek/code/Perspective_transform/Bottom_cropped/Double_Cropped_data/training_labels'

process_all_files(gt_folder, output_folder)
# %%





"""Same as the first function:
To double crop the bboxes, but here I am adding a class of (0) at the beginning of each bbox line"""

import numpy
import os

#%%


""" Crop bboxes by subtracting 1192 from the y_value of the center coordinates or the top left coords
"""

def process_file(input_path, output_path):
    with open(input_path, 'r') as infile:
        lines = infile.readlines()

    updated_lines = []
    for line in lines:
        # Split the line into x, y, w, h
        x, y, w, h = map(float, line.split())

        # Subtract 1192 from the y-coordinate
        updated_y = y - 1192.0

        # Build the updated line and append to the list
        updated_line = f"{0} {x} {updated_y} {w} {h}\n"
        updated_lines.append(updated_line)

    with open(output_path, 'w') as outfile:
        outfile.writelines(updated_lines)


def process_all_files(gt_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(gt_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(gt_folder, filename)
            output_path = os.path.join(output_folder, filename)
            process_file(input_path, output_path)


#%%

gt_folder = 'F:/Malek/code/evaluation/New_evaluation/Small_cropped/centered_gt'
output_folder = 'F:/Malek/code/evaluation/New_evaluation/Small_cropped/Cropped_GT'

process_all_files(gt_folder, output_folder)
# %%
