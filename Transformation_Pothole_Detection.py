"""# Exploratory Data Analysis (EDA)
Starting this project with EDA in order to understand the data 
"""
#%% Importing all the needed libraries 

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import cv2

from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imageio

#%%

from IPython.display import Image

data_dir = 'Transformed_data' # data directory

# printing all the folders/files within the data directory
print(os.listdir(data_dir))
# %%
def get_file_path_df(folder):
  """
  Function to get path of .jpg files in the specified folder
  takes folder input and returns pandas dataframe
  """
  image_path = [] 
  for root, dirs, files in os.walk(folder):
    for f in files:
      if f.endswith(".JPG"): # checking for JPG extension 
        image_path.append(os.path.join(root, f)) # appending to image_path list
  # df = dataframe
  df_cols = {'path': image_path} # creating a dictionary
  df = pd.DataFrame(df_cols) # creating a dataframe
  return df

#%% getting absolute path of all images and storing in a dataframe called data_df
data_df = get_file_path_df(data_dir)

#%% displying random 5 rows
data_df.head()

#%% This data frame represents all the data we have (Training (pos/neg) and Test data)
data_df.shape
# %%

"""## Observation """

def get_train_test_df(data_frame):
  """
  Function to get train and test dataframe
  takes dataframe input and returns train, and test dataframe
  """
  train_path = []
  train_label = []
  test_path = []

  for path in data_frame['path'].values:
    # splitting the path by '/' (eg: data/dataset 1 simplex/dataset 1 simplex/Train data/Positive data/G0027850.JPG) 
    path_split = path.split('\\') 
    data_type = path_split[1] # accessing folder name (1) to access Train or Test data folder

    if data_type == 'Train data': 
      # train images
      train_path.append(path)
      train_label.append(path_split[2]) # appending label (Positive or Negative)
    else:
      # test images
      test_path.append(path)
    
  # creating dictionary of train data
  train_dict = df_cols = {'path': train_path, 'label': train_label} 
  train_df = pd.DataFrame(train_dict) # creating train dataframe

  test_dict = {'path': test_path} # creating dictionary of test data
  test_df = pd.DataFrame(test_dict) # creating test dataframe
  return train_df, test_df

#%%
# Getting our train, and testing dataframes
train_df, test_df = get_train_test_df(data_df)

#%% Testing our train dataframe
train_df.sample(5)

#%% test dataframe
test_df.sample(5)
# %%
"""Let's just take a look at the data distribution by doing a visualization of data distribution
 among train and test data"""

# creating a bar plot
plot = sns.barplot(x = ['Train', 'Test'], 
                  y= [train_df.shape[0], test_df.shape[0]])
plt.title('Data Distribution')
plt.xlabel('Dataset')
plt.ylabel('Count')

# adding class count to the barplot
for patch in plot.patches:
    # formatting and positioning the count values
    plot.annotate('{:n}'.format(patch.get_height()), 
                  (patch.get_x()+0.25, patch.get_height()+1)) 
plt.show()

# %%

"""Based on the plot above, the data contains 3777 training images, and 628 testing images.

Next, we are checking the class distribution (Positive/Negative) 
"""
#%%
def get_class_distrib(df):
  """
  This function gets the class distribution within a dataset where it
  takes a dataframe as an input and plots the class distribtution
  """
  label_count = df['label'].value_counts()
  plot = sns.barplot(x = list(label_count.index), y= list(label_count.values))
  plt.title('Class Distribution')
  plt.xlabel('Class Label')
  plt.ylabel('Count')

  # adding class count to the barplot
  for patch in plot.patches:
      # formatting and positioning the count values
      plot.annotate('{:n}'.format(patch.get_height()), 
                    (patch.get_x()+0.25, patch.get_height()+1)) 
  plt.show()

#%%
print("Class distribution of training data")
get_class_distrib(train_df)


#Based on the above plot of the training data, the data has two labels or classes (Positive) that means there
#is a pothole in the image where we have (1119) positive images, and (Negative)
#which means that there is no pothole in the image and we have (2658) Negative images.
# %%

## Checking the distribution of the images height and width


# def get_image_size_distrib(df):
#   """
#   This function is to get distribution of image height and width where it
#   takes a dataframe as an input and
#   outputs mean and median value of height and width
#   """
  
#   heights = []
#   widths = []
  

#   for path in df['path'].values:
#       # img = cv2.imread(os.path.join(img_dir, path))
#       img = cv2.imread(path) 
#       shape = img.shape # (h, w, d) = image.shape
#       heights.append(shape[0])
#       widths.append(shape[1]) 

#   plt.figure(figsize=(10,3))
#   sns.distplot(heights)
#   plt.title('Image height distribution')
#   plt.xlabel('Height')
#   plt.show()
#   height_median = np.median(heights)
#   print('Median height:', height_median)
#   height_mean = np.mean(heights)
#   print('Mean height:', height_mean)
#   print()
#   plt.figure(figsize=(10,3))
#   sns.distplot(widths)
#   plt.title('Image width distribution')
#   plt.xlabel('Width')
#   plt.show()
#   width_median = np.median(widths)
#   print('Median width:', width_median)
#   width_mean = np.mean(widths)
#   print('Mean width:', width_mean)
  
#%%
# get_image_size_distrib(train_df)
# %%
"""Just to know that all of the images are of the same size before cropping:


*   Images hight: 800 pixels
*   Image width: 800 pixles

Now, preparing the image annotations (Bounding Boxes)"""
# %%

"""Creat a dataframe that contains all the bboxes of the data, using different text files"""

import os
import pandas as pd

# Specify the directory containing your bounding box files
bbox_directory = "Transformed_data/bboxes"

# Create an empty list to store data from each file
data_list = []

# Iterate over each file in the directory
for filename in os.listdir(bbox_directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(bbox_directory, filename)
        
        # Read the content of the bounding box file
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Process each line and extract relevant information
        for line in lines:
            parts = line.strip().split()
            
            # Since the format is "class x y w h"
            # class_label = int(parts[0])
            x = abs(float(parts[0]))
            y = abs(float(parts[1]))
            w = abs(float(parts[2]))
            h = abs(float(parts[3]))
            
            # Extract the image_id from the filename
            image_id = os.path.splitext(filename)[0]
            
            # Append data to the list
            # data_list.append([image_id, class_label, x, y, w, h])
            data_list.append([image_id, x, y, w, h])

# Create a DataFrame from the list
# columns = ['image_id', 'class', 'x', 'y', 'w', 'h']
columns = ['image_id', 'x', 'y', 'w', 'h']
annotation_df = pd.DataFrame(data_list, columns=columns)

# Display the DataFrame
print(annotation_df.head())

# %%

# Visualizing some of the data

def visualize_data(img_df, annot_df, data_type):
  """
  Function to visualize images
  takes image dataframe, BBox dataframe and datatype input
  if data_type is test then print images without label else print with labels
  and BBoxes
  """
  
  rows = np.random.randint(img_df.shape[0], size=5) # generating 5 random numbers
  sample_imgs = img_df.iloc[rows, :] # random images

  counter = 0
  plt.figure(figsize = (12, 50))

  # checking if lowercase data_type = test
  if data_type.lower() == 'test':
      for idx, row in sample_imgs.iterrows():
        path = row['path'] # file path
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED) # reading the image
        plt.subplot(5, 1, 1 + counter)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        counter += 1
  else:
      for idx, row in sample_imgs.iterrows():
        # file path
        path = row['path']  #(F:/Malek/code/Transf_dataset/Train data/positive/G0027850.JPG
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED) # reading the image
        lab = row['label'] # file label either positive or negative
        coords = [] # BBox coordinates
        # if the image contains potholes
        if lab == 'positive':
          lab = 'Pothole'
          img_id = path.split('\\')[-1]
          img_id = img_id.split('.')[0]
          coords = annot_df[annot_df['image_id'] == img_id]
          coords = coords[['x', 'y', 'w', 'h']].values

        else:
          lab = 'No Pothole' 
        # The lower part in the categorization part (Far, Med, Near)
        plt.subplot(5, 1, 1 + counter)
        plt.title(lab)
        plt.xticks([])
        plt.yticks([])
        # Plot BBoxes
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        counter += 1
  plt.show()
#%%
"""### Visualizing training data sample"""

# visualizing train data
visualize_data(train_df, annotation_df, 'Train')

"""### Next visualizing some test data samples """

#%% visualizing test data
visualize_data(test_df, annotation_df, 'Test')

# %%
"""# 2- Data Preprocessing """

# plot an image randomly
row = np.random.randint(data_df.shape[0], size=1)[0]
sample_img = data_df.iloc[row, :]

plt.figure(figsize = (12, 50))
path = sample_img['path'] # image path
img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.yticks(np.arange(0, img.shape[0], 100))
plt.show()
#%%
"""Based on the image shown, it shows that the area of interest is between 0-800 and anything above that shows the
dashboard of the car where it might be wrongly detected as pothole"""

annotation_df.shape # annotation_df dim is (3097,6) where 6 represents: (img id, class, x, y, w, h)

#%%
# remove duplicates if exists
annotation_df.drop_duplicates(keep='first', inplace=True)
annotation_df.reset_index(inplace=True, drop=True)
annotation_df.head()
#%%
# Checking the shape of the annotation df after removing all dublicates 
annotation_df.shape

# %%
"""Here we are adding both (x_max, y_max) which basically represent the width and height of the bounding boxes
 for each pothole, where it shows the bottom right coordinates of the bounding box"""


#%% add x_max and y_max columns (right extreme point)

annotation_df['x_max'] = annotation_df['x'] + annotation_df['w']
annotation_df['y_max'] = annotation_df['y'] + annotation_df['h']
print("Data frame after updating columns")
print()
annotation_df.head()

# %%

"""Based on the range we got before where the image's area of interests is between 0-800 pixles. Hence,
 we need to check the range for all images and the bounding boxes within if exists  """

# maximum right exterme y cooridinate
y_max = annotation_df['y_max'].values.max()
print('Maximum value of y-coordinate of a BBox:', y_max)

#%%
# minimum left top y cooridinate
y = annotation_df['y'].values.min()
print('Minimum value of y-coordinate of a BBox:', y)

"""The value of y, for the furthest bbox is 30"""
# %%
# let's check the percentile values of y coordinate

# print 0 to 100 percentile values of BBox (y_coord + height) with step size of 10. 
for i in range(0,100,10):
    # var = annotation_df['y'].values + annotation_df['h'].values
    var = annotation_df['y_max'].values
    var = np.sort(var,axis = None)
    print("{} percentile value is {}".format(i,var[int(len(var)*(float(i)/100))]))
print ("100 percentile value is ",var[-1])

# %%
"""Based on the results above we can see that 90 percent of bounding boxes have y-max (bottom right) coordinate values less than
 or equal to 762 pixel

### Now let's check between 90% and 100% with a lower step size
and let's also check between 0% and 15%
"""

#%% print 90 to 100 percentile values of BBox (y_coord + height) with step size of  1. 

for i in range(90,100,1):
    # var = annotation_df['y'].values + annotation_df['h'].values
    var = annotation_df['y_max'].values
    var = np.sort(var,axis = None)
    print("{} percentile value is {}".format(i,var[int(len(var)*(float(i)/100))]))
print ("100 percentile value is ",var[-1])



# %%
"""**Counting on the results we can know the following:**


*  99% of the bounding boxes have y_max-coordinates (Bottom right) value less than or eqaul to 1108 pixles
*  Our area of interests is between 0-1108, so we need to get rid of any bounding box
 that has y-coordinate larger than 1108 and crop all images within the area of interests we have.  


"""

#%% remove all the bounding boxes having y coordinate greater than 1964 pixel
y_max = 1967
annotation_df_new = annotation_df[annotation_df['y_max'] <= y_max]
annotation_df_new.head()
# %%
annotation_df_new.shape

# %%

"""Crop all images to be within the desired coords"""

def crop_imgs(data_frame, y_crop):
    """ 
    This function crops images based on the max y-value we got before

    Input: 
    1. data frame containing path info
    2. y coordinate value to crop

    Output:
    Cropped images
    """
    # dirictories to store the cropped images
    dirs = ['Auto_Transformed_processed_data/train/positive',
            'Auto_Transformed_processed_data/train/negative',
            'Auto_Transformed_processed_data/test']

    # if folder does not exist, then create it
    for d in dirs:
        if not os.path.exists(d):
                os.makedirs(d)

    # loop over each image and crop them
    for path in data_frame['path'].values:
        path_split = path.split('\\') 
        data_type = path_split[1] # accessing folder name
        img_type = path_split[2] # image type (positive/negative)
        img_id = path_split[-1]

        if data_type == 'Train data': # if data belong to trainset
            if img_type == 'positive':
                img_type_new = 'positive'
            else: 
                img_type_new = 'negative'
            new_path = f'Auto_Transformed_processed_data/train/{img_type_new}/{img_id}'
        else:
            new_path = f'Auto_Transformed_processed_data/test/{img_id}'

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img_crop = img[:y_crop+1, :] # cropping
        cv2.imwrite(new_path, img_crop)
    print("Done")
    
#%%
crop_imgs(data_df, y_max)
# %%
processed_data_dir = 'Auto_Transformed_processed_data' # data directory

# printing all the folders/files within the data directory
print(os.listdir(processed_data_dir))

# %%
#%% getting absolute path of all images and storing in a dataframe called processed_data_df
processed_data_df = get_file_path_df(processed_data_dir)

# displying random 5 rows
processed_data_df.sample(5)

processed_data_df.shape
# %%
def get_train_test_df(data_frame):
  """
  This function gets the training and testing dataframes where it
  takes a dataframe as input and returns train, and test dataframes
  """
  train_path = []
  train_label = []
  test_path = []

  for path in data_frame['path'].values:
    # splitting the path by '/' (eg: data/Train data/Positive data/G0027850.JPG) 
    path_split = path.split('\\') 
    data_type = path_split[1] # accessing folder name (Train/Test)

    if data_type == 'train': 
      # train images
      train_path.append(path)
      train_label.append(path_split[2]) # appending label (Positive/Negative)
    else:
      # test images
      test_path.append(path)
    
  # creating dictionary of train data
  train_dict = {'path': train_path, 'label': train_label} 
  train_df = pd.DataFrame(train_dict) # creating train dataframe

  test_dict = {'path': test_path} # creating dictionary of test data
  test_df = pd.DataFrame(test_dict) # creating test dataframe
  return train_df, test_df


#%% creating train and test dataframe
processed_train_df, processed_test_df = get_train_test_df(processed_data_df)

#%% The train datafram shape after processing 
processed_train_df.shape

#%% The test datafram shape after processing 
processed_test_df.shape
# %%
"""# 3- Data Augmentation"""

tot_neg = processed_train_df['label'].value_counts()['negative']
tot_pos = processed_train_df['label'].value_counts()['positive']

# This shows the difference 
imbalance = abs(tot_pos - tot_neg)
print(imbalance)
# %%
#%% This setup of augmentation parameters will pick two of five given 
# augmenters and apply them in random order

# augmentor = iaa.SomeOf(2, [    
#     iaa.Affine(scale=(0.8, 1.2)),
#     iaa.Affine(rotate=(-20, 20)),
#     iaa.Fliplr(1),
#     iaa.Affine(shear=(-5, 5)),
#     iaa.GaussianBlur(sigma=(1.0, 3.0)),
#     iaa.GammaContrast((0.6, 1.5))
# ])
#%%


# augmentor = iaa.SomeOf(6, [  # Increase to 4 augmentations
#     iaa.Affine(scale=(0.8, 1.2)),
#     iaa.Affine(rotate=(-20, 20)),
#     iaa.Fliplr(1),
#     iaa.Affine(shear=(-5, 5)),
#     iaa.GaussianBlur(sigma=(1.0, 3.0)),
#     iaa.GammaContrast((0.6, 1.5)),
#     iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255)),
#     iaa.CropAndPad(percent=(-0.1, 0.1)),
#     iaa.Multiply((0.8, 1.2), per_channel=0.2),  # Randomly adjust brightness
#     iaa.ContrastNormalization((0.5, 2.0), per_channel=0.3)  # Adjust contrast
#     # iaa.OneOf([
#     #     iaa.MotionBlur(k=(3, 7)),  # Simulate motion blur
#     #     iaa.MedianBlur(k=(3, 7)),  # Simulate median blur
#     #     iaa.AverageBlur(k=(3, 7))  # Simulate average blur
#     ])
#%%
augmentor = iaa.SomeOf(6, [  # Increase to 4 augmentations
    iaa.Affine(scale=(0.8, 1.2)), # Could cause Neg. values and moving bbox coords out of img
    iaa.Affine(rotate=(-20, 20)), # Could cause Neg. values and moving bbox coords out of img
    iaa.Fliplr(1), # Could cause Neg. values and moving bbox coords out of img
    iaa.Affine(shear=(-5, 5)), # Could cause Neg. values and moving bbox coords out of img
    iaa.GaussianBlur(sigma=(1.0, 3.0)),
    iaa.GammaContrast((0.6, 1.5)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255)),
    iaa.CropAndPad(percent=(-0.1, 0.1)), # Could cause Neg. values and moving bbox coords out of img
    iaa.Multiply((0.8, 1.2), per_channel=0.2),  # Randomly adjust brightness
    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.3)  # Adjust contrast
# augmentor = iaa.OneOf([
#     iaa.MotionBlur(k=(3, 7)),  # Simulate motion blur
#     iaa.MedianBlur(k=(3, 7)),  # Simulate median blur
#     iaa.AverageBlur(k=(3, 7))  # Simulate average blur
])

#%% This is used to augment one data sample at a time

def augment_img_bbox(annot_df, path, augmentor, img_id, suffix):
        """ 
        This function is to augment images and corresponding bounding boxes.
        Inputs:
                1. annot_df: Bounding box coordinates
                2. path: image path
                3. augmentor: set of data augmentors
                4. img_id: image id
        Output:
                Data frame containing augmented BBox coordinates

        """
        # details of an image to augment
        # First, getting the image id of the image that you want to work on
        bbox_coords = annot_df[annot_df['image_id'] == img_id]

        # Getting the values of the BBox coordinates of an orginal image
        bb_array = bbox_coords.loc[:, ['x', 'y', 'x_max', 'y_max']].values # This represents ((top left) and (bottom right) coordinates)
        # read an image
        image = imageio.imread(path)
        
        # We give it the top left and bottom right coordinate of the bbox
        #and it returns an object containing a list of bboxes instances or values
        bbs = BoundingBoxesOnImage.from_xyxy_array(bb_array, shape=image.shape)

        # image and BBox after applying augmentation
        image_aug, bbs_aug = augmentor(image=image, bounding_boxes=bbs)
        # disregard bounding boxes which have fallen out of image pane    
        bbs_aug = bbs_aug.remove_out_of_image()
        # clip bounding boxes which are partially outside of image pane
        bbs_aug = bbs_aug.clip_out_of_image()

        path = path.split('\\')[:-1]
        path = ('/').join(path)

        # save augmented image
        imageio.imwrite(f'{path}/{img_id}_{suffix}.JPG', image_aug)
        # convert bounding boxes to an array
        bbs_array = bbs_aug.to_xyxy_array()
        img_id_array = np.empty([bbs_array.shape[0], 1])
        # concatinate image_id and BBox coordinate
        final = np.concatenate((img_id_array, bbs_array), axis=1)
        # create a data frame
        df = pd.DataFrame(final, columns=['image_id', 'x', 'y', 'x_max', 'y_max'])
        df['image_id'] = f'{img_id}_{suffix}'
                
        return df
#%% This is to loop over all the dataset and augment every image and bb

def augment_data(train_img_df, annot_df, augmentor, num_sample):
    """ 
    This function loop through all the data and uses the previous function to augment images
    and the corresponding bounding boxes
    Inputs:
        1. train_img_df: images to augment
        2. annot_df: Bounding box data frame
        3. augmentor: set of data augmentors
        4: num_sample: number of samples to augment
    Output:
        Data frame containing augmented images and BBox coordinates
    """
    # initialize empty data frame
    final_df = pd.DataFrame()

    for i in range(num_sample):
        pos = False

        # repeat until find an image with potholes
        while not pos:
            idx = np.random.randint(train_img_df.shape[0], size=1)[0]
            row = train_img_df.iloc[idx, :]
            path = row['path']
            path = train_img_df['path'][idx]
            path_split = path.split('\\')
            sample_type = path_split[-2]
            if sample_type == 'positive':
                pos = True

        img_id = path_split[-1]
        img_id = img_id.split('.')[0]
        # augment the image and corresponding BBoxes
        df = augment_img_bbox(annot_df, path, augmentor, img_id, i)

        final_df = pd.concat([final_df, df], ignore_index=True)

    return final_df







# %%
augmented_BBox_df1 = augment_data(processed_train_df, annotation_df_new, augmentor, imbalance)
# %%

augmented_BBox_df1.shape
#%%
augmented_BBox_df1.head()
#%%
annotation_df_new.head()

#%% concatinate augmented images BBox coordinates
# aug_BBox = annotation_df_new.drop(['class', 'w', 'h'], axis=1)
aug_BBox = annotation_df_new.drop(['w', 'h'], axis=1)
augmented_BBox_df = pd.concat([aug_BBox, augmented_BBox_df1], ignore_index=True)
augmented_BBox_df.head()
#%%
augmented_BBox_df.shape

#%% save csv file containing all the BBox coordinates
augmented_BBox_df.to_csv("augmented_Auto_Transf_BBox_df.csv", index=False)
# %%
"""# Train and Test dataset"""

processed_data_dir = 'Auto_Transformed_processed_data' # data directory

processed_data_df = get_file_path_df(processed_data_dir)
processed_train_df, processed_test_df = get_train_test_df(processed_data_df)

# creating a bar plot
plot = sns.barplot(x = ['Train', 'Test'], 
                   y = [processed_train_df.shape[0], processed_test_df.shape[0]])
plt.title('Data Distribution')
plt.xlabel('Dataset')
plt.ylabel('Count')

# adding class count to the barplot
for patch in plot.patches:
    # formatting and positioning the count values
    plot.annotate('{:n}'.format(patch.get_height()), 
                  (patch.get_x()+0.25, patch.get_height()+1)) 
# %%
"""As it shown above, that data contains:



*   5316 Training data
*   628 Testing data


Next, checking the class distribution

"""
#%%
def get_class_distrib(df):
  """
  A function to get class distribution within a dataset
  takes dataframe input and plots the class distribtution
  """
  label_count = df['label'].value_counts()
  plot = sns.barplot(x = list(label_count.index), y =list(label_count.values))
  plt.title('Class Distribution')
  plt.xlabel('Class Label')
  plt.ylabel('Count')

  # adding class count to the barplot
  for patch in plot.patches:
      # formatting and positioning the count values
      plot.annotate('{:n}'.format(patch.get_height()), 
                    (patch.get_x()+0.25, patch.get_height()+1)) 
  plt.show()
#%%
print("Class distribution of training data")
get_class_distrib(processed_train_df)

#%%
"""As it appeares that the new data contains an equal numeber for both positive and negative data.

## Visualizing the data after augmentation is apllied
"""
# %%
# import random
# import imgaug as ia
# folder_old = 'F:/Malek/code/Auto_tran_data/Train data/positive'
# folder_new = 'F:/Malek/code/Auto_Transformed_processed_data/train/positive'

# def get_image_BBox(path, annot_df, img_id):
#   """
#   Function reads an image & return corresponding BBox coordinates
#   Inputs:
#     1. path: image path
#     2. annot_df: BBox data frame
#     3. img_id: image id
#   Output: 
#     image and corresponding BBox coordinates
#   """
#   # read an image
#   image = imageio.imread(path)

#   # extract BBox coordinates from a data frame
#   bbox_coords = annot_df[annot_df['image_id'] == img_id]
#   bbox_coords = bbox_coords[['x', 'y', 'x_max', 'y_max']].values
#   bbox_coords = BoundingBoxesOnImage.from_xyxy_array(bbox_coords, shape=image.shape)

#   return image, bbox_coords
# %%
# def visualize_data(img_dir_old, annot_df_old, img_dir_new, annot_df_new):
#     """
#     Function to visualize images before and after augmentation
#     Inputs:
#         1. img_dir_old: directory of images
#         2. annot_df_old: BBox data frame
#         3. img_dir_new: directory of augmented images
#         4. annot_df_new: BBox data frame of augmented images
#     Outputs:
#         image & augmented image
#     """

#     # 5 random images
#     smpls = random.sample(os.listdir(img_dir_old), 2)

#     plt.figure(figsize = (20, 40))
    
#     for old_img in smpls:
#         # image path
#         old_path = f'{img_dir_new}/{old_img}'
#         # image id
#         old_img_id = old_img.split('.')[0]
#         f = ''
#         for file in os.listdir(img_dir_new):
#             split_name = file.split('_')
            
#             if len(split_name) <= 1:
#                 continue
#             else:
#                 if split_name[0] == old_img_id:
#                     f = file
#                     break
#         if f:
#             new_path = f'{img_dir_new}/{f}'
#             new_img_id = f.split('.')[0]
#         else:
#             new_path = old_path
#             new_img_id = old_img_id
            
#         print(old_path)
#         print(new_path)

#         # image and corresponding BBoxes
#         img_old, bbox_old = get_image_BBox(old_path, annot_df_old, old_img_id)
#         img_new, bbox_new = get_image_BBox(new_path, annot_df_new, new_img_id)

#         plt.imshow(np.hstack([
#             bbox_old.draw_on_image(img_old, size=5, color=[0, 0, 255]),
#             bbox_new.draw_on_image(img_new, size=5, color=[0, 0, 255])
#             ]))
# #
# visualize_data(folder_old, annotation_df_new, folder_new, augmented_BBox_df)

# %%


"""# YOLO 5 Training"""

# Importing libraries

import os
import pandas as pd
from sklearn import model_selection
import numpy as np
import shutil
import cv2
import random
import matplotlib.pyplot as plt
from IPython.display import Image
# %%
# Using only positive data for training
data_dir = 'Auto_Transformed_processed_data/train/positive'
# %%
def get_file_path_df(folder):
  """
  Function to get name and path of .jpg files in the specified folder
  takes folder input and returns pandas dataframe
  """
  image_path = [] 
  image_name = [] 

  for root, dirs, files in os.walk(folder):
    for f in files:
      if f.endswith(".JPG"): # checking for JPG extension 
        image_name.append(f.split('.')[0])
        image_path.append(os.path.join(root, f)) # appending to image_path list
  # df = dataframe
  df_cols = {'image_name':image_name, 'path': image_path} # creating a dictionary
  df = pd.DataFrame(df_cols) # creating a dataframe
  return df

#%% getting absolute path of all images and storing in a dataframe called data_df
data_df = get_file_path_df(data_dir)

# displying random 5 rows
data_df.sample(5)
#%%

data_df.shape

#%% splitting the data into train and validation 
train_df, valid_df = model_selection.train_test_split(data_df, test_size=0.2, random_state=21, shuffle=True)

#%% Number of images in training set
print(train_df.shape)

#%% Number of images in validation set
print(valid_df.shape)

#%% loading the csv file containing bounding box coordinates

bbox_df = pd.read_csv("augmented_Auto_Transf_BBox_df.csv")
bbox_df.head()
# %%
def process_bbox(img_df, bbox_df, data_type, img_w, img_h):
    """
    Function to convert bounding box coordinates into YOLO format
    and also to arrange the images and bounding boxes in specified folders
    
    Parameters:
        1. img_df: data frame containing image name and path
        2. bbox_df: data frame containing bounding box coordinates
        3. data_type: type of data train / valid
        4. img_w: image width
        5. img_h: image height
    """
    # drictories for images and labels
    dirs = ['images/train',
            'images/valid',
            'labels/train',
            'labels/valid',
           ]
    
    # if folder does not exist create them
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
    
    # loop over each row of image data frame
    for _, row in img_df.iterrows():
        yolo_bbox = []
        
        image_name = row['image_name']
        bbox_coords = bbox_df[bbox_df['image_id'] == image_name]
        bbox_array = bbox_coords.loc[:, ['x', 'y', 'x_max', 'y_max']].values
        
        for bbox in bbox_array:
            x_min = bbox[0]
            y_min = bbox[1]
            x_max = bbox[2]
            y_max = bbox[3]

            # We divide by the image width and height to normalize the values
            x_center = (x_min + x_max) / 2 / img_w
            y_center = (y_min + y_max) / 2 / img_h
        
            w = (x_max - x_min) / img_w
            h = (y_max - y_min) / img_h
            
            yolo_bbox.append([0, # object class 
                              x_center, # bbox x-center
                              y_center, # bbox y-center
                              w, # bbox width
                              h]) # bbox height
            
        yolo_bbox = np.array(yolo_bbox)
        label_path = f"labels/{data_type}/{image_name}.txt"
        
        # saving txt file containing class label and bbox coordinates
        np.savetxt(label_path, yolo_bbox)
        
        img_source = row['path']
        img_desti = f"images/{data_type}/{image_name}.JPG"
        # moving images to the specific folder
        shutil.move(img_source, img_desti)
    print("Done")
# %%
image_width = 3680
image_height = 1965

process_bbox(train_df, bbox_df, 'train', image_width, image_height)
#%%
process_bbox(valid_df, bbox_df, 'valid', image_width, image_height)
# %%



"""# Training


Note: A yaml file with the path for the training data and lables needs to be created with the name of pothole.yaml
"""
!python train.py --img 640 --batch 16 --epochs 4 --data pothole.yaml --weights yolov5m.pt --device 0


python detect.py --source ../processed_data/test --weights runs/train/exp6/weights/best.pt --conf 0.5 --save-txt --save-conf



# Usage - Single-GPU training:
#     $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
#     $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch
# Usage - Multi-GPU DDP training:
#     $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3
