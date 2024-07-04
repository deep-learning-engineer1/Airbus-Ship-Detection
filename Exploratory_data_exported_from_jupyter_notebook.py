# Exploring data and sumarizing it
# For visualization we will use matplotlib
# Our mission is to draw boxes from csv file on images.

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import pandas as pd
import os

directory_csv = "/kaggle/input/airbus-ship-detection-data-visualization-css/sample_submission_v2.csv"
directory_images = "/kaggle/input/ship-data/my_shipdetection"
new_directory = os.mkdir("/kaggle/working/images")

# We need Computer Vision for drawing box 
import cv2 as cv

# Images
number_of_images = 85112
dataset_images = tf.keras.preprocessing.image_dataset_from_directory(
    directory = directory_images,
    color_mode = "rgb",
    batch_size = 32,
    shuffle=False
)

# CSV file
dataset_csv = pd.read_csv(directory_csv)
print(dataset_csv)

red = (0, 0, 256)

for i in range(number_of_images):
    encoded_pixels = dataset_csv["EncodedPixels"][i]
    if encoded_pixels == "":
        print("image doesn't have any ship")
        continue
    image = dataset_images[i]
    cv.line(image, encoded_pixels, red)
print("All images are ready.")

# i need to add annotated images to all images.
