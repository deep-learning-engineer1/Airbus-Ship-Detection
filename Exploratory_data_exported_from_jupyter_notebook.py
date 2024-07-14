# Exploring data and sumarizing it
# Our mission is to draw boxes from csv file on images.

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from PIL import Image, ImageDraw

directory_csv = "/kaggle/input/airbus-ship-detection/train_ship_segmentations_v2.csv"
directory_images = "/kaggle/input/airbus-ship-detection/train_v2/000155de5.jpg"
new_directory = os.mkdir("/kaggle/working/annotated_images")

image = Image.open(directory_csv)
width_image = image.width
height_image = image.height
image_shape = (width_image, height_image)

# 
file = open(directory_images, 'r')
file = json.load(file)
file[0]

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
print(dataset_csv["EncodedPixels"].head(10))


red = (0, 0, 256)

'''for i in range(number_of_images):
    encoded_pixels = dataset_csv["EncodedPixels"][i]
    if encoded_pixels == "" or encoded_pixels == "NaN":
        print("image doesn't have any ship")
        continue
'''
rle_decoder(image, run_length):
    open with(image, mode = "rb") as file:
        
        
        
        
print("All images are ready.")
