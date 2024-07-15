# Exploring data and sumarizing it
# Our mission is to draw boxes from csv file on images.

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from PIL import Image, ImageDraw

#directory_csv = "/kaggle/input/airbus-ship-detection-train-set-70/train_ship_segmentations_v3.csv"
directory_images = "/kaggle/input/airbus-ship-detection-train-set-70/train_v3/train_v3/Images"
#new_directory = os.mkdir("/kaggle/working/annotated_images")

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
number_of_images = 192556
dataset_images = tf.keras.preprocessing.image_dataset_from_directory(
    directory = directory_images,
    color_mode = "rgb",
    batch_size = 32,
    shuffle=False
)

# CSV file
dataset_csv = pd.read_csv(directory_csv)

for i in range(number_of_images):
    encoded_pixels = dataset_csv["EncodedPixels"][i]
    if encoded_pixels == "" or encoded_pixels == "NaN":
        print("image doesn't have any ship")
        continue
    decode_rle_photo(encoded_pixels, "JPG")
    

def decode_rle_photo(rle_data, image_format):
    # Placeholder for RLE data validation (assuming format specific)
    if not validate_rle_photo_format(rle_data, image_format):
        raise ValueError("Invalid RLE data for image format")

    # Use Pillow to open the image from RLE data (conversion might be needed)
    try:
        image = Image.open(rle_data)
        return image
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None
        
print("All images are ready.")
