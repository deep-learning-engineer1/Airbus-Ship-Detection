import tensorflow as tf
import tensorflow.keras.layers
import tensorflow.keras.models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt

your_directory = ""
length = 572
width = 572

data = tf.keras.preprocessing.image_dataset_from_directory(
    directory = your_directory,
    color_mode = "rgb",
    image_size = (length, width),
    batch_size = 32,
    shuffle=True
) 

Unet_model = tf.keras.load_model("./")

def predict():
  Unet_model.predict(data)
