import tensorflow as tf
import tensorflow.keras.layers
import tensorflow.keras.models

your_directory = ""
length = 
width = 

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
  
  pass
