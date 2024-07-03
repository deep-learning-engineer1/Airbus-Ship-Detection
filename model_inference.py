import tensorflow as tf
import tensorflow.keras.layers
import tensorflow.keras.models

Unet_model = tf.keras.load_model("./")

def preprocess():
  pass
  
def predict():
  preprocess()
  Unet_model.predict()
  pass
