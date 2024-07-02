import tensorflow as tf
import tensorflow.keras.layers
import tensorflow.keras.models
from tf.keras.preprocessing import images_dataset_from_directory

length = 
width = 
#Importing datasets
train_data = tf.keras.preprocessing.images_dataset_from_directory(
    directory = "./",
    color_mode = "rgb",
    image_size = (length, width),
    batch_size = 32,
    shuffle=True
) 

test_data = tf.keras.preprocessing.images_dataset_from_directory(
    directory = "./",
    color_mode = "rgb",
    image_size = (length, width),
    batch_size = 32,
    shuffle=True
)

Unet_model = tf.keras.models.Sequential([
  #Encoder Part

  #Block №1
  encoder_conv2d_layer1 = tf.keras.layers.Conv2D(),
  tf.keras.Dense(activation="relu")
  tf.keras.layers.MaxPooling2D(),

  #Block №2
  encoder_conv2d_layer2 = tf.keras.layers.Conv2D(),
  tf.keras.Dense(activation="relu")
  tf.keras.layers.MaxPooling2D(),

  #Block №3
  encoder_conv2d_layer3 = tf.keras.layers.Conv2D(),
  tf.keras.Dense(activation="relu")
  tf.keras.layers.MaxPooling2D(),

  #Decoder Part

  #Block №1
  tf.keras.layers.Concatenate()([encoder_conv2d_layer1, decoder_conv2d_layer1])
  decoder_conv2d_layer1 = tf.keras.layers.Conv2D(),
  

  #Block №2
  tf.keras.layers.Concatenate()([encoder_conv2d_layer2, decoder_conv2d_layer2])
  decoder_conv2d_layer2 = tf.keras.layers.Conv2D(),
  

  #Block №3
  tf.keras.layers.Concatenate()([encoder_conv2d_layer3, decoder_conv2d_layer3])
  decoder_conv2d_layer3 = tf.keras.layers.Conv2D(),
  
])

def train_model():
  model.compile(optimizer='nadam', loss=['binary_crossentropy'], metrics=['accuracy'])
  model.fit(train_data, validation_data=test_data, epochs= 18, batch_size=32)
  model.summary()

train_model()

#Saving model for inference
tf.keras.models.save_model("Airbus-Ship_Detection")
