import tensorflow as tf
import tensorflow.keras.layers
import tensorflow.keras.models
from tensorflow.keras.preprocessing import image_dataset_from_directory
​
​
encoder_conv2d_layer1 = tf.keras.layers.Conv2D(filters = 512, kernel_size = (3, 3), activation = "relu")
encoder_conv2d_layer2 = tf.keras.layers.Conv2D(filters = 512, kernel_size = (3, 3), activation = "relu")
encoder_conv2d_layer3 = tf.keras.layers.Conv2D(filters = 512, kernel_size = (3, 3), activation = "relu")
​
upsample_layer1 = tf.keras.layers.UpSampling2D()
upsample_layer2 = tf.keras.layers.UpSampling2D()
upsample_layer3 = tf.keras.layers.UpSampling2D()
​
length = 32
width = 32
#Importing datasets
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory = "/kaggle/input/ship-data/my_shipdetection",
    color_mode = "rgb",
    image_size = (length, width),
    batch_size = 32,
    shuffle=True
) 
​
test_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory = "/kaggle/input/ship-data/my_shipdetection",
    color_mode = "rgb",
    image_size = (length, width),
    batch_size = 32,
    shuffle=True
)
​
# data = tf.keras.preprocessing.image_dataset_from_directory(
#   directory = "/ship-data/my_shipdetection/subset_images",
#   color_mode = "rgb",
#   image_size = (length, width),
#   batch_size = 32,
#   shuffle=True)
# train_data = data*0.8
# test_data = data*0.2
​
​
Unet_model = tf.keras.models.Sequential([
  #Encoder Part
​
  #Block №1
  encoder_conv2d_layer1,
  tf.keras.layers.Dense(256, activation="relu"),
  tf.keras.layers.MaxPooling2D(),
​
  #Block №2
  encoder_conv2d_layer2,
  tf.keras.layers.Dense(256, activation="relu"),
  tf.keras.layers.MaxPooling2D(),
​
  #Block №3
  encoder_conv2d_layer3,
  tf.keras.layers.Dense(256 ,activation="relu"),
  tf.keras.layers.MaxPooling2D(),
​
  #BottleNeck
  tf.keras.layers.Conv2D(filters, kernel_size),
​
  #Decoder Part
​
  #Block №1
  upsample_layer1,
  tf.keras.layers.Concatenate()([encoder_conv2d_layer1, upsample_layer1]),
  tf.keras.layers.Conv2D(filters, kernel_size),
  tf.keras.layers.Dense(256, activation="relu"),
  
​
  #Block №2
  upsample_layer2,
  tf.keras.layers.Concatenate()([encoder_conv2d_layer2, upsample_layer2]),
  tf.keras.layers.Conv2D(filters, kernel_size),
  tf.keras.layers.Dense(256, activation="relu"),
  
​
  #Block №3
  upsample_layer3,
  tf.keras.layers.Concatenate()([encoder_conv2d_layer3, upsample_layer3]),
  tf.keras.layers.Conv2D(filters, kernel_size),
  tf.keras.layers.Dense(256, activation="relu")
  
])
​
def train_model():
  Unet_model.compile(optimizer='nadam', loss=['binary_crossentropy'], metrics=['accuracy'])
  Unet_model.fit(train_data, validation_data=test_data, epochs= 18, batch_size=32)
  Unet_model.summary()
​
train_model()
​
#Saving model for inference
tf.keras.models.save_model("Airbus-Ship-Detection")
​
