import tensorflow as tf
import tensorflow.keras.layers
import tensorflow.keras.models
from tensorflow.keras.preprocessing import image_dataset_from_directory

encoder_conv2d_layer1 = tf.keras.layers.Conv2D(16, kernel_size = (3, 3), strides=1, activation = "relu")
encoder_conv2d_layer2 = tf.keras.layers.Conv2D(32, kernel_size = (3, 3), strides=1, activation = "relu")
encoder_conv2d_layer3 = tf.keras.layers.Conv2D(64, kernel_size = (3, 3), strides=1, activation = "relu")
encoder_conv2d_layer4 = tf.keras.layers.Conv2D(128, kernel_size = (3, 3), strides=1, activation = "relu")

upsample_layer1 = tf.keras.layers.UpSampling2D()
upsample_layer2 = tf.keras.layers.UpSampling2D()
upsample_layer3 = tf.keras.layers.UpSampling2D()
upsample_layer4 = tf.keras.layers.UpSampling2D()

length = 572
width = 572
#Importing datasets
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory = "/kaggle/input/ship-data/my_shipdetection",
    color_mode = "rgb",
    batch_size = 32,
    shuffle=True
) 

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory = "/kaggle/input/ship-data/my_shipdetection",
    color_mode = "rgb",
    batch_size = 32,
    shuffle=True
)

'''data = tf.keras.preprocessing.image_dataset_from_directory(
   directory = "/kaggle/input/ship-data/my_shipdetection",
   color_mode = "rgb",
   image_size = (length, width),
   batch_size = 32,
   shuffle=True)
train_data = data[68089]
test_data = data[17023]'''


Unet_model = tf.keras.models.Sequential([

  #-----------------------------------------------------------------------------------------------
    
  #Encoder Part

  #Block №1
  encoder_conv2d_layer1,
  encoder_conv2d_layer1,
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),

  #Block №2
  encoder_conv2d_layer2,
  encoder_conv2d_layer2,
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),

  #Block №3
  encoder_conv2d_layer3,
  encoder_conv2d_layer3,
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),

  #Block №4
  encoder_conv2d_layer4,
  encoder_conv2d_layer4,
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),

  #-----------------------------------------------------------------------------------------------

  #BottleNeck
  tf.keras.layers.Conv2D(256, kernel_size = (3, 3), activation = "relu", padding="same"),
  tf.keras.layers.Conv2D(256, kernel_size = (3, 3), activation = "relu", padding="same"),
  tf.keras.layers.Conv2DTranspose(256, kernel_size = (3, 3), activation = "relu", padding="same"),

  #----------------------------------------------------------------------------------------------- 

  #Decoder Part

  #Block №1
  upsample_layer1,
  tf.keras.layers.Concatenate([encoder_conv2d_layer3, upsample_layer3]),
  tf.keras.layers.Conv2DTranspose(128, kernel_size = (3, 3), activation = "relu", padding="same"),
  

  #Block №2
  upsample_layer2,
  tf.keras.layers.Concatenate([encoder_conv2d_layer3, upsample_layer3]),
  tf.keras.layers.Conv2DTranspose(64, kernel_size = (3, 3), activation = "relu", padding="same"),

  #Block №3
  upsample_layer3,
  tf.keras.layers.Concatenate([encoder_conv2d_layer3, upsample_layer3]),
  tf.keras.layers.Conv2DTranspose(32, kernel_size = (3, 3), activation = "relu", padding="same"),
  

  #Block №4
  upsample_layer4,
  tf.keras.layers.Concatenate([encoder_conv2d_layer3, upsample_layer3]),
  tf.keras.layers.Conv2DTranspose(16, kernel_size = (3, 3), padding="same"),
  tf.keras.layers.Conv2DTranspose(16, kernel_size = (3, 3), padding="same"),
  tf.keras.layers.Conv2DTranspose(16, kernel_size = (1, 1), padding="same")

  #------------------------------------------------------------------------------------------------
])

def train_model():
  Unet_model.compile(optimizer='nadam', loss=['binary_crossentropy'], metrics=['accuracy'])
  Unet_model.fit(train_data, validation_data=test_data, epochs= 18, batch_size=32)
  Unet_model.summary()

train_model()

#Saving model for inference
tf.keras.models.save_model("Airbus-Ship-Detection", save_format="tf")
