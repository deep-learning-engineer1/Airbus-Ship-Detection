import tensorflow as tf
import tensorflow.keras.layers
import tensorflow.keras.models

Unet_model = tf.keras.models.Sequential([
  #Encoder Part

  #Block №1
  encoder_conv2d_layer1 = tf.keras.layers.Conv2D(),
  tf.keras.layers.MaxPooling2D(),

  #Block №2
  encoder_conv2d_layer2 = tf.keras.layers.Conv2D(),
  tf.keras.layers.MaxPooling2D(),

  #Block №3
  encoder_conv2d_layer3 = tf.keras.layers.Conv2D(),
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
  model.fit(training_set, validation_data=test_set, epochs= 14, batch_size=12)
  model.summary()

train_model()
#tf.keras.layers.Concatenate
tf.keras.models.save_model("Airbus-Ship_Detection")
