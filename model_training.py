import tensorflow as tf
import tensorflow.keras.layers
import tensorflow.keras.models

model = tf.keras.models.Sequential([

  #Block №1
  conv_layer1 = tf.keras.layers.Conv2D(),
  concatenate_layer1 = tf.keras.layers.Concatenate(),
  concatenate_layer1 = conv_layer1,
  max_pool_layer1 = tf.keras.layers.MaxPooling2D(),
  max_pool_layer1 = conv_layer1,


  #Block №2
  conv_layer2 = tf.keras.layers.Conv2D(),
  concatenate_layer2 = tf.keras.layers.Concatenate(),
  concatenate_layer2 = conv_layer2,
  max_pool_layer2 = tf.keras.layers.MaxPooling2D(),
  max_pool_layer2 = conv_layer2,

  #Block №3
  conv_layer2 = tf.keras.layers.Conv2D(),
  concatenate_layer2 = tf.keras.layers.Concatenate(),
  concatenate_layer2 = conv_layer2,
  max_pool_layer2 = tf.keras.layers.MaxPooling2D(),
  max_pool_layer2 = conv_layer2,
])

#tf.keras.layers.Concatenate

