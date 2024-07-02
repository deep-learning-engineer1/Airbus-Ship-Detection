import tensorflow as tf
import tensorflow.keras.layers
import tensorflow.keras.models

model = tf.keras.models.Sequential([
  #Encoder Part

  #Block №1
  tf.keras.layers.Conv2D(),
  tf.keras.layers.MaxPooling2D,

  
  #Block №2

  #Block №4

  #Concatenate Part

  #Decoder Part

  #Block №1
  
  #Block №2
  
  #Block №3
])

def train_model():
  model.compile(optimizer='nadam', loss=['binary_crossentropy'], metrics=['accuracy'])
  model.fit(training_set, validation_data=test_set, epochs= 14, batch_size=12)
  model.summary()

train_model()
#tf.keras.layers.Concatenate
tf.keras.models.save_model("Airbus-Ship_Detection")
