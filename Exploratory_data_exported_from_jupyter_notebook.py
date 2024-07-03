#Exploring data and sumarazing it
#for visualization we will take matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

number_of_images = 85112

dataset = tensorflow.keras.preprocessing.image_dataset_from_directory(
    directory = "/kaggle/input/ship-data/my_shipdetection",
    color_mode = "rgb",
    batch_size = 32,
    shuffle=False
)

plt.plot()
plt.axis(0, number_of_images)
plt.show()

