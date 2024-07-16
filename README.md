# Airbus-Ship-Detection
This model has arhitecture U-Net. Which is used for segmentation and works on encoder and decoder.
First of all, I need to explore and prepare data, what I did in "Exploratory_data_exported_from_Jupyter_Notebook". In this file I opened all images and CSV file with coordinates for bounding boxes. After that I draw bounding boxes on images. When I annotated all images, I was ready to start building and training model.
I created official U-Net model and well-trained it. For this model I used neurons: Conv2D, MaxPooling2D, TransposeConv2D. The most interesting neuron from this list is "TransposeConv2D", this is opposite to Conv2D, we used this type of neurons in decoder part. Code for model is in github file "model_training.py".
And the last part of project is model inference, you can find in gihub file, that called "model_inference". You create directory with all images, and run script. Model will add annotated images to new directory.
