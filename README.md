# Koach Sample Project

A CNN using the **MNIST dataset** to identify the number from its image with an accuracy of **99.16%**, created in Python using TensorFlow.


## Model Architecture
* Input Convolutional Layer
* Pooling Layer
* Convolutional Layer
* Pooling Layer
* Flattening Layer
* Dense Layer to combine everything
* Output Dense Layer

### Adam Optimizer:
* Learning Rate: 0.001
* Loss: sparse_categorical_crossentropy


## Packages
* TensorFlow
* Keras
* SciKit Learn
* Numpy
* Matplotlib
* Pickle


## Files:
* **main.py** - model created and saved
* **predict.py** - used to predict any image, just put any image and put its title in line 9 of the code as image_path (ensure the number is of a lighter shade and the background is darker)
* **plot.py** - used to plot the training and validation loss graph
* **mnist_cnn_model.h5** - the CNN model
* **model_history.pkl** - stores model traning history for plotting graph



##

<img width="849" alt="terminal" src="https://github.com/user-attachments/assets/f82aa77e-361b-465e-8044-d4cd3d88abb0">


<img width="849" alt="graph" src="https://github.com/user-attachments/assets/60d4b00b-7ef6-4719-9e0f-93758ae147b0">


### Batch Normalization
This layer is used to normalize the input ensuring stability and prevents overfitting. It also increases the training speed. 

### Dropout Layers
This layer randomly drops out some neurons to prevent overfitting and helps in generalization.

### Use of these Layers
I have not used these layers in this project as it is a simple project with a simple dataset so it's not that useful. It was only slightly be beneficial. For more complex models, these layers will be helpful. 