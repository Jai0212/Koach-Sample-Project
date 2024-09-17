# Koach Sample Project

A CNN using the **MNIST dataset** to identify the number from its image with an accuracy of **99.18%**, created in Python using TensorFlow.


## Model Architecture
* Convolutional Layer
* Pooling Layer
* Batch Normalization
* Dropout Layer
* Flattening Layer
* Dense Layer

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
* **model_history.pkl** - stores model training history for plotting graph


##

<img width="849" alt="terminal" src="https://github.com/user-attachments/assets/d3e05241-5f85-4352-a95c-419cd8c3b07e">


<img width="849" alt="graph" src="https://github.com/user-attachments/assets/dff17153-9083-40de-92d1-aa0978b60e47">


### Batch Normalization
This layer is used to normalize the input ensuring stability and preventing overfitting. It also increases the training speed. 

### Dropout Layers
This layer randomly drops out some neurons to prevent overfitting and helps in generalization.

### Use of these Layers
I have used these layers in this project to improve the model as it prevents overfitting and helps in generalization thereby improving accuracy. Since it is a simple project with a simple dataset it's not super useful. It was only slightly beneficial. For more complex models, these layers will be of great use.
