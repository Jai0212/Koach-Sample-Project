import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import pickle

# MNIST dataset loaded from Keras
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Data preporcessed by normalizing the pixel values between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Data preporcessed by reshaping images to add a channel dimension as it is needed for a CNN with TensorFlow
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# Data partitioning to generate training, validation and test datasets
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42
)

# CNN Model
model = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)

# Model compiled using "adam" optimizer, learning rate of 0.001, loss evaluated using "sparse_categorical_crossentropy" and accuracy as the metric
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Model trained for 10 epochs
history = model.fit(
    train_images, train_labels, epochs=10, validation_data=(val_images, val_labels)
)

# Model evaluated on the test dataset
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc}")

# Model Saved
model.save("mnist_cnn_model.h5")

# Stores model training history
with open("model_history.pkl", "wb") as f:
    pickle.dump(history.history, f)
