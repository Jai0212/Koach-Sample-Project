import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model("mnist_cnn_model.h5")
from PIL import Image
import numpy as np

# Load the image
image_path = "test.png" # IMPORTANT: Change this to the new file name to predict the number
image = Image.open(image_path).convert("L")  # Converts image to grayscale

# Resize the image to 28x28 pixels and convert to numpy array
image = image.resize((28, 28))
image_array = np.array(image)

# Normalize the image
image_array = image_array / 255.0

# Array reshaped to match the input shape of the model
image_array = image_array.reshape((1, 28, 28, 1))

# Predict the number in the image
predictions = model.predict(image_array)
predicted_class = np.argmax(predictions, axis=1)

print(f"Predicted Digit: {predicted_class[0]}")
