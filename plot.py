import matplotlib.pyplot as plt
import pickle

# Training history of the model is resotred to plot the training and validation loss curves
with open("model_history.pkl", "rb") as f:
    history_dict = pickle.load(f)

# Epochs and training & validation loss data extracted from the model history
train_loss = history_dict["loss"]
val_loss = history_dict["val_loss"]
epochs = range(1, len(train_loss) + 1)

# Matplotlib to plot the graph
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, "bo-", label="Training Loss")
plt.plot(epochs, val_loss, "ro-", label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()
