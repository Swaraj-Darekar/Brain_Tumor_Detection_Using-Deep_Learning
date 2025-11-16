from src.data_loader import train_paths, train_labels, train_dir
from src.utils import datagen
from src.model import build_model




from tensorflow.keras.models import load_model
from src.data_loader import train_paths, train_labels 
from src.utils import datagen    



import matplotlib.pyplot as plt
import pickle
import os


IMAGE_SIZE = 128

model = build_model(IMAGE_SIZE, train_dir)

batch_size = 20
epochs = 5
steps = int(len(train_paths) / batch_size)

# Train model
history = model.fit(datagen(train_paths, train_labels, batch_size=batch_size, epochs=epochs),
                    epochs=epochs, steps_per_epoch=steps)

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/brain_tumor_model.h5")

# Save history (optional)
with open("models/history.pkl", "wb") as f:
    pickle.dump(history.history, f)






# Plot accuracy & loss
plt.figure(figsize=(8,4))
plt.grid(True)
plt.plot(history.history['sparse_categorical_accuracy'], '.g-', linewidth=2)
plt.plot(history.history['loss'], '.r-', linewidth=2)
plt.title('Model Training History')
plt.xlabel('epoch')
plt.xticks([x for x in range(epochs)])
plt.legend(['Accuracy', 'Loss'], loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig("results/training_plot.png")


