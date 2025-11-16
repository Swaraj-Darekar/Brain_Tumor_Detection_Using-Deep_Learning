import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import os

model = load_model('models/model.h5', compile=False)
class_labels = sorted(os.listdir('data/archive/Training'))

def detect_and_display(img_path, model=model, image_size=128):
    img = load_img(img_path, target_size=(image_size, image_size))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    predicted_label = class_labels[predicted_class_index]
    result = "No Tumor" if predicted_label.lower() == 'notumor' else f"Tumor: {predicted_label.capitalize()}"

    plt.imshow(load_img(img_path))
    plt.axis('off')
    plt.title(f"{result} (Confidence: {confidence_score * 100:.2f}%)")
    plt.show()

# Example
# detect_and_display('data/archive/Testing/meningioma/Te-meTr_0001.jpg')
