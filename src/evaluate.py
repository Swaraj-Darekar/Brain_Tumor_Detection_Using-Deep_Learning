import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import load_model
from src.utils import open_images, encode_label
from src.data_loader import test_paths, test_labels

# Load model
model = load_model("models/brain_tumor_model.h5")

# Prepare data
test_images = open_images(test_paths)
test_labels_encoded = encode_label(test_labels)
test_predictions = model.predict(test_images)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(test_labels_encoded, np.argmax(test_predictions, axis=1)))

# Confusion Matrix
conf_matrix = confusion_matrix(test_labels_encoded, np.argmax(test_predictions, axis=1))
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=os.listdir('data/archive/Training'),
            yticklabels=os.listdir('data/archive/Training'))
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.savefig("results/confusion_matrix.png")

# ROC Curve
test_labels_bin = label_binarize(test_labels_encoded, classes=np.arange(len(os.listdir('data/archive/Training'))))
fpr, tpr, roc_auc = {}, {}, {}
for i in range(len(os.listdir('data/archive/Training'))):
    fpr[i], tpr[i], _ = roc_curve(test_labels_bin[:, i], test_predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
for i in range(len(os.listdir('data/archive/Training'))):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.savefig("results/roc_curve.png")
