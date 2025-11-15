import os
import numpy as np
import random
from PIL import Image, ImageEnhance
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import load_img


train_dir = 'data/archive/Training'
test_dir = 'data/archive/Testing'


# Load and suffle the train data  

train_paths = []
train_labels = []

for label in os.listdir(train_dir):
    # print(label)
    
    for image in os.listdir(os.path.join(train_dir, label)):
        train_paths.append(os.path.join(train_dir, label , image))
        train_labels.append(label)
        
train_paths , train_labels = shuffle(train_paths, train_labels)

# print(train_paths)
    


# Load and suffle the test data

test_paths = []
test_labels = []

for label in os.listdir(test_dir):
    # print(label)
    
    for image in os.listdir(os.path.join(test_dir, label)):
        test_paths.append(os.path.join(test_dir, label, image))
        test_labels.append(label)
        
test_paths , test_labels = shuffle(test_paths, test_labels)

# print(test_paths)

