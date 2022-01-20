import numpy as np
import cv2
import glob
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img

# Reads the images from the folders as shown below
# The test and train images are moved to one folder and later train test data is split randomly

GLIOMA = "Glioma\\"
MENINGIONMA = "meningioma\\"
NOTUMOUR = "notumour\\"
PITUTARY = "pitutary\\"

def read_data(path):
    paths = glob.glob(path + "*.jpg")
    images = []

    for img in paths:
        image = load_img(img, target_size=(128,128))
        image = np.array(image)/255.0
        #image = image.flatten()
        images.append(image)

    return np.array(images)

# Dataset 

glioma_dataset = read_data(GLIOMA)
meningioma_dataset = read_data(MENINGIONMA)
notumour_dataset = read_data(NOTUMOUR)
pitutary_dataset = read_data(PITUTARY)

print(f"Glioma -> {glioma_dataset.shape}")
print(f"No Tumour -> {notumour_dataset.shape}")
print(f"Pitutary -> {pitutary_dataset.shape}")
print(f"Meningioma -> {meningioma_dataset.shape}")

# Label generation
# Glioma - 1, Pitutary - 2, Meningioma - 3, No Tumour - 0

glioma_labels = tf.ones(shape=(glioma_dataset.shape[0],1))
meningioma_labels = 2*tf.ones(shape=(meningioma_dataset.shape[0],1))
pitutary_labels = 3*tf.ones(shape=(pitutary_dataset.shape[0],1))
notumour_labels = 0*tf.ones(shape=(notumour_dataset.shape[0],1))

# print(glioma_labels.shape, meningioma_labels.shape, pitutary_labels.shape, notumour_labels.shape)

# Glioma - 1st 1621 images
# Meningioma - next 1645 images
# Pitutary - next 1757 images
# notumour - last 2000 images

data = np.concatenate([glioma_dataset, meningioma_dataset, pitutary_dataset, notumour_dataset], axis=0)
labels = np.concatenate([glioma_labels, meningioma_labels, pitutary_labels, notumour_labels], axis=0)
print(data.shape)
# print(glioma_dataset[0][:,:,0])

# this data is used for further processing i.e. feature engineering and machine learning.
np.save("Data.npy", data)
np.save("Labels.npy",labels)



