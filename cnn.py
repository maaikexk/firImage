import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import numpy as np
# import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage import transform
import random


def preprocess(file_names):

    images = []

    for im_path in file_names:
        img = load_img("data/" + im_path)
        # img = rgb2gray(img)
        img = img_to_array(img).astype("float64")
        img = transform.resize(img, (224, 224))
        img *= 1. / 255
        img = np.expand_dims(img, axis=0)

        images.append(img)

    return np.array(images)


def sampleTriplets (images, indices_train, sim_matrix, sample_size):
    len_images = len(images)

    triplets = []
    for i in range(sample_size):
        while True:
            index_target = indices_train[random.randint(0, len_images - 1)]
            indices_pos = [index for index in indices_train if sim_matrix[index_target, index] and index != index_target]
            indices_neg = [index for index in indices_train if not sim_matrix[index_target, index]]

            if len(indices_pos) != 0 and len(indices_neg) != 0:
                break

        index_pos = indices_pos[random.randint(0, len(indices_pos)-1)]
        index_neg = indices_neg[random.randint(0, len(indices_neg)-1)]

        triplets.append((images[index_target], images[index_pos], images[index_neg]))

    return triplets


