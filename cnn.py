import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import numpy as np
# import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage import transform
import random

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *


def preprocess(file_names):

    images = []

    for im_path in file_names:
        img = load_img("data/" + im_path, color_mode="grayscale")
        # img = rgb2gray(img)
        img = img_to_array(img).astype("float64")
        img = transform.resize(img, (24, 24))
        img *= 1. / 255
        img = np.expand_dims(img, axis=0)

        images.append(img)

    return np.array(images)


def sample_triplets(images, indices_train, sim_matrix, sample_size=64):
    len_images = len(indices_train)

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


def triplet_loss(y_true, y_pred):

    alpha = 1.

    anchor_out = y_pred[:, 0:100]
    positive_out = y_pred[:, 100:200]
    negative_out = y_pred[:, 200:300]

    pos_dist = K.sum(K.abs(anchor_out - positive_out), axis=1)
    neg_dist = K.sum(K.abs(anchor_out - negative_out), axis=1)

    probs = K.softmax([pos_dist, neg_dist], axis=0)

    return K.mean(K.abs(probs[0]) + K.abs(alpha - probs[1]))

def model_base (shape):
    input_layer = Input(shape)
    layers = Conv2D(32, 3, activation="relu")(input_layer)
    layers = Conv2D(32, 3, activation="relu")(layers)
    layers = MaxPool2D(2)(layers)
    layers = Conv2D(64, 3, activation="relu")(layers)
    layers = Conv2D(64, 3, activation="relu")(layers)
    layers = MaxPool2D(2)(layers)
    layers = Conv2D(128, 3, activation="relu")(layers)
    layers = Flatten()(layers)
    layers = Dense(100, activation="relu")(layers)
    model = Model(input_layer, layers)
    model.summary()
    return model

def model_head(model, loss_function, shape):
    triplet_model_a = Input(shape)
    triplet_model_n = Input(shape)
    triplet_model_p = Input(shape)

    triplet_model_out = Concatenate()([model(triplet_model_a), model(triplet_model_p), model(triplet_model_n)])
    triplet_model = Model([triplet_model_a, triplet_model_p, triplet_model_n], triplet_model_out)
    triplet_model.compile(loss=loss_function, optimizer="adam")

    return triplet_model
