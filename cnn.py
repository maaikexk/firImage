import numpy as np
import random
from scipy.spatial import distance
from skimage import transform
from tensorflow.keras import backend as k
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def preprocess(file_names, image_size):
    images = []
    for im_path in file_names:
        img = load_img("data/" + im_path, color_mode="grayscale")
        img = img_to_array(img).astype("float64")
        img = transform.resize(img, image_size)
        img *= 1. / 255
        img = np.expand_dims(img, axis=0)
        images.append(img)
    return np.array(images)


def sample_triplets(images, indices_train, sim_matrix, sample_size=64):
    while True:  # Keep yielding samples forever (as long as they are requested)
        a, p, n = [], [], []

        for _ in range(sample_size):  # Generate sample_size samples
            while True:  # Attempt to generate a sample until succeeded
                index_target = indices_train[random.randint(0, len(indices_train) - 1)]
                indices_pos = [i for i in indices_train if sim_matrix[index_target, i] and i != index_target]
                indices_neg = [i for i in indices_train if not sim_matrix[index_target, i]]

                if len(indices_pos) > 0 and len(indices_neg) > 0:
                    index_pos = indices_pos[random.randint(0, len(indices_pos) - 1)]
                    index_neg = indices_neg[random.randint(0, len(indices_neg) - 1)]
                    break  # Sample is successfully generated, exit while loop

            a.append(images[index_target])
            p.append(images[index_pos])
            n.append(images[index_neg])

        yield [np.array(a), np.array(p), np.array(n)], np.zeros((sample_size, 1)).astype("float32")


def triplet_loss(y_true, y_pred):
    alpha = 1.

    anchor_out, positive_out, negative_out = y_pred[:, 0:100], y_pred[:, 100:200], y_pred[:, 200:300]
    pos_dist = k.sum(k.abs(anchor_out - positive_out), axis=1)
    neg_dist = k.sum(k.abs(anchor_out - negative_out), axis=1)
    probs = k.softmax([pos_dist, neg_dist], axis=0)

    return k.mean(k.abs(probs[0]) + k.abs(alpha - probs[1]))


def model_base(shape):
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
    return model


def model_head(model, loss_function, shape):
    triplet_model_a, triplet_model_p, triplet_model_n = Input(shape), Input(shape), Input(shape)
    triplet_model_out = Concatenate()([model(triplet_model_a), model(triplet_model_p), model(triplet_model_n)])
    triplet_model = Model([triplet_model_a, triplet_model_p, triplet_model_n], triplet_model_out)
    triplet_model.compile(loss=loss_function, optimizer="adam")

    return triplet_model


def get_ranking(xs_train, x_test, desired_results=10):
    distances = [distance.euclidean(x_train, x_test) for x_train in xs_train]
    return np.argsort(distances)[:desired_results]
