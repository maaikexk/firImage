import h5py
import itertools
import json
import numpy as np
from scipy.spatial.distance import squareform
from sklearn.model_selection import train_test_split


def load_file_names():
    # Load image file names
    with open("data/W17/W17.json") as f:
        data = json.load(f)
        im_paths = data["im_paths"]
    return im_paths


def load_similarity_matrix():
    # Load similarity matrix
    with h5py.File("data/W17/W17_similarity.h5", "r") as f:
        gt_labels = squareform(f["sim"][:].flatten())
    return gt_labels


def split_train_test(data_size, test_fraction=0.2):
    indices = list(range(data_size))
    return train_test_split(indices, test_size=test_fraction, random_state=42)


def apply_split(array, train_indices, test_indices):
    array_train = [array[i] for i in train_indices]
    array_test = [array[i] for i in test_indices]
    return np.array(array_train), np.array(array_test)


def merge_descriptors(descriptors):
    return np.array(list(itertools.chain(*descriptors)))
