import itertools

import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import squareform
from skimage.color import rgb2gray
from skimage.feature import ORB
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


def load_descriptors(file_names, num_keypoints=200):
    # Load images
    descriptor_extractor = ORB(n_keypoints=num_keypoints)
    descriptors = []
    for im_path in file_names:
        img = plt.imread("data/" + im_path)
        img = rgb2gray(img)
        descriptor_extractor.detect_and_extract(img)
        descriptors.append(descriptor_extractor.descriptors)
    return np.array(descriptors)


def split_train_test(descriptors, test_size=0.2):
    indices = list(range(len(descriptors)))
    return train_test_split(indices, descriptors, test_size=test_size, random_state=42)


def merge_descriptors(descriptors):
    return np.array(list(itertools.chain(*descriptors)))
