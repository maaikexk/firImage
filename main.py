import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import squareform
from skimage.color import rgb2gray
from skimage.feature import ORB

# Load image file names
with open("data/W17/W17.json") as f:
    data = json.load(f)
    im_paths = data["im_paths"]
    poses = np.array(data["poses"])

# Load similarity matrix
with h5py.File("data/W17/W17_similarity.h5", "r") as f:
    gt_labels = squareform(f["sim"][:].flatten())

im_paths = im_paths[:10]  # TODO: Limit to only first 10 images

# Load images
descriptor_extractor = ORB(n_keypoints=200)
descriptors = []
for im_path in im_paths:
    img = plt.imread("data/" + im_path)
    img = rgb2gray(img)
    descriptor_extractor.detect_and_extract(img)
    descriptors.extend(descriptor_extractor.descriptors)
descriptors = np.array(descriptors)
