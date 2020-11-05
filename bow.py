import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance
from skimage.color import rgb2gray
from skimage.feature import ORB
from sklearn.cluster import KMeans


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


def clustering(all_descriptors, num_clusters=100):
    clusters = KMeans(n_clusters=num_clusters, random_state=42, n_init=1).fit(all_descriptors)
    centroids = clusters.cluster_centers_
    return centroids


def bag_of_words(all_descriptors, centroids):
    bow_vector = np.zeros(len(centroids))
    for descriptor in all_descriptors:
        distances = [distance.euclidean(descriptor, centroid) for centroid in centroids]
        index = distances.index(min(distances))
        bow_vector[index] += 1
    return bow_vector


def make_bow(ind_descriptors, centroids):
    return np.array([bag_of_words(d, centroids) for d in ind_descriptors])


def retrieve_images(bows_train, bow_test, desired_results=10, use_euclidean=True):
    if use_euclidean:
        distances = [distance.euclidean(bow_train, bow_test) for bow_train in bows_train]
    else:
        distances = [distance.cosine(bow_train, bow_test) for bow_train in bows_train]
    return np.argsort(distances)[:desired_results]
