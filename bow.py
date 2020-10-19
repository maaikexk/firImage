import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans


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


def retrieve_images(bows_train, bow_test, desired_results=10):
    distances = [distance.euclidean(bow_train, bow_test) for bow_train in bows_train]
    return np.argsort(distances)[:desired_results]
