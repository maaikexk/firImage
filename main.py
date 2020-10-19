import bow
import evaluate
import load_data


file_names = load_data.load_file_names()[:100]  # TODO: Limit to only first 10 images
sim_matrix = load_data.load_similarity_matrix()
descriptors = load_data.load_descriptors(file_names, num_keypoints=200)
indices_train, indices_test, descriptors_train, descriptors_test = load_data.split_train_test(descriptors)
all_descriptors_train = load_data.merge_descriptors(descriptors_train)

centroids = bow.clustering(all_descriptors_train, num_clusters=100)
bows_train = bow.make_bow(descriptors_train, centroids)
bows_test = bow.make_bow(descriptors_test, centroids)

all_retrieved = [bow.retrieve_images(bows_train, bow_test) for bow_test in bows_test]
all_retrieved_gt = [evaluate.map_indices(indices_train, retrieved_images)
                    for retrieved_images in all_retrieved]
all_relevant_gt = evaluate.all_relevant_images(sim_matrix, indices_test, all_retrieved_gt)

x = evaluate.mean_average_precision(all_relevant_gt, all_retrieved_gt)
print(x)
