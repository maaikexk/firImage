import h5py
import numpy as np

import bow
import cnn
import evaluate
import load_data


def bow_ranking(file_names, indices_train, indices_test, sim_matrix, compute_descriptors=False):
    if compute_descriptors:
        descriptors = bow.load_descriptors(file_names, num_keypoints=200)
        with h5py.File("descriptors.h5", "w") as file:
            file.create_dataset("descriptors", data=descriptors)
    else:
        with h5py.File("descriptors.h5", "r") as file:
            descriptors = np.array(file.get("descriptors"))

    descriptors_train, descriptors_test = load_data.apply_split(descriptors, indices_train, indices_test)
    all_descriptors_train = load_data.merge_descriptors(descriptors_train)

    centroids = bow.clustering(all_descriptors_train, num_clusters=100)
    bows_train = bow.make_bow(descriptors_train, centroids)
    bows_test = bow.make_bow(descriptors_test, centroids)

    all_retrieved = [bow.retrieve_images(bows_train, bow_test) for bow_test in bows_test]
    all_retrieved_gt = [evaluate.map_indices(indices_train, retrieved_images) for retrieved_images in all_retrieved]
    all_relevant_gt = evaluate.all_relevant_images(sim_matrix, indices_test, all_retrieved_gt)

    print("----- BOW RESULTS -----")
    print("- MAP:  ", evaluate.mean_average_precision(all_relevant_gt, all_retrieved_gt))
    print("- MP@10:", evaluate.mean_precision_at_k(all_relevant_gt, all_retrieved_gt, 10))
    print()


def cnn_ranking(file_names, indices_train, indices_test, sim_matrix,
                image_size, epochs, steps_per_epoch, train_cnn=False):
    images = cnn.preprocess(file_names, image_size)
    images = [img[0] for img in images]

    base_model = cnn.model_base(images[0].shape)
    triplet_model = cnn.model_head(base_model, cnn.triplet_loss, images[0].shape)

    if train_cnn:
        triplet_model.fit_generator(cnn.sample_triplets(images, indices_train, sim_matrix),
                                    epochs=epochs, steps_per_epoch=steps_per_epoch)
        triplet_model.save("triplet.h5")
    else:
        triplet_model.load_weights("triplet.h5")

    embeddings_train = triplet_model.layers[3].predict(np.array([images[x] for x in indices_train]))
    embeddings_test = triplet_model.layers[3].predict(np.array([images[x] for x in indices_test]))

    rankings_cnn = [cnn.get_ranking(embeddings_train, embedding_test) for embedding_test in embeddings_test]
    cnn_retrieved_gt = [evaluate.map_indices(indices_train, retrieved_images) for retrieved_images in rankings_cnn]
    cnn_relevant_gt = evaluate.all_relevant_images(sim_matrix, indices_test, cnn_retrieved_gt)


    print("----- TCNN RESULTS -----")
    print("- MAP:  ", evaluate.mean_average_precision(cnn_relevant_gt, cnn_retrieved_gt))
    print("- MP@10:", evaluate.mean_precision_at_k(cnn_relevant_gt, cnn_retrieved_gt, 10))
    print()


def print_statistics(sim_matrix):
    sim_totals = [sum(row) - 1 for row in sim_matrix]
    print("Total similar images:", (sum(sim_totals)) / 2)
    print("Mean similar images:", sum(sim_totals) / len(sim_totals))
    print("Std similar images:", np.std(sim_totals))
    print("Min similar images:", min(sim_totals))
    print("Max similar images:", max(sim_totals))


def main(file_limit=10948):
    file_names = load_data.load_file_names()[:file_limit]
    indices_train, indices_test = load_data.split_train_test(file_limit)
    sim_matrix = load_data.load_similarity_matrix()
    bow_ranking(file_names, indices_train, indices_test, sim_matrix)
    cnn_ranking(file_names, indices_train, indices_test, sim_matrix,
                image_size=(24, 24), epochs=250, steps_per_epoch=100)


main()
