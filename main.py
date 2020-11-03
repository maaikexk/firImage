import bow
import evaluate
import load_data
import cnn
import numpy as np


file_names = load_data.load_file_names()[:100] # TODO: Limit to only first 10 images
sim_matrix = load_data.load_similarity_matrix()
descriptors = load_data.load_descriptors(file_names, num_keypoints=200)
indices_train, indices_test, descriptors_train, descriptors_test = load_data.split_train_test(descriptors)


def bow_ranking():
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


def cnn_ranking():

    imgs = cnn.preprocess(file_names)
    imgs = [img[0] for img in imgs]
    # triplets = cnn.sampleTriplets(temp_img, indices_train, sim_matrix, 2)
    base_model = cnn.model_base(imgs[0].shape)
    # head_model.summary()
    triplet_model = cnn.model_head(base_model, cnn.triplet_loss, imgs[0].shape)

    # sample_triplets = cnn.sample_triplets(temp_img, indices_train, sim_matrix)
    triplet_model.fit_generator(cnn.sample_triplets(imgs, indices_train, sim_matrix), steps_per_epoch=10, epochs=100)
    triplet_model.save("triplet.h5")

    model_embeddings_test = triplet_model.layers[3].predict(np.array([imgs[x] for x in indices_test]))
    model_embeddings_train = triplet_model.layers[3].predict(np.array([imgs[x] for x in indices_train]))

    rankings_cnn = [cnn.get_ranking(model_embeddings_train, model_embedding_test) for model_embedding_test in model_embeddings_test]

    cnn_retrieved_gt = [evaluate.map_indices(indices_train, retrieved_images)
                        for retrieved_images in rankings_cnn]
    cnn_relevant_gt = evaluate.all_relevant_images(sim_matrix, indices_test, cnn_retrieved_gt)

    x = evaluate.mean_average_precision(cnn_relevant_gt, cnn_retrieved_gt)
    print(x)


bow_ranking()
cnn_ranking()