def map_indices(indices_gt, retrieved_images):
    return [indices_gt[i] for i in retrieved_images]


def relevant_images(sim_matrix, target_index, retrieved_indices):
    return [r for r in retrieved_indices if sim_matrix[r, target_index]]


def all_relevant_images(sim_matrix, indices_test, all_retrieved_images_gt):
    return [relevant_images(sim_matrix, indices_test[i], all_retrieved_images_gt[i]) for i in range(len(indices_test))]


def precision_at_k(relevant, retrieved, k):
    # Assuming at least one retrieved document
    top_k = set(retrieved[:k])
    return len(set(relevant) & top_k) / len(top_k)


def all_precision_at_k(all_relevant, all_retrieved, k):
    return [precision_at_k(all_relevant[i], all_retrieved[i], k) for i in range(len(all_relevant))]


def average_precision(relevant, retrieved):
    return sum([precision_at_k(relevant, retrieved, k) for k in range(1, len(retrieved) + 1)]) / len(retrieved)


def mean_average_precision(all_relevant, all_retrieved):
    return sum([average_precision(all_relevant[i], all_retrieved[i])
                for i in range(len(all_retrieved))]) / len(all_retrieved)
