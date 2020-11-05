import matplotlib.pyplot as plt


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


def mean_precision_at_k(all_relevant, all_retrieved, k):
    return sum(all_precision_at_k(all_relevant, all_retrieved, k)) / len(all_retrieved)


def average_precision(relevant, retrieved):
    return sum([precision_at_k(relevant, retrieved, k) for k in range(1, len(retrieved) + 1)]) / len(retrieved)


def all_average_precision(all_relevant, all_retrieved):
    return [average_precision(all_relevant[i], all_retrieved[i]) for i in range(len(all_retrieved))]


def mean_average_precision(all_relevant, all_retrieved):
    return sum(all_average_precision(all_relevant, all_retrieved)) / len(all_retrieved)


def plot_distribution_precision_at_ten(all_relevant, all_retrieved, file_name):
    pat10s = all_precision_at_k(all_relevant, all_retrieved, 10)
    precisions = [i / 10 for i in range(11)]
    bars = [0. for _ in range(11)]
    for pat10 in pat10s:
        bars[int(pat10 * 10)] += 1
    bars = [bar / len(all_retrieved) for bar in bars]
    plt.figure()
    plt.bar(precisions, bars, width=0.08, color="#2979ff")
    plt.xlabel("precision at 10")
    plt.ylabel("fraction")
    plt.savefig(file_name, dpi=300)


def plot_distribution_average_precision(all_relevant, all_retrieved, file_name):
    aps = sorted(all_average_precision(all_relevant, all_retrieved), reverse=True)
    percentiles = [i * 100 / (len(aps) - 1) for i in range(len(aps))]
    plt.figure()
    plt.plot(percentiles, aps, linewidth=2.5, color="#2979ff")
    plt.xlabel("percentile")
    plt.ylabel("average precision")
    plt.savefig(file_name, dpi=300)
