import matplotlib.pyplot as plt
import numpy as np


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


def plot_distribution_precision_at_ten(all_relevant_bow_euclidean, all_retrieved_bow_euclidean,
                                       all_relevant_bow_cosine, all_retrieved_bow_cosine,
                                       all_relevant_tcnn, all_retrieved_tcnn):
    pat10s_bow_euclidean = all_precision_at_k(all_relevant_bow_euclidean, all_retrieved_bow_euclidean, 10)
    pat10s_bow_cosine = all_precision_at_k(all_relevant_bow_cosine, all_retrieved_bow_cosine, 10)
    pat10s_tcnn = all_precision_at_k(all_relevant_tcnn, all_retrieved_tcnn, 10)
    precisions = np.array([i / 10 for i in range(11)])
    bars_bow_euclidean = [0. for _ in range(11)]
    bars_bow_cosine = [0. for _ in range(11)]
    bars_tcnn = [0. for _ in range(11)]
    for pat10 in pat10s_bow_euclidean:
        bars_bow_euclidean[int(pat10 * 10)] += 1
    for pat10 in pat10s_bow_cosine:
        bars_bow_cosine[int(pat10 * 10)] += 1
    for pat10 in pat10s_tcnn:
        bars_tcnn[int(pat10 * 10)] += 1
    bars_bow_euclidean = [bar / len(all_retrieved_bow_euclidean) for bar in bars_bow_euclidean]
    bars_bow_cosine = [bar / len(all_retrieved_bow_cosine) for bar in bars_bow_cosine]
    bars_tcnn = [bar / len(all_retrieved_tcnn) for bar in bars_tcnn]
    plt.figure()
    plt.bar(precisions - 0.025, bars_bow_euclidean, width=0.025, color="#ff1744", label="BOW (Euclidean)")
    plt.bar(precisions, bars_bow_cosine, width=0.025, color="#00e676", label="BOW (Cosine)")
    plt.bar(precisions + 0.025, bars_tcnn, width=0.025, color="#2979ff", label="TCNN")
    plt.legend()
    plt.xlabel("precision at 10")
    plt.ylabel("fraction")
    plt.savefig("plot_pat10", dpi=300)


def plot_distribution_average_precision(all_relevant_bow_euclidean, all_retrieved_bow_euclidean,
                                        all_relevant_bow_cosine, all_retrieved_bow_cosine,
                                        all_relevant_tcnn, all_retrieved_tcnn):
    aps_bow_euclidean = sorted(all_average_precision(
        all_relevant_bow_euclidean, all_retrieved_bow_euclidean), reverse=True)
    aps_bow_cosine = sorted(all_average_precision(
        all_relevant_bow_cosine, all_retrieved_bow_cosine), reverse=True)
    aps_tcnn = sorted(all_average_precision(
        all_relevant_tcnn, all_retrieved_tcnn), reverse=True)
    percentiles = [i * 100 / (len(aps_tcnn) - 1) for i in range(len(aps_tcnn))]
    plt.figure()
    plt.plot(percentiles, aps_bow_euclidean, linewidth=2.5, color="#ff1744", label="BOW (Euclidean)")
    plt.plot(percentiles, aps_bow_cosine, linewidth=2.5, color="#00e676", label="BOW (Cosine)")
    plt.plot(percentiles, aps_tcnn, linewidth=2.5, color="#2979ff", label="TCNN")
    plt.legend()
    plt.xlabel("percentile")
    plt.ylabel("average precision")
    plt.savefig("plot_ap", dpi=300)
