import torch
import numpy as np
from scipy.optimize import linear_sum_assignment


def set_random_seed(seed):
    """
    Set the random number seed for reproducibility.

    Args:
        seed (int): Seed value.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_synthetic_data(n, m, k, avg_doc_length=1000, device=None):
    """
    Generate synthetic document-term matrix.

    Args:
        n (int): Number of documents.
        m (int): Number of terms (vocabulary size).
        k (int): Number of topics.
        avg_doc_length (int): Average document length.
        device (torch.device): Device to place the tensor on.

    Returns:
        X (torch.Tensor): Document-term matrix.
        true_L (np.ndarray): True document-topic matrix.
        true_F (np.ndarray): True topic-term matrix.
    """
    device = device or torch.device("cpu")

    # True document-topic matrix L (n x k)
    true_L = np.random.dirichlet(alpha=np.ones(k), size=n)  # shape (n, k)

    # True topic-term matrix F (k x m)
    true_F = np.random.dirichlet(alpha=np.ones(m), size=k)  # shape (k, m)

    # Simulate variable document lengths
    doc_lengths = np.random.poisson(lam=avg_doc_length, size=n)  # shape (n,)

    # Initialize document-term matrix X
    X = np.zeros((n, m), dtype=np.int32)

    # For each document i
    for i in range(n):
        # Sample topic counts for document i
        topic_probs = true_L[i]
        topic_counts = np.random.multinomial(doc_lengths[i], topic_probs)

        # Initialize term counts for document i
        term_counts = np.zeros(m, dtype=np.int32)

        # For each topic j
        for j in range(k):
            if topic_counts[j] > 0:
                # Sample term counts for topic j
                term_probs = true_F[j]
                term_counts_j = np.random.multinomial(topic_counts[j], term_probs)
                # Add term counts to document i
                term_counts += term_counts_j

        # Assign term counts to X[i,:]
        X[i, :] = term_counts

    return torch.tensor(X, device=device, dtype=torch.float32), true_L, true_F


def align_topics(true_F, learned_F):
    # Normalize topic-term distributions
    true_F_norm = true_F / np.linalg.norm(true_F, axis=1, keepdims=True)
    learned_F_norm = learned_F / np.linalg.norm(learned_F, axis=1, keepdims=True)

    # Compute the cosine similarity matrix
    similarity_matrix = np.dot(true_F_norm, learned_F_norm.T)
    # Compute the cost matrix for assignment (use negative similarity)
    cost_matrix = -similarity_matrix
    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    return col_ind


def sort_documents(L_matrix):
    """
    Sort documents grouped by dominant topics for visualization.

    Args:
        L_matrix (np.ndarray): Document-topic distribution matrix.

    Returns:
        sorted_indices (list): List of document indices sorted for plotting.
    """
    n, k = L_matrix.shape
    # Normalize L
    L_normalized = L_matrix / L_matrix.sum(axis=1, keepdims=True)

    # Determine dominant topics and their proportions
    dominant_topics = np.argmax(L_normalized, axis=1)
    dominant_props = L_normalized[np.arange(n), dominant_topics]

    # Combine indices, dominant topics, and proportions
    doc_info = list(zip(np.arange(n), dominant_topics, dominant_props))

    # Group documents by dominant topic
    from collections import defaultdict

    grouped_docs = defaultdict(list)
    for idx, topic, prop in doc_info:
        grouped_docs[topic].append((idx, prop))

    # Sort documents within each group by proportion of the dominant topic
    sorted_indices = []
    for topic in range(k):
        docs_in_topic = grouped_docs.get(topic, [])
        # Sort by proportion in descending order
        docs_sorted = sorted(docs_in_topic, key=lambda x: x[1], reverse=True)
        sorted_indices.extend([idx for idx, _ in docs_sorted])

    return sorted_indices
