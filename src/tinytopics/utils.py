from typing import Tuple
from collections.abc import Sequence, MutableMapping
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


def set_random_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility across Torch and NumPy.

    Args:
        seed (int): Random seed value.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_synthetic_data(
    n: int,
    m: int,
    k: int,
    avg_doc_length: int = 1000,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Generate synthetic document-term matrix for testing the model.

    Args:
        n (int): Number of documents.
        m (int): Number of terms (vocabulary size).
        k (int): Number of topics.
        avg_doc_length (int, optional): Average number of terms per document. Default is 1000.
        device (torch.device, optional): Device to place the output tensors on.

    Returns:
        (torch.Tensor): Document-term matrix.
        (np.ndarray): True document-topic distribution (L).
        (np.ndarray): True topic-term distribution (F).
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

    for i in tqdm(range(n), desc="Generating Documents"):
        # Compute document-specific term distribution by mixing topic-term distributions
        doc_term_probs = true_L[i] @ true_F  # shape (m,)
        # Single multinomial draw for all terms in the document
        X[i, :] = np.random.multinomial(doc_lengths[i], doc_term_probs)

    return torch.tensor(X, device=device, dtype=torch.float32), true_L, true_F


def align_topics(true_F: np.ndarray, learned_F: np.ndarray) -> np.ndarray:
    """
    Align learned topics with true topics for visualization,
    using cosine similarity and linear sum assignment.

    Args:
        true_F (np.ndarray): Ground truth topic-term matrix.
        learned_F (np.ndarray): Learned topic-term matrix.

    Returns:
        (np.ndarray): Permutation of learned topics aligned with true topics.
    """

    def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    true_F_norm = normalize_matrix(true_F)
    learned_F_norm = normalize_matrix(learned_F)

    similarity_matrix = np.dot(true_F_norm, learned_F_norm.T)
    cost_matrix = -similarity_matrix
    _, col_ind = linear_sum_assignment(cost_matrix)

    return col_ind


def sort_documents(L_matrix: np.ndarray) -> Sequence[int]:
    """
    Sort documents grouped by dominant topics for visualization.

    Args:
        L_matrix (np.ndarray): Document-topic distribution matrix.

    Returns:
        Indices of documents sorted by dominant topics.
    """
    n, k = L_matrix.shape
    L_normalized = L_matrix / L_matrix.sum(axis=1, keepdims=True)

    def get_document_info() -> Sequence[Tuple[int, int, float]]:
        dominant_topics = np.argmax(L_normalized, axis=1)
        dominant_props = L_normalized[np.arange(n), dominant_topics]
        return list(zip(range(n), dominant_topics, dominant_props))

    def group_by_topic(
        doc_info: Sequence[Tuple[int, int, float]],
    ) -> MutableMapping[int, list]:
        groups: MutableMapping[int, list] = defaultdict(list)
        for idx, topic, prop in doc_info:
            groups[topic].append((idx, prop))
        return groups

    def sort_topic_groups(grouped_docs: MutableMapping[int, list]) -> Sequence[int]:
        sorted_indices = []
        for topic in range(k):
            docs_in_topic = grouped_docs.get(topic, [])
            docs_sorted = sorted(docs_in_topic, key=lambda x: x[1], reverse=True)
            sorted_indices.extend(idx for idx, _ in docs_sorted)
        return sorted_indices

    doc_info = get_document_info()
    grouped_docs = group_by_topic(doc_info)
    return sort_topic_groups(grouped_docs)


class NumpyDiskDataset(Dataset):
    """
    A PyTorch Dataset class for loading document-term matrices from disk.

    The dataset can be initialized with either a path to a `.npy` file or
    a NumPy array. When a file path is provided, the data is accessed
    lazily using memory mapping, which is useful for handling large datasets
    that do not fit entirely in (CPU) memory.
    """

    def __init__(
        self, data: str | Path | np.ndarray, indices: Sequence[int] | None = None
    ) -> None:
        """
        Args:
            data: Either path to `.npy` file (str or Path) or numpy array.
            indices: Optional sequence of indices to use as valid indices.
        """
        if isinstance(data, (str, Path)):
            data_path = Path(data)
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            # Get shape without loading full array
            self.shape: tuple[int, int] = tuple(np.load(data_path, mmap_mode="r").shape)
            self.data_path: Path = data_path
            self.mmap_data: np.ndarray | None = None
        else:
            self.shape: tuple[int, int] = data.shape
            self.data_path: None = None
            self.data: np.ndarray = data

        self.indices: Sequence[int] = indices or range(self.shape[0])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        real_idx = self.indices[idx]

        if self.data_path is not None:
            # Load mmap data lazily
            if self.mmap_data is None:
                self.mmap_data = np.load(self.data_path, mmap_mode="r")
            return torch.tensor(self.mmap_data[real_idx], dtype=torch.float32)
        else:
            return torch.tensor(self.data[real_idx], dtype=torch.float32)

    @property
    def num_terms(self) -> int:
        """Return vocabulary size (number of columns)."""
        return self.shape[1]
