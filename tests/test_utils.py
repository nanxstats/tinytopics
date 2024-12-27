import pytest
import torch
import numpy as np

from tinytopics.utils import (
    set_random_seed,
    generate_synthetic_data,
    align_topics,
    sort_documents,
)


def test_set_random_seed():
    """Test random seed setting for reproducibility."""
    seed = 42

    # Set seed and generate random numbers
    set_random_seed(seed)
    torch_rand1 = torch.rand(5)
    np_rand1 = np.random.rand(5)

    # Set same seed again and generate random numbers
    set_random_seed(seed)
    torch_rand2 = torch.rand(5)
    np_rand2 = np.random.rand(5)

    # Check if random numbers are identical
    assert torch.allclose(torch_rand1, torch_rand2)
    assert np.allclose(np_rand1, np_rand2)


@pytest.mark.parametrize(
    "n, m, k, avg_doc_length",
    [
        (10, 20, 3, 100),  # Small dataset
        (100, 200, 5, 1000),  # Medium dataset
    ],
)
def test_generate_synthetic_data(n, m, k, avg_doc_length):
    """Test synthetic data generation with different parameters."""
    set_random_seed(42)

    X, true_L, true_F = generate_synthetic_data(
        n=n, m=m, k=k, avg_doc_length=avg_doc_length
    )

    # Check shapes
    assert X.shape == (n, m)
    assert true_L.shape == (n, k)
    assert true_F.shape == (k, m)

    # Check if matrices contain valid probability distributions
    assert torch.all(X >= 0)
    assert np.allclose(true_L.sum(axis=1), 1.0)
    assert np.allclose(true_F.sum(axis=1), 1.0)


def test_generate_synthetic_data_reproducibility():
    """Test that synthetic data generation is reproducible with seeds."""
    n, m, k = 10, 20, 3
    avg_doc_length = 100

    # Generate with a seed
    set_random_seed(42)
    X1, L1, F1 = generate_synthetic_data(n=n, m=m, k=k, avg_doc_length=avg_doc_length)

    # Generate with the same seed
    set_random_seed(42)
    X2, L2, F2 = generate_synthetic_data(n=n, m=m, k=k, avg_doc_length=avg_doc_length)

    # Generate with a different seed
    set_random_seed(43)
    X3, L3, F3 = generate_synthetic_data(n=n, m=m, k=k, avg_doc_length=avg_doc_length)

    # Check that same seeds produce identical results
    assert torch.allclose(X1, X2)
    assert np.allclose(L1, L2)
    assert np.allclose(F1, F2)

    # Check that different seeds produce different results
    assert not torch.allclose(X1, X3)
    assert not np.allclose(L1, L3)
    assert not np.allclose(F1, F3)


def test_align_topics():
    """Test topic alignment functionality."""
    # Create synthetic topic matrices
    true_F = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])

    # Create learned matrix with shuffled topics
    learned_F = np.array([[0.1, 0.1, 0.8], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])

    alignment = align_topics(true_F, learned_F)

    # Check if alignment is valid permutation
    assert len(alignment) == len(true_F)
    assert len(set(alignment)) == len(alignment)
    assert all(0 <= i < len(true_F) for i in alignment)


def test_sort_documents():
    """Test document sorting by dominant topics."""
    # Create sample document-topic matrix
    L_matrix = np.array(
        [
            [0.8, 0.1, 0.1],  # Doc 0: Topic 0 dominant
            [0.1, 0.7, 0.2],  # Doc 1: Topic 1 dominant
            [0.2, 0.1, 0.7],  # Doc 2: Topic 2 dominant
            [0.7, 0.2, 0.1],  # Doc 3: Topic 0 dominant
        ]
    )

    sorted_indices = sort_documents(L_matrix)

    # Check if output is valid
    assert len(sorted_indices) == len(L_matrix)
    assert set(sorted_indices) == set(range(len(L_matrix)))

    # Check if documents are properly grouped by dominant topics
    dominant_topics = np.argmax(L_matrix[sorted_indices], axis=1)
    assert all(
        dominant_topics[i] <= dominant_topics[i + 1]
        for i in range(len(dominant_topics) - 1)
    )
