import pytest
import torch
import numpy as np

from tinytopics.utils import set_random_seed, generate_synthetic_data
from tinytopics.data import NumpyDiskDataset
from tinytopics.fit import fit_model

# Test data dimensions
N_DOCS = 50
N_TERMS = 100
N_TOPICS = 5
N_EPOCHS = 3


@pytest.fixture
def sample_data(tmp_path):
    """Generate sample data and return both tensor and file path."""
    set_random_seed(42)
    X, _, _ = generate_synthetic_data(n=N_DOCS, m=N_TERMS, k=N_TOPICS)

    file_path = tmp_path / "test_data.npy"
    np.save(file_path, X.cpu().numpy())

    return X, file_path


def test_disk_dataset_reproducibility(sample_data):
    """Test that training with same disk dataset and seed gives identical results."""
    X, file_path = sample_data
    dataset = NumpyDiskDataset(file_path)

    set_random_seed(42)
    model1, losses1 = fit_model(dataset, k=N_TOPICS, num_epochs=N_EPOCHS)

    set_random_seed(42)
    model2, losses2 = fit_model(dataset, k=N_TOPICS, num_epochs=N_EPOCHS)

    assert np.allclose(losses1, losses2)
    assert torch.allclose(model1.get_normalized_L(), model2.get_normalized_L())
    assert torch.allclose(model1.get_normalized_F(), model2.get_normalized_F())


def test_disk_dataset_different_seeds(sample_data):
    """Test that training with same disk dataset but different seeds gives different results."""
    _, file_path = sample_data
    dataset = NumpyDiskDataset(file_path)

    set_random_seed(42)
    model1, losses1 = fit_model(dataset, k=N_TOPICS, num_epochs=N_EPOCHS)

    set_random_seed(43)
    model2, losses2 = fit_model(dataset, k=N_TOPICS, num_epochs=N_EPOCHS)

    assert not np.allclose(losses1, losses2)
    assert not torch.allclose(model1.get_normalized_L(), model2.get_normalized_L())
    assert not torch.allclose(model1.get_normalized_F(), model2.get_normalized_F())


def test_tensor_vs_disk_same_seed(sample_data):
    """Test that training with tensor and disk dataset gives identical results with same seed."""
    X, file_path = sample_data
    dataset = NumpyDiskDataset(file_path)

    set_random_seed(42)
    model1, losses1 = fit_model(X, k=N_TOPICS, num_epochs=N_EPOCHS)

    set_random_seed(42)
    model2, losses2 = fit_model(dataset, k=N_TOPICS, num_epochs=N_EPOCHS)

    assert np.allclose(losses1, losses2)
    assert torch.allclose(model1.get_normalized_L(), model2.get_normalized_L())
    assert torch.allclose(model1.get_normalized_F(), model2.get_normalized_F())


def test_tensor_vs_disk_different_seeds(sample_data):
    """Test that training with tensor and disk dataset gives different results with different seeds."""
    X, file_path = sample_data
    dataset = NumpyDiskDataset(file_path)

    set_random_seed(42)
    model1, losses1 = fit_model(X, k=N_TOPICS, num_epochs=N_EPOCHS)

    set_random_seed(43)
    model2, losses2 = fit_model(dataset, k=N_TOPICS, num_epochs=N_EPOCHS)

    assert not np.allclose(losses1, losses2)
    assert not torch.allclose(model1.get_normalized_L(), model2.get_normalized_L())
    assert not torch.allclose(model1.get_normalized_F(), model2.get_normalized_F())
