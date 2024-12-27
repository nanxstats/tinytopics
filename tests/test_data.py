import pytest
import torch
import numpy as np

from tinytopics.data import NumpyDiskDataset


def test_numpy_disk_dataset_from_array():
    """Test NumpyDiskDataset with direct numpy array input."""
    data = np.random.rand(10, 5).astype(np.float32)

    dataset = NumpyDiskDataset(data)

    # Test basic properties
    assert len(dataset) == 10
    assert dataset.num_terms == 5
    assert dataset.shape == (10, 5)

    # Test data access
    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, torch.Tensor)
        assert item.shape == (5,)
        assert torch.allclose(item, torch.tensor(data[i], dtype=torch.float32))


def test_numpy_disk_dataset_from_file(tmp_path):
    """Test NumpyDiskDataset with .npy file input."""
    data = np.random.rand(10, 5).astype(np.float32)
    file_path = tmp_path / "test_data.npy"
    np.save(file_path, data)

    dataset = NumpyDiskDataset(file_path)

    # Test basic properties
    assert len(dataset) == 10
    assert dataset.num_terms == 5
    assert dataset.shape == (10, 5)

    # Test data access
    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, torch.Tensor)
        assert item.shape == (5,)
        assert torch.allclose(item, torch.tensor(data[i], dtype=torch.float32))


def test_numpy_disk_dataset_with_indices():
    """Test NumpyDiskDataset with custom indices."""
    data = np.random.rand(10, 5).astype(np.float32)
    indices = [3, 1, 4]

    dataset = NumpyDiskDataset(data, indices=indices)

    # Test basic properties
    assert len(dataset) == len(indices)
    assert dataset.num_terms == 5
    assert dataset.shape == (10, 5)

    # Test data access
    for i, orig_idx in enumerate(indices):
        item = dataset[i]
        assert isinstance(item, torch.Tensor)
        assert item.shape == (5,)
        assert torch.allclose(item, torch.tensor(data[orig_idx], dtype=torch.float32))


def test_numpy_disk_dataset_file_not_found():
    """Test NumpyDiskDataset with non-existent file."""
    with pytest.raises(FileNotFoundError):
        NumpyDiskDataset("non_existent_file.npy")


def test_numpy_disk_dataset_memory_efficiency(tmp_path):
    """Test that NumpyDiskDataset uses memory mapping efficiently."""
    shape = (1000, 500)  # 500K elements
    data = np.random.rand(*shape).astype(np.float32)
    file_path = tmp_path / "large_data.npy"
    np.save(file_path, data)

    dataset = NumpyDiskDataset(file_path)

    # Access data in random order
    indices = np.random.permutation(shape[0])[:100]  # Sample 100 random rows
    for idx in indices:
        item = dataset[idx]
        assert torch.allclose(item, torch.tensor(data[idx], dtype=torch.float32))

    # Memory mapping should be initialized only after first access
    assert dataset.mmap_data is not None
