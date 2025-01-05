import pytest
import torch
import numpy as np

from tinytopics.data import NumpyDiskDataset, TorchDiskDataset


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


def test_torch_disk_dataset_from_file(tmp_path):
    """Test TorchDiskDataset with .pt file input."""
    data = torch.rand(10, 5, dtype=torch.float32)
    file_path = tmp_path / "test_data.pt"
    torch.save(data, file_path)

    dataset = TorchDiskDataset(file_path)

    # Test basic properties
    assert len(dataset) == 10
    assert dataset.num_terms == 5
    assert dataset.shape == (10, 5)

    # Test data access
    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, torch.Tensor)
        assert item.shape == (5,)
        assert torch.allclose(item, data[i])
        assert item.device == torch.device("cpu")


def test_torch_disk_dataset_with_indices(tmp_path):
    """Test TorchDiskDataset with custom indices."""
    data = torch.rand(10, 5, dtype=torch.float32)
    file_path = tmp_path / "test_data.pt"
    torch.save(data, file_path)
    indices = [3, 1, 4]

    dataset = TorchDiskDataset(file_path, indices=indices)

    # Test basic properties
    assert len(dataset) == len(indices)
    assert dataset.num_terms == 5
    assert dataset.shape == (10, 5)

    # Test data access
    for i, orig_idx in enumerate(indices):
        item = dataset[i]
        assert isinstance(item, torch.Tensor)
        assert item.shape == (5,)
        assert torch.allclose(item, data[orig_idx])


def test_torch_disk_dataset_with_device(tmp_path):
    """Test TorchDiskDataset with specific device."""
    data = torch.rand(10, 5, dtype=torch.float32)
    file_path = tmp_path / "test_data.pt"
    torch.save(data, file_path)

    device = torch.device("cpu")  # Use CPU for testing
    dataset = TorchDiskDataset(file_path, device=device)

    # Test data access with device
    for i in range(len(dataset)):
        item = dataset[i]
        assert item.device == device
        assert torch.allclose(item, data[i])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not available")
def test_torch_disk_dataset_cuda_device(tmp_path):
    """Test TorchDiskDataset with CUDA device."""
    data = torch.rand(10, 5, dtype=torch.float32)
    file_path = tmp_path / "test_data.pt"
    torch.save(data, file_path)

    device = torch.device("cuda")
    dataset = TorchDiskDataset(file_path, device=device)

    # Test data access with CUDA device
    for i in range(len(dataset)):
        item = dataset[i]
        assert item.device == device
        assert torch.allclose(item.cpu(), data[i])


def test_torch_disk_dataset_file_not_found():
    """Test TorchDiskDataset with non-existent file."""
    with pytest.raises(FileNotFoundError):
        TorchDiskDataset("non_existent_file.pt")


def test_torch_disk_dataset_invalid_content(tmp_path):
    """Test TorchDiskDataset with invalid file content."""
    file_path = tmp_path / "invalid_data.pt"
    invalid_data = {"not_a_tensor": 42}
    torch.save(invalid_data, file_path)

    with pytest.raises(ValueError, match="must contain a single tensor"):
        TorchDiskDataset(file_path)


def test_torch_disk_dataset_memory_efficiency(tmp_path):
    """Test that TorchDiskDataset uses memory mapping efficiently."""
    shape = (1000, 500)  # 500K elements
    data = torch.rand(*shape, dtype=torch.float32)
    file_path = tmp_path / "large_data.pt"
    torch.save(data, file_path)

    dataset = TorchDiskDataset(file_path)

    # Access data in random order
    indices = torch.randperm(shape[0])[:100]  # Sample 100 random rows
    for idx in indices:
        item = dataset[idx]
        assert torch.allclose(item, data[idx])

    # Memory mapping should be initialized only after first access
    if dataset.mmap_supported:
        assert dataset.mmap_data is not None
    else:
        assert hasattr(dataset, "data")
