from collections.abc import Sequence
from pathlib import Path

import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset


class IndexTrackingDataset(Dataset):
    """Dataset wrapper that tracks indices through shuffling"""

    def __init__(self, dataset: Dataset | Tensor) -> None:
        self.dataset = dataset
        self.shape: tuple[int, int] = (
            dataset.shape
            if hasattr(dataset, "shape")
            else (len(dataset), dataset[0].shape[0])
        )
        self.is_tensor: bool = isinstance(dataset, torch.Tensor)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.dataset[idx], torch.tensor(idx)


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
