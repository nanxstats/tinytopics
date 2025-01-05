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
    A PyTorch Dataset class for loading document-term matrices from `.npy` files.

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


class TorchDiskDataset(Dataset):
    """
    A PyTorch Dataset class for loading document-term matrices from `.pt` files.

    The dataset can be initialized with either a path to a `.pt` file or
    a PyTorch tensor. When a file path is provided, the data is accessed
    lazily using memory mapping, which is useful for handling large datasets
    that do not fit entirely in (CPU) memory.
    The input `.pt` file should contain a single tensor with document-term
    matrix data.
    """

    def _validate_tensor_data(self, tensor_data: any) -> torch.Tensor:
        """Validate that the loaded data is a single tensor and return it.

        Args:
            tensor_data: Data loaded from `.pt` file.

        Returns:
            Validated tensor data.

        Raises:
            ValueError: If data is not a tensor.
        """
        if not isinstance(tensor_data, torch.Tensor):
            raise ValueError(
                f"File {self.data_path} must contain a single tensor, "
                f"got {type(tensor_data)}"
            )
        return tensor_data

    def __init__(
        self,
        data: str | Path,
        indices: Sequence[int] | None = None,
    ) -> None:
        """
        Args:
            data: Path to `.pt` file (str or Path).
            indices: Optional sequence of indices to use as valid indices.
        """
        self.data_path = Path(data)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        # Try loading with mmap first to get shape
        try:
            tensor_data = self._validate_tensor_data(
                torch.load(
                    self.data_path, map_location="cpu", weights_only=True, mmap=True
                )
            )
            self.shape = tuple(tensor_data.shape)
            self.mmap_supported = True
            self.mmap_data: torch.Tensor | None = None

        except RuntimeError:
            # Fallback to regular loading if mmap not supported
            tensor_data = self._validate_tensor_data(
                torch.load(self.data_path, map_location="cpu", weights_only=True)
            )
            self.shape = tuple(tensor_data.shape)
            self.data = tensor_data
            self.mmap_supported = False

        self.indices = indices or range(self.shape[0])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        real_idx = self.indices[idx]

        if self.mmap_supported:
            if self.mmap_data is None:
                self.mmap_data = torch.load(
                    self.data_path, map_location="cpu", weights_only=True, mmap=True
                )
            return self.mmap_data[real_idx]
        else:
            return self.data[real_idx]

    @property
    def num_terms(self) -> int:
        """Return vocabulary size (number of columns)."""
        return self.shape[1]
