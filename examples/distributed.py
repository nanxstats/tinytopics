# Distributed training
# Run:
# accelerate config
# accelerate launch distributed.py

import os

import torch
import numpy as np
from accelerate.utils import set_seed
from tqdm.auto import tqdm

import tinytopics as tt
from tinytopics.fit_distributed import fit_model_distributed


def save_data_to_disk(X, data_path, chunk_size=10_000):
    """Save tensor data to disk in chunks using memory mapping.

    Args:
        X (torch.Tensor): Input tensor to save
        data_path (str): Path to save the .npy file
        chunk_size (int): Size of chunks to save at a time
    """
    n, m = X.shape
    # Create memory-mapped array
    array = np.lib.format.open_memmap(
        data_path, mode="w+", dtype=np.float32, shape=(n, m)
    )

    # Save in chunks
    n_chunks = n // chunk_size
    X_numpy = X.cpu().numpy()

    for i in tqdm(range(n_chunks), desc="Saving data chunks"):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        array[start_idx:end_idx] = X_numpy[start_idx:end_idx]

    array.flush()


def main():
    set_seed(42)
    n, m, k = 100_000, 100_000, 20
    data_path = "X.npy"

    if os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        X = torch.from_numpy(np.load(data_path))
    else:
        print("Generating synthetic data...")
        X, true_L, true_F = tt.generate_synthetic_data(
            n, m, k, avg_doc_length=256 * 256
        )
        print(f"Saving data to {data_path}")
        save_data_to_disk(X, data_path)
        X = torch.from_numpy(np.load(data_path))

    model, losses = fit_model_distributed(X, k=k)

    tt.plot_loss(losses, output_file="loss.png")


if __name__ == "__main__":
    main()
