# Generate data for distributed training
# Run:
# python examples/distributed_data.py

import os

import numpy as np

import tinytopics as tt


def main():
    n, m, k = 100_000, 100_000, 20
    data_path = "X.npy"

    if os.path.exists(data_path):
        print(f"Data already exists at {data_path}")
        return

    print("Generating synthetic data...")
    tt.set_random_seed(42)
    X, true_L, true_F = tt.generate_synthetic_data(n, m, k, avg_doc_length=256 * 256)

    print(f"Saving data to {data_path}")
    X_numpy = X.cpu().numpy()
    np.save(data_path, X_numpy)


if __name__ == "__main__":
    main()
