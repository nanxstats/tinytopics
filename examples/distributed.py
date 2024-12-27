# Distributed training
# Run:
# accelerate config
# accelerate launch distributed.py

import os

import torch
import numpy as np
from accelerate.utils import set_seed

import tinytopics as tt
from tinytopics.fit_distributed import fit_model_distributed


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
        np.save(data_path, X.cpu().numpy())

    model, losses = fit_model_distributed(X, k=k)

    tt.plot_loss(losses, output_file="loss.png")


if __name__ == "__main__":
    main()
