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
    X, true_L, true_F = tt.generate_synthetic_data(
        n=n, m=m, k=k, avg_doc_length=256 * 256
    )

    print(f"Saving data to {data_path}")
    X_numpy = X.cpu().numpy()
    np.save(data_path, X_numpy)


if __name__ == "__main__":
    main()

import os

from accelerate import Accelerator
from accelerate.utils import set_seed

import tinytopics as tt


def main():
    accelerator = Accelerator()
    set_seed(42)
    k = 20
    data_path = "X.npy"

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"{data_path} not found. Run distributed_data.py first."
        )

    print(f"Loading data from {data_path}")
    X = tt.NumpyDiskDataset(data_path)

    # All processes should have the data before proceeding
    accelerator.wait_for_everyone()

    model, losses = tt.fit_model_distributed(X, k=k)

    # Only the main process should plot the loss
    if accelerator.is_main_process:
        tt.plot_loss(losses, output_file="loss.png")


if __name__ == "__main__":
    main()
