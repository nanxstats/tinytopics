# Distributed training
# Run:
# accelerate config
# accelerate launch examples/distributed.py

import os

import numpy as np
from accelerate.utils import set_seed
from accelerate import Accelerator

import tinytopics as tt
from tinytopics.fit_distributed import fit_model_distributed


def main():
    accelerator = Accelerator()
    set_seed(42)
    n, m, k = 100_000, 100_000, 20
    data_path = "X.npy"

    # Only generate data on main process
    if accelerator.is_main_process:
        if os.path.exists(data_path):
            print(f"Data already exists at {data_path}")
        else:
            print("Generating synthetic data...")
            X, true_L, true_F = tt.generate_synthetic_data(
                n, m, k, avg_doc_length=256 * 256
            )

            print(f"Saving data to {data_path}")
            X_numpy = X.cpu().numpy()
            np.save(data_path, X_numpy)

    # Wait for main process to finish generating data
    accelerator.wait_for_everyone()

    print(f"Loading data from {data_path}")
    X = tt.NumpyDiskDataset(data_path)

    # Ensure all processes have the data before proceeding
    accelerator.wait_for_everyone()

    model, losses = fit_model_distributed(X, k=k)

    # Only the main process should plot the loss
    if accelerator.is_main_process:
        tt.plot_loss(losses, output_file="loss.png")


if __name__ == "__main__":
    main()
