# Distributed training
# Run:
# accelerate config
# accelerate launch examples/distributed.py

import os

from accelerate import Accelerator
from accelerate.utils import set_seed

import tinytopics as tt
from tinytopics.fit_distributed import fit_model_distributed


def main():
    accelerator = Accelerator()
    set_seed(42)
    k = 20
    data_path = "X.npy"

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file {data_path} not found. Run distributed_data.py first."
        )

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
