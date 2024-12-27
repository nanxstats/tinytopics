# Distributed training
# Run:
# accelerate config
# accelerate launch examples/distributed.py

import torch
import numpy as np
from accelerate.utils import set_seed
from accelerate import Accelerator
import tinytopics as tt
from tinytopics.fit_distributed import fit_model_distributed


def main():
    accelerator = Accelerator()
    set_seed(42)
    k = 20
    data_path = "X.npy"

    print(f"Loading data from {data_path}")
    X = torch.from_numpy(np.load(data_path))

    model, losses = fit_model_distributed(X, k=k)

    # Only the main process should plot the loss
    if accelerator.is_main_process:
        tt.plot_loss(losses, output_file="loss.png")


if __name__ == "__main__":
    main()
