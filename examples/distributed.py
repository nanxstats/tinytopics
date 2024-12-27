# Distributed training
# Run:
# accelerate config
# accelerate launch distributed.py

from accelerate.utils import set_seed

import tinytopics as tt
from tinytopics.fit_distributed import fit_model_distributed


def main():
    set_seed(42)
    n, m, k = 5000, 1000, 10
    X, true_L, true_F = tt.generate_synthetic_data(n, m, k, avg_doc_length=256 * 256)
    model, losses = fit_model_distributed(X, k=10)

    tt.plot_loss(losses, output_file="loss.png")


if __name__ == "__main__":
    main()
