from typing import Tuple
from collections.abc import Sequence

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from .models import NeuralPoissonNMF


def poisson_nmf_loss(X: Tensor, X_reconstructed: Tensor) -> Tensor:
    """
    Compute the Poisson NMF loss function (negative log-likelihood).

    Args:
        X: Original document-term matrix.
        X_reconstructed: Reconstructed matrix from the model.

    Returns:
        The computed Poisson NMF loss.
    """
    epsilon: float = 1e-10
    return (
        X_reconstructed - X * torch.log(torch.clamp(X_reconstructed, min=epsilon))
    ).sum()


def fit_model(
    X: Tensor,
    k: int,
    num_epochs: int = 200,
    batch_size: int = 16,
    base_lr: float = 0.01,
    max_lr: float = 0.05,
    T_0: int = 20,
    T_mult: int = 1,
    weight_decay: float = 1e-5,
    device: torch.device | None = None,
) -> Tuple[NeuralPoissonNMF, Sequence[float]]:
    """
    Fit topic model using sum-to-one constrained neural Poisson NMF,
    optimized with AdamW and a cosine annealing with warm restarts scheduler.

    Args:
        X: Document-term matrix.
        k: Number of topics.
        num_epochs: Number of training epochs. Default is 200.
        batch_size: Number of documents per batch. Default is 16.
        base_lr: Minimum learning rate after annealing. Default is 0.01.
        max_lr: Starting maximum learning rate. Default is 0.05.
        T_0: Number of epochs until the first restart. Default is 20.
        T_mult: Factor by which the restart interval increases after each restart. Default is 1.
        weight_decay: Weight decay for the AdamW optimizer. Default is 1e-5.
        device: Device to run the training on. Defaults to CUDA if available, otherwise CPU.

    Returns:
        A tuple containing:
            - The trained NeuralPoissonNMF model
            - List of training losses for each epoch
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    n, m = X.shape

    model = NeuralPoissonNMF(n=n, m=m, k=k, device=device)
    optimizer = AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult, eta_min=base_lr
    )

    losses: Sequence[float] = []
    num_batches: int = n // batch_size

    with tqdm(total=num_epochs, desc="Training Progress") as pbar:
        for epoch in range(num_epochs):
            permutation = torch.randperm(n, device=device)
            epoch_loss: float = 0.0

            for i in range(num_batches):
                indices = permutation[i * batch_size : (i + 1) * batch_size]
                batch_X = X[indices, :]

                optimizer.zero_grad()
                X_reconstructed = model(indices)
                loss = poisson_nmf_loss(batch_X, X_reconstructed)
                loss.backward()

                optimizer.step()
                # Update per batch for cosine annealing with restarts
                scheduler.step(epoch + i / num_batches)

                epoch_loss += loss.item()

            epoch_loss /= num_batches
            losses.append(epoch_loss)
            pbar.set_postfix({"Loss": f"{epoch_loss:.4f}"})
            pbar.update(1)

    return model, losses
