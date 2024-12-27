import torch
from torch import Tensor


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
